import warnings
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import numpy.typing as npt
import matplotlib as mpl
from matplotlib import colormaps
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import seaborn as sns
from scipy.stats import sem
from scipy.ndimage import gaussian_filter1d
from statsmodels.stats.oneway import anova_oneway
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# from mayavi import mlab
from scipy.spatial import ConvexHull
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import statsmodels.api as sm
import statsmodels.formula.api as smf
from joblib import Parallel, delayed

from .preprocess import remove_groups_missing_conditions
from .statistics import compute_confidence_interval
from .spikes import get_spikes, compute_spike_rates
from .utils import create_rolling_window


def polar_histogram(
    x: npt.ArrayLike,
    bins: int = 16,
    density: bool = True,
    offset: float = 0,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """
    Create a polar histogram.

    Parameters
    ----------
    x : array-like
        Input data.
    bins : int, optional
        Number of bins to use for the histogram.
    density : bool, optional
        If True, the histogram is normalized to form a probability density.
    offset : float, optional
        Offset angle for the bars.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw the histogram onto.
    **kwargs : keyword arguments
        Additional arguments passed to the bar function.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object with the polar histogram.
    """

    # Create a new figure if no Axes object is provided
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})

    # Warp input to [-pi, pi]
    x = np.asarray(x)
    x = (x + np.pi) % (2 * np.pi) - np.pi

    # Compute histogram
    n, bins = np.histogram(x, bins=np.linspace(-np.pi, np.pi, bins + 1))
    widths = np.diff(bins)
    if density:
        area = n / x.size
        radius = (area / np.pi) ** 0.5
    else:
        radius = n
    ax.bar(bins[:-1], radius, align="edge", width=widths, **kwargs)
    ax.set_theta_offset(offset)
    if density:
        ax.set_yticks([])

    return ax


def plot_dropout(
    data: pd.DataFrame,
    unit: str,
    condition: str | list[str],
    group: str | list[str] | None = None,
    n_conditions: int | None = None,
    normalize: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Axes:

    # Create a new figure if no Axes object is provided
    if ax is None:
        _, ax = plt.subplots()

    # process inputs
    if not isinstance(condition, list):
        condition = [condition]

    if group is not None:
        groups = data.groupby(group)
    else:
        groups = [(None, data)]

    for grp, df in groups:
        # infer the number of conditional groups if not provided
        if n_conditions is None:
            n_conditions = df.groupby(condition).ngroups

        # get min. trial count per condition
        n_trials = (
            df.groupby(unit)[condition].value_counts().groupby(unit).min().to_dict()
        )

        # check for missing conditional groups
        for uid, df in df.groupby(unit):
            if df.groupby(condition).ngroups < n_conditions:
                n_trials[uid] = 0

        # plot dropout curve
        n_trials = np.array(list(n_trials.values()))
        thresholds = np.arange(max(n_trials) + 1)
        n_remaining = np.array([np.sum(n_trials >= t) for t in thresholds])
        if normalize:
            n_remaining = n_remaining / len(n_trials) * 100
        ax.plot(thresholds, n_remaining, marker="o", lw=2, label=grp)

    # misc. settings
    if group is not None:
        ax.legend(frameon=False)
    ax.set(
        xlabel="Min. Trials per Condition",
        ylabel="Units Remaining (%)" if normalize else "Units Remaining",
    )
    sns.despine(ax=ax)

    return ax


def remove_duplicate_legend_entries(ax: plt.Axes | None = None) -> plt.Axes:
    """
    Remove duplicate entries from a figure legend.

    This function removes duplicate labels from the legend of a given
    Axes object. If no Axes object is provided, the current Axes object
    is used.

    Parameters
    ----------
    ax : `matplotlib.pyplot.Axes` or None, default=None
        The Axes object containing the legend. If not provided, the current Axes object is used.

    Returns
    -------
    ax : `matplotlib.pyplot.Axes`
        The Axes object with the legend entries removed.
    """

    if ax is None:
        ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    return ax


def plot_spikes(
    spikes: list[npt.ArrayLike],
    *,
    group_labels: npt.ArrayLike | None = None,
    group_order: npt.ArrayLike | None = None,
    stats: npt.ArrayLike | None = None,
    ascending: bool = False,
    plot_stats: bool = False,
    ax: plt.Axes | None = None,
    cmap: str | npt.ArrayLike | dict | mpl.colors.Colormap = "tab10",
) -> plt.Axes:
    """
    Plot spike rasters.

    This function plots a spike raster plot with optional grouping and statistical overlay. If groups are specified,
    spikes are colored according to their group. If stats are provided, they are plotted alongside the spike raster.

    Parameters
    ----------
    spikes : list of array-like
        A list of arrays where each array contains spike times for a unit.
    group_labels : array-like or None, default=None
        An array specifying the group label for each unit. Must have the same length as `spikes`.
        If None, no grouping is used.
    group_order : array-like or None, default=None
        An array specifying the order of groups. Must contain all unique group labels.
        Ignored if group_labels is None. If None, groups are ordered by their first appearance in group_labels.
    stats : array-like or None, default=None
        An array of statistics to plot alongside the spike raster. Must have the same length as `spikes`.
        If None, no statistics are plotted.
    ascending : bool, default=False
        If True, sorts the spikes and statistics in ascending order based on stats. If False, sorts in descending order.
    plot_stats : bool, default=False
        If True, plots the statistics alongside the spike raster. Only relevant if the statistics are time-based.
    ax : `matplotlib.pyplot.Axes` or None, default=None
        A matplotlib Axes object. If None, a new figure and axes are created.

    Returns
    -------
    ax : `matplotlib.pyplot.Axes`
        The Axes object containing the plot.
    """

    # Convert parameters to numpy arrays
    spikes = np.asarray(spikes, dtype=object)
    if group_labels is not None:
        group_labels = np.asarray(group_labels)
        if group_order is None:
            group_order = np.unique(group_labels)
    if stats is not None:
        stats = np.asarray(stats)

    # Create a new figure if no Axes object is provided
    if ax is None:
        _, ax = plt.subplots()

    # Sort spikes by stats
    if stats is not None:
        ind = np.argsort(stats)
        if not ascending:
            ind = ind[::-1]
        spikes = spikes[ind]
        stats = stats[ind]
        if group_labels is not None:
            group_labels = group_labels[ind]

    # Plot spikes
    # linelengths = len(spikes) / (ax.bbox.height * 1.5)
    linelengths = len(spikes) / (ax.bbox.height / 2)
    linewidths = ax.bbox.width / 500

    if group_labels is None:
        ax.eventplot(spikes, linelengths=linelengths, linewidths=linewidths)
        if plot_stats and stats is not None:
            ax.plot(stats, np.arange(len(spikes)), color="black", lw=1.5, alpha=0.75)
    else:
        # order by group
        group_inds = [np.nonzero(group_labels == grp)[0] for grp in group_order[::-1]]
        group_lens = [len(ind) for ind in group_inds]
        group_inds = np.concatenate(group_inds)
        spikes = spikes[group_inds]
        if stats is not None:
            stats = stats[group_inds]

        # Define colors and legend handles
        if isinstance(cmap, dict):
            cmap = [cmap[grp] for grp in group_order[::-1]]
        else:
            cmap = sns.color_palette(cmap, len(group_order))[::-1]
        colors = np.concatenate(
            [[cmap[i]] * grp_len for i, grp_len in enumerate(group_lens) if grp_len > 0]
        )
        handles = [
            plt.Line2D(
                [0], [0], marker="|", markersize=8, markeredgewidth=3.5, c=c, ls="None"
            )
            for c in cmap
        ]

        # Plots
        ax.eventplot(
            spikes, colors=colors, linelengths=linelengths, linewidths=linewidths
        )
        ax.legend(
            handles[::-1],
            group_order,
            loc="upper left",
            bbox_to_anchor=(1, 1),
            title=None,
            frameon=False,
        )

        if plot_stats and stats is not None:
            start_ind = 0
            for _, grp_len in enumerate(group_lens):
                end_ind = start_ind + grp_len
                ax.plot(
                    stats[start_ind:end_ind],
                    np.arange(grp_len) + start_ind,
                    color="black",
                    lw=1.5,
                    alpha=0.75,
                )
                start_ind = end_ind

    # Misc. settings
    ax.set_ylim(0, len(spikes))

    return ax


def plot_PSTH(
    spike_rates: npt.ArrayLike,
    timestamps: npt.ArrayLike | None = None,
    *,
    group_labels: npt.ArrayLike | None = None,
    group_order: npt.ArrayLike | None = None,
    error_type: str | tuple[str, float] | None = "se",
    sig_test: bool = False,
    alpha: float = 0.05,
    raw_labels: npt.ArrayLike | None = None,
    nuisance: npt.ArrayLike | None = None,
    smooth_width: float | None = None,
    ax: plt.Axes | None = None,
    cmap: str | npt.ArrayLike | dict | mpl.colors.Colormap = "tab10",
    n_permutes: int = 0,
    n_jobs: int = -1,
    **kwargs,
) -> plt.Axes:
    """
    Plot peristimulus time histogram (PSTH).

    This function plots a peristimulus time histogram (PSTH) with optional grouping of
    spike rates and the addition of error bars. If groups are specified, each group's
    spike rates are plotted separately with different colors.

    Parameters
    ----------
    spike_rates : array-like
        A 2D array where each row contains the spike rates for a unit over time.
    timestamps : array-like or None, default=None
        An array of time points at which spike rates are measured. If None, the indices of `spike_rates` are used.
    group_labels : array-like or None, default=None
        An array specifying the group label for each unit. Must have the same length as `spike_rates`.
        If None, no grouping is used.
    group_order : array-like or None, default=None
        An array specifying the order of groups. Must contain all unique group labels.
        Ignored if group_labels is None. If None, groups are ordered by their first appearance in group_labels.
    error_type : str or tuple of (str, float) or None, default=("ci", 95)
        Specifies the type of error intervals to plot. If a string, it must be one of 'std', 'sem', or 'ci' (confidence interval).
        If a tuple, the first element must be one of 'std', 'sem', or 'ci', and the second element is a scaling factor for the error bars.
        If None, no error intervals are plotted.
    ax : `matplotlib.pyplot.Axes` or None, default=None
        A matplotlib Axes object. If None, a new figure and axes are created.

    Returns
    -------
    ax : `matplotlib.pyplot.Axes`
        The Axes object containing the plot.
    """

    # Preprocess data
    spike_rates = np.asarray(spike_rates)
    if timestamps is None:
        timestamps = np.arange(spike_rates.shape[1])
    timestamps = np.asarray(timestamps)
    if smooth_width is not None:
        smooth_width /= timestamps[1] - timestamps[0]
        to_plot = gaussian_filter1d(spike_rates, sigma=smooth_width)
    else:
        to_plot = spike_rates.copy()
    if nuisance is not None:
        for v in np.unique(nuisance):
            mask = nuisance == v
            to_plot[mask] = (to_plot[mask] - to_plot[mask].mean(axis=0)) / np.sqrt(
                to_plot[mask].mean(axis=0) + 1e-6
            )
    if group_labels is not None and group_order is None:
        group_order = np.unique(group_labels)

    # Collect into dataframe
    n, T = to_plot.shape
    sns.lineplot(
        data={
            "time": np.tile(timestamps, n),
            "rate": to_plot.flatten(),
            "label": np.repeat(group_labels, T) if group_labels is not None else None,
        },
        x="time",
        y="rate",
        hue="label",
        hue_order=group_order,
        errorbar=error_type,
        palette=cmap,
        ax=ax,
        **kwargs
    )

    # Plot significance
    if group_labels is not None and sig_test:
        y_min, y_max = ax.get_ylim()
        if raw_labels is None:
            raw_labels = group_labels

        def _helper(permute, mask=None):
            srs = np.random.permutation(spike_rates) if permute else spike_rates
            pvals = np.zeros(srs.shape[1])
            for i, rate in enumerate(srs.T):
                if mask is not None and not mask[i]:
                    continue
                if nuisance is not None:
                    model = smf.glm(
                        "y ~ x + z",
                        data={
                            "y": rate,
                            "x": raw_labels,
                            "z": nuisance,
                        },
                        family=sm.families.Poisson(),
                    )
                else:
                    model = smf.glm(
                        "y ~ x",
                        data={"y": rate, "x": raw_labels},
                        family=sm.families.Poisson(),
                    )
                try:
                    model = model.fit(scale="X2", disp=0, use_t=True)
                    wald = model.wald_test_terms(scalar=True)
                    pvals[i] = wald.table.loc["x", "pvalue"]
                except:
                    pvals[i] = np.nan
            return pvals

        # compute p-values
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            pvals = _helper(False)
        mask = pvals < alpha
        if n_permutes > 0:
            null = list(
                Parallel(n_jobs=n_jobs)(
                    delayed(_helper)(True, mask) for _ in range(n_permutes)
                )
            )
            null = np.array(null)
            pvals = (pvals >= null).mean(axis=0)

        # annotate significance
        mask = pvals < alpha
        m = mask.astype(int)
        starts = np.where(np.diff(m) == 1)[0] + 1
        ends = np.where(np.diff(m) == -1)[0] + 1
        if mask[0]:
            starts = np.r_[0, starts]
        if mask[-1]:
            ends = np.r_[ends, len(mask)]
        for s, e in zip(starts, ends):
            ax.plot(
                [timestamps[s], timestamps[e - 1]],
                [y_max, y_max],
                c="r",
                alpha=0.75,
                lw=4,
            )
        line = ax.add_line(mpl.lines.Line2D([0], [0], c="r", lw=3, alpha=0.75))
        handles, labels = ax.get_legend_handles_labels()
        handles.append(line)
        labels.append(f"p < {alpha}")
        ax.legend(handles, labels)

        # reset y-limits
        ax.set_ylim(y_min)

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title=None, frameon=False)
    return ax


def plot_spikes_with_PSTH(
    spikes: npt.ArrayLike,
    alignments: npt.ArrayLike,
    *,
    window: tuple[float, float] = (-1, 2.5),
    bl_alignments: npt.ArrayLike | None = None,
    bl_window: tuple[float, float] | None = None,
    step: float = 0.025,
    width: float = 0.5,
    group_labels: npt.ArrayLike | None = None,
    group_order: npt.ArrayLike | None = None,
    nuisance: npt.ArrayLike | None = None,
    stats: npt.ArrayLike | None = None,
    ascending: bool = False,
    plot_stats: bool = False,
    error_type: str | tuple[str, float] | None = ("ci", 95),
    sig_test: bool = False,
    alpha: float = 0.05,
    raw_labels: npt.ArrayLike | None = None,
    smooth_width: float | None = None,
    axes: tuple[plt.Axes, plt.Axes] | None = None,
    cmap: str | npt.ArrayLike | dict | mpl.colors.Colormap = "tab10",
    unit_conversion: float = 1,
    alignment: str = "center",
    n_permutes: int = 0,
    n_jobs: int = -1,
    **kwargs,
) -> tuple[plt.Axes, plt.Axes]:
    """
    Plot spike rasters and peristimulus time histograms (PSTHs) together.

    This function plots a spike raster plot and a PSTH with optional grouping and statistical overlay.
    If groups are specified, spikes are colored according to their group. If stats are provided, they are plotted
    alongside the spike raster. Error bars can be added to the PSTH.

    Parameters
    ----------
    spikes : array-like
        A list of all spike times.
    alignments : array-like
        A list of alignment times for each spike raster.
    window : tuple of (float, float)
        A tuple specifying the window around the alignment times to plot the PSTH.
    step : float, default=0.025
        The time step to use for the PSTH.
    width : float, default=0.5
        The width of the PSTH window.
    group_labels : array-like or None, default=None
        An array specifying the group label for each unit. Must have the same length as `spikes`.
        If None, no grouping is used.
    group_order : array-like or None, default=None
        An array specifying the order of groups. Must contain all unique group labels.
        Ignored if group_labels is None. If None, groups are ordered by their first appearance in group_labels.
    stats : array-like or None, default=None
        An array of statistics to plot alongside the spike raster. Must have the same length as `spikes`.
        If None, no statistics are plotted.
    ascending : bool, default=False
        If True, sorts the spikes and statistics in ascending order based on stats. If False, sorts in descending order.
    plot_stats : bool, default=False
        If True, plots the statistics alongside the spike raster. Only relevant if the statistics are time-based.
    error_type : str or tuple of (str, float) or None, default=("ci", 95)
        Specifies the type of error intervals to plot. If a string, it must be one of 'std', 'sem', or 'ci' (confidence interval).
        If a tuple, the first element must be one of 'std', 'sem', or 'ci', and the second element is a scaling factor for the error bars.
        If None, no error intervals are plotted.
    axes : tuple of `matplotlib.pyplot.Axes` or None, default=None
        A tuple of two Axes objects. If None, new figures and axes are created.
    cmap : str or array-like or dict or `matplotlib.colors.Colormap`, default='tab10'
        The name of the colormap to use for coloring the groups.
    unit_conversion : float, default=1
        The conversion factor to apply to the spike times. Useful for converting spike times to seconds.

    Returns
    -------
    axes : tuple of `matplotlib.pyplot.Axes`
        A tuple of Axes objects containing the spike raster and PSTH plots.
    """

    # Create a new figure if no Axes objects are provided
    if axes is None:
        _, axes = plt.subplots(
            2,
            1,
            figsize=(4, 6),
            height_ratios=[3, 1],
            sharex=True,
            gridspec_kw={"hspace": 0.05},
        )
    axes = np.ravel(axes)

    # Process inputs
    spikes = np.asarray(spikes)
    alignments = np.asarray(alignments)

    # Compute per-trial spikes
    trial_spikes = get_spikes(
        spikes, alignments + window[0], alignments + window[1], alignments
    )

    # Compute per-trial spike rates
    w_starts, w_ends, w_refs = create_rolling_window(
        window[0], window[1], step, width, alignment=alignment
    )
    trial_sr = np.array(
        [
            compute_spike_rates(
                spikes, w_starts + ts, w_ends + ts, unit_conversion=unit_conversion
            )
            for ts in alignments
        ]
    )

    # baseline normalize
    if bl_window is not None:
        if bl_alignments is None:
            bl_alignments = alignments
        bl_sr = np.asarray(
            compute_spike_rates(
                spikes,
                bl_window[0] + bl_alignments,
                bl_window[1] + bl_alignments,
                unit_conversion=unit_conversion,
            )
        )
        trial_sr = trial_sr - bl_sr[:, None] + bl_sr.mean()

    # Plot spike raster
    plot_spikes(
        trial_spikes,
        group_labels=group_labels,
        group_order=group_order,
        stats=stats,
        ascending=ascending,
        plot_stats=plot_stats,
        ax=axes[0],
        cmap=cmap,
    )

    # Plot PSTH
    plot_PSTH(
        trial_sr,
        w_refs,
        group_labels=group_labels,
        raw_labels=raw_labels,
        nuisance=nuisance,
        group_order=group_order,
        error_type=error_type,
        sig_test=sig_test,
        alpha=alpha,
        smooth_width=smooth_width,
        ax=axes[1],
        cmap=cmap,
        n_permutes=n_permutes,
        n_jobs=n_jobs,
        **kwargs
    )

    # Misc. settings
    axes[0].set_xlim(*window)
    axes[1].set_xlim(*window)
    axes[0].get_xaxis().set_visible(False)
    axes[0].set_ylabel("Trial" if stats is None else "Trial (sorted)")
    if group_labels is not None:
        axes[1].get_legend().remove()

    if sig_test:
        line = axes[1].add_line(mpl.lines.Line2D([0], [0], c="r", lw=3, alpha=0.75))
        axes[1].legend(
            [line],
            ["Ad-Hoc\n(p < 0.05)"],
            loc="upper left",
            bbox_to_anchor=(1, 1),
            title=None,
            frameon=False,
        )
    axes[1].set_xlabel("Time [s]")

    ylabel = "Firing Rate"
    if bl_window is not None:
        ylabel = "Bln. Firing Rate"
    if nuisance is None:
        ylabel += " (Hz)"
    else:
        ylabel += " (resid.)"
    axes[1].set_ylabel(ylabel)

    sns.despine(ax=axes[0])
    sns.despine(ax=axes[1])

    return axes


def plot_metrics(
    data: pd.DataFrame,
    metric: str = "scores",
    *,
    null: pd.DataFrame | None = None,
    x_group: str | None = None,
    y_group: str | None = None,
    x_order: list[str] | None = None,
    y_order: list[str] | None = None,
    y_emph: str | list[str] | None = None,
    quantiles: tuple[float, float] = (0.05, 0.95),
    annotate: bool = False,
    use_p: bool = False,
    marker: str = "o",
    chance: float | None = 0.5,
    palette: str | npt.ArrayLike | dict | mpl.colors.Colormap = "tab10",
    markersize: int = 10,
    linewidth: float = 3,
    ax: plt.Axes | None = None,
    **kwargs,
):
    q_low, q_high = sorted(quantiles)

    if x_group is not None:
        x_order = list(pd.unique(data[x_group])) if x_order is None else list(x_order)

    if annotate and (null is None or x_group is None):
        raise ValueError("annotate=True requires both null and x_group.")
    if y_emph is not None and y_group is None:
        raise ValueError("y_emph requires y_group.")

    if ax is None:
        _, ax = plt.subplots(figsize=(1 if x_group is None else len(x_order), 6))

    # ------------------------------------------------------------------
    # Colors / emphasized groups
    # ------------------------------------------------------------------

    all_ys = list(pd.unique(data[y_group])) if y_group is not None else []
    emph_order = (
        list(all_ys if y_order is None else y_order) if y_group is not None else [None]
    )
    active = list(emph_order)

    if y_emph is not None:
        selected = (
            set(data.loc[data[y_emph].fillna(False).astype(bool), y_group])
            if isinstance(y_emph, str)
            else set(y_emph)
        )
        active = [g for g in emph_order if g in selected]
        if not active:
            raise ValueError("y_emph did not select any y_group conditions.")

    colors = None
    if y_group is not None:
        colors = {g: "grey" for g in all_ys}
        if isinstance(palette, dict):
            colors.update({g: palette[g] for g in active})
        elif isinstance(palette, mpl.colors.Colormap):
            xs = np.linspace(0, 1, max(2, len(active)))
            colors.update({g: palette(xs[i]) for i, g in enumerate(active)})
        else:
            colors.update(dict(zip(active, sns.color_palette(palette, len(active)))))

    # ------------------------------------------------------------------
    # Observed values
    # ------------------------------------------------------------------
    point_kws = dict(
        x=x_group,
        y=metric,
        hue=y_group,
        order=x_order,
        errorbar=None,
        markers=marker,
        markersize=markersize,
        linewidth=linewidth,
        markeredgecolor="black",
        ax=ax,
    )

    if y_emph is not None:
        # Muted groups come from the data, not y_order. y_order only
        # controls which emphasized groups are shown and in what order.
        muted = [g for g in all_ys if g not in active]

        if muted:
            muted_point_kws = {**point_kws, "markeredgecolor": "grey"}
            sns.pointplot(
                data=data[data[y_group].isin(muted)],
                hue_order=muted,
                palette=colors,
                linestyle="",
                alpha=0.25,
                legend=False,
                zorder=100,
                **muted_point_kws,
                **({"dodge": kwargs["dodge"]} if "dodge" in kwargs else {}),
            )

        plot_data = data[data[y_group].isin(active)]
        hue_order = active
        plot_palette = colors
    else:
        plot_data = data
        hue_order = emph_order if y_group is not None else None
        plot_palette = colors or palette

    sns.pointplot(
        data=plot_data,
        hue_order=hue_order,
        palette=plot_palette,
        zorder=200,
        **point_kws,
        **kwargs,
    )

    # ------------------------------------------------------------------
    # Current plotted range
    # ------------------------------------------------------------------
    ys = [
        np.asarray(line.get_ydata(), float)
        for line in ax.lines
        if len(line.get_ydata())
    ]
    ys = np.concatenate(ys or [np.array([0.0, 1.0])])
    ys = ys[np.isfinite(ys)]
    y_min, y_max = ys.min(), ys.max()

    # ------------------------------------------------------------------
    # Null distributions
    # ------------------------------------------------------------------
    null_dists = {}
    null_bounds = {}

    if null is not None:
        old_xlim = ax.get_xlim()
        label = f"Perm. Null\n({q_low:.0%}-{q_high:.0%} pctl.)"

        groups = (
            [(0, None, null[metric])]
            if x_group is None
            else [
                (x, g, null.loc[null[x_group].eq(g), metric])
                for x, g in enumerate(x_order)
            ]
        )

        for i, (x, g, values) in enumerate(groups):
            values = values.dropna()
            if values.empty:
                continue

            null_dists[g] = values.to_numpy(float)
            low, high = values.quantile([q_low, q_high])
            null_bounds[g] = (float(low), float(high))

            ax.fill_between(
                [x - 0.25, x + 0.25],
                [low, low],
                [high, high],
                color="black",
                alpha=0.25,
                zorder=0,
                label=label if i == 0 else None,
            )

            y_min = min(y_min, low)
            y_max = max(y_max, high)

        ax.set_xlim(old_xlim)

    # ------------------------------------------------------------------
    # Chance
    # ------------------------------------------------------------------
    if chance is not None:
        ax.axhline(
            chance,
            color="red",
            linestyle="--",
            zorder=0,
            label="Chance",
        )
        y_min = min(y_min, chance)
        y_max = max(y_max, chance)

    # ------------------------------------------------------------------
    # Significance annotations
    # ------------------------------------------------------------------
    if annotate:
        estimator = kwargs.get("estimator", "mean")

        def estimate(values):
            return float(
                estimator(values) if callable(estimator) else values.agg(estimator)
            )

        def significance(obs, dist, bounds):
            if use_p:
                p = (np.sum(dist <= obs) + 1) / (len(dist) + 1)
                return "*" if p <= q_low or p >= q_high else ""
            low, high = bounds
            return "*" if obs <= low or obs >= high else ""

        span = y_max - y_min
        if span <= 0:
            span = max(abs(y_max), 1.0) * 0.1

        offset = 0.10 * span
        spacing = 0.06 * span
        annotation_tops = []

        for x, xv in enumerate(x_order):
            if xv not in null_dists:
                continue

            dist = null_dists[xv]
            bounds = null_bounds[xv]
            estimates = {}
            sigs = []

            for g in active:
                mask = data[x_group].eq(xv)
                if g is not None:
                    mask &= data[y_group].eq(g)

                values = data.loc[mask, metric].dropna()
                if values.empty:
                    continue

                obs = estimate(values)
                estimates[g] = obs

                stars = significance(obs, dist, bounds)
                if stars:
                    sigs.append(("black" if g is None else colors[g], stars))

            if not estimates or not sigs:
                continue

            # Clear observed points, null interval, and chance line.
            highest = max(max(estimates.values()), bounds[1])
            if chance is not None:
                highest = max(highest, chance)

            for level, (color, stars) in enumerate(reversed(sigs)):
                y = highest + offset + level * spacing
                ax.text(
                    x,
                    y,
                    stars,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=16,
                    fontweight="bold",
                    clip_on=False,
                    zorder=1000,
                )
                annotation_tops.append(y)

        if annotation_tops:
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom, max(top, max(annotation_tops) + spacing))

    # ------------------------------------------------------------------
    # Legend / cleanup
    # ------------------------------------------------------------------
    handles, labels = ax.get_legend_handles_labels()
    legend = dict(zip(labels, handles))

    if legend:
        ax.legend(
            legend.values(),
            legend.keys(),
            loc="upper left",
            bbox_to_anchor=(1, 1),
            frameon=False,
        )

    sns.despine(ax=ax)
    return ax


def plot_metrics_continuous(
    data: pd.DataFrame,
    metric: str = "scores",
    *,
    null: pd.DataFrame | None = None,
    ts_group: str | None = None,
    y_group: str | None = None,
    y_order: list[str] | None = None,
    y_emph: str | list[str] | None = None,
    quantiles: tuple[float, float] = (0.05, 0.95),
    annotate: bool = False,
    use_p: bool = False,
    chance: float | None = 0.5,
    min_samples: int = 1,
    palette: str | npt.ArrayLike | dict | mpl.colors.Colormap = "tab10",
    linewidth: float = 3,
    smooth_width: float | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
):
    q_low, q_high = sorted(quantiles)

    if not 0 <= q_low <= q_high <= 1:
        raise ValueError("quantiles must be between 0 and 1.")
    if ts_group is None:
        raise ValueError("ts_group is required.")
    if annotate and null is None:
        raise ValueError("annotate=True requires null.")
    if y_emph is not None and y_group is None:
        raise ValueError("y_emph requires y_group.")

    data = data.sort_values(by=ts_group)
    orig_data = data.copy()
    if smooth_width is not None:
        data[metric] = data.groupby(y_group)[metric].transform(
            gaussian_filter1d, sigma=smooth_width
        )

    times = np.sort(data[ts_group].dropna().unique()).astype(float)
    if not len(times) or not np.isfinite(times).all():
        raise ValueError("ts_group must contain finite numeric timepoints.")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    all_ys = list(pd.unique(data[y_group])) if y_group is not None else []
    emph_order = (
        list(all_ys if y_order is None else y_order) if y_group is not None else [None]
    )
    active = list(emph_order)

    if y_emph is not None:
        selected = (
            set(data.loc[data[y_emph].fillna(False).astype(bool), y_group])
            if isinstance(y_emph, str)
            else set(y_emph)
        )
        active = [g for g in emph_order if g in selected]
        if not active:
            raise ValueError("y_emph did not select any y_group conditions.")

    colors = None
    if y_group is not None:
        colors = {g: "grey" for g in all_ys}
        if isinstance(palette, dict):
            colors.update({g: palette[g] for g in active})
        elif isinstance(palette, mpl.colors.Colormap):
            xs = np.linspace(0, 1, max(2, len(active)))
            colors.update({g: palette(xs[i]) for i, g in enumerate(active)})
        else:
            colors.update(dict(zip(active, sns.color_palette(palette, len(active)))))

    line_kws = dict(
        x=ts_group,
        y=metric,
        hue=y_group,
        errorbar=None,
        linewidth=linewidth,
        sort=True,
        ax=ax,
    )
    line_kws.update(kwargs)

    if y_emph is not None:
        # Muted groups come from the data, not y_order. y_order only
        # controls which emphasized groups are shown and in what order.
        muted = [g for g in all_ys if g not in active]
        if muted:
            muted_kws = {
                **line_kws,
                "hue_order": muted,
                "palette": colors,
                "alpha": 0.25,
                "legend": False,
                "zorder": 100,
            }
            sns.lineplot(
                data=data[data[y_group].isin(muted)],
                **muted_kws,
            )
        plot_data = data[data[y_group].isin(active)]
        hue_order = active
    else:
        plot_data = data
        hue_order = emph_order if y_group is not None else None

    first_line = len(ax.lines)
    sns.lineplot(
        data=plot_data,
        **{
            **line_kws,
            "hue_order": hue_order,
            "palette": colors if colors is not None else palette,
            "zorder": kwargs.get("zorder", 200),
        },
    )

    observed_lines = ax.lines[first_line:]
    single_color = next(
        (line.get_color() for line in observed_lines if len(line.get_ydata())),
        kwargs.get("color", "black"),
    )

    ys = [
        np.asarray(line.get_ydata(), dtype=float)
        for line in ax.lines
        if len(line.get_ydata())
    ]
    ys = np.concatenate(ys) if ys else np.array([])
    ys = ys[np.isfinite(ys)]
    y_min, y_max = (ys.min(), ys.max()) if ys.size else (0.0, 1.0)

    dist = np.array([], dtype=float)
    bounds = None

    if null is not None:
        dist = null[metric].dropna().to_numpy(dtype=float)
        dist = dist[np.isfinite(dist)]

        if dist.size:
            low, high = np.quantile(dist, [q_low, q_high])
            bounds = (low, high)
            ax.axhspan(
                low,
                high,
                color="black",
                alpha=0.25,
                zorder=0,
                label=f"Perm. Null\n({q_low:.0%}-{q_high:.0%} pctl.)",
            )
            y_min = min(y_min, low)
            y_max = max(y_max, high)

    if chance is not None:
        ax.axhline(
            chance,
            color="red",
            linestyle="--",
            zorder=0,
            label="Chance",
        )
        y_min = min(y_min, chance)
        y_max = max(y_max, chance)

    if annotate and bounds is not None:
        estimator = kwargs.get("estimator", "mean")
        if estimator is None:
            raise ValueError("annotate=True requires an estimator.")

        def estimate(values):
            values = values.dropna()
            if values.empty:
                return np.nan
            return float(
                estimator(values) if callable(estimator) else values.agg(estimator)
            )

        def significant(obs):
            if use_p:
                p = (np.sum(dist <= obs) + 1) / (len(dist) + 1)
                return p <= q_low or p >= q_high
            return obs <= bounds[0] or obs >= bounds[1]

        def singleton_interval(i):
            if len(times) == 1:
                return times[i] - 0.5, times[i] + 0.5
            left_step = times[i] - times[i - 1] if i else times[1] - times[0]
            right_step = (
                times[i + 1] - times[i] if i < len(times) - 1 else times[-1] - times[-2]
            )
            return times[i] - left_step / 2, times[i] + right_step / 2

        span = y_max - y_min
        if span <= 0:
            span = max(abs(y_max), 1.0) * 0.1

        offset = 0.10 * span
        spacing = 0.06 * span
        annotation_tops = []
        level = 0

        for g in reversed(active):
            group_data = (
                orig_data if y_group is None else orig_data[orig_data[y_group].eq(g)]
            )
            estimates = (
                group_data.groupby(ts_group)[metric]
                .agg(estimate)
                .reindex(times)
                .to_numpy(dtype=float)
            )
            sig = np.array(
                [np.isfinite(obs) and significant(obs) for obs in estimates],
                dtype=bool,
            )

            indices = np.flatnonzero(sig)
            if not indices.size:
                continue

            runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
            runs = [run for run in runs if len(run) >= min_samples]
            if not runs:
                continue
            
            segments = [
                (
                    singleton_interval(run[0])
                    if len(run) == 1
                    else (times[run[0]], times[run[-1]])
                )
                for run in runs
            ]

            y = y_max + offset + level * spacing
            ax.hlines(
                y=np.full(len(segments), y),
                xmin=[start for start, _ in segments],
                xmax=[end for _, end in segments],
                colors=single_color if y_group is None else colors[g],
                linewidth=linewidth * 1.5,
                clip_on=False,
                zorder=1000,
            )
            annotation_tops.append(y)
            level += 1

        if annotation_tops:
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom, max(top, max(annotation_tops) + spacing))

    handles, labels = ax.get_legend_handles_labels()
    legend = dict(zip(labels, handles))
    if legend:
        ax.legend(
            legend.values(),
            legend.keys(),
            loc="upper left",
            bbox_to_anchor=(1, 1),
            frameon=False,
        )

    sns.despine(ax=ax)
    return ax


# def plot_projection(
#     data: pd.DataFrame,
#     unit: str,
#     response: str,
#     condition: str | list[str],
#     cmap: str = "tab10",
#     interactive: bool = False,
#     ax: plt.Axes | None = None,
#     n_jobs: int | None = None,
# ) -> plt.Axes:
#     """
#     Plot a 3D projection of the data using PCA.

#     This function plots a 3D projection of the data using PCA. Each point in the plot represents the mean response of
#     a unit to different conditions. The points are colored based on the conditions.

#     Parameters
#     ----------
#     data : pd.DataFrame
#         The data to plot.
#     unit : str
#         The column name of the unit identifier.
#     response : str
#         The column name of the response variable.
#     condition : str or list of str
#         The column name(s) of the condition(s) to plot.
#     cmap : str, default='tab10'
#         The name of the colormap to use for coloring the conditions.
#     interactive : bool, default=False
#         If True, the plot is displayed interactively using Mayavi.
#     ax : `matplotlib.pyplot.Axes` or None, default=None
#         A matplotlib Axes object. If None, a new figure and axes are created.

#     Returns
#     -------
#     ax : `matplotlib.pyplot.Axes`
#         The Axes object containing the plot.
#     """

#     # check inputs
#     if isinstance(condition, str):
#         condition = [condition]
#     cmap = colormaps.get_cmap(cmap)

#     # create a new figure if no Axes object is provided
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(8, 8))

#     # process data
#     data = remove_groups_missing_conditions(data, unit, condition)
#     unit_std = data.groupby(unit)[response].std()
#     to_remove = unit_std[unit_std == 0].index
#     data = data[~data[unit].isin(to_remove)]

#     # normalize data
#     unit_mean = data.groupby(unit)[response].mean()
#     unit_std = data.groupby(unit)[response].std()
#     data[response] = (data[response] - data[unit].map(unit_mean)) / data[unit].map(
#         unit_std
#     )

#     # aggregate population data and apply MDS transformation
#     proj = MDS(n_components=3, n_init=64, max_iter=2400, n_jobs=n_jobs)
#     pop_mean = (
#         data.groupby([unit] + condition)[response]
#         .mean()
#         .groupby(condition)
#         .agg(list)
#         .reset_index()
#     )
#     X_mean = proj.fit_transform(np.stack(pop_mean[response]))

#     # generate plotting styles
#     y = pop_mean[condition].to_numpy(dtype=str)
#     offset, colors_lv1 = 0, None

#     # define color for first level of condition (if applicable)
#     if len(condition) > 1:
#         u_lv1, _ = np.unique(y[:, 0], return_inverse=True)
#         offset = len(u_lv1)
#         colors_lv1 = {u: cmap(i) for i, u in enumerate(u_lv1)}

#     # define color for all conditions
#     start = 0 if len(condition) == 1 else 1
#     _, l_all = np.unique(y[:, start:], return_inverse=True, axis=0)
#     colors_all = [cmap(l + offset) for l in l_all]

#     # set up the figure
#     fig = mlab.figure(size=(1000, 1000))

#     xmin, xmax = np.min(X_mean[:, 0]), np.max(X_mean[:, 0])
#     ymin, ymax = np.min(X_mean[:, 1]), np.max(X_mean[:, 1])
#     zmin, zmax = np.min(X_mean[:, 2]), np.max(X_mean[:, 2])
#     max_val = max(xmax - xmin, ymax - ymin, zmax - zmin)

#     # plot edges
#     tube_radius = 0.015 * max_val
#     if len(condition) > 1:
#         edge_sets = defaultdict(list)
#         for i, j in combinations(range(len(y)), 2):
#             li, lj = y[i], y[j]
#             Xi, Xj = X_mean[i], X_mean[j]
#             if li[0] != lj[0] and np.all(li[1:] == lj[1:]):
#                 mlab.plot3d(
#                     [Xi[0], Xj[0]],
#                     [Xi[1], Xj[1]],
#                     [Xi[2], Xj[2]],
#                     tube_radius=tube_radius,
#                     color=(1, 1, 1),
#                     opacity=0.25,
#                 )
#             # elif li[0] == lj[0] and np.sum(li[1:] == lj[1:]) == 1:
#             elif li[0] == lj[0]:
#                 edge_sets[li[0]].extend([Xi, Xj])
#                 mlab.plot3d(
#                     [Xi[0], Xj[0]],
#                     [Xi[1], Xj[1]],
#                     [Xi[2], Xj[2]],
#                     tube_radius=tube_radius,
#                     color=colors_lv1[li[0]][0:3],
#                 )
#     else:
#         edge_sets = defaultdict(list)
#         for i, j in combinations(range(len(y)), 2):
#             li, lj = y[i], y[j]
#             Xi, Xj = X_mean[i], X_mean[j]
#             mlab.plot3d(
#                 [Xi[0], Xj[0]],
#                 [Xi[1], Xj[1]],
#                 [Xi[2], Xj[2]],
#                 tube_radius=tube_radius / 2,
#                 color=(1, 1, 1),
#                 opacity=0.25,
#             )

#     # plot faces
#     if len(condition) > 2:
#         for cond, edges in edge_sets.items():
#             vertices = np.unique(edges, axis=0)
#             triangles = ConvexHull(vertices).simplices
#             mlab.triangular_mesh(
#                 vertices[:, 0],
#                 vertices[:, 1],
#                 vertices[:, 2],
#                 triangles,
#                 color=colors_lv1[cond][0:3],
#                 opacity=0.25,
#             )

#     # plot points
#     point_radius = 0.1 * max_val
#     for i in range(len(X_mean)):
#         xx, yy, zz = X_mean[i]
#         mlab.points3d(xx, yy, zz, color=colors_all[i][0:3], scale_factor=point_radius)

#     # convert to image for Matplotlib
#     mlab.view(distance="auto", focalpoint="auto")
#     fig.scene._lift()
#     img = mlab.screenshot(figure=fig, antialiased=True)
#     img = img[100:-100, 100:-100]
#     if interactive:
#         mlab.show()
#     else:
#         mlab.close()

#     # plot projection screenshot
#     ax.imshow(img)
#     ax.axis("off")

#     # add legend
#     if len(condition) > 2:
#         for k, v in colors_lv1.items():
#             ax.plot([], [], lw=5, color=v, label=k)
#     elif len(condition) > 1:
#         for k, v in colors_lv1.items():
#             ax.plot([], [], color=v, label=k)
#     for i, label in enumerate(y[:, start:]):
#         ax.plot([], [], "o", color=colors_all[i], label=" × ".join(label))
#     remove_duplicate_legend_entries(ax)
#     sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1), frameon=False)

#     return ax


# def plot_projection_2d(
#     data: pd.DataFrame,
#     unit: str,
#     response: str,
#     condition: str | list[str],
#     cmap: str = "tab10",
#     interactive: bool = False,
#     ax: plt.Axes | None = None,
#     n_jobs: int | None = None,
# ) -> plt.Axes:
#     """
#     Plot a 3D projection of the data using PCA.

#     This function plots a 3D projection of the data using PCA. Each point in the plot represents the mean response of
#     a unit to different conditions. The points are colored based on the conditions.

#     Parameters
#     ----------
#     data : pd.DataFrame
#         The data to plot.
#     unit : str
#         The column name of the unit identifier.
#     response : str
#         The column name of the response variable.
#     condition : str or list of str
#         The column name(s) of the condition(s) to plot.
#     cmap : str, default='tab10'
#         The name of the colormap to use for coloring the conditions.
#     interactive : bool, default=False
#         If True, the plot is displayed interactively using Mayavi.
#     ax : `matplotlib.pyplot.Axes` or None, default=None
#         A matplotlib Axes object. If None, a new figure and axes are created.

#     Returns
#     -------
#     ax : `matplotlib.pyplot.Axes`
#         The Axes object containing the plot.
#     """

#     # check inputs
#     if isinstance(condition, str):
#         condition = [condition]
#     cmap = colormaps.get_cmap(cmap)

#     # create a new figure if no Axes object is provided
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(8, 8))

#     # process data
#     data = remove_groups_missing_conditions(data, unit, condition)
#     unit_std = data.groupby(unit)[response].std()
#     to_remove = unit_std[unit_std == 0].index
#     data = data[~data[unit].isin(to_remove)]

#     # normalize data
#     unit_mean = data.groupby(unit)[response].mean()
#     unit_std = data.groupby(unit)[response].std()
#     data[response] = (data[response] - data[unit].map(unit_mean)) / data[unit].map(
#         unit_std
#     )

#     # aggregate population data and apply MDS transformation
#     proj = MDS(n_components=2, n_init=64, max_iter=2400, n_jobs=n_jobs)
#     pop_mean = (
#         data.groupby([unit] + condition)[response]
#         .mean()
#         .groupby(condition)
#         .agg(list)
#         .reset_index()
#     )
#     X_mean = proj.fit_transform(np.stack(pop_mean[response]))

#     # generate plotting styles
#     y = pop_mean[condition].to_numpy(dtype=str)
#     offset, colors_lv1 = 0, None

#     # define color for first level of condition (if applicable)
#     if len(condition) > 1:
#         u_lv1, _ = np.unique(y[:, 0], return_inverse=True)
#         offset = len(u_lv1)
#         colors_lv1 = {u: cmap(i) for i, u in enumerate(u_lv1)}

#     # define color for all conditions
#     start = 0 if len(condition) == 1 else 1
#     _, l_all = np.unique(y[:, start:], return_inverse=True, axis=0)
#     colors_all = [cmap(l + offset) for l in l_all]

#     # set up the figure
#     xmin, xmax = np.min(X_mean[:, 0]), np.max(X_mean[:, 0])
#     ymin, ymax = np.min(X_mean[:, 1]), np.max(X_mean[:, 1])
#     max_val = max(xmax - xmin, ymax - ymin)

#     # plot edges
#     if len(condition) > 1:
#         edge_sets = defaultdict(list)
#         for i, j in combinations(range(len(y)), 2):
#             li, lj = y[i], y[j]
#             Xi, Xj = X_mean[i], X_mean[j]
#             if li[0] != lj[0] and np.all(li[1:] == lj[1:]):
#                 ax.plot([Xi[0], Xj[0]], [Xi[1], Xj[1]], color=(1, 1, 1), alpha=0.25)
#             # elif li[0] == lj[0] and np.sum(li[1:] == lj[1:]) == 1:
#             elif li[0] == lj[0]:
#                 edge_sets[li[0]].extend([Xi, Xj])
#                 ax.plot([Xi[0], Xj[0]], [Xi[1], Xj[1]], color=colors_lv1[li[0]])
#     else:
#         edge_sets = defaultdict(list)
#         for i, j in combinations(range(len(y)), 2):
#             li, lj = y[i], y[j]
#             Xi, Xj = X_mean[i], X_mean[j]
#             ax.plot([Xi[0], Xj[0]], [Xi[1], Xj[1]], color=(1, 1, 1), alpha=0.25)

#     # plot points
#     for i in range(len(X_mean)):
#         ax.plot(*X_mean[i], "o", color=colors_all[i])

#     # add legend
#     ax.axis("off")
#     if len(condition) > 2:
#         for k, v in colors_lv1.items():
#             ax.plot([], [], lw=5, color=v, label=k)
#     elif len(condition) > 1:
#         for k, v in colors_lv1.items():
#             ax.plot([], [], color=v, label=k)
#     for i, label in enumerate(y[:, start:]):
#         ax.plot([], [], "o", color=colors_all[i], label=" × ".join(label))
#     remove_duplicate_legend_entries(ax)
#     sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1), frameon=False)

#     return ax


def plot_dendrogram(
    data: pd.DataFrame,
    unit: str,
    response: str,
    condition: str | list[str],
    cmap: str = "tab10",
    ax: plt.Axes | None = None,
    n_jobs: int | None = None,
) -> plt.Axes:

    # check inputs
    if isinstance(condition, str):
        condition = [condition]
    n_conds = len(condition)
    cmap = colormaps.get_cmap(cmap)

    # create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # process data
    data = remove_groups_missing_conditions(data, unit, condition)
    unit_std = data.groupby(unit)[response].std()
    to_remove = unit_std[unit_std == 0].index
    data = data[~data[unit].isin(to_remove)]

    # normalize data
    unit_mean = data.groupby(unit)[response].mean()
    unit_std = data.groupby(unit)[response].std()
    data[response] = (data[response] - data[unit].map(unit_mean)) / data[unit].map(
        unit_std
    )

    # aggregate population data
    pop_mean = (
        data.groupby([unit] + condition)[response]
        .mean()
        .groupby(condition)
        .agg(list)
        .reset_index()
    )
    pop_mean = pop_mean.set_index(condition)

    # cluster points
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=0)
    model = model.fit(np.stack(pop_mean[response]))

    # compute linkage matrix
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts])

    # plot dendrogram
    dendrogram(linkage_matrix, orientation="right", ax=ax)
    sns.despine(ax=ax)
    ax.set_xlabel("Euclidean Distance")
    labels = np.array([item.get_text() for item in ax.get_yticklabels()], dtype=int)
    if n_conds == 1:
        labels = pop_mean.index[labels]
    else:
        labels = [" + ".join(map(str, pop_mean.index[l])) for l in labels]
    ax.set_yticklabels(labels)

    return ax


def draw_sig_outlines(sig, ax, **kwargs):
    sig = np.asarray(sig, bool)
    nrows, ncols = sig.shape
    edges = {}

    def add(p0, p1):
        k = tuple(sorted((p0, p1)))
        if k in edges:
            del edges[k]
        else:
            edges[k] = (p0, p1)

    for i in range(nrows):
        for j in range(ncols):
            if not sig[i, j]:
                continue
            p = [(j, i), (j + 1, i), (j + 1, i + 1), (j, i + 1)]
            add(p[0], p[1])
            add(p[1], p[2])
            add(p[2], p[3])
            add(p[3], p[0])

    adj = defaultdict(list)
    for a, b in edges.values():
        adj[a].append(b)
        adj[b].append(a)

    visited = set()
    polys = []
    for k, (a0, a1) in edges.items():
        if k in visited:
            continue
        poly = [a0]
        prev, cur = a0, a1
        while True:
            poly.append(cur)
            visited.add(tuple(sorted((prev, cur))))
            nxt = None
            for nb in adj[cur]:
                if nb == prev:
                    continue
                ek = tuple(sorted((cur, nb)))
                if ek in edges and ek not in visited:
                    nxt = nb
                    break
            if nxt is None or nxt == poly[0]:
                break
            prev, cur = cur, nxt
        polys.append(poly)

    for poly in polys:
        ax.add_patch(
            Polygon(poly, closed=True, fill=False, antialiased=False, **kwargs)
        )