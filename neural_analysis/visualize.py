from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import numpy.typing as npt
import matplotlib as mpl
from matplotlib import colormaps
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem
from scipy.ndimage import gaussian_filter1d
from statsmodels.stats.oneway import anova_oneway
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mayavi import mlab
from scipy.spatial import ConvexHull
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

from .preprocess import remove_groups_missing_conditions
from .statistics import compute_confidence_interval


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

    for grp, df in data.groupby(group):
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
        ax.plot(thresholds, n_remaining, lw=2, label=grp)

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

    # Plot without grouping
    if group_labels is None:
        ax.eventplot(spikes, colors="black", linelengths=len(spikes) / 100)
        if plot_stats and stats is not None:
            ax.plot(stats, np.arange(len(spikes)), color="black", lw=2)

    # Plot with grouping
    else:
        # order by group
        group_inds = [np.nonzero(group_labels == grp)[0] for grp in group_order[::-1]]
        group_lens = [len(ind) for ind in group_inds]
        group_inds = np.concatenate(group_inds)
        spikes = spikes[group_inds]
        if stats is not None:
            stats = stats[group_inds]

        # Define colors and legend handles
        cmap = sns.color_palette(cmap, len(group_order))[::-1]
        colors = np.concatenate(
            [[cmap[i]] * grp_len for i, grp_len in enumerate(group_lens)]
        )
        handles = [plt.Line2D([0], [0], color=c, lw=3) for c in cmap]

        # Plots
        ax.eventplot(spikes, colors=colors, linelengths=len(spikes) / 100)
        ax.legend(handles[::-1], group_order)
        if plot_stats and stats is not None:
            start_ind = 0
            for i, grp_len in enumerate(group_lens):
                end_ind = start_ind + grp_len
                ax.plot(
                    stats[start_ind:end_ind],
                    np.arange(grp_len) + start_ind,
                    # color=cmap[i],
                    color="black",
                    lw=2,
                )
                start_ind = end_ind

    # Plot zero line
    ax.axvline(0, color="black", linestyle="--")

    # Misc. settings
    ax.set_ylim(0, len(spikes))

    return ax


def plot_PSTH(
    spike_rates: npt.ArrayLike,
    timestamps: npt.ArrayLike | None = None,
    *,
    group_labels: npt.ArrayLike | None = None,
    group_order: npt.ArrayLike | None = None,
    error_type: str | tuple[str, float] | None = ("ci", 95),
    sig_test: bool = False,
    smooth_width: float | None = None,
    ax: plt.Axes | None = None,
    cmap: str | npt.ArrayLike | dict | mpl.colors.Colormap = "tab10",
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

    spike_rates = np.asarray(spike_rates)
    if timestamps is None:
        timestamps = np.arange(spike_rates.shape[1])
    timestamps = np.asarray(timestamps)
    if smooth_width is not None:
        smooth_width /= timestamps[1] - timestamps[0]

    # Create a new figure if no Axes object is provided
    if ax is None:
        _, ax = plt.subplots()

    # Process errorbar parameter
    error_func = None
    if error_type is not None:
        if isinstance(error_type, str):
            if error_type == "std":
                error_func = lambda x: np.std(x, axis=0)
            elif error_type == "sem":
                error_func = lambda x: sem(x, axis=0)
            elif error_type == "ci":
                error_func = lambda x: compute_confidence_interval(
                    x, confidence=0.95, axis=0
                )
            else:
                raise ValueError("Unknown errorbar type.")
        else:
            etype, escale = error_type
            if etype == "std":
                error_func = lambda x: escale * np.std(x, axis=0)
            elif etype == "sem":
                error_func = lambda x: escale * sem(x, axis=0)
            elif etype == "ci":
                error_func = lambda x: compute_confidence_interval(
                    x, confidence=escale / 100, axis=0
                )
            else:
                raise ValueError("Unknown errorbar type.")

    # Plot without grouping
    if group_labels is None:
        mean_rates = np.mean(spike_rates, axis=0)
        if smooth_width is not None:
            mean_rates = gaussian_filter1d(mean_rates, sigma=smooth_width)
        ax.plot(timestamps, mean_rates)
        if error_type is not None:
            error_rates = error_func(spike_rates)
            if smooth_width is not None:
                error_rates = gaussian_filter1d(error_rates, sigma=smooth_width)
            ax.fill_between(
                timestamps,
                mean_rates - error_rates,
                mean_rates + error_rates,
                alpha=0.2,
            )

    # Plot with grouping
    else:
        group_labels = np.asarray(group_labels)
        colors = sns.color_palette(cmap, len(np.unique(group_labels)))
        if group_order is None:
            group_order = np.unique(group_labels)
        group_inds = [np.nonzero(group_labels == grp)[0] for grp in group_order]
        for k, (grp, grp_ind) in enumerate(zip(group_order, group_inds)):
            grp_rates = spike_rates[grp_ind]
            mean_rates = np.mean(grp_rates, axis=0)
            if smooth_width is not None:
                mean_rates = gaussian_filter1d(mean_rates, sigma=smooth_width)
            ax.plot(timestamps, mean_rates, label=grp, color=colors[k])
            if error_type is not None:
                error_rates = error_func(grp_rates)
                if smooth_width is not None:
                    error_rates = gaussian_filter1d(error_rates, sigma=smooth_width)
                ax.fill_between(
                    timestamps,
                    mean_rates - error_rates,
                    mean_rates + error_rates,
                    alpha=0.2,
                    color=colors[k],
                )
        ax.legend()

    # Plot zero line
    ax.axvline(0, color="black", linestyle="--")

    # Plot significance
    if group_labels is not None and sig_test:
        y_max = ax.get_ylim()[1]
        spike_rates = np.sqrt(spike_rates).T
        spike_rates += np.random.normal(0, 1e-6, spike_rates.shape)
        pvals = np.array(
            [anova_oneway(rate, group_labels).pvalue for rate in spike_rates]
        )

        sig_mask = pvals < 0.001
        ax.plot(
            timestamps[sig_mask],
            [y_max] * np.sum(sig_mask),
            ".",
            color=(1, 0, 0),
            zorder=100,
        )

        sig_mask = (0.001 <= pvals) & (pvals < 0.01)
        ax.plot(
            timestamps[sig_mask],
            [y_max] * np.sum(sig_mask),
            ".",
            color=(1, 0.33, 0.33),
            zorder=50,
        )

        sig_mask = (0.01 <= pvals) & (pvals < 0.05)
        ax.plot(
            timestamps[sig_mask],
            [y_max] * np.sum(sig_mask),
            ".",
            color=(1, 0.67, 0.67),
            zorder=0,
        )
    return ax


def plot_spikes_with_PSTH(
    spikes: list[npt.ArrayLike],
    spike_rates: npt.ArrayLike,
    timestamps: npt.ArrayLike,
    *,
    group_labels: npt.ArrayLike | None = None,
    group_order: npt.ArrayLike | None = None,
    stats: npt.ArrayLike | None = None,
    ascending: bool = False,
    plot_stats: bool = False,
    error_type: str | tuple[str, float] | None = ("ci", 95),
    sig_test: bool = False,
    smooth_width: float | None = None,
    axes: tuple[plt.Axes, plt.Axes] | None = None,
    cmap: str | npt.ArrayLike | dict | mpl.colors.Colormap = "tab10",
) -> tuple[plt.Axes, plt.Axes]:
    """
    Plot spike rasters and peristimulus time histograms (PSTHs) together.

    This function plots a spike raster plot and a PSTH with optional grouping and statistical overlay.
    If groups are specified, spikes are colored according to their group. If stats are provided, they are plotted
    alongside the spike raster. Error bars can be added to the PSTH.

    Parameters
    ----------
    spikes : list of array-like
        A list of arrays where each array contains spike times for a unit.
    spike_rates : array-like
        A 2D array where each row contains the spike rates for a unit over time.
    timestamps : array-like
        An array of time points at which spike rates are measured.
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

    # Plot spike raster
    plot_spikes(
        spikes,
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
        spike_rates,
        timestamps,
        group_labels=group_labels,
        group_order=group_order,
        error_type=error_type,
        sig_test=sig_test,
        smooth_width=smooth_width,
        ax=axes[1],
        cmap=cmap,
    )

    # Misc. settings
    axes[0].set_xlim(timestamps[0], timestamps[-1])
    axes[0].get_xaxis().set_visible(False)
    axes[0].set_ylabel("Trial" if stats is None else "Trial (sorted)")

    axes[1].set_xlim(timestamps[0], timestamps[-1])
    try:
        axes[1].get_legend().remove()
    except AttributeError:
        pass
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Firing Rate [Hz]")

    sns.despine(ax=axes[0])
    sns.despine(ax=axes[1])
    if axes[0].get_legend() is not None:
        sns.move_legend(
            axes[0], "upper left", bbox_to_anchor=(1, 1), title=None, frameon=False
        )

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
    sig_test: bool = False,
    marker: str = "o",
    chance: float | None = 0.5,
    palette: str | npt.ArrayLike | dict | mpl.colors.Colormap = "tab10",
    ax: plt.Axes | None = None,
    **kwargs,
):

    # Process parameters
    if x_group is not None and x_order is None:
        x_order = np.unique(data[x_group])
    if y_group is not None and y_order is None:
        y_order = np.unique(data[y_group])

    # Create a new figure if no Axes object is provided
    if ax is None:
        x_n = 1 if x_group is None else len(x_order)
        _, ax = plt.subplots(figsize=(x_n, 6))

    # Plot metrics
    if y_emph is None:
        sns.pointplot(
            data,
            x=x_group,
            y=metric,
            hue=y_group,
            order=x_order,
            hue_order=y_order,
            errorbar=None,
            markers=marker,
            markersize=10,
            markeredgecolor="black",
            lw=3,
            palette=palette,
            ax=ax,
            zorder=100,
            **kwargs,
        )

    else:
        if isinstance(y_emph, str):
            y_emph = data.loc[data[y_emph], y_group].unique()
        y_order = [y for y in y_order if y in y_emph]
        y_emph = np.isin(data[y_group], y_emph)
        sns.pointplot(
            data[~y_emph],
            x=x_group,
            y=metric,
            hue=y_group,
            order=x_order,
            errorbar=None,
            markers=marker,
            markerfacecolor="white",
            markeredgecolor="black",
            ls="",
            # dodge=0.25,
            alpha=0.25,
            ax=ax,
            zorder=100,
            legend=False,
        )
        sns.pointplot(
            data[y_emph],
            x=x_group,
            y=metric,
            hue=y_group,
            order=x_order,
            hue_order=y_order,
            errorbar=None,
            markers=marker,
            markersize=10,
            markeredgecolor="black",
            lw=3,
            palette=palette,
            ax=ax,
            zorder=200,
            **kwargs,
        )

    # Get plot limits
    ys = [line.get_ydata() for line in ax.get_lines()]
    y_min = min([min(y) for y in ys if len(y) > 0])
    y_max = max([max(y) for y in ys if len(y) > 0])

    # Plot null distribution
    if null is not None:
        old_xlim = ax.get_xlim()
        if x_group is None:
            low, high = null[metric].quantile([0.05, 0.95])
            ax.fill_between(
                [-0.25, 0.25],
                [low, low],
                [high, high],
                color="black",
                alpha=0.25,
                zorder=0,
                label="Perm. Null\n(5-95th pctl.)",
            )
            y_min, y_max = min(y_min, low), max(y_max, high)
        else:
            labeled = False
            null = null.groupby(x_group)[metric].quantile([0.05, 0.95]).unstack()
            for grp, (low, high) in null.iterrows():
                if grp not in x_order:
                    continue
                grp = list(x_order).index(grp)
                ax.fill_between(
                    [grp - 0.25, grp + 0.25],
                    [low, low],
                    [high, high],
                    color="black",
                    alpha=0.25,
                    zorder=0,
                    label=None if labeled else "Perm. Null\n(5-95th pctl.)",
                )
                labeled = True
                y_min, y_max = min(y_min, low), max(y_max, high)
        ax.set_xlim(old_xlim)

    # Plot chance level
    if chance is not None:
        ax.axhline(chance, color="red", linestyle="--", zorder=0, label="Chance")

    # Plot significance test results
    if x_group is not None and sig_test:
        x_test_base = np.mean(np.arange(len(x_order)))
        y_test = y_max + 0.05 * (y_max - y_min)
        if y_group is None:
            p = anova_oneway(np.sqrt(data[metric]), data[x_group]).pvalue
            if p < 0.05:
                ax.text(
                    np.mean(np.arange(len(x_order))),
                    y_test,
                    "*",
                    ha="center",
                    va="bottom",
                    fontsize=16,
                    weight="bold",
                    color="black",
                    zorder=300,
                )
        else:
            cmap = sns.color_palette(palette, len(y_order))
            to_plot = []
            for i, y in enumerate(y_order):
                df = data[data[y_group] == y]
                p = anova_oneway(np.sqrt(df[metric]), df[x_group]).pvalue
                if p < 0.05 / len(y_order):
                    to_plot.append(cmap[i])
            x_tests = (
                (
                    np.linspace(
                        1 - len(to_plot) / 2.0, len(to_plot) / 2.0 - 1, len(to_plot)
                    )
                    * 0.25
                    + x_test_base
                )
                if len(to_plot) > 1
                else [x_test_base]
            )
            for x, color in zip(x_tests, to_plot):
                ax.text(
                    x,
                    y_test,
                    "*",
                    ha="center",
                    va="bottom",
                    fontsize=16,
                    weight="bold",
                    color=color,
                    zorder=300,
                )

    # Misc. settings
    ax.legend()
    sns.despine(ax=ax)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title=None, frameon=False)

    return ax


def plot_projection(
    data: pd.DataFrame,
    unit: str,
    response: str,
    condition: str | list[str],
    cmap: str = "tab10",
    interactive: bool = False,
    ax: plt.Axes | None = None,
    n_jobs: int | None = None,
) -> plt.Axes:
    """
    Plot a 3D projection of the data using PCA.

    This function plots a 3D projection of the data using PCA. Each point in the plot represents the mean response of
    a unit to different conditions. The points are colored based on the conditions.

    Parameters
    ----------
    data : pd.DataFrame
        The data to plot.
    unit : str
        The column name of the unit identifier.
    response : str
        The column name of the response variable.
    condition : str or list of str
        The column name(s) of the condition(s) to plot.
    cmap : str, default='tab10'
        The name of the colormap to use for coloring the conditions.
    interactive : bool, default=False
        If True, the plot is displayed interactively using Mayavi.
    ax : `matplotlib.pyplot.Axes` or None, default=None
        A matplotlib Axes object. If None, a new figure and axes are created.

    Returns
    -------
    ax : `matplotlib.pyplot.Axes`
        The Axes object containing the plot.
    """

    # check inputs
    if isinstance(condition, str):
        condition = [condition]
    cmap = colormaps.get_cmap(cmap)

    # create a new figure if no Axes object is provided
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

    # aggregate population data and apply MDS transformation
    proj = MDS(n_components=3, n_init=64, max_iter=2400, n_jobs=n_jobs)
    pop_mean = (
        data.groupby([unit] + condition)[response]
        .mean()
        .groupby(condition)
        .agg(list)
        .reset_index()
    )
    X_mean = proj.fit_transform(np.stack(pop_mean[response]))

    # generate plotting styles
    y = pop_mean[condition].to_numpy(dtype=str)
    offset, colors_lv1 = 0, None

    # define color for first level of condition (if applicable)
    if len(condition) > 1:
        u_lv1, _ = np.unique(y[:, 0], return_inverse=True)
        offset = len(u_lv1)
        colors_lv1 = {u: cmap(i) for i, u in enumerate(u_lv1)}

    # define color for all conditions
    start = 0 if len(condition) == 1 else 1
    _, l_all = np.unique(y[:, start:], return_inverse=True, axis=0)
    colors_all = [cmap(l + offset) for l in l_all]

    # set up the figure
    fig = mlab.figure(size=(1000, 1000))

    xmin, xmax = np.min(X_mean[:, 0]), np.max(X_mean[:, 0])
    ymin, ymax = np.min(X_mean[:, 1]), np.max(X_mean[:, 1])
    zmin, zmax = np.min(X_mean[:, 2]), np.max(X_mean[:, 2])
    max_val = max(xmax - xmin, ymax - ymin, zmax - zmin)

    # plot edges
    tube_radius = 0.015 * max_val
    if len(condition) > 1:
        edge_sets = defaultdict(list)
        for i, j in combinations(range(len(y)), 2):
            li, lj = y[i], y[j]
            Xi, Xj = X_mean[i], X_mean[j]
            if li[0] != lj[0] and np.all(li[1:] == lj[1:]):
                mlab.plot3d(
                    [Xi[0], Xj[0]],
                    [Xi[1], Xj[1]],
                    [Xi[2], Xj[2]],
                    tube_radius=tube_radius,
                    color=(1, 1, 1),
                    opacity=0.25,
                )
            # elif li[0] == lj[0] and np.sum(li[1:] == lj[1:]) == 1:
            elif li[0] == lj[0]:
                edge_sets[li[0]].extend([Xi, Xj])
                mlab.plot3d(
                    [Xi[0], Xj[0]],
                    [Xi[1], Xj[1]],
                    [Xi[2], Xj[2]],
                    tube_radius=tube_radius,
                    color=colors_lv1[li[0]][0:3],
                )
    else:
        edge_sets = defaultdict(list)
        for i, j in combinations(range(len(y)), 2):
            li, lj = y[i], y[j]
            Xi, Xj = X_mean[i], X_mean[j]
            mlab.plot3d(
                [Xi[0], Xj[0]],
                [Xi[1], Xj[1]],
                [Xi[2], Xj[2]],
                tube_radius=tube_radius / 2,
                color=(1, 1, 1),
                opacity=0.25,
            )

    # plot faces
    if len(condition) > 2:
        for cond, edges in edge_sets.items():
            vertices = np.unique(edges, axis=0)
            triangles = ConvexHull(vertices).simplices
            mlab.triangular_mesh(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                triangles,
                color=colors_lv1[cond][0:3],
                opacity=0.25,
            )

    # plot points
    point_radius = 0.1 * max_val
    for i in range(len(X_mean)):
        xx, yy, zz = X_mean[i]
        mlab.points3d(xx, yy, zz, color=colors_all[i][0:3], scale_factor=point_radius)

    # convert to image for Matplotlib
    mlab.view(distance="auto", focalpoint="auto")
    fig.scene._lift()
    img = mlab.screenshot(figure=fig, antialiased=True)
    img = img[100:-100, 100:-100]
    if interactive:
        mlab.show()
    else:
        mlab.close()

    # plot projection screenshot
    ax.imshow(img)
    ax.axis("off")

    # add legend
    if len(condition) > 2:
        for k, v in colors_lv1.items():
            ax.plot([], [], lw=5, color=v, label=k)
    elif len(condition) > 1:
        for k, v in colors_lv1.items():
            ax.plot([], [], color=v, label=k)
    for i, label in enumerate(y[:, start:]):
        ax.plot([], [], "o", color=colors_all[i], label=" × ".join(label))
    remove_duplicate_legend_entries(ax)
    sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1), frameon=False)

    return ax


def plot_projection_2d(
    data: pd.DataFrame,
    unit: str,
    response: str,
    condition: str | list[str],
    cmap: str = "tab10",
    interactive: bool = False,
    ax: plt.Axes | None = None,
    n_jobs: int | None = None,
) -> plt.Axes:
    """
    Plot a 3D projection of the data using PCA.

    This function plots a 3D projection of the data using PCA. Each point in the plot represents the mean response of
    a unit to different conditions. The points are colored based on the conditions.

    Parameters
    ----------
    data : pd.DataFrame
        The data to plot.
    unit : str
        The column name of the unit identifier.
    response : str
        The column name of the response variable.
    condition : str or list of str
        The column name(s) of the condition(s) to plot.
    cmap : str, default='tab10'
        The name of the colormap to use for coloring the conditions.
    interactive : bool, default=False
        If True, the plot is displayed interactively using Mayavi.
    ax : `matplotlib.pyplot.Axes` or None, default=None
        A matplotlib Axes object. If None, a new figure and axes are created.

    Returns
    -------
    ax : `matplotlib.pyplot.Axes`
        The Axes object containing the plot.
    """

    # check inputs
    if isinstance(condition, str):
        condition = [condition]
    cmap = colormaps.get_cmap(cmap)

    # create a new figure if no Axes object is provided
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

    # aggregate population data and apply MDS transformation
    proj = MDS(n_components=2, n_init=64, max_iter=2400, n_jobs=n_jobs)
    pop_mean = (
        data.groupby([unit] + condition)[response]
        .mean()
        .groupby(condition)
        .agg(list)
        .reset_index()
    )
    X_mean = proj.fit_transform(np.stack(pop_mean[response]))

    # generate plotting styles
    y = pop_mean[condition].to_numpy(dtype=str)
    offset, colors_lv1 = 0, None

    # define color for first level of condition (if applicable)
    if len(condition) > 1:
        u_lv1, _ = np.unique(y[:, 0], return_inverse=True)
        offset = len(u_lv1)
        colors_lv1 = {u: cmap(i) for i, u in enumerate(u_lv1)}

    # define color for all conditions
    start = 0 if len(condition) == 1 else 1
    _, l_all = np.unique(y[:, start:], return_inverse=True, axis=0)
    colors_all = [cmap(l + offset) for l in l_all]

    # set up the figure
    xmin, xmax = np.min(X_mean[:, 0]), np.max(X_mean[:, 0])
    ymin, ymax = np.min(X_mean[:, 1]), np.max(X_mean[:, 1])
    max_val = max(xmax - xmin, ymax - ymin)

    # plot edges
    if len(condition) > 1:
        edge_sets = defaultdict(list)
        for i, j in combinations(range(len(y)), 2):
            li, lj = y[i], y[j]
            Xi, Xj = X_mean[i], X_mean[j]
            if li[0] != lj[0] and np.all(li[1:] == lj[1:]):
                ax.plot([Xi[0], Xj[0]], [Xi[1], Xj[1]], color=(1, 1, 1), alpha=0.25)
            # elif li[0] == lj[0] and np.sum(li[1:] == lj[1:]) == 1:
            elif li[0] == lj[0]:
                edge_sets[li[0]].extend([Xi, Xj])
                ax.plot([Xi[0], Xj[0]], [Xi[1], Xj[1]], color=colors_lv1[li[0]])
    else:
        edge_sets = defaultdict(list)
        for i, j in combinations(range(len(y)), 2):
            li, lj = y[i], y[j]
            Xi, Xj = X_mean[i], X_mean[j]
            ax.plot([Xi[0], Xj[0]], [Xi[1], Xj[1]], color=(1, 1, 1), alpha=0.25)

    # plot points
    for i in range(len(X_mean)):
        ax.plot(*X_mean[i], "o", color=colors_all[i])

    # add legend
    ax.axis("off")
    if len(condition) > 2:
        for k, v in colors_lv1.items():
            ax.plot([], [], lw=5, color=v, label=k)
    elif len(condition) > 1:
        for k, v in colors_lv1.items():
            ax.plot([], [], color=v, label=k)
    for i, label in enumerate(y[:, start:]):
        ax.plot([], [], "o", color=colors_all[i], label=" × ".join(label))
    remove_duplicate_legend_entries(ax)
    sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1), frameon=False)

    return ax


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
