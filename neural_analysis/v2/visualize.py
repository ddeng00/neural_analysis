import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem, kruskal, f_oneway

from .statistics import compute_confidence_interval


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
        ax.eventplot(spikes, colors="black")
        if plot_stats and stats is not None:
            ax.plot(stats, np.arange(len(spikes)), color="black")

    # Plot with grouping
    else:
        # order by group
        group_inds = [np.nonzero(group_labels == grp)[0] for grp in group_order]
        group_lens = [len(ind) for ind in group_inds]
        group_inds = np.concatenate(group_inds)
        spikes = spikes[group_inds]
        if stats is not None:
            stats = stats[group_inds]

        # Define colors and legend handles
        cmap = sns.color_palette("tab10", len(group_order))
        colors = np.concatenate(
            [[cmap[i]] * grp_len for i, grp_len in enumerate(group_lens)]
        )
        handles = [plt.Line2D([0], [0], color=c) for c in cmap]

        # Plots
        ax.eventplot(spikes, colors=colors)
        ax.legend(handles, group_order)
        if plot_stats and stats is not None:
            start_ind = 0
            for i, grp_len in enumerate(group_lens):
                end_ind = start_ind + grp_len
                ax.plot(
                    stats[start_ind:end_ind],
                    np.arange(grp_len) + start_ind,
                    color=cmap[i],
                )
                start_ind = end_ind

    # Plot zero line
    ax.axvline(0, color="black", linestyle="--")

    return ax


def plot_PSTH(
    timestamps: npt.ArrayLike,
    spike_rates: npt.ArrayLike,
    *,
    group_labels: npt.ArrayLike | None = None,
    group_order: npt.ArrayLike | None = None,
    error_type: str | tuple[str, float] | None = ("ci", 95),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot peristimulus time histogram (PSTH).

    This function plots a peristimulus time histogram (PSTH) with optional grouping of
    spike rates and the addition of error bars. If groups are specified, each group's
    spike rates are plotted separately with different colors.

    Parameters
    ----------
    timestamps : array-like
        An array of time points at which spike rates are measured.
    spike_rates : array-like
        A 2D array where each row contains the spike rates for a unit over time.
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
        ax.plot(timestamps, mean_rates)
        if error_type is not None:
            error_rates = error_func(spike_rates)
            ax.fill_between(
                timestamps,
                mean_rates - error_rates,
                mean_rates + error_rates,
                alpha=0.2,
            )

    # Plot with grouping
    else:
        group_labels = np.asarray(group_labels)
        if group_order is None:
            group_order = np.unique(group_labels)
        group_inds = [np.nonzero(group_labels == grp)[0] for grp in group_order]
        for grp, grp_ind in zip(group_order, group_inds):
            grp_rates = spike_rates[grp_ind]
            mean_rates = np.mean(grp_rates, axis=0)
            ax.plot(timestamps, mean_rates, label=grp)
            if error_type is not None:
                error_rates = error_func(grp_rates)
                ax.fill_between(
                    timestamps,
                    mean_rates - error_rates,
                    mean_rates + error_rates,
                    alpha=0.2,
                )
        ax.legend()

    # Plot zero line
    ax.axvline(0, color="black", linestyle="--")

    return ax


def plot_spikes_with_PSTH(
    spikes: list[npt.ArrayLike],
    timestamps: npt.ArrayLike,
    spike_rates: npt.ArrayLike,
    *,
    group_labels: npt.ArrayLike | None = None,
    group_order: npt.ArrayLike | None = None,
    stats: npt.ArrayLike | None = None,
    ascending: bool = False,
    plot_stats: bool = False,
    error_type: str | tuple[str, float] | None = ("ci", 95),
    axes: tuple[plt.Axes, plt.Axes] | None = None,
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
    timestamps : array-like
        An array of time points at which spike rates are measured.
    spike_rates : array-like
        A 2D array where each row contains the spike rates for a unit over time.
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
            height_ratios=[3, 1],
            sharex=True,
            gridspec_kw={"hspace": 0.1},
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
    )

    # Plot PSTH
    plot_PSTH(
        timestamps,
        spike_rates,
        group_labels=group_labels,
        group_order=group_order,
        error_type=error_type,
        ax=axes[1],
    )

    # Misc. settings
    plt.xlim(timestamps[0], timestamps[-1])
    axes[1].get_legend().remove()

    return axes


def plot_metrics(
    data: npt.ArrayLike,
    metric: str,
    *,
    null: npt.ArrayLike | None = None,
    x_group: str | None = None,
    y_group: str | None = None,
    x_order: npt.ArrayLike | None = None,
    y_order: npt.ArrayLike | None = None,
    y_emph: npt.ArrayLike | None = None,
    test_significance: bool = False,
    marker: str = "o",
    ax: plt.Axes | None = None,
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
            ax=ax,
            zorder=100,
        )

    else:
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
            dodge=0.25,
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
            ax=ax,
            zorder=200,
        )

    # Get plot limits
    ys = [line.get_ydata() for line in ax.get_lines()]
    y_min = min([min(y) for y in ys if len(y) > 0])
    y_max = max([max(y) for y in ys if len(y) > 0])

    # Plot null distribution
    if null is not None:
        old_xlim = ax.get_xlim()
        null = null.groupby(x_group)[metric].quantile([0.05, 0.95]).unstack()
        for grp, (low, high) in null.iterrows():
            grp = list(x_order).index(grp)
            ax.fill_between(
                [grp - 0.25, grp + 0.25],
                [low, low],
                [high, high],
                color="black",
                alpha=0.25,
                zorder=0,
            )
            y_min, y_max = min(y_min, low), max(y_max, high)
        ax.set_xlim(old_xlim)

    # Plot chance level
    ax.axhline(0.5, color="red", linestyle="--", zorder=0, label="Chance")

    # Plot significance test results
    if x_group is not None and test_significance:
        x_test_base = np.mean(np.arange(len(x_order)))
        y_test = y_max + 0.05 * (y_max - y_min)
        if y_group is None:
            samples = [data[data[x_group] == x][metric].values for x in x_order]
            p = f_oneway(*samples).pvalue
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
            cmap = sns.color_palette("tab10", len(y_order))
            to_plot = []
            for i, y in enumerate(y_order):
                samples = [
                    data[(data[x_group] == x) & (data[y_group] == y)][metric].values
                    for x in x_order
                ]
                p = f_oneway(*samples).pvalue
                if p < 0.05 / len(np.unique(data[y_group])):
                    to_plot.append(cmap[i])
            x_tests = (
                np.linspace(
                    1 - len(to_plot) / 2.0, len(to_plot) / 2.0 - 1, len(to_plot)
                )
                * 0.25
                + x_test_base
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
