import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import seaborn as sns


def remove_legend_duplicates(ax: plt.Axes) -> None:
    """
    Remove duplicate legend entries.

    Parameters
    ----------
    ax : matplotlib Axes
        Axes object with legend.
    """

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    return ax


def modify_legend(
    ax: plt.Axes,
    loc: str = "upper right",
    title: bool | str | None = True,
    labels: list[str] | None = None,
    frameon: bool = False,
) -> None:
    """
    Modify legend location, title, and labels.

    Parameters
    ----------
    ax : matplotlib Axes
        Axes object with legend.
    loc : str, default="upper right"
        New location of legend. Must be one of
        "upper right", "center right", "lower right", "top".
    title : bool or str or None, default=True
        Modify legend title.
        If True, the title is the current legend title.
        If False or None, no title is added.
        If str, the title is the given string.
    labels : list of str or None, default=None
        Modify legend labels. If None, the current labels are kept.
    frameon : bool, default=False
        Whether to draw a frame around the legend.
    """

    # check input
    loc = loc.lower()
    if loc not in ["upper right", "center right", "lower right", "top"]:
        raise ValueError("Invalid location.")
    if type(title) is not str:
        title = ax.get_legend().get_title().get_text() if title else ""

    # modify legend labels
    if labels is not None:
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title=title)

    # move legend
    n_labels = len(ax.get_legend_handles_labels()[1])
    if loc == "upper right":
        sns.move_legend(
            ax, "upper left", bbox_to_anchor=(1, 1), title=title, frameon=frameon
        )
    elif loc == "center right":
        sns.move_legend(
            ax,
            "center left",
            bbox_to_anchor=(1, 0.5),
            title=title,
            frameon=frameon,
        )
    elif loc == "lower right":
        sns.move_legend(
            ax, "lower left", bbox_to_anchor=(1, 0), title=title, frameon=frameon
        )
    elif loc == "top":
        sns.move_legend(
            ax,
            "lower center",
            bbox_to_anchor=(0.5, 1),
            ncol=n_labels,
            frameon=frameon,
        )
    else:
        raise NotImplementedError("Not implemented yet.")


def annot_h(
    x1: int | float,
    x2: int | float,
    ax: plt.Axes,
    y: int | float | None = None,
    text: str | None = None,
    color: str = "black",
) -> plt.Axes:
    """
    Annotate a horizontal line between two x-values.

    Parameters
    ----------
    x1 : int or float
        First x-value.
    x2 : int or float
        Second x-value.
    ax : matplotlib Axes
        Axes object to annotate. Must not be empty.
    y : int or float, default=None
        y-value to place annotation line.
        If None, the line is placed above the current y-limit.
    text : str, default=None
        Text to annotate.
    color : str, default="black"
        Color of the annotation.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with annotation.
    """

    # check input
    if not ax.has_data():
        raise ValueError("Axes must not be empty.")
    if x1 > x2:
        x1, x2 = x2, x1

    # plot annotation
    ylim = ax.get_ylim()
    h = (ylim[1] - ylim[0]) / 50
    if y is None:
        y = ylim[1] + h
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=color)
    if text is not None:
        ax.text((x1 + x2) * 0.5, y + h, text, ha="center", va="bottom", c=color)

    return ax


def annot_v(
    y1: int | float,
    y2: int | float,
    ax: plt.Axes,
    x: int | float | None = None,
    text: str | None = None,
    color: str = "black",
) -> plt.Axes:
    """
    Annotate a vertical line between two y-values.

    Parameters
    ----------
    y1 : int or float
        First y-value.
    y2 : int or float
        Second x-value.
    ax : matplotlib Axes
        Axes object to annotate. Must not be empty.
    x : int or float, default=None
        x-value to place annotation line.
        If None, the line is placed above the current x-limit.
    text : str, default=None
        Text to annotate.
    color : str, default="black"
        Color of the annotation.

    Returns
    -------
    ax : matplotlib Axes
        Axes object with annotation.
    """

    # check input
    if not ax.has_data():
        raise ValueError("Axes must not be empty.")
    if y1 > y2:
        y1, y2 = y2, y1

    # plot annotation
    xlim = ax.get_xlim()
    w = (xlim[1] - xlim[0]) / 50
    if x is None:
        x = xlim[1] + w
    ax.plot([x, x + w, x + w, x], [y1, y1, y2, y2], lw=1.5, c=color)
    if text is not None:
        ax.text(x + w, (y1 + y2) * 0.5, text, ha="right", va="center", c=color)

    return ax


def plot_spike_raster(
    spikes: list[npt.ArrayLike],
    reaction_times: npt.ArrayLike = None,
    groups: npt.ArrayLike = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Plot spike raster for all trials.

    Parameters
    ----------
    spikes : list of array-like
        Spike times for each trial.
    rts : array-like, default=None
        Reaction times for each trial.
        If provided, trials are reordered by RT.
    groups : array-like, default=None
        Group labels for each trial.
    ax : `matplotlib.pyplot.Axes`, default=None
        Axes object to plot on. If None, a new figure is created.

    Returns
    -------
    ax : `matplotlib.pyplot.Axes`
        Axes object with spike raster.
    """

    # create new figure if necessary
    if ax is None:
        _ = plt.figure()
        ax = plt.gca()

    # check input
    if len(spikes) == 0:
        raise ValueError("No trials provided.")
    if reaction_times is not None and len(spikes) != len(reaction_times):
        raise ValueError("Number of trials and reaction times must match.")
    if groups is not None and len(spikes) != len(groups):
        raise ValueError("Number of trials and groups must match.")
    spikes = np.asarray(spikes, dtype=object)
    if reaction_times is not None:
        reaction_times = np.asarray(reaction_times)
    if groups is not None:
        groups = np.asarray(groups)

    # reorder trials by RT
    if reaction_times is not None:
        reaction_times = np.asarray(reaction_times)
        order = np.argsort(reaction_times)
        if reaction_times[0] > 0:
            order = order[::-1]
        spikes = spikes[order]
        reaction_times = reaction_times[order]
        groups = groups[order] if groups is not None else None

    # reorder trials by group
    if groups is not None:
        group_names, groups = np.unique(groups, return_inverse=True)
        order = np.argsort(groups, kind="stable")
        spikes = spikes[order]
        reaction_times = reaction_times[order] if reaction_times is not None else None
        groups = groups[order]

    # plot spike raster
    if groups is None:
        ax.eventplot(spikes, color="black")
    else:
        colors = [sns.color_palette()[i] for i in groups]
        ax.eventplot(spikes, colors=colors)
        handles = [
            ax.plot([], c=sns.color_palette()[i], label=name)[0]
            for i, name in enumerate(group_names)
        ]
        ax.legend(handles=handles)

    # plot RT
    if reaction_times is not None:
        if groups is None:
            ax.plot(reaction_times, np.arange(len(reaction_times)), color="black")
        else:
            for i in range(len(group_names)):
                ax.plot(
                    reaction_times[groups == i],
                    np.arange(len(reaction_times))[groups == i],
                    color="k",
                )

    # plot zero line
    ax.axvline(0, color="black", ls="--")

    return ax


def plot_PSTH(
    timesteps: npt.ArrayLike,
    spike_rates: npt.ArrayLike,
    groups: npt.ArrayLike = None,
    TOIs: npt.ArrayLike = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Plot PSTH.

    Parameters
    ----------
    timesteps : array-like
        Time steps.
    spike_rates : array-like
        Spike rates across trials for each time step.
    groups : array-like, default=None
        Group labels for each time step.
    TOIs : array-like, default=None
        Times of interest for each trial.
        If provided, vertical lines are plotted at each time.
    ax : `matplotlib.pyplot.Axes`, default=None
        Axes object to plot on. If None, a new figure is created.

    Returns
    -------
    ax : `matplotlib.pyplot.Axes`
        Axes object with PSTH.
    """

    # create new figure if necessary
    if ax is None:
        _ = plt.figure()
        ax = plt.gca()

    # check input
    timesteps, spike_rates = np.asarray(timesteps), np.asarray(spike_rates)
    if timesteps.shape[0] != spike_rates.shape[1]:
        raise ValueError("Number of time steps and spike rates must match.")
    if spike_rates.shape[0] != len(groups):
        raise ValueError("Number of trials and groups must match.")

    # plot PSTH
    sns.lineplot(x=timesteps, y=np.mean(spike_rates, axis=0), hue=groups, ax=ax)


    # if se_spike_rates is not None:
    #     ax.fill_between(
    #         timesteps,
    #         mean_spike_rates - se_spike_rates,
    #         mean_spike_rates + se_spike_rates,
    #         color="black",
    #         alpha=0.2,
    #     )

    # plot vertical lines at TOIs
    if TOIs is not None:
        for toi in TOIs:
            ax.axvline(toi, color="black", ls="--")

    return ax
