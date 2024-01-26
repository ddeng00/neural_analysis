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
