import matplotlib.pyplot as plt
import seaborn as sns


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
