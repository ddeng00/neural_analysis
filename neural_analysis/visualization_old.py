from collections.abc import Iterable
from typing import Callable, Any

import numpy as np
from matplotlib import cm
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd

from .utils_old import confidence_interval, get_spikes, spike_rates_sw, split_by


def nested_subplots(
    n_cells: int = 1,
    grid_nrows: int | None = None,
    grid_ncols: int | None = None,
    cell_nrows: int = 1,
    cell_ncols: int = 1,
    grid_spec_params: dict[str, Any] = {},
    cell_spec_params: dict[str, Any] = {},
    plot_func: Callable | None = None,
    plot_func_params: dict[str, Any] = {},
    figsize: tuple[float, float] | str = "auto",
    scale: float = 4.0,
) -> tuple[plt.Figure, list[list[plt.Axes]]]:
    """
    Initialize empty subplots in a grid.

    Parameters
    ----------
    TODO

    Returns
    -------
    fig : `matplotlib.pyplot.Figure`
    grid_axes : list of list of `matplotlib.pyplot.Axes`
    """

    if grid_nrows is None and grid_ncols is None:
        raise "One of grid_nrows or grid_ncols must be provided."
    elif grid_nrows is None and grid_ncols is not None:
        grid_nrows = n_cells // grid_ncols + 1
    elif grid_nrows is not None and grid_ncols is None:
        grid_ncols = n_cells // grid_nrows + 1
    elif grid_nrows * grid_ncols < n_cells:
        raise "Grid dimensions too small to fit all cells."

    if figsize == "auto":
        figsize = (grid_ncols * scale, grid_nrows * scale)
    fig = plt.figure(figsize=figsize)

    subplot_spec = gridspec.GridSpec(grid_nrows, grid_ncols, **grid_spec_params)
    grid_axes = []
    for i in range(n_cells):
        subspecs = gridspec.GridSpecFromSubplotSpec(
            cell_nrows, cell_ncols, subplot_spec=subplot_spec[i], **cell_spec_params
        )
        axes = [plt.Subplot(fig, s) for s in subspecs]

        if len(axes) == 1:
            ax = axes[0]
            if plot_func is not None:
                ax = plot_func(ax=ax, **plot_func_params)
            fig.add_subplot(ax)
            grid_axes.append(ax)
        else:
            if plot_func is not None:
                axes = plot_func(axes=axes, **plot_func_params)
            [fig.add_subplot(ax) for ax in axes]
            grid_axes.append(axes)

    return fig, grid_axes


def raster_with_PSTH(
    spike_timings: Iterable[float],
    start_times: Iterable[float],
    end_times: Iterable[float],
    alignments: Iterable[float],
    roi: list[tuple[Iterable[float], Iterable[float]]] | None = None,
    time_scale: float = 1e-3,
    window: float = 250.0,
    step: float = 10.0,
    inclusive: bool = True,
    ma_size: int = 1,
    confidence_level: float = 0.95,
    title: str | None = None,
    labels: Iterable[int | float | str] | None = None,
    legend: bool = True,
    axes: tuple[plt.Axes, plt.Axes] = None,
) -> tuple[plt.Axes, plt.Axes]:
    """
    Graph stack raster with PST plots.

    Parameters
    ----------
    spike_timings : iterable of float
        Timings of all spikes of interest.
    start_times : iterable of float
        Start times of each time window.
    end_times : iterable of float
        End times of each time window.
    alignments : iterable of float
        Alignment times all other times will be relative to.
    roi : list of (iterable of float, iterable of float) or None, default = None
        Time boundaries for region(s) of interest to visualize.
    time_scale: float, default = 1e-3
        Scale to convert given time unit to seconds.
    window : float, default = 500.0
        Size of sliding window.
    step : float, default = 16.0
        Size of step to take for sliding window operation.
    inclusive : bool, default = True
        If True, sliding window will include spikes outside the defined start/end boundaries when computing spike rates.
        Otherwise, only spikes within bounardies will be considered and windows with size less than `window` will be used near bonudaries.
    ma_size : int, default = 1
        Size of the rolling window to apply moving average smoothing.
    confidence_level : float
        Desired confidence level for estimating mean spike rate.
    title : str or None, default = None
        Title of the PTSH.
    labels : iterable of int or iterable of float or iterable of str or None:
        Labels for each time window that is mapped to determine the color of plot elements.
    legend : bool, default = True,
        If True, graph the legend unless `labels` are not provided.
    axes : (`matplotlib.pyplot.Axes`, `matplotlib.pyplot.Axes`) | None, default = None
        Plots in which to graph the raster plot and PSTH, respectively.

    Returns
    -------
    (`matplotlib.pyplot.Axes`, `matplotlib.pyplot.Axes`)
        Plots containing the raster plot and PSTH, respectively (same as `axes` if provided).
    """

    # create figure if not provided
    if axes is None:
        _, axes = plt.subplots(2, 1, height_ratios=[2, 1])
    ax1, ax2 = axes

    # plot raster plot
    ax1 = raster_plot(
        spike_timings,
        start_times,
        end_times,
        alignments,
        roi=roi,
        title=title,
        labels=labels,
        legend=legend,
        ax=ax1,
    )
    ax1.set_xticklabels([])
    ax1.set_xticks([])
    ax1.set_xlabel("")

    # plot PSTH
    ax2 = PST_plot(
        spike_timings,
        start_times,
        end_times,
        alignments,
        roi=roi,
        time_scale=time_scale,
        window=window,
        step=step,
        inclusive=inclusive,
        ma_size=ma_size,
        confidence_level=confidence_level,
        labels=labels,
        legend=False,
        ax=ax2,
    )
    ax2.set_xlim(*ax1.get_xlim())

    return axes


def raster_plot(
    spike_timings: Iterable[float],
    start_times: Iterable[float],
    end_times: Iterable[float],
    alignments: Iterable[float],
    roi: list[tuple[Iterable[float], Iterable[float]]] | None = None,
    title: str | None = None,
    labels: Iterable[int | float | str] | None = None,
    legend: bool = True,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Graph raster plot of the spiking data.

    Parameters
    ----------
    spike_timings : iterable of float
        Timings of all spikes of interest.
    start_times : iterable of float
        Start times of each time window.
    end_times : iterable of float
        End times of each time window.
    alignments : iterable of float
        Alignment times all other times will be relative to.
    roi : list of (iterable of float, iterable of float) or None, default = None
        Time boundaries for region(s) of interest to visualize.
    title : str or None, default = None
        Title of the raster plot.
    labels : iterable of int or iterable of float or iterable of str or None:
        Labels for each time window that is mapped to determine the color of plot elements.
    legend : bool, default = True,
        If True, graph the legend unless `labels` are not provided.
    ax : `matplotlib.pyplot.Axes` | None, default = None
        Plot in which to graph the raster plot.

    Returns
    -------
    `matplotlib.pyplot.Axes`
        Plot containing the raster plot (same as `ax` if provided).
    """

    # create new figure if not provided
    if ax is None:
        _ = plt.figure()
        ax = plt.gca()

    # get spikes
    pre_align = (alignments - start_times).mean()
    post_align = (end_times - alignments).mean()
    spikes = get_spikes(spike_timings, start_times, end_times, alignments)

    # plot raster
    if labels is None:
        ax.eventplot(spikes)
        ax.set_ylabel("Trial Number")
    else:
        labels = np.asarray(labels)
        sorted_idx = np.argsort(labels)

        labels = labels[sorted_idx]
        spikes = [spikes[i] for i in sorted_idx]
        unique = np.unique(labels)

        cmap = {k: cm.tab10(i) for i, k in enumerate(unique)}
        colors = [[cmap[labels[i]]] * len(s) for i, s in enumerate(spikes)]

        ax.eventplot(spikes, colors=colors)
        ax.set_ylabel("Trial Number (Sorted)")

        if legend:
            handles = [Line2D([], [], color=cmap[k], label=k) for k in unique]
            ax.legend(loc="upper right", handles=handles)

    # plot region of interests
    if roi is not None:
        for b1, b2 in roi:
            b1 = np.mean(b1 - alignments)
            b2 = np.mean(b2 - alignments)
            ax.axvspan(b1, b2, color="grey", alpha=0.1)
            ax.axvline(b1, color="grey", alpha=0.5)
            ax.axvline(b2, color="grey", alpha=0.5)

    # adjust plot elements
    ax.set_xlabel("Time [ms]")
    ax.set_xlim(-pre_align, post_align)
    if title is not None:
        ax.set_title(title)

    return ax


def PST_plot(
    spike_timings: Iterable[float],
    start_times: Iterable[float],
    end_times: Iterable[float],
    alignments: Iterable[float],
    roi: list[tuple[Iterable[float], Iterable[float]]] | None = None,
    time_scale: float = 1e-3,
    window: float = 250.0,
    step: float = 10.0,
    inclusive: bool = True,
    ma_size: int = 1,
    confidence_level: float = 0.95,
    title: str | None = None,
    labels: Iterable[int | float | str] | None = None,
    legend: bool = True,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Graph peristimulus time histogram based on the provided spikes.

    Parameters
    ----------
    spike_timings : iterable of float
        Timings of all spikes of interest.
    start_times : iterable of float
        Start times of each time window.
    end_times : iterable of float
        End times of each time window.
    alignments : iterable of float
        Alignment times all other times will be relative to.
    roi : list of (iterable of float, iterable of float) or None, default = None
        Time boundaries for region(s) of interest to visualize.
    time_scale: float, default = 1e-3
        Scale to convert given time unit to seconds.
    window : float, default = 500.0
        Size of sliding window.
    step : float, default = 16.0
        Size of step to take for sliding window operation.
    inclusive : bool, default = True
        If True, sliding window will include spikes outside the defined start/end boundaries when computing spike rates.
        Otherwise, only spikes within bounardies will be considered and windows with size less than `window` will be used near bonudaries.
    ma_size : int, default = 1
        Size of the rolling window to apply moving average smoothing.
    confidence_level : float
        Desired confidence level for estimating mean spike rate.
    title : str or None, default = None
        Title of the PTSH.
    labels : iterable of int or iterable of float or iterable of str or None:
        Labels for each time window that is mapped to determine the color of plot elements.
    legend : bool, default = True,
        If True, graph the legend unless `labels` are not provided.
    ax : `matplotlib.pyplot.Axes` | None, default = None
        Plot in which to graph the PSTH.

    Returns
    -------
    `matplotlib.pyplot.Axes`
        Plot containing the RSTH (same as `ax` if provided).
    """

    # create new figure if not provided
    if ax is None:
        _ = plt.figure()
        ax = plt.gca()

    # estimate spike rates
    pre_align = (alignments - start_times).mean()
    post_align = (end_times - alignments).mean()
    timestamps, spike_rates = spike_rates_sw(
        spike_timings,
        alignments - pre_align,
        alignments + post_align,
        alignments=alignments,
        time_scale=time_scale,
        window=window,
        step=step,
        inclusive=inclusive,
    )
    spike_rates = pd.DataFrame(spike_rates, columns=timestamps[0])
    if ma_size > 1:
        spike_rates = spike_rates.T.rolling(ma_size, center=True)
        spike_rates = spike_rates.mean().T.dropna(axis=1)

    # plot PSTH
    if labels is None:
        mean = spike_rates.mean()
        ci = spike_rates.apply(lambda x: confidence_interval(x, confidence_level))
        ax.plot(mean)
        ax.fill_between(ci.columns.astype(float), ci.iloc[0], ci.iloc[1], alpha=0.2)
    else:
        spike_rates["labels"] = labels
        for label, l_spikes in split_by(spike_rates, by="labels").items():
            mean = l_spikes.mean()
            ci = l_spikes.apply(lambda x: confidence_interval(x, confidence_level))
            lines = ax.plot(mean, label=label)
            ax.fill_between(
                ci.columns.astype(float),
                ci.iloc[0],
                ci.iloc[1],
                alpha=0.2,
                color=lines[0].get_color(),
            )
        if legend:
            ax.legend()

    # plot region of interests
    if roi is not None:
        for b1, b2 in roi:
            b1 = np.mean(b1 - alignments)
            b2 = np.mean(b2 - alignments)
            ax.axvspan(b1, b2, color="grey", alpha=0.1)
            ax.axvline(b1, color="grey", alpha=0.5)
            ax.axvline(b2, color="grey", alpha=0.5)

    # adjust plot elements
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Spike Rate [Hz]")
    ax.set_xlim(-pre_align, post_align)
    if title is not None:
        ax.set_title(title)

    return ax
