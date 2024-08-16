import numpy as np
import numpy.typing as npt


def get_spikes(
    spikes: npt.ArrayLike,
    starts: npt.ArrayLike,
    ends: npt.ArrayLike,
    alignments: npt.ArrayLike | None = None,
    *,
    sorted: bool = True,
) -> list[np.ndarray]:
    """
    Return spikes occurring in trial time windows.

    Parameters
    ----------
    spikes : array-like of shape (n_events,)
        Array of spike times.
    starts : array-like of shape (n_trials,)
        Array of start times for the trial windows.
    ends : array-like of shape (n_trials,)
        Array of end times for the trial windows.
    alignments : array-like of shape (n_trials,) or None, default=None
        Array of times within each trial window for alignment.
        If None, align to the start times.
    sorted : bool, default=True
        Whether the timings are sorted in ascending order.

    Returns
    -------
    list of `numpy.ndarray`
        List of arrays, each containing the spikes occurring within the respective trial time window.
    """

    spikes = np.asarray(spikes)
    starts = np.asarray(starts)
    ends = np.asarray(ends)

    if not sorted:
        spikes = np.sort(spikes)
        inds = np.argsort(starts)
        starts, ends = starts[inds], ends[inds]
    if alignments is None:
        alignments = starts

    spikes_in_windows = []
    for start, end, alignment in zip(starts, ends, alignments):
        start = np.searchsorted(spikes, start, side="left")
        end = np.searchsorted(spikes, end, side="right")
        spikes_in_windows.append(spikes[start:end] - alignment)

    return spikes_in_windows


def count_spikes(
    spikes: npt.ArrayLike,
    starts: npt.ArrayLike,
    ends: npt.ArrayLike,
    *,
    sorted: bool = True,
) -> list[np.ndarray]:
    """
    Return the count of spikes occurring in trial time windows.

    Parameters
    ----------
    spikes : array-like of shape (n_events,)
        Array of spike times.
    starts : array-like of shape (n_trials,)
        Array of start times for the trial windows.
    ends : array-like of shape (n_trials,)
        Array of end times for the trial windows.
    sorted : bool, default=True
        Whether the timings are sorted in ascending order.

    Returns
    -------
    list of int
        List of integers, each representing the count of spikes occurring within the respective
    """

    spikes = np.asarray(spikes)
    starts = np.asarray(starts)
    ends = np.asarray(ends)

    if not sorted:
        spikes = np.sort(spikes)
        inds = np.argsort(starts)
        starts, ends = starts[inds], ends[inds]

    counts_in_windows = []
    for start, end in zip(starts, ends):
        start = np.searchsorted(spikes, start, side="left")
        end = np.searchsorted(spikes, end, side="right")
        counts_in_windows.append(end - start)

    return counts_in_windows


def compute_spike_rates(
    spikes: npt.ArrayLike,
    starts: npt.ArrayLike,
    ends: npt.ArrayLike,
    *,
    sorted: bool = True,
    unit_conversion: float = 1.0,
) -> list[float]:
    """
    Compute the spike rates in trial time windows.

    Parameters
    ----------
    spikes : array-like of shape (n_events,)
        Array of spike times.
    starts : array-like of shape (n_trials,)
        Array of start times for the trial windows.
    ends : array-like of shape (n_trials,)
        Array of end times for the trial windows.
    sorted : bool, default=True
        Whether the timings are sorted in ascending order.
    unit_conversion : float, default=1.0
        Conversion factor to convert spike times to seconds.

    Returns
    -------
    list of float
        List of floats, each representing the spike rate (spikes per unit time) within the respective trial time window.
    """

    counts = count_spikes(spikes, starts, ends, sorted=sorted)
    durations = np.subtract(ends, starts) * unit_conversion
    spike_rates = [cnt / dur for cnt, dur in zip(counts, durations)]

    return spike_rates
