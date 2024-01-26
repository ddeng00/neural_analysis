import multiprocessing as mp
import os

import numpy as np
import numpy.typing as npt
import scipy.stats as stats


def _get_spikes_in_window(
    spike_times: npt.ArrayLike,
    start_time: float,
    end_time: float,
) -> npt.NDArray:
    """
    Return spikes occurring in a trial time window.

    Parameters
    ----------
    spike_times : array-like of shape (n_spikes,)
        Timings of all spikes of interest. Assumed to be strictly increasing.
    start_time : float
        Start time of the trial time window.
    end_time : float
        End time of the trial time window.

    Returns
    -------
    spikes : ndarray of shape (n_spikes_in_window,)
        Spikes occurring in the trial time window.
    truncated : ndarray of shape (n_spikes_in_window,)
        Truncated spike times. Only returned if return_truncated is True.
    """

    if len(spike_times) == 0:
        return 0

    # find the index of the first spike that occurs within the time window.
    start_ind = np.searchsorted(spike_times, start_time, side="left")
    # truncate the spike times to exclude spikes that occur before the time window.
    truncated = spike_times[start_ind:]
    # find the index of the last spike that occurs within the time window.
    end_ind = np.searchsorted(truncated, end_time, side="right")
    spikes = truncated[:end_ind]

    return spikes


def get_spikes(
    spike_times: npt.ArrayLike,
    start_times: npt.ArrayLike,
    end_times: npt.ArrayLike,
    target_times: npt.ArrayLike | None = None,
    n_jobs: int = -1,
) -> list[npt.NDArray]:
    """
    Return spikes occurring in trial time windows.

    Parameters
    ----------
    spike_times : array-like of shape (n_spikes,)
        Timings of all spikes of interest. Assumed to be strictly increasing.
    start_times : array-like of shape (n_trials,)
        Start times of each trial time window. Assumed to be strictly increasing.
    end_times : array-like of shape (n_trials,)
        End times of each trial time window. Assumed to be strictly increasing.
    start_times : array-like of shape (n_trials,)
        Target times to align spike timings to. Assumed to be strictly increasing.
        If None, use start_times.
    n_jobs : int, default = -1
        Number of processes to use. If -1, use all available CPUs. If 0 or 1, do not use multiprocessing.

    Returns
    -------
    spikes : list of ndarrays of shape (n_spikes_in_window,)
        Spikes occurring in each trial time window.
    """

    # Validate n_jobs
    if n_jobs < -1:
        raise ValueError("n_jobs must be greater than or equal to -1.")
    elif n_jobs == -1:
        n_jobs = os.cpu_count()
    elif n_jobs > 1:
        n_jobs = min(n_jobs, os.cpu_count())

    # Validate inputs
    start_times, end_times = np.asarray(start_times), np.asarray(end_times)
    if len(start_times) != len(end_times):
        raise ValueError("start_times and end_times must have the same length.")
    if target_times is None:
        target_times = start_times
    elif len(start_times) != len(target_times):
        raise ValueError(
            "target_times must have the same length as start_times/end_times."
        )
    target_times, spike_times = np.asarray(target_times), np.asarray(spike_times)

    # If n_jobs is 0 or 1, do not use multiprocessing
    if n_jobs == 0 or n_jobs == 1:
        spikes = []
        for start_time, end_time in zip(start_times, end_times):
            spikes.append(_get_spikes_in_window(spike_times, start_time, end_time))
    else:
        with mp.Pool(n_jobs) as pool:
            spikes = pool.starmap(
                _get_spikes_in_window,
                zip([spike_times] * len(start_times), start_times, end_times),
            )
    spikes = [np.asarray(s) - target_times[i] for i, s in enumerate(spikes)]

    return spikes


def count_spikes(
    spike_times: npt.ArrayLike,
    start_times: npt.ArrayLike,
    end_times: npt.ArrayLike,
    n_jobs: int = -1,
) -> npt.NDArray:
    """
    Count spike occurances in trial time windows.

    Parameters
    ----------
    spike_times : array-like of shape (n_spikes,)
        Timings of all spikes of interest. Assumed to be strictly increasing.
    start_times : array-like of shape (n_trials,)
        Start times of each trial time window. Assumed to be strictly increasing.
    end_times : array-like of shape (n_trials,)
        End times of each trial time window. Assumed to be strictly increasing.
    n_jobs : int, default = -1
        Number of processes to use. If -1, use all available CPUs. If 0 or 1, do not use multiprocessing.

    Returns
    -------
    spike_counts : ndarray of shape (n_trials,)
        Number of spikes in each time window.
    """

    spikes = get_spikes(spike_times, start_times, end_times, n_jobs=n_jobs)
    spike_counts = np.asarray([len(s) for s in spikes])
    return spike_counts


def PSTH(
    spike_times: npt.ArrayLike,
    start_times: npt.ArrayLike,
    end_times: npt.ArrayLike,
    target_times: npt.ArrayLike | None = None,
    window_size: float = 500.0,
    step_size: float = 7.8,
    include_oob: bool = True,
    return_trials: bool = False,
    return_rates: bool = False,
    n_jobs: int = 1,
) -> tuple[npt.NDArray, ...]:
    """
    Calculate peri-stimulus time histogram (PSTH) from spike times.

    Parameters
    ----------
    spike_times : array-like of shape (n_spikes,)
        Timings of all spikes of interest in [ms]. Assumed to be strictly increasing.
    start_times : array-like of shape (n_trials,)
        Start times of each trial time window in [ms]. Assumed to be strictly increasing.
    end_times : array-like of shape (n_trials,)
        End times of each trial time window in [ms]. Assumed to be strictly increasing.
    target_times : array-like of shape (n_trials,) or None, default = None
        Target times to align spike timings to in [ms]. If None, use start_times.
    window_size : float, default = 500.0
        Size of time window for pooling spikes in [ms].
    step_size : float, default = 7.8
        Amount to time to shift window for each step in [ms].
    include_oob : bool, default = True
        Whether to include out-of-bounds spikes in windows.
    return_rates : bool, default = False
        Whether to return spike rates instead of spike counts.
    return_trials : bool, default = False
        Whether to return spike counts/rates for each trial.
    n_jobs : int, default = -1
        Number of processes to use. If -1, use all available CPUs. If 0 or 1, do not use multiprocessing.

    Returns
    -------
    timesteps : ndarray of shape (n_windows,)
        Timesteps of each PSTH bin in [ms].
    spikes : ndarray of shape (n_trials, n_windows)
        Spike counts/rates in each PSTH bin per trial. Only returned if return_trials is True.
    mean_spikes : ndarray of shape (n_windows,)
        Average spike counts of each PSTH bin.
        If return_rates is True, this is the average spike rate in [Hz]. Otherwise, this is the average spike count in [ms].
    se_spikes : ndarray of shape (n_windows,)
        Standard error of the mean of spike counts/rates in each PSTH bin.
    """

    # Validate inputs
    start_times, end_times = np.asarray(start_times), np.asarray(end_times)
    if len(start_times) != len(end_times):
        raise ValueError("start_times and end_times must have the same length.")
    if target_times is None:
        target_times = start_times
    elif len(start_times) != len(target_times):
        raise ValueError(
            "target_times must have the same length as start_times/end_times."
        )
    target_times, spike_times = np.asarray(target_times), np.asarray(spike_times)

    # modify start and end times to have consistent trial window size
    pre_target_dur = np.min(target_times - start_times)
    post_target_dur = np.min(end_times - target_times)
    trial_dur = pre_target_dur + post_target_dur
    start_times = target_times - pre_target_dur
    end_times = target_times + post_target_dur

    # info for sliding window
    half_ws = window_size / 2
    n_windows = int(trial_dur / step_size) + 1
    timesteps = np.arange(n_windows) * step_size

    # calculate start and end times of each sliding window
    rep_starts = np.repeat(start_times, n_windows)
    rep_ends = np.repeat(end_times, n_windows)
    win_centers = rep_starts + np.tile(timesteps, len(start_times))
    win_starts = win_centers - half_ws
    win_ends = win_centers + half_ws
    if not include_oob:  # exclude out-of-bounds spikes
        win_starts = np.maximum(win_starts, rep_starts)
        win_ends = np.minimum(win_ends, rep_ends)
    win_sizes = win_ends - win_starts

    # get spike counts for each window
    spikes = count_spikes(spike_times, win_starts, win_ends, n_jobs=n_jobs)
    if return_rates:
        spikes = spikes / (win_sizes / 1000)  # convert to Hz
    spikes = np.asarray(spikes, dtype=np.float64).reshape(-1, n_windows)

    # calcualte mean and se of spikes
    mean_spikes = spikes.mean(axis=0)
    se_spikes = stats.sem(spikes, axis=0)

    # align timesteps to target times
    timesteps = timesteps - pre_target_dur

    if return_trials:
        return timesteps, spikes, mean_spikes, se_spikes
    return timesteps, mean_spikes, se_spikes
