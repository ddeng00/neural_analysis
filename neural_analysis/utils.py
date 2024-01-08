from collections.abc import Iterable
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat
import scipy.stats as stats


def validate_file(path: Path | str) -> Path:
    """
    Validate a file path and return a `pathlib.Path` object.

    Parameters
    ----------
    path : "pathlib.Path" or str
        File path to be validated.

    Returns
    -------
    path : "pathlib.Path"
        Validated file path.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File '{path}' does not exist.")
    if not path.is_file():
        raise FileNotFoundError(f"'{path}' is not a file.")
    return path


def validate_dir(path: Path | str) -> Path:
    """
    Validate a directory path and return a `pathlib.Path` object.

    Parameters
    ----------
    path : "pathlib.Path" or str
        Directory path to be validated.

    Returns
    -------
    path : "pathlib.Path"
        Validated directory path.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Directory '{path}' does not exist.")
    if not path.is_dir():
        raise FileNotFoundError(f"'{path}' is not a directory.")
    return path


def natsort_path_key(path: Path | str) -> list[object]:
    """
    Return key for natural sort order.

    Parameters
    ----------
    path : "pathlib.Path" or str
        Path to be sorted by natural sort order.

    Returns
    -------
    list of object
        Key for natural sort order.
    """

    path = Path(path)
    ns = re.compile("([0-9]+)")
    return [int(s) if s.isdigit() else s.lower() for s in ns.split(path.name)]


def sanitize_filename(filename: str) -> str:
    return "".join(c for c in filename if (c.isalnum() or c in "._- "))


def read_mat(path: Path | str) -> dict[str:list]:
    """
    Read a `.mat` file into python `dict` format.

    Metadata from the original `.mat` file is dropped.

    Parameters
    ----------
    path : `pathlib.Path` or str
        Path to `.mat` file to be read.

    Returns
    -------
    dict of {str : list}
        Dictionary containing relevant data from the specified `.mat` file.
    """

    mat = loadmat(path, simplify_cells=True)
    return {k: v for k, v in mat.items() if not k.startswith("__")}


def confidence_interval(
    data: Iterable[int | float], confidence_level: float = 0.95
) -> tuple[float, float]:
    """
    Calculate the confidence interval for the sample mean.

    Parameters
    ----------
    data : iterable of int or float
        Sample data.
    confidence_level : float [0, 1], default = 0.95
        Desired confidence level.

    Returns
    -------
    lower : float
        Lower bounds of the confidence interval.
    upper : float
        Upper bounds of the confidence interval.
    """

    mean = np.mean(data)
    t = stats.t.ppf((1 + confidence_level) / 2, df=len(data) - 1)
    moe = stats.sem(data) * t
    lower, upper = mean - moe, mean + moe
    return lower, upper


def split_by(
    data: pd.DataFrame | pd.Series,
    by: str | Iterable[str] | None = None,
) -> dict[object : pd.DataFrame | pd.Series | pd.Index]:
    """
    Split a `pandas.DataFrame` or `pandas.Series` into groups.

    The column(s) or value(s) used for group assignment is dropped.

    Parameters
    ----------
    data : `pandas.DataFrame` or `pandas.Series`
        Data to be split.
    by : str or iterable of str or None, default = None
        Column label or labels to split `data` by.
        Ignored when `data` is a `pandas.Series`.

    Returns
    -------
    groups : dict of {object : `pandas.DataFrame` or `pandas.Series` or `pandas.Index`}
        Dictionary of separated groups with group value(s) as keys.
    """

    if isinstance(data, pd.DataFrame):
        if by is None:
            raise "Needs to specify at least one column."
        groups = data.groupby(by).groups
        groups = {val: data.loc[idx].drop(by, axis=1) for val, idx in groups.items()}
    elif isinstance(data, pd.Series):
        groups = data.groupby(data).groups
    else:
        raise "Unsupported type."

    return groups


def get_spikes(
    spike_timings: Iterable[float],
    start_times: Iterable[float],
    end_times: Iterable[float],
    alignments: Iterable[float] | None = None,
) -> list[list[float]]:
    """
    Return timings of all spikes occured during specified time window.

    Parameters
    ----------
    spike_timings : iterable of float
        Timings of all spikes of interest.
    start_times : iterable of float
        Start times of each time window.
    end_times : iterable of float
        End times of each time window.
    alignments : iterable of float or None, default = None
        If provided, spike timings of each window will be relative to the alignments.

    Returns
    -------
    spikes_in_windows : list of list of float
        A list of spike timings for each time window.
    """

    if len(start_times) != len(end_times):
        raise "The lengths of trial start and end times do no match."
    if alignments is None:
        alignments = np.zeros_like(start_times)
    elif len(alignments) != len(start_times):
        raise "The number of alingment times much match the number of trials."

    spike_timings = np.asarray(spike_timings)
    spikes_in_windows = []
    for start, end, offset in zip(start_times, end_times, alignments):
        mask = (spike_timings >= start) & (spike_timings <= end)
        timings = spike_timings[mask] - offset
        # spike_timings = spike_timings[~mask]
        spikes_in_windows.append(np.sort(timings).tolist())

    return spikes_in_windows


def spike_counts(
    spike_timings: Iterable[float],
    start_times: Iterable[float],
    end_times: Iterable[float],
) -> list[float]:
    """
    Count spike occurances in specified time windows.

    Parameters
    ----------
    spike_timings : iterable of float
        Timings of all spikes of interest.
    start_times: iterable of float
        Start times of each time window.
    end_times : iterable of float
        End times of each time window.
    alignments : iterable of float or None, default = None
        If provided, spike timings of each window will be relative to the alignments.

    Returns
    -------
    list of float
        Spike counts for each time window.
    """

    spikes_in_windows = get_spikes(spike_timings, start_times, end_times)
    return [len(s) for s in spikes_in_windows]


def spike_rates_sw(
    spike_timings: Iterable[float],
    start_times: Iterable[float],
    end_times: Iterable[float],
    alignments: Iterable[float] | None = None,
    time_scale: float = 1e-3,
    window: float = 250.0,
    step: float = 10.0,
    inclusive: bool = True,
) -> tuple[list[list[float]], list[list[float]]]:
    """
    Estimate spike rates in specified time windows using sliding window.

    Parameters
    ----------
    spike_timings : iterable of float
        Timings of all spikes of interest.
    start_times : iterable of float
        Start times of each time window.
    end_times : iterable of float
        End times of each time window.
    alignments : iterable of float or None, default = None
        If provided, spike timings of each window will be relative to the alignments.
    time_scale: float, default = 1e-3
        Scale to convert given time unit to seconds.
    window : float, default = 500.0
        Size of sliding window.
    step : float, default = 16.0
        Size of step to take for sliding window operation.
    inclusive : bool, default = True
        If True, sliding window will include spikes outside the defined start/end boundaries when computing spike rates.
        Otherwise, only spikes within bounardies will be considered and windows with size less than `window` will be used near bonudaries.

    Returns
    -------
    timestamps : list of list of float
        List of timestamps for each estimated spike rates per time window.
    spike_rates : list of list of float
        List of estimated spike rates for each time window.
    """

    # TODO: validate parameters

    half_w = window / 2
    spike_timings = np.asarray(spike_timings)
    timestamps, spike_rates = [], []
    for start, end, align in zip(start_times, end_times, alignments):
        w_timestamps = np.arange(start, end, step)
        w_starts = w_timestamps - half_w
        w_ends = w_timestamps + half_w
        if not inclusive:
            w_starts = np.clip(w_starts, start, end)
            w_ends = np.clip(w_ends, start, end)
        w_cnts = spike_counts(spike_timings, w_starts, w_ends)
        w_rates = np.divide(w_cnts, w_ends - w_starts)
        w_rates = w_rates / time_scale
        timestamps.append((w_timestamps - align).round(5).tolist())
        spike_rates.append(w_rates.round(5).tolist())
    return timestamps, spike_rates
