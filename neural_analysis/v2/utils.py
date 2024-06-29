from pathlib import Path
import re

import numpy as np
import numpy.typing as npt
from scipy.io import loadmat


def remove_if_exists(path: Path | str) -> None:
    """
    Remove a file or directory if it exists.

    Parameters
    ----------
    path : `pathlib.Path` or str
        File or directory to be removed.
    """

    path = Path(path)
    if path.exists():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()


def validate_file(path: Path | str) -> Path:
    """
    Validate a file path and return a `pathlib.Path` object.

    Parameters
    ----------
    path : `pathlib.Path` or str
        File path to be validated.

    Returns
    -------
    path : `pathlib.Path`
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
    path : `pathlib.Path` or str
        Directory path to be validated.

    Returns
    -------
    path : `pathlib.Path`
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
    path : `pathlib.Path` or str
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
    """Return a sanitized version of a filename."""
    return "".join(c for c in filename if (c.isalnum() or c in "._- "))


def read_mat(path: Path | str) -> dict[str, list]:
    """
    Read a `.mat` file into a Python dictionary.

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

    mat = loadmat(str(path), simplify_cells=True)
    return {k: v for k, v in mat.items() if not k.startswith("__")}


def create_rolling_windows(
    starts: npt.ArrayLike,
    ends: npt.ArrayLike,
    window_size: float,
    step_size: float,
    *,
    exclude_oob: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate rolling windows based on the given start and end times.

    Parameters
    ----------
    starts : array-like of shape (n_trials,)
        Array of start times for each trial.
    ends : array-like of shape (n_trials,)
        Array of end times for each trial.
    window_size : float
        Size of each rolling window.
    step_size : float
        Step size between consecutive windows.
    exclude_oob : bool, default=False
        If True, adjust windows to stay within the bounds of the trials.

    Returns
    -------
    tuple of `numpy.ndarray`
        A tuple containing:
        - Array of start times for each rolling window.
        - Array of end times for each rolling window.
        - Array of center times for each rolling window.
    """

    rolling_starts, rolling_ends, rolling_centers = [], [], []
    half_win_size = window_size / 2

    for start, end in zip(starts, ends):
        center = start
        w_starts, w_ends, w_times = [], [], []
        while center <= end:
            w_start = center - half_win_size
            w_end = center + half_win_size
            if exclude_oob:
                w_start = max(w_start, start)
                w_end = min(w_end, end)
            w_starts.append(w_start)
            w_ends.append(w_end)
            w_times.append(center)
            center += step_size
        rolling_starts.append(w_starts)
        rolling_ends.append(w_ends)
        rolling_centers.append(w_times)

    return rolling_starts, rolling_ends, rolling_centers


def isin_2d(x1, x2):
    return (x1[:, None] == x2).all(-1).any(-1)
