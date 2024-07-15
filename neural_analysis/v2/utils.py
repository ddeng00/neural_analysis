from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat


def remove_subsets(lst: list[list]) -> list[list]:
    lst = sorted(lst, key=len)
    result = []
    for i, x in enumerate(lst):
        if not any(set(x).issubset(set(lst[j])) for j in range(i + 1, len(lst))):
            result.append(x)
    return result


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


def create_rolling_window(
    start: float, stop: float, step: float, width: float, exluce_oob=False
):
    """
    Generate rolling windows based on the given start and end times.

    Parameters
    ----------
    start : float
        Start time.
    stop : float
        End time.
    step : float
        Step size between consecutive windows.
    width : float
        Size of each rolling window.
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

    center = start
    starts, ends, centers = [], [], []
    half_width = width / 2

    while center <= stop:
        w_start = center - half_width
        w_end = center + half_width
        if exluce_oob:
            w_start = max(w_start, start)
            w_end = min(w_end, stop)
        starts.append(w_start)
        ends.append(w_end)
        centers.append(center)
        center += step

    return np.array(starts), np.array(ends), np.array(centers)


def isin_2d(x1, x2):
    """
    Check if any of the rows in a 2D array `x1` are present in a 2D array `x2`.

    Parameters:
    -----------
    x1 : numpy.ndarray
        A 2D numpy array where each row is a vector to check for presence in `x2`.
    x2 : numpy.ndarray
        A 2D numpy array where each row is a vector to check against.

    Returns:
    --------
    bool
        True if any row in `x1` is present in `x2`, False otherwise.

    Example:
    --------
    >>> import numpy as np
    >>> x1 = np.array([[1, 2, 3], [4, 5, 6]])
    >>> x2 = np.array([[7, 8, 9], [1, 2, 3]])
    >>> isin_2d(x1, x2)
    True
    """

    return (x1[:, None] == x2).all(-1).any(-1)


def pvalue_to_decimal(pvalue: float, levels: list[float] = [0.05, 0.01, 0.001]) -> str:
    """
    Convert p-value to a string representation using asterisks.

    Parameters
    ----------
    pvalue : float
        The p-value to convert.
    levels : list of float, default=[0.05, 0.01, 0.001]
        The significance levels for conversion. The number of asterisks corresponds to the number of levels the p-value crosses.
        For example, if levels are [0.05, 0.01, 0.001] and p-value is 0.007, it would return '**' because 0.007 < 0.05 and 0.007 < 0.01.

    Returns
    -------
    str
        A string of asterisks representing the significance level of the p-value.
    """
    levels = sorted(levels, reverse=True)
    ret_val = "ns"
    for i, level in enumerate(levels):
        if pvalue <= level:
            ret_val = "*" * (i + 1)
    return ret_val
