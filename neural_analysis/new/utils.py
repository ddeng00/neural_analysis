from pathlib import Path
import re

import numpy as np
import numpy.typing as npt
from scipy.io import loadmat
import scipy.stats as stats


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


def read_mat(path: Path | str) -> dict[str:list]:
    """
    Read a `.mat` file into a python dictionary.

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
    samples: npt.ArrayLike, confidence_level: float = 0.95
) -> float:
    """
    Return the confidence interval of the sample data.

    Parameters
    ----------
    samples : array-like
        Sample data.
    confidence_level : float, default = 0.95
        Confidence level of the confidence interval.

    Returns
    -------
    moe: float
        Margin of error of the sample data.
    """

    t_score = stats.t.ppf((1 + confidence_level) / 2, df=len(samples) - 1)
    moe = stats.sem(samples) * t_score
    return moe
