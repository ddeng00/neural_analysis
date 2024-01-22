from pathlib import Path
import re

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.io import loadmat


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


def group_df_by(
    df: pd.DataFrame,
    by: str | list[str],
    drop: bool = True,
) -> dict[str, pd.DataFrame | pd.Series | pd.Index]:
    """
    Group a `pandas.DataFrame` by column(s) and return a dictionary of groups.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Data to be grouped.
    by : str or list of str
        Column label(s) to group by.
    drop : bool, default = True
        Whether to drop the column(s) used for grouping.

    Returns
    -------
    groups : dict of {str : `pandas.DataFrame` or `pandas.Series` or `pandas.Index`}
        Dictionary of separated groups with group value(s) as keys.
    """

    if isinstance(df, pd.DataFrame):
        if by is None:
            raise "Needs to specify at least one column."
        groups = df.groupby(by).groups
        if drop:
            groups = {val: df.loc[idx].drop(by, axis=1) for val, idx in groups.items()}
        else:
            groups = {val: df.loc[idx] for val, idx in groups.items()}
    elif isinstance(df, pd.Series):
        groups = df.groupby(df).groups
    else:
        raise "Unsupported type."

    return groups


def seq_where(arr: npt.ArrayLike, seq: npt.ArrayLike) -> npt.NDArray:
    """
    Find all occurrences of a sequence in an array.

    Parameters
    ----------
    arr : array-like
        Array to search.
    seq : array-like
        Sequence to find.

    Returns
    -------
    ind : ndarray
        Indices of all occurrences of `seq` in `arr`.
    """

    if len(arr) == 0 or len(seq) == 0:
        return np.array([])
    if len(seq) > len(arr):
        return np.array([])
    arr, seq = np.asarray(arr), np.asarray(seq)

    ind = []
    for i in range(len(arr) - len(seq) + 1):
        if np.all(arr[i : i + len(seq)] == seq):
            ind.append(i)
    ind = np.asarray(ind)

    return ind


def generate_group_names(vars: list[npt.ArrayLike], var_names: list[str]) -> list[str]:
    """
    Generate group labels for all combinations of variables.

    Parameters
    ----------
    vars : list of array-like
        A list of arrays with the values of each variable. All arrays must have the same length.
    var_names : list of str
        A list of names of each variable.

    Returns
    -------
    group_names : list of str
        A list of names of each variable combination.
    """

    if len(vars) != len(var_names):
        raise ValueError(
            f"The number of variables ({len(vars)}) must match the number of variable names ({len(var_names)})"
        )
    if len(vars) == 0:
        return []

    vars = [np.asarray(var) for var in vars]
    vars = [var.astype(int) if var.dtype == bool else var for var in vars]

    combiner = lambda var_set: "|".join(
        [f"{col}={var}" for var, col in zip(var_set, var_names)]
    )
    group_names = [combiner(set) for set in zip(*vars)]

    return group_names
