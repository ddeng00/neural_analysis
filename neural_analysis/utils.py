from pathlib import Path
import re

import numpy as np
import numpy.typing as npt
import pandas as pd
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
) -> dict[str, pd.DataFrame | pd.Series]:
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
    df : dict of {str : `pandas.DataFrame` or `pandas.Series`}
        Dictionary of separated groups with group value(s) as keys.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    df = dict(tuple(df.groupby(by)))
    if drop:
        df = {k: v.drop(by, axis=1) for k, v in df.items()}
    return df


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


def generate_group_names(
    vars: list[npt.ArrayLike],
    var_names: list[str],
    keep_name: bool = True,
    separator: str = " | ",
) -> list[str]:
    """
    Generate group labels for all combinations of variables.

    Parameters
    ----------
    vars : list of array-like
        A list of arrays with the values of each variable. All arrays must have the same length.
    var_names : list of str
        A list of names of each variable.
    keep_name : bool, default=True
        Whether to include the variable name in the group label.
    separator : str, default=" | "
        Separator to use between variable names and values.

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

    if keep_name:
        combiner = lambda var_set: separator.join(
            [f"{col}={var}" for var, col in zip(var_set, var_names)]
        )
    else:
        combiner = lambda var_set: separator.join([f"{var}" for var in var_set])
    group_names = [combiner(set) for set in zip(*vars)]

    return group_names


def make_new_groups(
    df: pd.DataFrame,
    columns: list[str],
    column_names: list[str] | None = None,
    group_name: str = "new_group",
    keep_name: bool = True,
    inplace: bool = False,
    separator: str = " | ",
) -> pd.DataFrame | None:
    """
    Add new group labels to `pandas.DataFrame` based on column values.

    Parameters
    ----------
    df : `pandas.DataFrame`
        DataFrame to add group labels to.
    columns : list of str
        Columns to use for grouping.
    column_names : list of str or None, default=None
        Names of the columns to use for grouping.
        If None, original column names are used.
    group_name : str, default="new_group"
        Name of the new group column.
    keep_name : bool, default=True
        Whether to include the variable name in the group label.
    inplace : bool, default=False
        Whether to modify `df` in-place. If False, a copy is returned.
    separator : str, default=" | "

    Returns
    -------
    df : `pandas.DataFrame` or None
        DataFrame with new group labels added. If `inplace` is True, returns None.
    """

    if group_name in df.columns:
        raise ValueError(f"Column '{group_name}' already exists.")

    if not inplace:
        df = df.copy()

    if column_names is None:
        column_names = columns

    df[group_name] = generate_group_names(
        [df[col] for col in columns],
        column_names,
        keep_name=keep_name,
        separator=separator,
    )

    if not inplace:
        return df


def pval_to_decimal(pvalue: float) -> str:
    """
    Convert p-value to decimal format (i.e., asterisks).

    Parameters
    ----------
    pvalue : float
        P-value to be converted.

    Returns
    -------
    str
        P-value in decimal format.

    """

    if pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    else:
        return "n.s."
