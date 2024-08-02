import numpy as np
import numpy.typing as npt
import pandas as pd


def inv_norm(arr: npt.ArrayLike, negate: bool = True) -> np.ndarray:
    """
    Inverse normalize an array.

    Parameters
    ----------
    arr : array_like
        Array to be inverse normalized.
    negate : bool, default=True
        If True, negate the array to maintain the original sign.

    Returns
    -------
    `numpy.ndarray`
        Inverse normalized array.
    """

    arr = np.divide(1, arr)
    arr = (arr - arr.mean()) / arr.std()
    return -arr if negate else arr


def remove_groups_missing_conditions(
    data: pd.DataFrame,
    unit: str,
    condition: str | list[str],
    n_conditions: int | None = None,
    n_samples_per_cond: int = 1,
    return_removed: bool = False,
) -> pd.DataFrame:

    # process inputs
    if not isinstance(condition, list):
        condition = [condition]

    # infer the number of conditional groups if not provided
    if n_conditions is None:
        n_conditions = data.groupby(condition).ngroups

    to_remove = []
    # check for missing conditional groups
    for grp, df in data.groupby(unit):
        if df.groupby(condition).ngroups < n_conditions:
            to_remove.append(grp)
    # check for insufficient conditional samples
    min_cond_cnts = data.groupby(unit)[condition].value_counts().groupby(unit).min()
    to_remove.extend(min_cond_cnts[min_cond_cnts < n_samples_per_cond].index)
    to_remove = list(set(to_remove))

    if return_removed:
        data_removed = data[data[unit].isin(to_remove)]
        data = data[~data[unit].isin(to_remove)]
        return data, data_removed
    return data[~data[unit].isin(to_remove)]


def construct_pseudopopulation(
    data: pd.DataFrame,
    var: str,
    group: str,
    conditions: str | list[str],
    n_samples_per_cond: int | None = None,
    all_groups_complete: bool = False,
    bootstrap: bool = False,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct a pseudopopulation from the given data.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data.
    var : str
        Column name containing the variable.
    group : str
        Column name containing the groups.
    conditions : List[str]
        Column names containing the conditions.
    n_samples_per_cond : int or None, optional
        Number of samples per condition. If None, the minimum number of samples
        across all conditions will be used.
    all_groups_complete : bool, default=False
        If True, all conditions are assumed to be present in each group. If False,
        groups missing conditions will be removed.
    bootstrap : bool, default=False
        If True, resample the data with replacement.
    random_state : int, RandomState or None, optional
        Random state for resampling.

    Returns
    -------
    X : `numpy.ndarray` of shape (n_complete_groups, n_samples)
        Array containing the pseudopopulation data.
    conds : `numpy.ndarray` of shape (n_complete_groups, n_conditions)
        Array containing the conditions for each group.
    """

    if not isinstance(conditions, list):
        conditions = [conditions]

    if n_samples_per_cond is None:
        n_samples_per_cond = data.groupby([group] + conditions).size().min()
    if not all_groups_complete:
        data = remove_groups_missing_conditions(
            data, group, conditions, n_samples_per_cond=n_samples_per_cond
        )

    # Note: groupby ensures that noise correlations are destroyed.
    resampled = data.groupby([group] + conditions).sample(
        n=n_samples_per_cond, replace=bootstrap, random_state=random_state
    )

    # Note: previous groupby ensures that conditions are sorted
    X = np.column_stack(resampled.groupby(group)[var].apply(np.vstack))
    X = X.astype(float)
    conds = resampled[conditions].iloc[: len(X)].to_numpy(str)

    return X, conds
