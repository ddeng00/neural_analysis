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
    unit: str,
    response: str | list[str],
    condition: str | list[str],
    n_samples_per_cond: int | None = None,
    all_groups_complete: bool = False,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[list[np.ndarray], np.ndarray]:
    """
    Construct a pseudopopulation from the given data.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data.
    unit : str
        Column name containing the units.
    response : str or List[str]
        Column name(s)  containing the response(s).
    condition : str or List[str]
        Column name(s) containing the conditions.
    n_samples_per_cond : int or None, optional
        Number of samples per condition. If None, the minimum number of samples
        across all conditions will be used.
    all_groups_complete : bool, default=False
        If True, all conditions are assumed to be present in each group. If False,
        groups missing conditions will be removed.
    random_state : int, RandomState or None, optional
        Random state for resampling.

    Returns
    -------
    X : `numpy.ndarray` of shape (n_complete_groups, n_samples)
        Array containing the pseudopopulation data.
    conds : `numpy.ndarray` of shape (n_complete_groups, n_conditions)
        Array containing the conditions for each group.
    """

    if not isinstance(condition, list):
        condition = [condition]

    if n_samples_per_cond is None:
        n_samples_per_cond = data.groupby([unit] + condition).size().min()
    if not all_groups_complete:
        data = remove_groups_missing_conditions(
            data, unit, condition, n_samples_per_cond=n_samples_per_cond
        )

    # Note: groupby ensures that noise correlations are destroyed.
    resampled = data.groupby([unit] + condition).sample(
        n=n_samples_per_cond, random_state=random_state
    )

    # Note: previous groupby ensures that conditions are sorted
    if not isinstance(response, list):
        X = np.column_stack(resampled.groupby(unit)[response].apply(np.vstack)).astype(
            float
        )
        conds = resampled[condition].iloc[: len(X)].to_numpy(str)
        return X, conds
    else:
        Xs = [
            np.column_stack(resampled.groupby(unit)[v].apply(np.vstack)).astype(float)
            for v in response
        ]
        conds = resampled[condition].iloc[: len(Xs[0])].to_numpy(str)
        return Xs, conds
