import numpy as np
import numpy.typing as npt
from sklearn.utils import check_random_state


def count_smallest_condition(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    groups: npt.ArrayLike | None = None,
    *,
    unique_groups: npt.ArrayLike | None = None,
) -> int:
    """
    Count the number of samples in the smallest condition.

    This function counts the number of samples in the smallest condition.
    If the number of expected conditions is not met, the function returns None.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data. Ignored, only for compatibility with other functions.
    y : array-like of shape (n_samples,)
        The target variable for supervised learning problems.
    groups : array-like of shape (n_samples,) or None, default=None
        Group/condition labels. If provided, resampling is done within each group.
    unique_groups : array-like of shape (n_groups,) or None, default=None
        Unique group labels. If provided, the function will check if all groups are present and return 0 if not.

    Returns
    -------
    smallest_condition : int
        Number of samples in the smallest condition. If not all expected conditions are present, returns 0.
    """

    # Check if all groups are present
    if unique_groups is not None and set(groups) != set(unique_groups):
        return 0

    # Count samples in each condition
    conds = y if groups is None else np.column_stack((y, groups))
    _, cond_cnts = np.unique(conds, axis=0, return_counts=True)
    return cond_cnts.min()


def condition_balanced_resample(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    groups: npt.ArrayLike | None = None,
    *,
    unique_groups: npt.ArrayLike | None = None,
    n_samples: int | None = None,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Resample data while balancing conditions based on groups and class labels.

    This function resamples the data to ensure balanced class distribution within groups.
    If the number of samples is not provided, the function will resample to match the
    size of the smallest conditional subset. If insufficient samples are available or the
    number of expected conditions is not met, the function returns None.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        The target variable for supervised learning problems.
    groups : array-like of shape (n_samples,) or None, default=None
        Group/condition labels. If provided, resampling is done within each group.
    unique_groups : array-like of shape (n_groups,) or None, default=None
        Unique group labels. If provided, the function will check if all groups are present and return None if not.
    n_samples : int or None, default=None
        Number of samples per class per condition to sample.
        If None, resample to the size of the smallest conditional subset.
    random_state : int or `numpy.random.RandomState` or None, default=None
        Random state for reproducibility.

    Returns
    -------
    X_resampled : ndarray
        Resampled training data.
    y_resampled : ndarray
        Resampled target variable.
    groups_resampled : ndarray, optional
        Resampled group labels if groups are provided.
    """

    rng = check_random_state(random_state)
    X, y = np.asarray(X), np.asarray(y)

    # Check if all groups are present
    if unique_groups is not None and set(groups) != set(unique_groups):
        return None

    # Process conditions
    conds = y if groups is None else np.column_stack((y, groups))
    unique_conds, cond_invs, cond_cnts = np.unique(
        conds, axis=0, return_inverse=True, return_counts=True
    )
    if not n_samples:
        n_samples = cond_cnts.min()
    elif n_samples > cond_cnts.min():
        return None

    # Resample conditions
    to_keep = []
    for i in range(len(unique_conds)):
        cond_inds = np.nonzero(cond_invs == i)[0]
        cond_inds = rng.choice(cond_inds, n_samples, replace=False)
        to_keep.append(cond_inds)
    to_keep = np.concatenate(to_keep)

    if groups is None:
        return X[to_keep], y[to_keep]
    else:
        return X[to_keep], y[to_keep], groups[to_keep]
