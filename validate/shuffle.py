import numpy as np
import numpy.typing as npt
from sklearn.utils import check_random_state

from ..utils import isin_2d


def permute_data(
    X: npt.ArrayLike, random_state: int | np.random.RandomState | None = None
) -> np.ndarray:
    """
    Permute the rows of a 2D array.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data. Must be a 2D array.

    random_state : int or `numpy.random.RandomState` or None, default=None
        Random state for reproducibility.

    Returns
    -------
    X_permuted : numpy.ndarray
        The permuted input data.
    """

    rng = check_random_state(random_state)
    perm_inds = rng.permutation(X.shape[0])
    return X[perm_inds]


def rotate_data_within_groups(
    X: npt.ArrayLike,
    groups: npt.ArrayLike,
    random_state: int | np.random.RandomState | None = None,
) -> np.ndarray:
    """
    Rotate the columns of a 2D array within groups.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data. Must be a 2D array.
    groups : array-like of shape (n_samples, n_conditions)
        Group labels for the samples.
    random_state : int or `numpy.random.RandomState` or None, default=None
        Random state for reproducibility.

    Returns
    -------
    X_rotated : numpy.ndarray
        The rotated input data.
    """

    rng = check_random_state(random_state)
    X = np.copy(X)
    for group in np.unique(groups, axis=0):
        group_mask = isin_2d(groups, group)
        X[group_mask] = X[group_mask][:, rng.permutation(X.shape[1])]
    return X
