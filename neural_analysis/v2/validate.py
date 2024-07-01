from collections import defaultdict

import numpy as np
import numpy.typing as npt
from sklearn.model_selection._split import (
    GroupsConsumerMixin,
    BaseCrossValidator,
    StratifiedKFold,
    _RepeatedSplits,
)
from sklearn.utils import check_random_state

from .utils import isin_2d


class MultiStratifiedKFold(GroupsConsumerMixin, StratifiedKFold):
    """
    K-Folds cross-validator with condition stratification.

    Provides train/test indices to split data into train/test sets while
    ensuring that class and group labels are evenly distributed across the splits.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle each class's samples before splitting into batches.
    random_state : int or `numpy.random.RandomState` or None, default=None
        Random state for reproducibility.
    """

    def __init__(
        self,
        *,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: int | np.random.RandomState | None = None,
    ):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(
        self, X: npt.ArrayLike, y: npt.ArrayLike, groups: npt.ArrayLike | None = None
    ):
        """
        Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples, n_conditions) or None, default=None
            Group/condition labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """

        if groups is not None:
            y = np.column_stack((y, groups))
            y = list(map(str, y))
        yield from super().split(X, y)


class RepeatedMultiStratifiedKFold(_RepeatedSplits):
    """
    Repeated K-Folds cross-validator with condition stratification.

    Repeats Condition Balanced K-Folds n times with different randomization in each repetition.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.
    random_state : int or None, default=None
        Random seed to use for random number generation.
    """

    def __init__(
        self, *, n_splits: int = 5, n_repeats: int = 10, random_state: int | None = None
    ):
        super().__init__(
            MultiStratifiedKFold,
            n_repeats=n_repeats,
            random_state=random_state,
            n_splits=n_splits,
        )


def shuffle_data(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    groups: npt.ArrayLike,
    random_state: int | np.random.RandomState | None = None,
):
    rng = check_random_state(random_state)
    x = np.copy(x)
    for group in np.unique(groups, axis=0):
        group_mask = isin_2d(groups, group)
        x[group_mask] = x[group_mask][:, rng.permutation(x.shape[1])]
    return x, y, groups


def shuffle_class_labels(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    groups: npt.ArrayLike | None = None,
    *,
    precomputed_group_masks: list[npt.ArrayLike] | None = None,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shuffle class labels.

    This function shuffles the class labels while keeping the data and group labels intact.
    Specifically, if group labels are provided, shuffling is done within each group.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        The target variable for supervised learning problems.
    groups : array-like of shape (n_samples, n_conditions) or None, default=None
        Group/condition labels. If provided, shuffling is done within each group.
    precomputed_group_masks : list of array-like or None, default=None
        Precomputed boolean masks for each group to speed up shuffling.
    random_state : int or `numpy.random.RandomState` or None, default=None
        Random state for reproducibility.

    Returns
    -------
    x : ndarray
        Original training data.
    y_shuffled : ndarray
        Shuffled target variable.
    groups : ndarray, optional
        Original group labels.
    """

    rng = check_random_state(random_state)
    x, y = np.asarray(x), np.asarray(y)
    if groups is not None:
        groups = np.asarray(groups)

    y_shuffled = y.copy()

    if groups is None:
        rng.shuffle(y_shuffled)
        return x, y_shuffled
    else:
        if precomputed_group_masks is not None:
            for group_mask in precomputed_group_masks:
                y_group = y_shuffled[group_mask]
                rng.shuffle(y_group)
                y_shuffled[group_mask] = y_group
        else:
            unique_groups = np.unique(groups)
            for group in unique_groups:
                group_mask = isin_2d(groups, group)
                y_group = y_shuffled[group_mask]
                rng.shuffle(y_group)
                y_shuffled[group_mask] = y_group

        return x, y_shuffled, groups


class LeaveNCrossingsOut(GroupsConsumerMixin, BaseCrossValidator):
    """
    Leave-N-Crossings-Out cross-validator.

    Provides train/test indices to split data into train/test sets where
    n group crossings over bipartite are left out as the test set in each iteration.

    Parameters
    ----------
    n_crossings : int, default=1
        Number of group crossings to leave out in each iteration.
    """

    def __init__(self, n_crossings: int = 1):
        self.n_crossings = n_crossings

    def _iter_test_indices(self, X, y=None, groups=None):
        left_grp_inds = defaultdict(list)
        right_grp_inds = defaultdict(list)
        for i, (grp, yi) in enumerate(zip(groups, y)):
            grp = tuple(grp)
            if yi == y[0]:
                left_grp_inds[grp].append(i)
            else:
                right_grp_inds[grp].append(i)

        for left_inds in left_grp_inds.values():
            for right_inds in right_grp_inds.values():
                yield np.concatenate([left_inds, right_inds])

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object, default=None
            Always ignored, exists for compatibility.
        y : object, default=None
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into train/test set.

        Returns
        -------
        n_splits : int
            The number of splitting iterations.
        """

        n_subgroups = len(np.unique(groups)) // 2
        n1 = n_subgroups**2
        n2 = (n_subgroups - 1) ** 2
        return n1 * n2

    def split(self, X, y, groups):
        """
        Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """

        yield from super().split(X, y, groups)
