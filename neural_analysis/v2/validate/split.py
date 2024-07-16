from collections import defaultdict

import numpy as np
import numpy.typing as npt
from sklearn.model_selection._split import (
    GroupsConsumerMixin,
    BaseCrossValidator,
    StratifiedKFold,
    _RepeatedSplits,
)


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

    def get_group_indices(self, X, y, groups):
        left_grp_inds = defaultdict(list)
        right_grp_inds = defaultdict(list)
        for i, (grp, yi) in enumerate(zip(groups, y)):
            grp = tuple(grp)
            if yi == y[0]:
                left_grp_inds[grp].append(i)
            else:
                right_grp_inds[grp].append(i)
        return left_grp_inds, right_grp_inds

    def _iter_test_indices(self, X, y, groups):
        left_grp_inds, right_grp_inds = self.get_group_indices(X, y, groups)
        for left_inds in left_grp_inds.values():
            for right_inds in right_grp_inds.values():
                yield np.concatenate([left_inds, right_inds])

    def get_n_splits(self, X, y, groups):
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
