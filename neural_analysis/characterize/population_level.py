from abc import ABC, abstractmethod
from warnings import warn
from itertools import combinations, permutations

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    cross_validate,
    cross_val_score,
    StratifiedKFold,
    RepeatedStratifiedKFold,
)
from scipy.spatial.distance import cosine
from sklearn.base import clone

from joblib import Parallel, delayed

from ..partition import make_balanced_dichotomies
from ..preprocess import remove_groups_missing_conditions, construct_pseudopopulation
from ..validate import (
    MultiStratifiedKFold,
    RepeatedMultiStratifiedKFold,
    LeaveNCrossingsOut,
    permute_data,
    rotate_data_within_groups,
)
from ..utils import isin_2d


class _BaseDichotomyEstimator(ABC):
    def __init__(
        self,
        data: pd.DataFrame,
        unit: str,
        response: str | list[str],
        condition: str | list[str],
        *,
        n_conditions: int | None = None,
        n_samples_per_cond: int | None = None,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:

        # store data
        self.data = data
        self.unit = unit
        self.response = response
        self.condition = condition

        # get random state
        self.random_state = np.random.RandomState(random_state)

        # infer number of samples per condition if not provided
        if n_samples_per_cond is None:
            n_samples_per_cond = data.groupby(condition).size().min()
        self.n_samples_per_cond = n_samples_per_cond

        # remove units missing conditional trials
        self.n_init = data[unit].nunique()
        data = remove_groups_missing_conditions(
            data,
            unit,
            condition,
            n_conditions=n_conditions,
            n_samples_per_cond=n_samples_per_cond,
        )
        self.n_valid = data[unit].nunique()

        # define dichotomies
        u_conds = data[condition].drop_duplicates().values
        dichots, dichot_names, dichot_diffs = make_balanced_dichotomies(
            u_conds, cond_names=condition, return_one_sided=True
        )
        self.dichotomies = dichots
        self.dichotomy_names = dichot_names
        self.dichotomy_difficulties = dichot_diffs

    @property
    @abstractmethod
    def shuffle_fn(self):
        raise NotImplementedError

    def resample_and_score(
        self,
        n_resamples: int = 1,
        *,
        permute: bool = False,
        n_splits: int = 5,
        n_repeats: int = 1,
        shuffle: bool = False,
        clf: ClassifierMixin = LinearSVC,
        clf_kwargs: dict | None = None,
        return_clfs: bool = False,
        n_jobs: int | None = None,
        random_state: int | np.random.RandomState | None = None,
    ):
        def helper(i):
            # generate new random state
            global_state = self.random_state.get_state()
            

            # construct condition-balanced pseudopopulation of units
            X, condition = construct_pseudopopulation(
                self.data,
                self.response,
                self.unit,
                self.condition,
                n_samples_per_cond=self.n_samples_per_cond,
                all_groups_complete=True,
                random_state=self.random_state,
            )

            # fit and validate each dichotomy
            res = {}
            for dichot, dichot_name in zip(self.dichotomies, self.dichotomy_names):
                y = isin_2d(condition, dichot).astype(int)
                res[dichot_name] = self.fit_and_validate(
                    X,
                    y,
                    condition,
                    n_splits=n_splits,
                    n_repeats=n_repeats,
                    shuffle=shuffle,
                    clf=clf,
                    clf_kwargs=clf_kwargs,
                    return_clfs=return_clfs,
                    n_jobs=n_jobs,
                    random_state=random_state,
                )
            if not permute:
                return res
            
            # estimate permutation null
            res_perm = {}
            X = self.shuffle_fn(X)
            for dichot, dichot_name in zip(self.dichotomies, self.dichotomy_names):
                y_perm = self.random_state.permutation(y)
                res_perm[dichot_name] = self.fit_and_validate(
                    X,
                    y_perm,
                    condition,
                    n_splits=n_splits,
                    n_repeats=n_repeats,
                    shuffle=shuffle,
                    clf=clf,
                    clf_kwargs=clf_kwargs,
                    return_clfs=return_clfs,
                    random_state=random_state,
                )

            return res, res_perm
        
        

    @staticmethod
    @abstractmethod
    def fit_and_validate(
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        condition: npt.ArrayLike | None = None,
        *,
        n_splits: int = 5,
        n_repeats: int = 1,
        shuffle: bool = False,
        clf: ClassifierMixin = LinearSVC,
        clf_kwargs: dict | None = None,
        return_clfs: bool = False,
        random_state: int | np.random.RandomState | None = None,
    ):
        raise NotImplementedError


def compute_decodability(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    condition: npt.ArrayLike | None = None,
    *,
    clf: ClassifierMixin = LinearSVC,
    n_splits: int = 5,
    n_repeats: int = 1,
    shuffle: bool = False,
    clf_kwargs: dict | None = None,
    return_weights: bool = False,
    n_jobs: int | None = None,
    random_state: int | np.random.RandomState | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Compute decodability (i.e., classification accuracy) using cross-validation.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        Target labels.
    condition : array-like of shape (n_samples, n_conditions) or None, default=None
        Group labels for the samples used while splitting the dataset into train/test set.
    clf : `sklearn.base.ClassifierMixin`, default=`sklearn.svm.SVC`
        Classifier to use.
    n_splits : int, default=5
        Number of folds.
    n_repeats : int, default=1
        Number of times cross-validation needs to be repeated.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches. Only relevant if `n_repeats` = 1.
    clf_kwargs : dict or None, default=None
        Additional arguments to pass to the classifier.
    return_weights : bool, default=False
        Whether to return the weights of the classifier. If True, the classifier must have a `coef_` attribute.
    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
    random_state : int, RandomState instance or None, default=None
        Random seed or random number generator.

    Returns
    -------
    scores : np.ndarray of shape (n_repeats * n_splits,)
        Array of test scores for each fold.
    weights : np.ndarray of shape (n_repeats * n_splits, n_features), optional
        Array of feature weights for each fold.
    """

    # initialize classifier
    clf = clf(**clf_kwargs) if clf_kwargs else clf()
    clf = make_pipeline(StandardScaler(), clf)

    # determine which cross-validation strategy to use
    if condition is None:
        if n_repeats == 1:
            cv = StratifiedKFold(
                n_splits=n_splits, shuffle=shuffle, random_state=random_state
            )
        else:
            cv = RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
            )
    else:
        if n_repeats == 1:
            cv = MultiStratifiedKFold(
                n_splits=n_splits, shuffle=shuffle, random_state=random_state
            )
        else:
            cv = RepeatedMultiStratifiedKFold(
                n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
            )

    # perform cross-validation
    results = cross_validate(
        clf,
        X,
        y,
        groups=condition,
        cv=cv,
        n_jobs=n_jobs,
        return_estimator=return_weights,
        error_score="raise",
    )
    if return_weights:
        try:
            weights = [est.coef_ for est in results["estimator"]]
            return np.asarray(results["test_score"]), np.asarray(weights)
        except AttributeError:
            warn("Weights are not available for the given classifier.", UserWarning)
            return np.asarray(results["test_score"]), None
    else:
        return np.asarray(results["test_score"])


def compute_decodability_ct_ind(
    Xs: list[npt.ArrayLike],
    ys: list[npt.ArrayLike],
    conditions: list[npt.ArrayLike],
    *,
    clf: ClassifierMixin = LinearSVC,
    n_splits: int = 5,
    n_repeats: int = 1,
    shuffle: bool = False,
    clf_kwargs: dict | None = None,
    skip_diagonal: bool = False,
    n_jobs: int | None = None,
) -> np.ndarray:
    """
    Compute temporal generalization accuracy across multiple independent datasets.

    Parameters
    ----------
    Xs : list of array-like of shape (n_samples, n_features)
        List of feature matrices.
    ys : list of array-like of shape (n_samples,)
        List of target vectors.
    clf : `sklearn.base.ClassifierMixin`, default=`sklearn.svm.SVC`
        Classifier to use.
    clf_kwargs : dict, optional
        Additional arguments to pass to the classifier.
    skip_diagonal : bool, default=False
        If True, skip computation of the diagonal elements of the score matrix.
    n_jobs : int or None, optional
        Number of jobs to run in parallel. By convention, n_jobs=-1 means using all processors.

    Returns
    -------
    score_matrix : `np.ndarray` for shape (n_samples, n_samples)
        A square matrix containing the generalization accuracies. Row indices correspond to the
        training datasets and column indices correspond to the test datasets.
    """

    # initialize score matrix and classifier
    n_samples = len(Xs)
    score_matrix = np.full((n_samples, n_samples), fill_value=np.nan)
    clf = clf(**clf_kwargs) if clf_kwargs else clf()
    clf = make_pipeline(StandardScaler(), clf)

    # define worker function
    def fit_and_score(i):
        clf_i = clone(clf)
        clf_i.fit(Xs[i], ys[i])
        for j in range(n_samples):
            if i == j:
                if skip_diagonal:
                    continue
                score_matrix[i, j] = compute_decodability(
                    Xs[i],
                    ys[i],
                    conditions[i],
                    clf=lambda _: clone(clf_i),
                    n_splits=n_splits,
                    n_repeats=n_repeats,
                    shuffle=shuffle,
                ).mean()
            else:
                score_matrix[i, j] = clf_i.score(Xs[j], ys[j])

            # if skip_diagonal and i == j:
            #     continue
            # score_matrix[i, j] = clf_i.score(Xs[j], ys[j])

    # compute generalization scores
    if n_jobs is None or n_jobs == 1:
        [fit_and_score(i) for i in range(n_samples)]
    else:
        Parallel(n_jobs=n_jobs)(delayed(fit_and_score)(i) for i in range(n_samples))
    return score_matrix


def compute_decodability_ct_rel(
    Xs: list[npt.ArrayLike],
    y: npt.ArrayLike | list[npt.ArrayLike],
    condition: npt.ArrayLike | list[npt.ArrayLike] | None = None,
    *,
    clf: ClassifierMixin = LinearSVC,
    n_splits: int = 5,
    n_repeats: int = 1,
    shuffle: bool = False,
    clf_kwargs: dict | None = None,
    skip_diagonal: bool = False,
    n_jobs: int | None = None,
) -> np.ndarray:
    """
    Compute temporal generalization accuracy across multiple related datasets.

    Parameters
    ----------
    Xs : list of array-like of shape (n_samples, n_features)
        List of feature matrices.
    y : array-like of shape (n_samples,) or list of array-like of shape (n_samples,)
        Target vector. If a list is provided, only the first element is used for compatibility.
    condition : array-like of shape (n_samples,) or list of array-like of shape (n_samples,) or None, default=None
        Group labels for the samples used while splitting the dataset into train/test set.
        If a list is provided, only the first element is used for compatibility.
    clf : `sklearn.base.ClassifierMixin`, default=`sklearn.svm.SVC`
        Classifier to use.
    n_splits : int, default=5
        Number of folds.
    n_repeats : int, default=1
        Number of times cross-validation needs to be repeated.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches. Only relevant if `n_repeats` = 1.
    clf_kwargs : dict or None, default=None
        Additional arguments to pass to the classifier.
    skip_diagonal : bool, default=False
        If True, skip computation of the diagonal elements of the score matrix.
    n_jobs : int or None, default=None
        Number of jobs to run in parallel.

    Returns
    -------
    score_matrix : `np.ndarray` for shape (n_samples, n_samples)
        A square matrix containing the generalization accuracies. Row indices correspond to the
        training datasets and column indices correspond to the test datasets.
    """

    # process inputs
    if isinstance(y, list):
        y = y[0]
    if condition is not None and isinstance(condition, list):
        condition = condition[0]

    # initialize score matrix and classifier
    n_samples = len(Xs)
    score_matrix = np.zeros((n_splits, n_samples, n_samples))
    clf = clf(**clf_kwargs) if clf_kwargs else clf()
    clf = make_pipeline(StandardScaler(), clf)

    # determine which cross-validation strategy to use
    if condition is None:
        if n_repeats == 1:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=None)
        else:
            cv = RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=n_repeats, random_state=None
            )
    else:
        if n_repeats == 1:
            cv = MultiStratifiedKFold(
                n_splits=n_splits, shuffle=shuffle, random_state=None
            )
        else:
            cv = RepeatedMultiStratifiedKFold(
                n_splits=n_splits, n_repeats=n_repeats, random_state=None
            )

    # define worker function
    def fit_and_score(i):
        clf_i = clone(clf)
        for k, (train_inds, test_inds) in enumerate(cv.split(Xs[i], y, condition)):
            X_train, y_train = Xs[i][train_inds], y[train_inds]
            clf_i.fit(X_train, y_train)
            for j in range(n_samples):
                if skip_diagonal and i == j:
                    score_matrix[i, j] = np.nan
                X_test, y_test = Xs[j][test_inds], y[test_inds]
                score_matrix[k, i, j] = clf_i.score(X_test, y_test)

    # compute generalization scores
    if n_jobs is None or n_jobs == 1:
        [fit_and_score(i) for i in range(n_samples)]
    else:
        Parallel(n_jobs=n_jobs)(delayed(fit_and_score)(i) for i in range(n_samples))
    return score_matrix


def compute_ccgp(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    condition: npt.ArrayLike,
    *,
    clf: ClassifierMixin = LinearSVC,
    n_crossings: int = 1,
    clf_kwargs: dict | None = None,
    n_jobs: int | None = None,
) -> np.ndarray:
    """
    Compute cross-conditional generalization performance (CCGP) using cross-validation across conditions.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    condition : array-like of shape (n_samples,)
        Group labels for the samples used while splitting the dataset into train/test set.
    clf : ClassifierMixin, default=SVC
        Classifier to use.
    n_crossings : int, default=1
        Number of group crossings to leave out in each iteration.
    clf_kwargs : dict, optional
        Additional arguments to pass to the classifier.
    n_jobs : int, optional
        Number of jobs to run in parallel.

    Returns
    -------
    `np.ndarray`
        Array of cross-validation scores.
    """

    clf = clf(**clf_kwargs) if clf_kwargs else clf()
    clf = make_pipeline(StandardScaler(), clf)
    cv = LeaveNCrossingsOut(n_crossings=n_crossings)
    return cross_val_score(
        clf,
        X,
        y,
        groups=condition,
        cv=cv,
        n_jobs=n_jobs,
        error_score="raise",
    )


def compute_ccgp_ct(
    Xs: list[npt.ArrayLike],
    ys: list[npt.ArrayLike],
    conditions: list[npt.ArrayLike],
    *,
    clf: ClassifierMixin = LinearSVC,
    n_crossings: int = 1,
    clf_kwargs: dict | None = None,
    skip_diagonal: bool = False,
):
    clf = clf(**clf_kwargs) if clf_kwargs else clf()
    clf = make_pipeline(StandardScaler(), clf)

    # Get the group indices for each dataset
    left_inds, right_inds = None, None
    for X, y, condition in zip(Xs, ys, conditions):
        cv = LeaveNCrossingsOut(n_crossings=n_crossings)
        li, ri = cv.get_group_indices(X, y, condition)
        if left_inds is None:
            left_inds, right_inds = [li], [ri]
            n_splits = cv.get_n_splits(None, None, condition)
        else:
            left_inds.append(li)
            right_inds.append(ri)

    # Compute the cross-generalization scores
    n_samples = len(Xs)
    score_matrix = np.zeros((n_splits, n_samples, n_samples))

    for i in range(n_samples):
        l1, r1 = left_inds[i], right_inds[i]

        for j, (k1, k2) in enumerate(zip(l1.keys(), r1.keys())):
            train_inds = np.array(
                [i for i in range(len(X)) if i not in l1[k1] + r1[k2]]
            )
            clf = clf.fit(Xs[i][train_inds], ys[i][train_inds])

            for k in range(n_samples):
                if skip_diagonal and i == k:
                    continue
                l2, r2 = left_inds[k], right_inds[k]
                test_inds = np.array(l2[k1] + r2[k2])
                score_matrix[j, i, k] = clf.score(Xs[k][test_inds], ys[k][test_inds])

    return np.mean(score_matrix, axis=0)


def compute_ps(
    X: npt.ArrayLike, y: npt.ArrayLike, conditions: npt.ArrayLike
) -> np.ndarray:
    """
    Compute the parallelism score among conditional vectors.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    conditions : array-like of shape (n_samples,)
        Group labels for the samples.

    Returns
    -------
    np.ndarray
        Array of cosine similarities for the best coding direction.
    """

    cv = LeaveNCrossingsOut()
    left_inds, right_inds = cv.get_group_indices(X, y, conditions)

    X_left = [np.mean(X[inds], axis=0) for inds in left_inds.values()]
    X_right = [np.mean(X[inds], axis=0) for inds in right_inds.values()]
    X_left, X_right = np.asarray(X_left), np.asarray(X_right)

    best_score = -np.inf
    best_similarities = None
    for right_xs_perm in permutations(X_right):
        vecs = np.array(right_xs_perm) - X_left
        curr_sims = [1 - cosine(v1, v2) for v1, v2 in combinations(vecs, 2)]
        curr_score = np.mean(curr_sims)
        if curr_score > best_score:
            best_score = curr_score
            best_similarities = curr_sims

    return best_similarities
