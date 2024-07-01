from collections import defaultdict
from itertools import combinations, permutations

import numpy as np
import numpy.typing as npt
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, cross_val_score
from scipy.spatial.distance import cosine

from .validate import (
    MultiStratifiedKFold,
    RepeatedMultiStratifiedKFold,
    LeaveNCrossingsOut,
)


def balanced_decoding(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    conditions: npt.ArrayLike | None = None,
    clf: ClassifierMixin = SVC,
    n_splits: int = 5,
    n_repeats: int = 1,
    shuffle: bool = False,
    clf_kwargs: dict | None = {"kernel": "linear"},
    return_weights: bool = False,
    n_jobs: int | None = None,
    random_state: int | np.random.RandomState | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Cross-validate balanced classification accuracy.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        Target labels.
    conditions : array-like of shape (n_samples, n_conditions) or None, default=None
        Group labels for the samples used while splitting the dataset into train/test set.
    clf : `sklearn.base.ClassifierMixin`, default=`sklearn.svm.SVC`
        Classifier to use.
    n_splits : int, default=5
        Number of folds.
    n_repeats : int, default=1
        Number of times cross-validation needs to be repeated.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.
    clf_kwargs : dict or None, default=None
        Additional arguments to pass to the classifier.
    return_weights : bool, default=False
        Whether to return the weights of the classifier.
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

    clf = clf(**clf_kwargs) if clf_kwargs else clf()
    clf = make_pipeline(StandardScaler(), clf)
    if n_repeats > 1:
        cv = RepeatedMultiStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )
    else:
        cv = MultiStratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

    results = cross_validate(
        clf,
        x,
        y,
        groups=conditions,
        cv=cv,
        n_jobs=n_jobs,
        return_estimator=return_weights,
        error_score="raise",
    )

    if return_weights:
        weights = [est.coef_ for est in results["estimator"]]
        return np.asarray(results["test_score"]), np.asarray(weights)
    else:
        return np.asarray(results["test_score"])


def pairwise_generalization(
    xs: list[npt.ArrayLike],
    ys: list[npt.ArrayLike],
    *,
    clf: ClassifierMixin = SVC,
    clf_kwargs: dict | None = {"kernel": "linear"},
    skip_diagonal: bool = False,
) -> np.ndarray:
    """
    Compute generalization accuracy across multiple datasets.

    Parameters
    ----------
    xs : list of array-like of shape (n_samples, n_features)
        List of feature matrices.
    ys : list of array-like of shape (n_samples,)
        List of target vectors.
    clf : `sklearn.base.ClassifierMixin`, default=`sklearn.svm.SVC`
        Classifier to use.
    clf_kwargs : dict, optional
        Additional arguments to pass to the classifier.
    skip_diagonal : bool, default=False
        If True, skip computation of the diagonal elements of the score matrix.

    Returns
    -------
    score_matrix : `np.ndarray` for shape (n_samples, n_samples)
        A square matrix containing the generalization accuracies. Row indices correspond to the
        training datasets and column indices correspond to the test datasets.
    """

    clf = clf(**clf_kwargs) if clf_kwargs else clf()
    clf = make_pipeline(StandardScaler(), clf)
    n_samples = len(xs)
    score_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        clf = clf.fit(xs[i], ys[i])
        for j in range(n_samples):
            if skip_diagonal and i == j:
                continue
            score_matrix[i, j] = clf.score(xs[j], ys[j])
    return score_matrix


def between_crossings_generalization(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    conditions: npt.ArrayLike,
    *,
    clf: ClassifierMixin = SVC,
    n_crossings: int = 1,
    clf_kwargs: dict | None = {"kernel": "linear"},
    n_jobs: int | None = None,
) -> np.ndarray:
    """
    Perform cross-conditional generalization using Leave-N-Crossings-Out cross-validation.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    conditions : array-like of shape (n_samples,)
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
        x,
        y,
        groups=conditions,
        cv=cv,
        n_jobs=n_jobs,
        error_score="raise",
    )


def between_crossings_parallelism(
    x: npt.ArrayLike, y: npt.ArrayLike, conditions: npt.ArrayLike
) -> np.ndarray:
    """
    Compute the parallelism between group crossings.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
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

    left_cond_inds = defaultdict(list)
    right_cond_inds = defaultdict(list)
    for i, (cond, yi) in enumerate(zip(conditions, y)):
        cond = tuple(cond)
        if yi == y[0]:
            left_cond_inds[cond].append(i)
        else:
            right_cond_inds[cond].append(i)

    left_xs = [np.mean(x[inds], axis=0) for inds in left_cond_inds.values()]
    right_xs = [np.mean(x[inds], axis=0) for inds in right_cond_inds.values()]
    left_xs, right_xs = np.asarray(left_xs), np.asarray(right_xs)

    best_score = -np.inf
    best_similarities = None
    for right_xs_perm in permutations(right_xs):
        vecs = np.array(right_xs_perm) - left_xs
        curr_sims = [1 - cosine(v1, v2) for v1, v2 in combinations(vecs, 2)]
        curr_score = np.mean(curr_sims)
        if curr_score > best_score:
            best_score = curr_score
            best_similarities = curr_sims

    return best_similarities
