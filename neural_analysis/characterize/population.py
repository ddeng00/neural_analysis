from itertools import combinations, permutations
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    cross_validate,
    StratifiedKFold,
    RepeatedStratifiedKFold,
)
from scipy.spatial.distance import cosine

from ..validate import (
    MultiStratifiedKFold,
    RepeatedMultiStratifiedKFold,
    LeaveNCrossingsOut,
    permute_data,
    rotate_data_within_groups,
)

from ._base import (
    _BaseEstimator,
    _BaseRelatedSamplesGeneralizer,
    _BaseIndependentSamplesGeneralizer,
)


class Decodability(_BaseEstimator):
    """
    Dichotomy-based estimator for decoding analyses.

    Attributes
    ----------
    data : `pd.DataFrame`
        DataFrame containing the data.
    unit : str
        Name of the column containing the unit identifiers.
    response : str
        Name of the column containing the response variable.
    condition : str or list of str
        Name of the column(s) containing the condition labels.
    n_samples_per_cond : int
        Number of samples per condition.
    random_seed : int
        Random seed used for all computations.
    n_init : int
        Number of units in the initial dataset.
    n_valid : int
        Number of units in the dataset after removing units with missing conditional trials.
    dichotomies : list of tuple of array-like
        List of dichotomies.
    dichotomy_names : list of str
        List of dichotomy names.
    dichotomy_difficulties : list of float
        List of dichotomy difficulties, operationalized as the proportion of adjacent condition pairs.
    shuffle_fn : function
        Function that shuffles row order of data.
    """

    @staticmethod
    def shuffle(
        X: npt.ArrayLike, condition: npt.ArrayLike, rs: np.random.RandomState
    ) -> npt.ArrayLike:
        return permute_data(X, random_state=rs)

    @staticmethod
    def validate(
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        condition: npt.ArrayLike | None = None,
        *,
        n_splits: int = 5,
        n_repeats: int = 1,
        shuffle: bool = False,
        clf: BaseEstimator = LinearSVC,
        clf_kwargs: dict | None = None,
        return_clfs: bool = False,
        random_state: int | np.random.RandomState | None = None,
    ) -> dict[str, Any]:

        # initialize classifier
        clf = clf(**clf_kwargs) if clf_kwargs else clf()
        clf = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

        # determine which cross-validation strategy to use
        if condition is None and n_repeats == 1:
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state if shuffle else None,
            )
        elif condition is None and n_repeats > 1:
            cv = RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
            )
        elif condition is not None and n_repeats == 1:
            cv = MultiStratifiedKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state if shuffle else None,
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
            return_estimator=return_clfs,
            error_score="raise",
        )
        if return_clfs:
            return {
                "scores": np.mean(results["test_score"]),
                "clfs": [clf.named_steps["clf"] for clf in results["estimator"]],
            }
        else:
            return {"scores": np.mean(results["test_score"])}


class RelatedSamplesDecodability(_BaseRelatedSamplesGeneralizer):

    @staticmethod
    def shuffle(
        X: list[npt.ArrayLike], condition: npt.ArrayLike, rs: np.random.RandomState
    ) -> list[npt.ArrayLike]:
        perm_inds = np.arange(X[0].shape[0])
        perm_inds = permute_data(perm_inds, random_state=rs)
        return [Xi[perm_inds] for Xi in X]

    @staticmethod
    def validate(
        X1: npt.ArrayLike,
        X2: npt.ArrayLike,
        y: npt.ArrayLike,
        condition: npt.ArrayLike | None = None,
        *,
        n_splits: int = 5,
        n_repeats: int = 1,
        shuffle: bool = False,
        clf: BaseEstimator = LinearSVC,
        clf_kwargs: dict | None = None,
        return_clfs: bool = False,
        random_state: int | np.random.RandomState | None = None,
    ) -> dict[str, Any]:

        # initialize classifier
        clf = clf(**clf_kwargs) if clf_kwargs else clf()
        clf = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

        # determine which cross-validation strategy to use
        if condition is None and n_repeats == 1:
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state if shuffle else None,
            )
        elif condition is None and n_repeats > 1:
            cv = RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
            )
        elif condition is not None and n_repeats == 1:
            cv = MultiStratifiedKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state if shuffle else None,
            )
        else:
            cv = RepeatedMultiStratifiedKFold(
                n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
            )

        # perform cross-validation
        scores, clfs = [], []
        for train_inds, test_inds in cv.split(X1, y, groups=condition):
            clf.fit(X1[train_inds], y[train_inds])
            scores.append(clf.score(X2[test_inds], y[test_inds]))
            if return_clfs:
                clfs.append(clf.named_steps["clf"])
        if return_clfs:
            return {"scores": np.mean(scores), "clfs": clfs}
        else:
            return {"scores": np.mean(scores)}


class IndependentSamplesDecodability(_BaseIndependentSamplesGeneralizer):

    @staticmethod
    def shuffle(
        X: npt.ArrayLike, condition: npt.ArrayLike, rs: np.random.RandomState
    ) -> npt.ArrayLike:
        return permute_data(X, random_state=rs)

    @staticmethod
    def validate(
        X1: npt.ArrayLike,
        X2: npt.ArrayLike,
        y1: npt.ArrayLike,
        y2: npt.ArrayLike,
        condition1: npt.ArrayLike | None = None,
        condition2: npt.ArrayLike | None = None,
        *,
        same_group: bool = False,
        n_splits: int = 5,
        n_repeats: int = 1,
        shuffle: bool = False,
        clf: BaseEstimator = LinearSVC,
        clf_kwargs: dict | None = None,
        return_clfs: bool = False,
        random_state: int | np.random.RandomState | None = None,
    ) -> dict[str, Any]:

        # initialize classifier
        clf = clf(**clf_kwargs) if clf_kwargs else clf()
        clf = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

        # check if cross-validation is possible
        if n_splits == 1:
            clf.fit(X1, y1)
            score = clf.score(X2, y2)
            if return_clfs:
                return {"scores": score, "clfs": clf.named_steps["clf"]}
            return {"scores": score}

        # determine which cross-validation strategy to use
        if condition1 is None and n_repeats == 1:
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state if shuffle else None,
            )
        elif condition1 is None and n_repeats > 1:
            cv = RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
            )
        elif condition1 is not None and n_repeats == 1:
            cv = MultiStratifiedKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state if shuffle else None,
            )
        else:
            cv = RepeatedMultiStratifiedKFold(
                n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
            )

        # test generalization via cross-validation
        scores, clfs = [], []

        # if same_group:
        #     for train_inds, test_inds in cv.split(X1, y1, groups=condition1):
        #         clf.fit(X1[train_inds], y1[train_inds])
        #         scores.append(clf.score(X1[test_inds], y1[test_inds]))
        #         if return_clfs:
        #             clfs.append(clf.named_steps["clf"])
        # else:
        #     for train_inds, _ in cv.split(X1, y1, groups=condition1):
        #         clf.fit(X1[train_inds], y1[train_inds])
        #         for _, test_inds in cv.split(X2, y2, groups=condition2):
        #             scores.append(clf.score(X2[test_inds], y2[test_inds]))
        #             if return_clfs:
        #                 clfs.append(clf.named_steps["clf"])
        # if return_clfs:
        #     return {"scores": np.mean(scores), "clfs": clfs}
        # else:
        #     return {"scores": np.mean(scores)}

        for train_inds, test_inds in cv.split(X1, y1, groups=condition1):
            clf.fit(X1[train_inds], y1[train_inds])
            if same_group:
                scores.append(clf.score(X1[test_inds], y1[test_inds]))
            else:
                scores.append(clf.score(X2, y2))
            if return_clfs:
                clfs.append(clf.named_steps["clf"])
        if return_clfs:
            return {"scores": np.mean(scores), "clfs": clfs}
        else:
            return {"scores": np.mean(scores)}


class CCGP(_BaseEstimator):
    """
    Dichotomy-based estimator for cross-conditional generalization performance (CCGP) analyses.

    Attributes
    ----------
    data : `pd.DataFrame`
        DataFrame containing the data.
    unit : str
        Name of the column containing the unit identifiers.
    response : str
        Name of the column containing the response variable.
    condition : str or list of str
        Name of the column(s) containing the condition labels.
    n_samples_per_cond : int
        Number of samples per condition.
    random_seed : int
        Random seed used for all computations.
    n_init : int
        Number of units in the initial dataset.
    n_valid : int
        Number of units in the dataset after removing units with missing conditional trials.
    dichotomies : list of tuple of array-like
        List of dichotomies.
    dichotomy_names : list of str
        List of dichotomy names.
    dichotomy_difficulties : list of float
        List of dichotomy difficulties, operationalized as the proportion of adjacent condition pairs.
    shuffle_fn : function
        Function that rotates data such that unit responses are decorrelated across conditions.
    """

    @staticmethod
    def shuffle(X: npt.ArrayLike, condition: npt.ArrayLike, rs: np.random.RandomState):
        X = X.copy()
        [rs.shuffle(Xi) for Xi in X]
        return X

    @staticmethod
    def validate(
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        condition: npt.ArrayLike,
        *,
        n_splits: None = None,
        n_repeats: None = None,
        shuffle: None = None,
        clf: BaseEstimator = LinearSVC,
        clf_kwargs: dict | None = None,
        return_clfs: bool = False,
        random_state: None = None,
    ):
        """
        Fit and validate a classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target vector.
        condition : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into train/test set.
        n_splits : None
            Ignored. Only present for compatibility with the base class.
        n_repeats : None
            Ignored. Only present for compatibility with the base class.
        shuffle : None
            Ignored. Only present for compatibility with the base class.
        clf : `BaseEstimator`, default=LinearSVC
            Classifier to use.
        clf_kwargs : dict or None, default=None
            Additional arguments to pass to the classifier.
        return_clfs : bool, default=False
            Whether to return the fitted classifiers.
        random_state : int or np.random.RandomState or None, default=None
            Random state to use.

        Returns
        -------
        results : dict of list
            Dictionary containing the scores and fitted classifiers (if requested) for each dichotomy.
            The keys are "scores" and "clfs".
        """

        # initialize classifier and cross-validation strategy
        clf = clf(**clf_kwargs) if clf_kwargs else clf()
        clf = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        cv = LeaveNCrossingsOut()

        # perform cross-validation
        results = cross_validate(
            clf,
            X,
            y,
            groups=condition,
            cv=cv,
            return_estimator=return_clfs,
            error_score="raise",
        )
        if return_clfs:
            return {
                "scores": np.mean(results["test_score"]),
                "clfs": [clf.named_steps["clf"] for clf in results["estimator"]],
            }
        else:
            return {"scores": np.mean(results["test_score"])}


class IndependentSamplesCCGP(_BaseIndependentSamplesGeneralizer):

    @staticmethod
    def shuffle(
        X: npt.ArrayLike, condition: npt.ArrayLike, rs: np.random.RandomState
    ) -> npt.ArrayLike:
        return rotate_data_within_groups(X, condition, random_state=rs)

    @staticmethod
    def validate(
        X1: npt.ArrayLike,
        X2: npt.ArrayLike,
        y1: npt.ArrayLike,
        y2: npt.ArrayLike,
        condition1: npt.ArrayLike | None = None,
        condition2: npt.ArrayLike | None = None,
        *,
        same_group: bool = False,
        n_splits: None = None,
        n_repeats: None = None,
        shuffle: None = None,
        clf: BaseEstimator = LinearSVC,
        clf_kwargs: dict | None = None,
        return_clfs: bool = False,
        random_state: None = None,
    ) -> dict[str, Any]:

        # initialize classifier
        clf = clf(**clf_kwargs) if clf_kwargs else clf()
        clf = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        cv = LeaveNCrossingsOut()

        # perform cross-validation (same group)
        if same_group:
            results = cross_validate(
                clf,
                X1,
                y1,
                groups=condition1,
                cv=cv,
                return_estimator=return_clfs,
                error_score="raise",
            )
            if return_clfs:
                return {
                    "scores": np.mean(results["test_score"]),
                    "clfs": [clf.named_steps["clf"] for clf in results["estimator"]],
                }
            else:
                return {"scores": np.mean(results["test_score"])}

        # perform cross-validation (different groups)
        scores, clfs = [], []
        left_inds_1, right_inds_1 = cv.get_group_indices(X1, y1, condition1)
        left_inds_2, right_inds_2 = cv.get_group_indices(X2, y2, condition2)
        for test_k1, test_v1 in left_inds_1.items():
            for test_k2, test_v2 in right_inds_1.items():
                test_inds = np.concatenate([test_v1, test_v2])
                X_test = X1[test_inds]
                y_test = y1[test_inds]

                exclude_inds = np.concatenate(
                    [left_inds_2[test_k1], right_inds_2[test_k2]]
                )
                X_train = np.delete(X2, exclude_inds, axis=0)
                y_train = np.delete(y2, exclude_inds)

                scores.append(clf.fit(X_train, y_train).score(X_test, y_test))
                if return_clfs:
                    clfs.append(clf.named_steps["clf"].copy())
        if return_clfs:
            return {"scores": np.mean(scores), "clfs": clfs}
        else:
            return {"scores": np.mean(scores)}


class PS(_BaseEstimator):
    def __init__(
        self,
        data: pd.DataFrame,
        unit: str,
        response: str,
        condition: str | list[str],
        *,
        n_conditions: int | None = None,
        n_samples_per_cond: None = None,
        dichot_map: dict[str, Any] | None = None,
        verbose: bool = False,
        remove_const: bool | None = None,
        random_seed: int | None = None,
    ) -> None:

        super().__init__(
            data,
            unit,
            response,
            condition,
            n_conditions=n_conditions,
            n_samples_per_cond=1,
            dichot_map=dichot_map,
            verbose=verbose,
            remove_const=True,
            random_seed=random_seed,
        )

        # normalize data
        self.data = self.data.copy()
        unit_mean = self.data.groupby(unit)[response].mean()
        unit_std = self.data.groupby(unit)[response].std()
        self.data[response] = (
            self.data[response] - self.data[unit].map(unit_mean)
        ) / self.data[unit].map(unit_std)

        # generate conditional averages
        self.data = self.data.groupby([unit] + condition)[response].mean().reset_index()

    @staticmethod
    def shuffle(X: npt.ArrayLike, condition: npt.ArrayLike, rs: np.random.RandomState):
        return rotate_data_within_groups(X, condition, random_state=rs)

    @staticmethod
    def validate(
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        condition: npt.ArrayLike,
        **kwargs,
    ):
        """
        Fit and validate a classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Conditional average matrix.
        y : array-like of shape (n_samples,)
            Target vector.
        condition : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into train/test set.
        kwargs : dict
            Ignored. Only present for compatibility with the base class.

        Returns
        -------
        results : dict of list
            Dictionary containing the scores for each dichotomy.
            The key is "scores".
        """

        cv = LeaveNCrossingsOut()
        left_inds, right_inds = cv.get_group_indices(X, y, condition)
        left_inds = np.concatenate(list(left_inds.values()))
        right_inds = np.concatenate(list(right_inds.values()))
        X_left, X_right = X[left_inds], X[right_inds]

        best_score = -np.inf
        for right_xs_perm in permutations(X_right):
            vecs = np.array(right_xs_perm) - X_left
            curr_sims = [1 - cosine(v1, v2) for v1, v2 in combinations(vecs, 2)]
            curr_score = np.mean(curr_sims)
            if curr_score > best_score:
                best_score = curr_score

        return {"scores": best_score}
