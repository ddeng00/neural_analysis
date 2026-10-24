from itertools import combinations, permutations
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    cross_validate,
    StratifiedKFold,
    RepeatedStratifiedKFold,
)
from sklearn.metrics import recall_score, roc_auc_score
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.spatial.distance import cosine

from ..validate import (
    MultiStratifiedKFold,
    RepeatedMultiStratifiedKFold,
    LeaveNCrossingsOut,
    permute_data,
    rotate_data_within_groups,
)
from ..preprocess import remove_groups_missing_conditions, construct_pseudopopulation
from ..utils import isin_2d

from ._base import (
    ScoreResult,
    _BaseEstimator,
    _BaseRelatedSamplesGeneralizer,
    _BaseIndependentSamplesGeneralizer,
)


def _rotate_ccgp_pair(
    X1: np.ndarray,
    condition1: np.ndarray,
    X2: np.ndarray,
    condition2: np.ndarray,
    random_state: int | np.random.RandomState | None,
    same_group: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same condition-wise feature rotations to two populations."""

    rs = (
        random_state
        if isinstance(random_state, np.random.RandomState)
        else np.random.RandomState(random_state)
    )
    X1_rotated = np.array(X1, copy=True)
    X2_rotated = X1_rotated if same_group else np.array(X2, copy=True)
    for condition in np.unique(condition1, axis=0):
        permutation = rs.permutation(X1.shape[1])
        condition = np.asarray(condition).reshape(1, -1)
        mask1 = isin_2d(condition1, condition)
        X1_rotated[mask1] = X1_rotated[mask1][:, permutation]
        if not same_group:
            mask2 = isin_2d(condition2, condition)
            X2_rotated[mask2] = X2_rotated[mask2][:, permutation]
    return X1_rotated, X2_rotated


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

    permute_labels = True

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
        use_roc: bool = False,
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
            scoring="roc_auc" if use_roc else None,
        )
        if return_clfs:
            return {
                "scores": np.mean(results["test_score"]),
                "clfs": results["estimator"],
            }
        else:
            return {"scores": np.mean(results["test_score"])}


class MulticlassDecodability:
    """Decode each condition variable from a balanced pseudopopulation.

    A single response column is sampled equally within each joint condition.
    Units missing conditions or sufficient trials are removed. Class labels
    follow the string representation used by construct_pseudopopulation.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        unit: str,
        response: str,
        condition: str | list[str],
        *,
        n_conditions: int | None = None,
        n_samples_per_cond: int | None = None,
        verbose: bool = False,
        remove_const: bool = False,
        random_seed: int | None = None,
    ) -> None:

        if not isinstance(response, str):
            raise TypeError("response must be a single column name.")
        if not isinstance(condition, list):
            condition = [condition]
        data = data[[unit, response] + condition].copy().infer_objects()
        for column in [unit] + condition:
            if isinstance(data[column].dtype, pd.CategoricalDtype):
                data[column] = data[column].astype(object)
        if data.empty or data[[unit] + condition].isna().any().any():
            raise ValueError("Data must contain nonmissing units and conditions.")
        if n_samples_per_cond is None:
            n_samples_per_cond = data.groupby([unit] + condition).size().min()
        if n_samples_per_cond < 1:
            raise ValueError("n_samples_per_cond must be positive.")
        self.n_samples_per_cond = n_samples_per_cond
        self.random_seed = (
            np.random.randint(1e6) if random_seed is None else random_seed
        )

        self.n_init = data[unit].nunique()
        data = remove_groups_missing_conditions(
            data,
            unit,
            condition,
            n_conditions=n_conditions,
            n_samples_per_cond=n_samples_per_cond,
        )
        if remove_const:
            data = data.groupby(unit).filter(lambda x: x[response].nunique() > 1)
        self.n_valid = data[unit].nunique()
        if not self.n_valid:
            raise ValueError("No units have sufficient trials for all conditions.")
        if any(data[column].nunique() < 2 for column in condition):
            raise ValueError("Each condition variable must have at least two classes.")
        self.verbose = verbose
        if verbose:
            prop_left = self.n_valid / self.n_init * 100
            print(f"Remaining Units: {self.n_valid}/{self.n_init} ({prop_left:.1f}%)")

        self.data = data
        self.unit = unit
        self.response = response
        self.condition = condition

    def score(
        self,
        n_resamples: int = 1,
        *,
        n_permutes: int = 0,
        n_splits: int = 5,
        n_repeats: int = 1,
        shuffle: bool = False,
        clf: BaseEstimator = LinearSVC,
        clf_kwargs: dict | None = None,
        return_clfs: bool = False,
        n_jobs: int | None = None,
    ) -> ScoreResult:
        """Return pooled held-out recall per variable, class, and resample.

        Null draws permute each variable's labels. Metrics include scores,
        variable, class, resample, permutation, is_null, and optional clfs.
        """

        if n_resamples < 1 or n_permutes < 0:
            raise ValueError("n_resamples must be positive and n_permutes nonnegative.")
        results = Parallel(return_as="generator", n_jobs=n_jobs, prefer="processes")(
            delayed(self._score_helper)(
                i,
                n_splits,
                n_repeats,
                shuffle,
                clf,
                clf_kwargs,
                return_clfs,
                n_permutes,
            )
            for i in range(n_resamples)
        )
        if self.verbose:
            results = tqdm(
                results,
                total=n_resamples,
                desc=f"{self.__class__.__name__} scoring",
                unit="resample",
            )
        records = [record for result in results for record in result]
        metrics = pd.DataFrame.from_records(records).convert_dtypes()
        metadata = {
            "metric": self.__class__.__name__,
            "n_resamples": n_resamples,
            "n_permutes": n_permutes,
            "n_splits": n_splits,
            "n_repeats": n_repeats,
            "shuffle": shuffle,
            "n_samples_per_condition": self.n_samples_per_cond,
            "n_units": self.n_valid,
            "random_seed": self.random_seed,
            "classes": {
                variable: np.unique(self.data[variable].to_numpy(str))
                for variable in self.condition
            },
        }
        return ScoreResult(metrics, metadata)

    def _score_helper(
        self,
        i: int,
        n_splits: int,
        n_repeats: int,
        shuffle: bool,
        clf: BaseEstimator,
        clf_kwargs: dict | None,
        return_clfs: bool,
        n_permutes: int,
    ) -> list[dict[str, Any]]:

        rs = np.random.RandomState(self.random_seed + i)
        X, condition = construct_pseudopopulation(
            self.data,
            self.unit,
            self.response,
            self.condition,
            n_samples_per_cond=self.n_samples_per_cond,
            all_groups_complete=True,
            random_state=rs,
        )
        records = []
        for k, variable in enumerate(self.condition):
            y = condition[:, k]
            for permutation in range(-1, n_permutes):
                yi = y if permutation == -1 else rs.permutation(y)
                result = self.validate(
                    X,
                    yi,
                    n_splits=n_splits,
                    n_repeats=n_repeats,
                    shuffle=shuffle,
                    clf=clf,
                    clf_kwargs=clf_kwargs,
                    return_clfs=return_clfs,
                    random_state=rs,
                )
                for label, score in zip(result["classes"], result["scores"]):
                    record = {
                        "scores": score,
                        "variable": variable,
                        "class": label,
                        "resample": i,
                        "permutation": None if permutation == -1 else permutation,
                        "is_null": permutation != -1,
                    }
                    if return_clfs:
                        record["clfs"] = result["clfs"]
                    records.append(record)
        return records

    @staticmethod
    def validate(
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        *,
        n_splits: int = 5,
        n_repeats: int = 1,
        shuffle: bool = False,
        clf: BaseEstimator = LinearSVC,
        clf_kwargs: dict | None = None,
        return_clfs: bool = False,
        random_state: int | np.random.RandomState | None = None,
    ) -> dict[str, Any]:
        """Pool predictions across folds and repeats before computing recall."""

        X, y = np.asarray(X), np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2 or counts.min() < n_splits:
            raise ValueError("Each of at least two classes must have n_splits samples.")
        if n_repeats < 1:
            raise ValueError("n_repeats must be positive.")
        clf = clf(**clf_kwargs) if clf_kwargs else clf()
        clf = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        if n_repeats == 1:
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state if shuffle else None,
            )
        else:
            cv = RepeatedStratifiedKFold(
                n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
            )

        ys, predictions, clfs = [], [], []
        for train_inds, test_inds in cv.split(X, y):
            fitted = clone(clf).fit(X[train_inds], y[train_inds])
            ys.append(y[test_inds])
            predictions.append(fitted.predict(X[test_inds]))
            if return_clfs:
                clfs.append(fitted)
        result = {
            "scores": recall_score(
                np.concatenate(ys),
                np.concatenate(predictions),
                labels=classes,
                average=None,
                zero_division=0,
            ),
            "classes": classes,
        }
        if return_clfs:
            result["clfs"] = clfs
        return result


class RelatedSamplesDecodability(_BaseRelatedSamplesGeneralizer):

    permute_labels = True

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
        use_roc: bool = False,
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
            if use_roc:
                decision = clf.decision_function(X2[test_inds])
                scores.append(roc_auc_score(y[test_inds], decision))
            else:
                scores.append(clf.score(X2[test_inds], y[test_inds]))
            if return_clfs:
                clfs.append(clf.named_steps["clf"])
        if return_clfs:
            return {"scores": np.mean(scores), "clfs": clfs}
        else:
            return {"scores": np.mean(scores)}


class IndependentSamplesDecodability(_BaseIndependentSamplesGeneralizer):

    permute_labels = True

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
        use_roc: bool = False,
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
        for train_inds, test_inds in cv.split(X1, y1, groups=condition1):
            clf.fit(X1[train_inds], y1[train_inds])
            if use_roc:
                if same_group:
                    decision = clf.decision_function(X1[test_inds])
                    scores.append(roc_auc_score(y1[test_inds], decision))
                else:
                    decision = clf.decision_function(X2)
                    scores.append(roc_auc_score(y2, decision))
            else:
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

    shuffle_in_validation = True

    @staticmethod
    def shuffle(X: npt.ArrayLike, condition: npt.ArrayLike, rs: np.random.RandomState):
        return rotate_data_within_groups(X, condition, random_state=rs)

    @staticmethod
    def validate(
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        condition: npt.ArrayLike,
        *,
        n_splits: None = None,
        n_repeats: None = None,
        shuffle: None = None,
        use_roc: bool = False,
        clf: BaseEstimator = LinearSVC,
        clf_kwargs: dict | None = None,
        return_clfs: bool = False,
        random_state: None = None,
        permute: bool = False,
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

        # initialize cross-validation strategy
        cv = LeaveNCrossingsOut()

        if permute:
            scores, clfs = [], []
            for train, test in cv.split(X, y, groups=condition):
                X_fold = CCGP.shuffle(X, condition, random_state)
                estimator = clf(**clf_kwargs) if clf_kwargs else clf()
                estimator = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
                estimator.fit(X_fold[train], y[train])
                scores.append(estimator.score(X_fold[test], y[test]))
                if return_clfs:
                    clfs.append(estimator)
            result = {"scores": np.mean(scores)}
            if return_clfs:
                result["clfs"] = clfs
            return result

        estimator = clf(**clf_kwargs) if clf_kwargs else clf()
        estimator = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])

        # perform cross-validation
        results = cross_validate(
            estimator,
            X,
            y,
            groups=condition,
            cv=cv,
            return_estimator=return_clfs,
            error_score="raise",
            scoring="roc_auc" if use_roc else None,
        )
        if return_clfs:
            return {
                "scores": np.mean(results["test_score"]),
                "clfs": results["estimator"],
            }
        else:
            return {"scores": np.mean(results["test_score"])}


class IndependentSamplesCCGP(_BaseIndependentSamplesGeneralizer):

    shuffle_in_validation = True

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
        use_roc: bool = False,
        clf: BaseEstimator = LinearSVC,
        clf_kwargs: dict | None = None,
        return_clfs: bool = False,
        random_state: None = None,
        permute: bool = False,
    ) -> dict[str, Any]:
        cv = LeaveNCrossingsOut()
        scores, clfs = [], []
        left_inds_1, right_inds_1 = cv.get_group_indices(X1, y1, condition1)
        for test_k1, test_v1 in left_inds_1.items():
            for test_k2, test_v2 in right_inds_1.items():
                train_mask = ~(
                    isin_2d(condition1, np.asarray(test_k1).reshape(1, -1))
                    | isin_2d(condition1, np.asarray(test_k2).reshape(1, -1))
                )
                target_X = X1 if same_group else X2
                target_y = y1 if same_group else y2
                target_condition = condition1 if same_group else condition2
                test_mask = isin_2d(
                    target_condition, np.asarray(test_k1).reshape(1, -1)
                ) | isin_2d(target_condition, np.asarray(test_k2).reshape(1, -1))

                train_X, test_X = X1, target_X
                if permute:
                    train_X, test_X = _rotate_ccgp_pair(
                        X1,
                        condition1,
                        target_X,
                        target_condition,
                        random_state,
                        same_group,
                    )

                estimator = clf(**clf_kwargs) if clf_kwargs else clf()
                estimator = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
                estimator.fit(train_X[train_mask], y1[train_mask])
                if use_roc:
                    decision = estimator.decision_function(test_X[test_mask])
                    scores.append(roc_auc_score(target_y[test_mask], decision))
                else:
                    scores.append(
                        estimator.score(test_X[test_mask], target_y[test_mask])
                    )
                if return_clfs:
                    clfs.append(estimator)
        if return_clfs:
            return {"scores": np.mean(scores), "clfs": clfs}
        else:
            return {"scores": np.mean(scores)}


class PS(_BaseEstimator):
    shuffle_each_dichotomy = True

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
        named_only: bool = False,
        verbose: bool = False,
        remove_const: bool | None = None,
        normalize: bool | None = None,
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
            named_only=named_only,
            verbose=verbose,
            remove_const=True,
            random_seed=random_seed,
        )

        # normalize data
        if normalize:
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

        n = len(X_left)

        delta = X_right[None, :, :] - X_left[:, None, :]
        unit_vecs = delta / np.linalg.norm(delta, axis=-1, keepdims=True)

        perms = np.array(list(permutations(range(n))), dtype=np.int8)

        best_score = -np.inf
        chunk_size = 100
        for start in range(0, len(perms), chunk_size):
            p = perms[start : start + chunk_size]
            vec_sums = unit_vecs[np.arange(n)[None, :], p].sum(axis=1)
            scores = (np.sum(vec_sums**2, axis=1) - n) / (n * (n - 1))
            best_score = max(best_score, scores.max())

        return {"scores": best_score}
