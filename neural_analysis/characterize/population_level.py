from abc import ABC, abstractmethod
from warnings import warn
from itertools import combinations, permutations, chain

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import (
    cross_validate,
    cross_val_score,
    StratifiedKFold,
    RepeatedStratifiedKFold,
)
from scipy.spatial.distance import cosine
from sklearn.base import clone
from joblib import Parallel, delayed
from tqdm import tqdm

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
    """
    Base class for dichotomy-based population-level estimators.

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
        Function to shuffle data. Must be implemented in subclasses.
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
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize the dichotomy-based estimator.

        Parameters
        ----------
        n_conditions : int or None, default=None
            Number of conditions. If None, the number of conditions is inferred from the data.
        """

        # store data
        self.data = data
        self.unit = unit
        self.response = response
        self.condition = condition

        # infer number of samples per condition if not provided
        if n_samples_per_cond is None:
            n_samples_per_cond = data.groupby(condition).size().min()
        self.n_samples_per_cond = n_samples_per_cond

        # check random seed
        self.random_seed = (
            np.random.randint(1e6) if random_seed is None else random_seed
        )

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
        show_progress: bool = False,
        return_clfs: bool = False,
        n_jobs: int | None = None,
    ) -> list[dict] | tuple[list[dict], list[dict]]:
        """
        Construct resampled pseudopopulations and compute scores for each dichotomy.

        Parameters
        ----------
        n_resamples : int, default=1
            Number of resamples to perform.
        permute : bool, default=False
            If True, also estimate the permutation null.
        n_splits : int, default=5
            Number of folds in cross-validation.
        n_repeats : int, default=1
            Number of times cross-validation needs to be repeated.
        shuffle : bool, default=False
            Whether to shuffle the data before splitting into batches. Only relevant if `n_repeats` = 1.
        clf : ClassifierMixin, default=LinearSVC
            Classifier to use.
        clf_kwargs : dict or None, default=None
            Additional arguments to pass to the classifier.
        show_progress : bool, default=False
            Whether to track progress using a progress bar.
        return_clfs : bool, default=False
            Whether to return the fitted classifiers.
        n_jobs : int or None, default=None
            Number of jobs to run in parallel.

        Returns
        -------
        results : list of dict
            List of dictionaries containing the scores for each dichotomy.
        null_results : list of dict
            List of dictionaries containing the scores for each dichotomy under the permutation null.
            Only available if `permute` is True.
        """

        def helper(i):
            # generate new random state
            rs = (
                np.random.RandomState(self.random_seed + i)
                if self.random_seed
                else None
            )

            # construct condition-balanced pseudopopulation of units
            X, condition = construct_pseudopopulation(
                self.data,
                self.response,
                self.unit,
                self.condition,
                n_samples_per_cond=self.n_samples_per_cond,
                all_groups_complete=True,
                random_state=rs,
            )

            # fit and validate each dichotomy
            res = []
            for dichot, dichot_name in zip(self.dichotomies, self.dichotomy_names):
                y = isin_2d(condition, dichot).astype(int)
                d_res = self.__class__._fit_and_validate(
                    X,
                    y,
                    condition,
                    n_splits=n_splits,
                    n_repeats=n_repeats,
                    shuffle=shuffle,
                    clf=clf,
                    clf_kwargs=clf_kwargs,
                    return_clfs=return_clfs,
                    random_state=rs,
                )
                d_res["dichotomy"] = dichot_name
                res.append(d_res)
            if not permute:
                return res

            # estimate permutation null
            res_perm = []
            X = self.shuffle_fn(X, rs)
            for dichot, dichot_name in zip(self.dichotomies, self.dichotomy_names):
                y = isin_2d(condition, dichot).astype(int)
                d_res = self.__class__._fit_and_validate(
                    X,
                    y,
                    condition,
                    n_splits=n_splits,
                    n_repeats=n_repeats,
                    shuffle=shuffle,
                    clf=clf,
                    clf_kwargs=clf_kwargs,
                    return_clfs=return_clfs,
                    random_state=rs,
                )
                d_res["dichotomy"] = dichot_name
                res_perm.append(d_res)

            return res, res_perm

        # perform resampling with parallelization
        if show_progress:
            results = list(
                tqdm(
                    Parallel(return_as="generator", n_jobs=n_jobs)(
                        delayed(helper)(i) for i in range(n_resamples)
                    ),
                    total=n_resamples,
                )
            )
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(helper)(i) for i in range(n_resamples)
            )

        # return results
        to_explode = ["scores", "clfs"] if return_clfs else "scores"
        if not permute:
            results = pd.DataFrame.from_records(chain.from_iterable(results))
            results = results.explode(to_explode, ignore_index=True)
            results = results.convert_dtypes()
            return results
        else:
            results, null_results = zip(*results)
            results = pd.DataFrame.from_records(chain.from_iterable(results))
            results = results.explode(to_explode, ignore_index=True)
            null_results = pd.DataFrame.from_records(chain.from_iterable(null_results))
            null_results = null_results.explode(to_explode, ignore_index=True)
            results, null_results = (
                results.convert_dtypes(),
                null_results.convert_dtypes(),
            )
            return results, null_results

    @staticmethod
    @abstractmethod
    def _fit_and_validate(
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
    ) -> dict[str, list]:
        """
        Fit and validate a classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Target vector.
        condition : array-like of shape (n_samples,) or None, default=None
            Group labels for the samples used while splitting the dataset into train/test set.
        n_splits : int, default=5
            Number of folds.
        n_repeats : int, default=1
            Number of times cross-validation needs to be repeated.
        shuffle : bool, default=False
            Whether to shuffle the data before splitting into batches. Only relevant if `n_repeats` = 1.
        clf : ClassifierMixin, default=LinearSVC
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
        raise NotImplementedError


class DichotomyDecoding(_BaseDichotomyEstimator):
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

    @property
    def shuffle_fn(self):
        return lambda X, rs: permute_data(X, random_state=rs)

    @staticmethod
    def _fit_and_validate(
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
                "scores": results["test_score"],
                "clfs": [clf.named_steps["clf"] for clf in results["estimator"]],
            }
        else:
            return {"scores": results["test_score"]}


class DichotomyCCGP(_BaseDichotomyEstimator):
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

    @property
    def shuffle_fn(self):
        return lambda X, groups, rs: rotate_data_within_groups(
            X, groups, random_state=rs
        )

    @staticmethod
    def _fit_and_validate(
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        condition: npt.ArrayLike,
        *,
        n_splits: None = None,
        n_repeats: None = None,
        shuffle: None = None,
        clf: ClassifierMixin = LinearSVC,
        clf_kwargs: dict | None = None,
        return_clfs: bool = False,
        random_state: int | np.random.RandomState | None = None,
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
        clf : ClassifierMixin, default=LinearSVC
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
                "scores": results["test_score"],
                "clfs": [clf.named_steps["clf"] for clf in results["estimator"]],
            }
        else:
            return {"scores": results["test_score"]}


class DichotomyPS(_BaseDichotomyEstimator):
    """
    Dichotomy-based estimator for parallelism score (PS) analyses.

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

    @property
    def shuffle_fn(self):
        return lambda X, groups, rs: rotate_data_within_groups(
            X, groups, random_state=rs
        )

    @staticmethod
    def _fit_and_validate(
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        condition: npt.ArrayLike,
        *,
        n_splits: None = None,
        n_repeats: None = None,
        shuffle: None = None,
        clf: None = None,
        clf_kwargs: None = None,
        return_clfs: None = None,
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
        clf : None
            Ignored. Only present for compatibility with the base class.
        clf_kwargs : None
            Ignored. Only present for compatibility with the base class.
        return_clfs : None
            Ignored. Only present for compatibility with the base class.
        random_state : None
            Ignored. Only present for compatibility with the base class.

        Returns
        -------
        results : dict of list
            Dictionary containing the scores for each dichotomy.
            The key is "scores".
        """

        cv = LeaveNCrossingsOut()
        left_inds, right_inds = cv.get_group_indices(X, y, condition)

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

        return {"scores": [best_similarities]}


# def compute_decodability_ct_ind(
#     Xs: list[npt.ArrayLike],
#     ys: list[npt.ArrayLike],
#     conditions: list[npt.ArrayLike],
#     *,
#     clf: ClassifierMixin = LinearSVC,
#     n_splits: int = 5,
#     n_repeats: int = 1,
#     shuffle: bool = False,
#     clf_kwargs: dict | None = None,
#     skip_diagonal: bool = False,
#     n_jobs: int | None = None,
# ) -> np.ndarray:
#     """
#     Compute temporal generalization accuracy across multiple independent datasets.

#     Parameters
#     ----------
#     Xs : list of array-like of shape (n_samples, n_features)
#         List of feature matrices.
#     ys : list of array-like of shape (n_samples,)
#         List of target vectors.
#     clf : `sklearn.base.ClassifierMixin`, default=`sklearn.svm.SVC`
#         Classifier to use.
#     clf_kwargs : dict, optional
#         Additional arguments to pass to the classifier.
#     skip_diagonal : bool, default=False
#         If True, skip computation of the diagonal elements of the score matrix.
#     n_jobs : int or None, optional
#         Number of jobs to run in parallel. By convention, n_jobs=-1 means using all processors.

#     Returns
#     -------
#     score_matrix : `np.ndarray` for shape (n_samples, n_samples)
#         A square matrix containing the generalization accuracies. Row indices correspond to the
#         training datasets and column indices correspond to the test datasets.
#     """

#     # initialize score matrix and classifier
#     n_samples = len(Xs)
#     score_matrix = np.full((n_samples, n_samples), fill_value=np.nan)
#     clf = clf(**clf_kwargs) if clf_kwargs else clf()
#     clf = make_pipeline(StandardScaler(), clf)

#     # define worker function
#     def fit_and_score(i):
#         clf_i = clone(clf)
#         clf_i.fit(Xs[i], ys[i])
#         for j in range(n_samples):
#             if i == j:
#                 if skip_diagonal:
#                     continue
#                 score_matrix[i, j] = compute_decodability(
#                     Xs[i],
#                     ys[i],
#                     conditions[i],
#                     clf=lambda _: clone(clf_i),
#                     n_splits=n_splits,
#                     n_repeats=n_repeats,
#                     shuffle=shuffle,
#                 ).mean()
#             else:
#                 score_matrix[i, j] = clf_i.score(Xs[j], ys[j])

#             # if skip_diagonal and i == j:
#             #     continue
#             # score_matrix[i, j] = clf_i.score(Xs[j], ys[j])

#     # compute generalization scores
#     if n_jobs is None or n_jobs == 1:
#         [fit_and_score(i) for i in range(n_samples)]
#     else:
#         Parallel(n_jobs=n_jobs)(delayed(fit_and_score)(i) for i in range(n_samples))
#     return score_matrix


# def compute_decodability_ct_rel(
#     Xs: list[npt.ArrayLike],
#     y: npt.ArrayLike | list[npt.ArrayLike],
#     condition: npt.ArrayLike | list[npt.ArrayLike] | None = None,
#     *,
#     clf: ClassifierMixin = LinearSVC,
#     n_splits: int = 5,
#     n_repeats: int = 1,
#     shuffle: bool = False,
#     clf_kwargs: dict | None = None,
#     skip_diagonal: bool = False,
#     n_jobs: int | None = None,
# ) -> np.ndarray:
#     """
#     Compute temporal generalization accuracy across multiple related datasets.

#     Parameters
#     ----------
#     Xs : list of array-like of shape (n_samples, n_features)
#         List of feature matrices.
#     y : array-like of shape (n_samples,) or list of array-like of shape (n_samples,)
#         Target vector. If a list is provided, only the first element is used for compatibility.
#     condition : array-like of shape (n_samples,) or list of array-like of shape (n_samples,) or None, default=None
#         Group labels for the samples used while splitting the dataset into train/test set.
#         If a list is provided, only the first element is used for compatibility.
#     clf : `sklearn.base.ClassifierMixin`, default=`sklearn.svm.SVC`
#         Classifier to use.
#     n_splits : int, default=5
#         Number of folds.
#     n_repeats : int, default=1
#         Number of times cross-validation needs to be repeated.
#     shuffle : bool, default=False
#         Whether to shuffle the data before splitting into batches. Only relevant if `n_repeats` = 1.
#     clf_kwargs : dict or None, default=None
#         Additional arguments to pass to the classifier.
#     skip_diagonal : bool, default=False
#         If True, skip computation of the diagonal elements of the score matrix.
#     n_jobs : int or None, default=None
#         Number of jobs to run in parallel.

#     Returns
#     -------
#     score_matrix : `np.ndarray` for shape (n_samples, n_samples)
#         A square matrix containing the generalization accuracies. Row indices correspond to the
#         training datasets and column indices correspond to the test datasets.
#     """

#     # process inputs
#     if isinstance(y, list):
#         y = y[0]
#     if condition is not None and isinstance(condition, list):
#         condition = condition[0]

#     # initialize score matrix and classifier
#     n_samples = len(Xs)
#     score_matrix = np.zeros((n_splits, n_samples, n_samples))
#     clf = clf(**clf_kwargs) if clf_kwargs else clf()
#     clf = make_pipeline(StandardScaler(), clf)

#     # determine which cross-validation strategy to use
#     if condition is None:
#         if n_repeats == 1:
#             cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=None)
#         else:
#             cv = RepeatedStratifiedKFold(
#                 n_splits=n_splits, n_repeats=n_repeats, random_state=None
#             )
#     else:
#         if n_repeats == 1:
#             cv = MultiStratifiedKFold(
#                 n_splits=n_splits, shuffle=shuffle, random_state=None
#             )
#         else:
#             cv = RepeatedMultiStratifiedKFold(
#                 n_splits=n_splits, n_repeats=n_repeats, random_state=None
#             )

#     # define worker function
#     def fit_and_score(i):
#         clf_i = clone(clf)
#         for k, (train_inds, test_inds) in enumerate(cv.split(Xs[i], y, condition)):
#             X_train, y_train = Xs[i][train_inds], y[train_inds]
#             clf_i.fit(X_train, y_train)
#             for j in range(n_samples):
#                 if skip_diagonal and i == j:
#                     score_matrix[i, j] = np.nan
#                 X_test, y_test = Xs[j][test_inds], y[test_inds]
#                 score_matrix[k, i, j] = clf_i.score(X_test, y_test)

#     # compute generalization scores
#     if n_jobs is None or n_jobs == 1:
#         [fit_and_score(i) for i in range(n_samples)]
#     else:
#         Parallel(n_jobs=n_jobs)(delayed(fit_and_score)(i) for i in range(n_samples))
#     return score_matrix


# def compute_ccgp_ct(
#     Xs: list[npt.ArrayLike],
#     ys: list[npt.ArrayLike],
#     conditions: list[npt.ArrayLike],
#     *,
#     clf: ClassifierMixin = LinearSVC,
#     n_crossings: int = 1,
#     clf_kwargs: dict | None = None,
#     skip_diagonal: bool = False,
# ):
#     clf = clf(**clf_kwargs) if clf_kwargs else clf()
#     clf = make_pipeline(StandardScaler(), clf)

#     # Get the group indices for each dataset
#     left_inds, right_inds = None, None
#     for X, y, condition in zip(Xs, ys, conditions):
#         cv = LeaveNCrossingsOut(n_crossings=n_crossings)
#         li, ri = cv.get_group_indices(X, y, condition)
#         if left_inds is None:
#             left_inds, right_inds = [li], [ri]
#             n_splits = cv.get_n_splits(None, None, condition)
#         else:
#             left_inds.append(li)
#             right_inds.append(ri)

#     # Compute the cross-generalization scores
#     n_samples = len(Xs)
#     score_matrix = np.zeros((n_splits, n_samples, n_samples))

#     for i in range(n_samples):
#         l1, r1 = left_inds[i], right_inds[i]

#         for j, (k1, k2) in enumerate(zip(l1.keys(), r1.keys())):
#             train_inds = np.array(
#                 [i for i in range(len(X)) if i not in l1[k1] + r1[k2]]
#             )
#             clf = clf.fit(Xs[i][train_inds], ys[i][train_inds])

#             for k in range(n_samples):
#                 if skip_diagonal and i == k:
#                     continue
#                 l2, r2 = left_inds[k], right_inds[k]
#                 test_inds = np.array(l2[k1] + r2[k2])
#                 score_matrix[j, i, k] = clf.score(Xs[k][test_inds], ys[k][test_inds])

#     return np.mean(score_matrix, axis=0)