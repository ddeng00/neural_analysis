from abc import ABC, abstractmethod
from itertools import combinations, permutations, chain, product
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


class _BaseEstimator(ABC):
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
        Function to shuffle data. Must take the feature matrix, group labels, and random state as inputs.
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

        # process conditions
        if not isinstance(condition, list):
            condition = [condition]

        # infer number of samples per condition if not provided
        if n_samples_per_cond is None:
            n_samples_per_cond = data.groupby([unit] + condition).size().min()
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

        # # remove units with constant responses
        # data = data.groupby(unit).filter(lambda x: x[response].nunique() > 1)
        self.n_valid = data[unit].nunique()

        # define dichotomies
        u_conds = data[condition].drop_duplicates().values
        dichots, dichot_names, dichot_diffs = make_balanced_dichotomies(
            u_conds, cond_names=condition, return_one_sided=True
        )
        self.dichotomies = dichots
        self.dichotomy_names = dichot_names
        self.dichotomy_difficulties = dichot_diffs

        # store data
        self.data = data
        self.unit = unit
        self.response = response
        self.condition = condition

    @staticmethod
    @abstractmethod
    def shuffle(
        X: npt.ArrayLike, condition: npt.ArrayLike, rs: np.random.RandomState
    ) -> npt.ArrayLike:
        raise NotImplementedError

    def score(
        self,
        n_resamples: int = 1,
        *,
        permute: bool = True,
        n_splits: int = 5,
        n_repeats: int = 1,
        shuffle: bool = False,
        clf: BaseEstimator = LinearSVC,
        clf_kwargs: dict | None = None,
        show_progress: bool = False,
        return_clfs: bool = False,
        n_jobs: int | None = None,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
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
        clf : `BaseEstimator`, default=LinearSVC
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
        results : `pd.DataFrame`
            DataFrame containing the scores and fitted classifiers (if requested) for each dichotomy.
            The DataFrame is exploded such that each row corresponds to a single score.
        null_results : `pd.DataFrame`
            DataFrame containing the scores and fitted classifiers (if requested) for each dichotomy
            under the permutation null. The DataFrame is exploded such that each row corresponds to a single score.
        """

        # perform resampling with parallelization
        if show_progress:
            results = list(
                tqdm(
                    Parallel(return_as="generator", n_jobs=n_jobs)(
                        delayed(self._score_helper)(
                            i,
                            n_splits,
                            n_repeats,
                            shuffle,
                            clf,
                            clf_kwargs,
                            return_clfs,
                            permute,
                        )
                        for i in range(n_resamples)
                    ),
                    total=n_resamples,
                )
            )
        else:
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._score_helper)(
                    i,
                    n_splits,
                    n_repeats,
                    shuffle,
                    clf,
                    clf_kwargs,
                    return_clfs,
                    permute,
                )
                for i in range(n_resamples)
            )

        # return results
        if not permute:
            results = pd.DataFrame.from_records(chain.from_iterable(results))
            results = results.convert_dtypes()
            return results
        else:
            results, null_results = zip(*results)
            results = pd.DataFrame.from_records(chain.from_iterable(results))
            null_results = pd.DataFrame.from_records(chain.from_iterable(null_results))
            results, null_results = (
                results.convert_dtypes(),
                null_results.convert_dtypes(),
            )
            return results, null_results

    def _score_helper(
        self,
        i: int,
        n_splits: int,
        n_repeats: int,
        shuffle: bool,
        clf: BaseEstimator,
        clf_kwargs: dict | None,
        return_clfs: bool,
        permute: bool,
    ) -> list[pd.DataFrame] | tuple[list[pd.DataFrame], list[pd.DataFrame]]:

        # generate new random state
        rs = np.random.RandomState(self.random_seed + i)

        # construct condition-balanced pseudopopulation of units
        X, condition = construct_pseudopopulation(
            self.data,
            self.unit,
            self.response,
            self.condition,
            n_samples_per_cond=self.n_samples_per_cond,
            all_groups_complete=True,
            random_state=rs,
        )
        ys = [isin_2d(condition, dichot).astype(int) for dichot in self.dichotomies]

        # fit and validate each dichotomy
        res = []
        for i, dichot_name in enumerate(self.dichotomy_names):
            d_res = self.__class__.validate(
                X,
                ys[i],
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
            d_res["named"] = "unnamed" not in dichot_name
            res.append(d_res)
        if not permute:
            return res

        # estimate permutation null
        res_perm = []
        X = self.__class__.shuffle(X, condition, rs)
        for i, dichot_name in enumerate(self.dichotomy_names):
            d_res = self.__class__.validate(
                X,
                ys[i],
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
            d_res["named"] = "unnamed" not in dichot_name
            res_perm.append(d_res)

        return res, res_perm

    @staticmethod
    @abstractmethod
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
        raise NotImplementedError


class _BaseRelatedSamplesGeneralizer(_BaseEstimator):

    def __init__(
        self,
        data: pd.DataFrame,
        unit: str,
        responses: list[str],
        condition: str | list[str],
        *,
        n_conditions: int | None = None,
        n_samples_per_cond: int | None = None,
        random_seed: int | None = None,
    ) -> None:

        # check if generalization is possible
        if len(responses) < 2:
            raise ValueError("At least two responses are required for generalization.")
        self.responses = responses
        super().__init__(
            data,
            unit,
            responses,
            condition,
            n_conditions=n_conditions,
            n_samples_per_cond=n_samples_per_cond,
            random_seed=random_seed,
        )
        del self.response

    def _score_helper(
        self,
        i: int,
        n_splits: int,
        n_repeats: int,
        shuffle: bool,
        clf: BaseEstimator,
        clf_kwargs: dict | None,
        return_clfs: bool,
        permute: bool,
    ) -> list[pd.DataFrame] | tuple[list[pd.DataFrame], list[pd.DataFrame]]:

        # generate new random state
        rs = np.random.RandomState(self.random_seed + i)

        # construct condition-balanced pseudopopulation of units
        Xs, condition = construct_pseudopopulation(
            self.data,
            self.unit,
            self.responses,
            self.condition,
            n_samples_per_cond=self.n_samples_per_cond,
            all_groups_complete=True,
            random_state=rs,
        )
        ys = [isin_2d(condition, dichot).astype(int) for dichot in self.dichotomies]

        # fit and validate each dichotomy
        res = []
        for k, dichot_name in enumerate(self.dichotomy_names):
            for i, j in product(range(len(self.responses)), repeat=2):
                d_res = self.__class__.validate(
                    Xs[i],
                    Xs[j],
                    ys[k],
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
                d_res["named"] = "unnamed" not in dichot_name
                d_res["train"] = self.responses[i]
                d_res["test"] = self.responses[j]
                res.append(d_res)
        if not permute:
            return res

        # estimate permutation null
        res_perm = []
        Xs = self.shuffle(Xs, condition, rs)
        for k, dichot_name in enumerate(self.dichotomy_names):
            for i, j in product(range(len(self.responses)), repeat=2):
                d_res = self.__class__.validate(
                    Xs[i],
                    Xs[j],
                    ys[k],
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
                d_res["named"] = "unnamed" not in dichot_name
                d_res["train"] = self.responses[i]
                d_res["test"] = self.responses[j]
                res_perm.append(d_res)

        return res, res_perm

    @staticmethod
    @abstractmethod
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
        raise NotImplementedError


class _BaseIndependentSamplesGeneralizer(_BaseEstimator):
    def __init__(
        self,
        data: pd.DataFrame,
        unit: str,
        response: str,
        condition: str | list[str],
        group: str | None = None,
        *,
        n_conditions: int | None = None,
        n_samples_per_cond: int | dict[Any, int] | None = None,
        random_seed: int | None = None,
    ) -> None:

        # process conditions
        if not isinstance(condition, list):
            condition = [condition]
        u_conds = data[condition].drop_duplicates().values
        self.n_init = data[unit].nunique()

        # process sample groups
        tmp_data, tmp_grp = [], []
        for k, v in data.groupby(group):
            tmp_data.append(v)
            tmp_grp.append(k)
        data = tmp_data
        group = tmp_grp
        if len(data) < 2:
            raise ValueError("At least two groups are required for generalization.")

        # infer number of samples per condition if not provided
        if n_samples_per_cond is None:
            n_samples_per_cond = [d.gropuby(condition).size().min() for d in data]
        elif isinstance(n_samples_per_cond, int):
            n_samples_per_cond = [n_samples_per_cond] * len(data)
        else:
            n_samples_per_cond = [n_samples_per_cond[k] for k in group]
        self.n_samples_per_cond = n_samples_per_cond

        # check random seed
        self.random_seed = (
            np.random.randint(1e6) if random_seed is None else random_seed
        )

        # remove units missing conditional trials
        data = [
            remove_groups_missing_conditions(
                v,
                unit,
                condition,
                n_conditions=n_conditions,
                n_samples_per_cond=nspc,
            )
            for v, nspc in zip(data, n_samples_per_cond)
        ]

        # # remove units with constant responses
        # data = [
        #     v.groupby(unit).filter(lambda x: x[response].nunique() > 1) for v in data
        # ]

        # align neurons across groups
        ids = [v[unit].unique() for v in data]
        ids = set.intersection(*map(set, ids))
        data = [v[v[unit].isin(ids)] for v in data]
        self.n_valid = len(ids)

        # define dichotomies
        dichots, dichot_names, dichot_diffs = make_balanced_dichotomies(
            u_conds, cond_names=condition, return_one_sided=True
        )
        self.dichotomies = dichots
        self.dichotomy_names = dichot_names
        self.dichotomy_difficulties = dichot_diffs

        # store data
        self.data = data
        self.unit = unit
        self.response = response
        self.condition = condition
        self.group = group

    def score(
        self,
        n_resamples: int = 1,
        *,
        permute: bool = True,
        n_splits: int | dict[Any, int] = 5,
        n_repeats: int | dict[Any, int] = 1,
        shuffle: bool | dict[Any, bool] = False,
        clf: BaseEstimator = LinearSVC,
        clf_kwargs: dict | None = None,
        show_progress: bool = False,
        return_clfs: bool = False,
        n_jobs: int | None = None,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:

        # process inputs
        if isinstance(n_splits, int):
            n_splits = [n_splits] * len(self.data)
        else:
            n_splits = [n_splits[k] for k in self.group]
        if isinstance(n_repeats, int):
            n_repeats = [n_repeats] * len(self.data)
        else:
            n_repeats = [n_repeats[k] for k in self.group]
        if isinstance(shuffle, bool):
            shuffle = [shuffle] * len(self.data)
        else:
            shuffle = [shuffle[k] for k in self.group]

        return super().score(
            n_resamples=n_resamples,
            permute=permute,
            n_splits=n_splits,
            n_repeats=n_repeats,
            shuffle=shuffle,
            clf=clf,
            clf_kwargs=clf_kwargs,
            show_progress=show_progress,
            return_clfs=return_clfs,
            n_jobs=n_jobs,
        )

    def _score_helper(
        self,
        i: int,
        n_splits: list[int],
        n_repeats: list[int],
        shuffle: list[bool],
        clf: BaseEstimator,
        clf_kwargs: dict | None,
        return_clfs: bool,
        permute: bool,
    ) -> list[pd.DataFrame] | tuple[list[pd.DataFrame], list[pd.DataFrame]]:

        # generate new random state
        rs = np.random.RandomState(self.random_seed + i)

        # construct condition-balanced pseudopopulation of units
        Xs, conditions = [], []
        for v, nspc in zip(self.data, self.n_samples_per_cond):
            X, condition = construct_pseudopopulation(
                v,
                self.unit,
                self.response,
                self.condition,
                n_samples_per_cond=nspc,
                all_groups_complete=True,
                random_state=rs,
            )
            Xs.append(X)
            conditions.append(condition)
        ys = [
            [isin_2d(condition, dichot).astype(int) for dichot in self.dichotomies]
            for condition in conditions
        ]

        # fit and validate each dichotomy
        res = []
        for i, j in product(range(len(self.data)), repeat=2):
            if n_splits[i] == 0:
                continue
            for k, dichot_name in enumerate(self.dichotomy_names):
                d_res = self.__class__.validate(
                    Xs[i],
                    Xs[j],
                    ys[i][k],
                    ys[j][k],
                    conditions[i],
                    conditions[j],
                    same_group=i == j,
                    n_splits=n_splits[i],
                    n_repeats=n_repeats[i],
                    shuffle=shuffle[i],
                    clf=clf,
                    clf_kwargs=clf_kwargs,
                    return_clfs=return_clfs,
                    random_state=rs,
                )
                d_res["dichotomy"] = dichot_name
                d_res["named"] = "unnamed" not in dichot_name
                d_res["train"] = self.group[i]
                d_res["test"] = self.group[j]
                res.append(d_res)
        if not permute:
            return res

        # estimate permutation null
        res_perm = []
        for i, j in product(range(len(self.data)), repeat=2):
            if n_splits[i] == 0:
                continue

            # pool data and shuffle
            if i == j:
                Xi = Xj = self.shuffle(Xs[i], conditions[i], rs)
            else:
                X_ij = np.vstack([Xs[i], Xs[j]])
                c_ij = np.vstack([conditions[i], conditions[j]])
                X_ij = self.shuffle(X_ij, c_ij, rs)
                Xi = X_ij[: len(Xs[i])]
                Xj = X_ij[len(Xs[i]) :]

            for k, dichot_name in enumerate(self.dichotomy_names):
                d_res = self.__class__.validate(
                    Xi,
                    Xj,
                    ys[i][k],
                    ys[j][k],
                    conditions[i],
                    conditions[j],
                    same_group=i == j,
                    n_splits=n_splits[i],
                    n_repeats=n_repeats[i],
                    shuffle=shuffle[i],
                    clf=clf,
                    clf_kwargs=clf_kwargs,
                    return_clfs=return_clfs,
                    random_state=rs,
                )
                d_res["dichotomy"] = dichot_name
                d_res["named"] = "unnamed" not in dichot_name
                d_res["train"] = self.group[i]
                d_res["test"] = self.group[j]
                res_perm.append(d_res)

        return res, res_perm

    @staticmethod
    @abstractmethod
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
        raise NotImplementedError


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
        random_state: int | np.random.RandomState | None = None,
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
        random_seed: int | None = None,
    ) -> None:

        super().__init__(
            data,
            unit,
            response,
            condition,
            n_conditions=n_conditions,
            n_samples_per_cond=1,
            random_seed=random_seed,
        )

        # remove units with constant responses
        data = self.data.copy()
        data = data.groupby(unit).filter(lambda x: x[response].nunique() > 1)
        self.n_valid = data[unit].nunique()

        # normalize data
        unit_mean = data.groupby(unit)[response].mean()
        unit_std = data.groupby(unit)[response].std()
        data[response] = (data[response] - data[unit].map(unit_mean)) / data[unit].map(
            unit_std
        )

        # generate conditional averages
        self.data = data.groupby([unit] + condition)[response].mean().reset_index()

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
