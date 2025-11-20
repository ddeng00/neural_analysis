from abc import ABC, abstractmethod
from itertools import chain, product
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed
from tqdm import tqdm

from ..partition import make_balanced_dichotomies
from ..preprocess import remove_groups_missing_conditions, construct_pseudopopulation
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
        dichot_map: dict[str, Any] | None = None,
        named_only: bool = False,
        verbose: bool = False,
        remove_const: bool = False,
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize the estimator.

        Parameters
        ----------
        n_conditions : int or None, default=None
            Number of conditions. If None, the number of conditions is inferred from the data.
        dichot_map : dict or None, default=None
            Custom/additional dichotomies to indicate.
        verbose : bool, default=False
            Whether to print verbose output.
        remove_const : bool, default=False
            Whether to remove units with constant responses.
        random_seed : int or None, default=None
            Random seed used for all computations. If None, a new one is generated.
        """

        # process conditions
        if not isinstance(condition, list):
            condition = [condition]
        if not isinstance(response, list):
            response = [response]

        # remove irrelevant columns
        data = data[[unit] + response + condition]
        data = data.infer_objects()

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

        # remove units with constant responses
        if remove_const:
            for res in response:
                data = data.groupby(unit).filter(lambda x: x[res].nunique() > 1)
        self.n_valid = data[unit].nunique()

        # print results
        self.verbose = verbose
        if verbose:
            prop_left = self.n_valid / self.n_init * 100
            print(f"Remaining Units: {self.n_valid}/{self.n_init} ({prop_left:.1f}%)")

        # define dichotomies
        u_conds = data[condition].drop_duplicates().values
        dichots, dichot_names, dichot_diffs = make_balanced_dichotomies(
            u_conds,
            cond_names=condition,
            return_one_sided=True,
            dichot_map=dichot_map,
            named_only=named_only,
        )
        self.dichotomies = dichots
        self.dichotomy_names = dichot_names
        self.dichotomy_difficulties = dichot_diffs

        # store data
        self.data = data
        self.unit = unit
        self.response = response if len(response) > 1 else response[0]
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
        n_permutes: int = 0,
        n_splits: int = 5,
        n_repeats: int = 1,
        shuffle: bool = False,
        clf: BaseEstimator = LinearSVC,
        clf_kwargs: dict | None = None,
        return_clfs: bool = False,
        n_jobs: int | None = None,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Construct resampled pseudopopulations and compute scores for each dichotomy.

        Parameters
        ----------
        n_resamples : int, default=1
            Number of resamples to perform.
        n_permutes : int, default=0
            Number of permutations to estimate the null distribution. If 0, no permutation is performed.
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
        if self.verbose:
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
                            n_permutes,
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
                    n_permutes,
                )
                for i in range(n_resamples)
            )

        # return results
        if not n_permutes:
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
        n_permutes: int,
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
        if n_permutes <= 0:
            return res

        # estimate permutation null
        res_perm = []
        for _ in range(n_permutes):
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
    """
    Base class for dichotomy-based population-level estimators generalizaing across related samples.

    Attributes
    ----------
    data : `pd.DataFrame`
        DataFrame containing the data.
    unit : str
        Name of the column containing the unit identifiers.
    responses : list of str
        Name(s) of the column containing the response variables to generalize across.
    condition : str or list of str
        Name of the column(s) containing the condition labels.
    n_samples_per_cond : int
        Number of samples per condition.
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
        responses: list[str],
        condition: str | list[str],
        *,
        n_conditions: int | None = None,
        n_samples_per_cond: int | None = None,
        dichot_map: dict[str, Any] | None = None,
        named_only: bool = False,
        verbose: bool = False,
        remove_const: bool = False,
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize the estimator.

        Parameters
        ----------
        n_conditions : int or None, default=None
            Number of conditions. If None, the number of conditions is inferred from the data.
        dichot_map : dict or None, default=None
            Custom/additional dichotomies to indicate.
        verbose : bool, default=False
            Whether to print verbose output.
        remove_const : bool, default=False
            Whether to remove units with constant responses.
        random_seed : int or None, default=None
            Random seed used for all computations. If None, a new one is generated.
        """

        # check if generalization is possible
        if len(responses) < 2:
            raise ValueError("At least two responses are required for generalization.")
        super().__init__(
            data,
            unit,
            responses,
            condition,
            n_conditions=n_conditions,
            n_samples_per_cond=n_samples_per_cond,
            dichot_map=dichot_map,
            named_only=named_only,
            verbose=verbose,
            remove_const=remove_const,
            random_seed=random_seed,
        )

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
    ) -> list[pd.DataFrame] | tuple[list[pd.DataFrame], list[pd.DataFrame]]:

        # generate new random state
        rs = np.random.RandomState(self.random_seed + i)

        # construct condition-balanced pseudopopulation of units
        Xs, condition = construct_pseudopopulation(
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
        for k, dichot_name in enumerate(self.dichotomy_names):
            for i, j in product(range(len(self.response)), repeat=2):
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
                d_res["train"] = self.response[i]
                d_res["test"] = self.response[j]
                res.append(d_res)
        if n_permutes <= 0:
            return res

        # estimate permutation null
        res_perm = []
        for _ in range(n_permutes):
            Xs = self.shuffle(Xs, condition, rs)
            for k, dichot_name in enumerate(self.dichotomy_names):
                for i, j in product(range(len(self.response)), repeat=2):
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
                    d_res["train"] = self.response[i]
                    d_res["test"] = self.response[j]
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
    """
    Base class for dichotomy-based population-level estimators generalizaing across independent samples.

    Attributes
    ----------
    data : list of `pd.DataFrame`
        List of DataFrames containing the data for each group.
    unit : str
        Name of the column containing the unit identifiers.
    response : str
        Name of the column containing the response variable.
    condition : str or list of str
        Name of the column(s) containing the condition labels.
    group: list of str
        List of group labels (same order as `data`).
    n_samples_per_cond : int
        Number of samples per condition.
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
        group: str | None = None,
        *,
        n_conditions: int | None = None,
        n_samples_per_cond: int | dict[Any, int] | None = None,
        dichot_map: dict[str, Any] | None = None,
        named_only: bool = False,
        verbose: bool = False,
        remove_const: bool = False,
        random_seed: int | None = None,
    ) -> None:
        """
        Initialize the estimator.

        Parameters
        ----------
        n_conditions : int or None, default=None
            Number of conditions. If None, the number of conditions is inferred from the data.
        dichot_map : dict or None, default=None
            Custom/additional dichotomies to indicate.
        verbose : bool, default=False
            Whether to print verbose output.
        remove_const : bool, default=False
            Whether to remove units with constant responses.
        random_seed : int or None, default=None
            Random seed used for all computations. If None, a new one is generated.
        """

        # process conditions
        if not isinstance(condition, list):
            condition = [condition]
        u_conds = data[condition].drop_duplicates().values
        self.n_init = data[unit].nunique()

        # remove irrelevant columns
        data = data[[unit] + [response] + condition + ([group] if group else [])]
        data = data.infer_objects()

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
            n_samples_per_cond = [d.groupby(condition).size().min() for d in data]
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

        # remove units with constant responses
        if remove_const:
            data = [
                v.groupby(unit).filter(lambda x: x[response].nunique() > 1)
                for v in data
            ]

        # align neurons across groups
        ids = [v[unit].unique() for v in data]
        ids = set.intersection(*map(set, ids))
        data = [v[v[unit].isin(ids)] for v in data]
        self.n_valid = len(ids)

        # print results
        self.verbose = verbose
        if verbose:
            prop_left = self.n_valid / self.n_init * 100
            print(f"Remaining Units: {self.n_valid}/{self.n_init} ({prop_left:.1f}%)")

        # define dichotomies
        dichots, dichot_names, dichot_diffs = make_balanced_dichotomies(
            u_conds, cond_names=condition, return_one_sided=True
        )
        self.dichotomies = dichots
        self.dichotomy_names = dichot_names
        self.dichotomy_difficulties = dichot_diffs

        # assign additional dichotomies
        if dichot_map is not None:
            for dichot_name, dichot in dichot_map.items():
                for i, d in enumerate(self.dichotomies):
                    if all(isin_2d(d, dichot)) or not any(isin_2d(d, dichot)):
                        self.dichotomy_names[i] = dichot_name
                        break

        # remove irrelevant dichotomies
        if named_only:
            sel_inds = [
                i
                for i, name in enumerate(self.dichotomy_names)
                if "unnamed" not in name
            ]
            self.dichotomies = [self.dichotomies[i] for i in sel_inds]
            self.dichotomy_names = [self.dichotomy_names[i] for i in sel_inds]
            self.dichotomy_difficulties = [
                self.dichotomy_difficulties[i] for i in sel_inds
            ]

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
        n_permutes: int = 0,
        n_splits: int | dict[Any, int] = 5,
        n_repeats: int | dict[Any, int] = 1,
        shuffle: bool | dict[Any, bool] = False,
        clf: BaseEstimator = LinearSVC,
        clf_kwargs: dict | None = None,
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
            n_permutes=n_permutes,
            n_splits=n_splits,
            n_repeats=n_repeats,
            shuffle=shuffle,
            clf=clf,
            clf_kwargs=clf_kwargs,
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
        n_permutes: int,
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
        if n_permutes <= 0:
            return res

        # estimate permutation null
        res_perm = []
        for _ in range(n_permutes):
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
