from collections import defaultdict
from os import cpu_count
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from ._population_helper import _cross_cond_helper, _cross_temp_helper
from ..utils import group_df_by


def pop_decode_var_cross_cond(
    data: pd.DataFrame,
    spike_rate_cols: str | list[str],
    variable_col: str,
    condition_col: str,
    neuron_col: str,
    min_trials: int | None = None,
    n_pseudo: int = 250,
    subsample_ratio: float = 1.0,
    n_permute: int = 10,
    n_jobs: int = -1,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estimate variable decoding generalizability across conditions based on pseudo-populations of neurons.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing spike rates, variable values, and neuron identities.
    spike_rate_cols : str or list[str]
        Column name(s) of spike rates in data.
    variable_col : str
        Column name of variable values in data.
    condition_col : str
        Column name of condition values in data.
    neuron_col : str
        Column name of neuron identities in data.
    min_trials : int or None, default=None
        Minimum number of trials to include in each pseudo-population.
    n_pseudo : int, default=250
        Number of random pseudo-populations to construct.
    subsample_ratio : float, default=0.75
        Ratio of neurons to include in pseudo-population.
    n_permute : int, default=10
        Number of permutation tests to perform for each pseudo-population.
        If 0, no permuatation test will be performed.
    n_jobs : int, default=-1
        Number of jobs to run in parallel.

    Returns
    -------
    accuracies : pd.DataFrame
        Dataframe of decoding accuracies.
    null_accuracies : pd.DataFrame
        Dataframe of null distribution accuracies.
        Only returned if permutation tests are performed.
    """

    # check input
    if isinstance(spike_rate_cols, str):
        spike_rate_cols = [spike_rate_cols]
    if n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs < 1:
        raise ValueError("Invalid n_jobs.")
    n_jobs = min(n_jobs, cpu_count())

    # group unique conditions by variable
    vc1, vc2 = [
        v[condition_col].unique().tolist()
        for v in group_df_by(data, variable_col).values()
    ]

    # pre-processing
    if min_trials is None:
        min_trials = (
            data.groupby([variable_col, condition_col])
            .value_counts([neuron_col])
            .groupby(neuron_col)
            .min()
            .min()
        )
    data = group_df_by(data, neuron_col)

    # initialize variables
    accuracies = defaultdict(list)
    null_accuracies = defaultdict(list)

    # start analysis
    with mp.Pool(n_jobs) as pool:
        pbar = tqdm(total=n_pseudo)
        results = []
        func = partial(
            _cross_cond_helper,
            data,
            spike_rate_cols,
            variable_col,
            condition_col,
            min_trials,
            subsample_ratio,
            n_permute,
            vc1,
            vc2,
        )
        for result in pool.imap_unordered(func, range(n_pseudo)):
            results.append(result)
            pbar.update()

    # gather results
    for acc, null in results:
        for k, v in acc.items():
            accuracies[k].extend(v)
        for k, v in null.items():
            null_accuracies[k].extend(v)

    # convert results to dataframe
    accuracies = pd.DataFrame(accuracies)
    if n_permute > 0:
        null_accuracies = pd.DataFrame(null_accuracies)

    # return results
    if n_permute > 0:
        return accuracies, null_accuracies
    else:
        return accuracies


def pop_decode_var_cross_temp(
    data: pd.DataFrame,
    spike_rate_cols: list[str],
    variable_col: str,
    neuron_col: str,
    min_trials: int | None = None,
    n_pseudo: int = 250,
    subsample_ratio: float = 1.0,
    n_permute: int = 10,
    skip_self: bool = False,
    n_jobs: int = -1,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estimate variable decoding generalizability across time based on pseudo-populations of neurons.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing spike rates, variable values, and neuron identities.
    spike_rate_cols : str or list[str]
        Column name(s) of spike rates in data.
    variable_col : str
        Column name of variable values in data.
    neuron_col : str
        Column name of neuron identities in data.
    min_trials : int or None, default=None
        Minimum number of trials to include in each pseudo-population.
    n_pseudo : int, default=250
        Number of random pseudo-populations to construct.
    subsample_ratio : float, default=0.75
        Ratio of neurons to include in pseudo-population.
    n_permute : int, default=10
        Number of permutation tests to perform for each pseudo-population.
        If 0, no permuatation test will be performed.
    skip_self : bool, default=True
        Whether to skip self-comparisons.
    n_jobs : int, default=-1
        Number of jobs to run in parallel.

    Returns
    -------
    accuracies : pd.DataFrame
        Dataframe of decoding accuracies.
    null_accuracies : pd.DataFrame
        Dataframe of null distribution accuracies.
        Only returned if permutation tests are performed.
    weights : pd.DataFrame
        Dataframe of neuron weights.
        Only returned if return_weights is True.
    """

    # check input
    if len(spike_rate_cols) < 2:
        raise ValueError("At least two spike rate columns are required.")
    if n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs < 1:
        raise ValueError("Invalid n_jobs.")
    n_jobs = min(n_jobs, cpu_count())

    # pre-processing
    if min_trials is None:
        min_trials = (
            data.groupby(variable_col)
            .value_counts([neuron_col])
            .groupby(neuron_col)
            .min()
            .min()
        )
    data = group_df_by(data, neuron_col)

    # initialize variables
    accuracies = defaultdict(list)
    null_accuracies = defaultdict(list)

    # start analysis
    with mp.Pool(n_jobs) as pool:
        pbar = tqdm(total=n_pseudo)
        results = []
        func = partial(
            _cross_temp_helper,
            data,
            spike_rate_cols,
            variable_col,
            min_trials,
            subsample_ratio,
            n_permute,
            skip_self,
        )
        for result in pool.imap_unordered(func, range(n_pseudo)):
            results.append(result)
            pbar.update()

    # gather results
    for acc, null in results:
        for k, v in acc.items():
            accuracies[k].extend(v)
        for k, v in null.items():
            null_accuracies[k].extend(v)

    # convert results to dataframe
    accuracies = pd.DataFrame(accuracies)
    if n_permute > 0:
        null_accuracies = pd.DataFrame(null_accuracies)

    # return results
    if n_permute > 0:
        return accuracies, null_accuracies
    else:
        return accuracies


def pop_decode_var(
    data: pd.DataFrame,
    spike_rate_cols: str | list[str],
    variable_col: str,
    neuron_col: str,
    min_trials: int | None = None,
    n_pseudo: int = 250,
    subsample_ratio: float = 1.0,
    n_splits: int = 5,
    n_permute: int = 10,
    show_progress: bool = True,
    show_accuracy: bool = False,
    return_weights: bool = False,
    n_jobs: int = -1,
) -> (
    pd.DataFrame
    | tuple[pd.DataFrame, pd.DataFrame]
    | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
):
    """
    Estimate variable decoding accuracy based on pseudo-populations of neurons.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing spike rates, variable values, and neuron identities.
    spike_rate_cols : str or list[str]
        Column name(s) of spike rates in data.
    variable_col : str
        Column name of variable values in data.
    neuron_col : str
        Column name of neuron identities in data.
    min_trials : int or None, default=None
        Minimum number of trials to include in each pseudo-population.
    n_pseudo : int, default=250
        Number of random pseudo-populations to construct.
    subsample_ratio : float, default=0.75
        Ratio of neurons to include in pseudo-population.
    n_splits : int, default=5
        Number of cross-validation splits to use.
    n_splits : int, default=5
        Number of cross-validation splits to use.
    n_permute : int, default=10
        Number of permutation tests to perform for each pseudo-population.
        If 0, no permuatation test will be performed.
    show_progress : bool, default=True
        Whether to show progress bar.
    show_accuracy : bool, default=True
        Whether to show decoding accuracy.
        Ignored if show_progress is False.
    return_weights : bool, default=False
        Whether to return neuron weights.
    n_jobs : int, default=-1
        Number of jobs to run in parallel.

    Returns
    -------
    accuracies : pd.DataFrame
        Dataframe of decoding accuracies.
    null_accuracies : pd.DataFrame
        Dataframe of null distribution accuracies.
        Only returned if permutation tests are performed.
    weights : pd.DataFrame
        Dataframe of neuron weights.
        Only returned if return_weights is True.
    """

    # check input
    if isinstance(spike_rate_cols, str):
        spike_rate_cols = [spike_rate_cols]

    # define decoding model
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LinearSVC()),
        ]
    )
    cv = StratifiedKFold(n_splits=n_splits)

    # pre-processing
    if min_trials is None:
        min_trials = (
            data.groupby(variable_col)
            .value_counts([neuron_col])
            .groupby(neuron_col)
            .min()
            .min()
        )
    data = group_df_by(data, neuron_col)

    # initialize variables
    accuracies = defaultdict(list)
    null_accuracies = defaultdict(list)
    weights = defaultdict(list)

    # start analysis
    acc_ave = {}
    with tqdm(total=n_pseudo * len(spike_rate_cols), disable=not show_progress) as pbar:
        for i in range(n_pseudo):
            # generate random pseudo-population
            pseudo = {
                neuron: df.groupby(variable_col)
                .sample(n=min_trials)
                .reset_index(drop=True)
                for neuron, df in data.items()
            }

            # select random subset of neurons
            neurons = list(pseudo.keys())
            if subsample_ratio < 1:
                to_remove = np.random.choice(
                    neurons,
                    size=int((1 - subsample_ratio) * len(neurons)),
                    replace=False,
                )
                pseudo = {
                    neuron: df
                    for neuron, df in pseudo.items()
                    if neuron not in to_remove
                }
            if subsample_ratio < 1:
                to_remove = np.random.choice(
                    neurons,
                    size=int((1 - subsample_ratio) * len(neurons)),
                    replace=False,
                )
                pseudo = {
                    neuron: df
                    for neuron, df in pseudo.items()
                    if neuron not in to_remove
                }

            # estimate accuracies for each spike window
            for window in spike_rate_cols:
                # gather data
                X, y = {}, None
                for neuron, df in pseudo.items():
                    X[neuron] = np.asarray(df[window])
                    if y is None:
                        y = df[variable_col]
                X = pd.DataFrame(X)

                # cross-validate
                cv_results = cross_validate(
                    model, X, y, cv=cv, n_jobs=n_jobs, return_estimator=True
                )
                cv_accuracies = cv_results["test_score"]
                accuracies[window].extend(cv_accuracies)
                if return_weights:
                    cv_weights = [
                        pd.Series(cv_model["clf"].coef_[0], index=X.columns)
                        for cv_model in cv_results["estimator"]
                    ]
                    weights[window].extend(cv_weights)

                # perform permutation tests
                for _ in range(n_permute):
                    null_scores = cross_val_score(
                        model,
                        X,
                        np.random.permutation(y),
                        cv=cv,
                        n_jobs=n_jobs,
                    )
                    null_accuracies[window].extend(null_scores)

                # update progress bar
                if show_progress:
                    if show_accuracy and i % 25 == 0:
                        acc_ave[window] = np.mean(accuracies[window])
                    pbar.update()

            if show_progress:
                progress = {
                    "n_trials": min_trials,
                    "n_selected": f"{len(X.columns)}/{len(neurons)}",
                }
                if show_accuracy:
                    progress.update(acc_ave)
                pbar.set_postfix(progress)

    # convert results to dataframe
    accuracies = pd.DataFrame(accuracies)
    if n_permute > 0:
        null_accuracies = pd.DataFrame(null_accuracies)
    if return_weights:
        tmp = []
        for k, v in weights.items():
            v = pd.concat(v, axis=1).T
            v["window"] = k
            tmp.append(v)
        weights = pd.concat(tmp)

    # return results
    if n_permute > 0 and return_weights:
        return accuracies, null_accuracies, weights
    elif n_permute > 0:
        return accuracies, null_accuracies
    elif return_weights:
        return accuracies, weights
    else:
        return accuracies