from collections import defaultdict
from os import cpu_count

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import (
    cross_val_score,
    cross_validate,
    BaseCrossValidator,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from .utils import group_df_by


def _pseudo_pop_decode_var_cross_time_helper(
    data,
    spike_rate_cols,
    variable_col,
    min_trials,
    subsample_ratio,
    n_permute,
    skip_self,
):

    # initialize variables
    accuracies = defaultdict(list)
    null_accuracies = defaultdict(list)

    # define decoding model
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LinearSVC()),
        ]
    )
    cv = StratifiedKFold()

    # generate random pseudo-population
    pseudo = {
        neuron: df.groupby(variable_col).sample(n=min_trials).reset_index(drop=True)
        for neuron, df in data.items()
    }

    # select random subset of neurons
    neurons = list(pseudo.keys())
    to_remove = np.random.choice(
        neurons,
        size=int((1 - subsample_ratio) * len(neurons)),
        replace=False,
    )
    pseudo = {neuron: df for neuron, df in pseudo.items() if neuron not in to_remove}

    # estimate accuracies across each pair of spike windows
    for window1 in spike_rate_cols:
        # gather base data
        X_base, y = {}, None
        for neuron, df in pseudo.items():
            X_base[neuron] = np.asarray(df[window1])
            if y is None:
                y = df[variable_col]
        X_base = pd.DataFrame(X_base)

        # estimate accuracies for each other spike window
        for window2 in spike_rate_cols:
            if skip_self and window1 == window2:
                continue

            name = f"{window1} → {window2}"

            # gather comparison data
            X_other = {}
            for neuron, df in pseudo.items():
                X_other[neuron] = np.asarray(df[window2])
            X_other = pd.DataFrame(X_other)

            # cross-temporal cross-validate
            for train_idx, test_idx in cv.split(X_base, y):
                X_train = X_base.iloc[train_idx]
                X_test = X_other.iloc[test_idx]
                model.fit(X_train, y.iloc[train_idx])
                accuracy = model.score(X_test, y.iloc[test_idx])
                accuracies[name].append(accuracy)

                # perform permutation tests
                for _ in range(n_permute):
                    y_perm = np.random.permutation(y)
                    for train_idx, test_idx in cv.split(X_base, y_perm):
                        X_train = X_base.iloc[train_idx]
                        X_test = X_other.iloc[test_idx]
                        model.fit(X_train, y_perm.iloc[train_idx])
                        null_accuracy = model.score(X_test, y_perm.iloc[test_idx])
                        null_accuracies[name].append(null_accuracy)

    return accuracies, null_accuracies


def pseudo_pop_decode_var_cross_temp(
    data: pd.DataFrame,
    spike_rate_cols: list[str],
    variable_col: str,
    neuron_col: str,
    n_pseudo: int = 250,
    subsample_ratio: float = 0.75,
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

    # pre-processing
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
    max_workers = cpu_count() if n_jobs == -1 else n_jobs
    results = process_map(
        _pseudo_pop_decode_var_cross_time_helper,
        [data] * n_pseudo,
        [spike_rate_cols] * n_pseudo,
        [variable_col] * n_pseudo,
        [min_trials] * n_pseudo,
        [subsample_ratio] * n_pseudo,
        [n_permute] * n_pseudo,
        [skip_self] * n_pseudo,
        max_workers=max_workers,
    )

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


def pseudo_pop_decode_var(
    data: pd.DataFrame,
    spike_rate_cols: str | list[str],
    variable_col: str,
    neuron_col: str,
    n_pseudo: int = 250,
    subsample_ratio: float = 0.75,
    cv: BaseCrossValidator = StratifiedKFold(),
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
    n_pseudo : int, default=250
        Number of random pseudo-populations to construct.
    subsample_ratio : float, default=0.75
        Ratio of neurons to include in pseudo-population.
    cv : `sklearn.model_selection.BaseCrossValidator`, default=`sklearn.model_selection.StratifiedKFold()`
        Cross-validation splitter.
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

    # pre-processing
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
            to_remove = np.random.choice(
                neurons,
                size=int((1 - subsample_ratio) * len(neurons)),
                replace=False,
            )
            pseudo = {
                neuron: df for neuron, df in pseudo.items() if neuron not in to_remove
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
