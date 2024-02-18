from collections import defaultdict
from os import cpu_count
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

from ._population_helper import (
    _pop_decode_var_cross_temp_helper,
    _pop_decode_var_helper,
    _pop_decode_var_cross_cond_helper,
    _pop_vec_sim_var_cross_cond_helper,
)
from ..utils import group_df_by


def pop_vec_sim_var_cross_cond(
    data: pd.DataFrame,
    spike_rate_cols: str | list[str],
    variable_col: str,
    condition_col: str,
    neuron_col: str,
    min_trials: int | None = None,
    n_samples: int = 250,
    subsample_ratio: float = 1.0,
    n_permute: int = 10,
    show_progress: bool = True,
    n_jobs: int = -1,
) -> dict[str : pd.DataFrame]:
    """
    Estimate coding direction similarity across conditions based on pseudo-populations of neurons.

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
    n_samples : int, default=250
        Number of neuron and trial samples to construct.
    subsample_ratio : float, default=0.75
        Ratio of neurons to include in pseudo-population for each sample construction.
    n_permute : int, default=10
        Number of permutation tests to perform for each pseudo-population.
        If 0, no permuatation test will be performed.
    show_progress : bool, default=True
        Whether to show progress bar.
    n_jobs : int, default=-1
        Number of jobs to run in parallel.

    Returns
    -------
    results : dict
        Dictionary containing similarity results. Keys include:
        - 'similarities': Dataframe of coding direction similarities.
        - 'null_similarities': Dataframe of null distribution similarities. Only available if permutation tests are performed.
    """

    # check input
    if isinstance(spike_rate_cols, str):
        spike_rate_cols = [spike_rate_cols]
    for col in spike_rate_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data.")
    if variable_col not in data.columns:
        raise ValueError(f"Column '{variable_col}' not found in data.")
    if condition_col not in data.columns:
        raise ValueError(f"Column '{condition_col}' not found in data.")
    if neuron_col not in data.columns:
        raise ValueError(f"Column '{neuron_col}' not found in data.")

    # set n_jobs
    if n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs < 1:
        raise ValueError("n_jobs must be -1 or a positive integer.")
    n_jobs = min(n_samples, n_jobs, cpu_count())
    chunksize = int(np.ceil(n_samples / n_jobs))

    # group unique conditions by variable
    cond_grp_1, cond_grp_2 = [
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
    similarities = defaultdict(list)
    if n_permute > 0:
        null_similarities = defaultdict(list)

    # start analysis
    with mp.Pool(n_jobs) as pool:
        pbar = tqdm(total=n_samples, disable=not show_progress)
        all_results = []
        func = partial(
            _pop_vec_sim_var_cross_cond_helper,
            data,
            spike_rate_cols,
            variable_col,
            condition_col,
            min_trials,
            subsample_ratio,
            n_permute,
            cond_grp_1,
            cond_grp_2,
        )
        for results in pool.imap_unordered(func, range(n_samples), chunksize=chunksize):
            all_results.append(results)
            if show_progress:
                pbar.update()
        pbar.close()

    # gather results
    for results in all_results:
        for k, v in results["accuracies"].items():
            similarities[k].extend(v)
        if n_permute > 0:
            for k, v in results["null_accuracies"].items():
                null_similarities[k].extend(v)

    # convert results to dataframe
    similarities = pd.DataFrame(similarities)
    if n_permute > 0:
        null_similarities = pd.DataFrame(null_similarities)

    # return results
    results = {"similarities": similarities}
    if n_permute > 0:
        results["null_similarities"] = null_similarities
    return results


def pop_decode_var_cross_cond(
    data: pd.DataFrame,
    spike_rate_cols: str | list[str],
    variable_col: str,
    condition_col: str,
    neuron_col: str,
    min_trials: int | None = None,
    n_samples: int = 250,
    subsample_ratio: float = 1.0,
    n_permute: int = 10,
    show_progress: bool = True,
    return_weights: bool = False,
    n_jobs: int = -1,
) -> dict[str : pd.DataFrame]:
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
    n_samples : int, default=250
        Number of neuron and trial samples to construct.
    subsample_ratio : float, default=0.75
        Ratio of neurons to include in pseudo-population for each sample construction.
    n_permute : int, default=10
        Number of permutation tests to perform for each pseudo-population.
        If 0, no permuatation test will be performed.
    show_progress : bool, default=True
        Whether to show progress bar.
    return_weights : bool, default=False
        Whether to return neuron weights.
    n_jobs : int, default=-1
        Number of jobs to run in parallel.

    Returns
    -------
    results : dict
        Dictionary containing decoding results. Keys include:
        - 'accuracies': Dataframe of decoding accuracies.
        - 'null_accuracies': Dataframe of null distribution accuracies. Only available if permutation tests are performed.
        - 'weights': Dataframe of neuron weights. Only available if return_weights is True.
    """

    # check input
    if isinstance(spike_rate_cols, str):
        spike_rate_cols = [spike_rate_cols]
    for col in spike_rate_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data.")
    if variable_col not in data.columns:
        raise ValueError(f"Column '{variable_col}' not found in data.")
    if condition_col not in data.columns:
        raise ValueError(f"Column '{condition_col}' not found in data.")
    if neuron_col not in data.columns:
        raise ValueError(f"Column '{neuron_col}' not found in data.")

    # set n_jobs
    if n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs < 1:
        raise ValueError("n_jobs must be -1 or a positive integer.")
    n_jobs = min(n_samples, n_jobs, cpu_count())
    chunksize = int(np.ceil(n_samples / n_jobs))

    # group unique conditions by variable
    cond_grp_1, cond_grp_2 = [
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
    if n_permute > 0:
        null_accuracies = defaultdict(list)
    if return_weights:
        weights = defaultdict(list)

    # start analysis
    with mp.Pool(n_jobs) as pool:
        pbar = tqdm(total=n_samples, disable=not show_progress)
        all_results = []
        func = partial(
            _pop_decode_var_cross_cond_helper,
            data,
            spike_rate_cols,
            variable_col,
            condition_col,
            min_trials,
            subsample_ratio,
            n_permute,
            cond_grp_1,
            cond_grp_2,
            return_weights,
        )
        for results in pool.imap_unordered(func, range(n_samples), chunksize=chunksize):
            all_results.append(results)
            if show_progress:
                pbar.update()
        pbar.close()

    # gather results
    for results in all_results:
        for k, v in results["accuracies"].items():
            accuracies[k].extend(v)
        if n_permute > 0:
            for k, v in results["null_accuracies"].items():
                null_accuracies[k].extend(v)
        if return_weights:
            for k, v in results["weights"].items():
                weights[k].extend(v)

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
    results = {"accuracies": accuracies}
    if n_permute > 0:
        results["null_accuracies"] = null_accuracies
    if return_weights:
        results["weights"] = weights
    return results


def pop_decode_var_cross_temp(
    data: pd.DataFrame,
    spike_rate_cols: list[str],
    variable_col: str,
    neuron_col: str,
    min_trials: int | None = None,
    n_splits: int = 5,
    n_samples: int = 250,
    subsample_ratio: float = 1.0,
    n_permute: int = 10,
    skip_self: bool = False,
    show_progress: bool = True,
    return_weights: bool = False,
    n_jobs: int = -1,
) -> dict[str : pd.DataFrame]:
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
    n_samples : int, default=250
        Number of neuron and trial samples to construct.
    subsample_ratio : float, default=0.75
        Ratio of neurons to include in pseudo-population for each sample construction.
    n_splits : int, default=5
        Number of cross-validation splits to use.
    n_permute : int, default=10
        Number of permutation tests to perform for each pseudo-population.
        If 0, no permuatation test will be performed.
    skip_self : bool, default=True
        Whether to skip self-comparisons.
    show_progress : bool, default=True
        Whether to show progress bar.
    return_weights : bool, default=False
        Whether to return neuron weights.
    n_jobs : int, default=-1
        Number of jobs to run in parallel.

    Returns
    -------
    results : dict
        Dictionary containing decoding results. Keys include:
        - 'accuracies': Dataframe of decoding accuracies.
        - 'null_accuracies': Dataframe of null distribution accuracies. Only available if permutation tests are performed.
        - 'weights': Dataframe of neuron weights. Only available if return_weights is True.
    """

    # check input
    if len(spike_rate_cols) < 2:
        raise ValueError(
            "At least two spike rate columns are required for cross-temporal comparison."
        )
    for col in spike_rate_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data.")
    if variable_col not in data.columns:
        raise ValueError(f"Column '{variable_col}' not found in data.")
    if neuron_col not in data.columns:
        raise ValueError(f"Column '{neuron_col}' not found in data.")

    # set n_jobs
    if n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs < 1:
        raise ValueError("n_jobs must be -1 or a positive integer.")
    n_jobs = min(n_samples, n_jobs, cpu_count())
    chunksize = int(np.ceil(n_samples / n_jobs))

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
    if n_permute > 0:
        null_accuracies = defaultdict(list)
    if return_weights:
        weights = defaultdict(list)

    # start analysis
    with mp.Pool(n_jobs) as pool:
        pbar = tqdm(total=n_samples, disable=not show_progress)
        all_results = []
        func = partial(
            _pop_decode_var_cross_temp_helper,
            data,
            spike_rate_cols,
            variable_col,
            min_trials,
            subsample_ratio,
            n_splits,
            n_permute,
            skip_self,
            return_weights,
        )
        for results in pool.imap_unordered(func, range(n_samples), chunksize=chunksize):
            all_results.append(results)
            if show_progress:
                pbar.update()
        pbar.close()

    # gather results
    for results in all_results:
        for k, v in results["accuracies"].items():
            accuracies[k].extend(v)
        if n_permute > 0:
            for k, v in results["null_accuracies"].items():
                null_accuracies[k].extend(v)
        if return_weights:
            for k, v in results["weights"].items():
                weights[k].extend(v)

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
    results = {"accuracies": accuracies}
    if n_permute > 0:
        results["null_accuracies"] = null_accuracies
    if return_weights:
        results["weights"] = weights
    return results


def pop_decode_var(
    data: pd.DataFrame,
    spike_rate_cols: str | list[str],
    variable_col: str,
    neuron_col: str,
    min_trials: int | None = None,
    n_samples: int = 250,
    subsample_ratio: float = 1.0,
    n_splits: int = 5,
    n_permute: int = 10,
    show_progress: bool = True,
    return_weights: bool = False,
    n_jobs: int = -1,
) -> dict[str : pd.DataFrame]:
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
    n_samples : int, default=250
        Number of neuron and trial samples to construct.
    subsample_ratio : float, default=0.75
        Ratio of neurons to include in pseudo-population for each sample construction.
    n_splits : int, default=5
        Number of cross-validation splits to use.
    n_splits : int, default=5
        Number of cross-validation splits to use.
    n_permute : int, default=10
        Number of permutation tests to perform for each pseudo-population.
        If 0, no permuatation test will be performed.
    show_progress : bool, default=True
        Whether to show progress bar.
    return_weights : bool, default=False
        Whether to return neuron weights.
    n_jobs : int, default=-1
        Number of jobs to run in parallel.

    Returns
    -------
    results : dict
        Dictionary containing decoding results. Keys include:
        - 'accuracies': Dataframe of decoding accuracies.
        - 'null_accuracies': Dataframe of null distribution accuracies. Only available if permutation tests are performed.
        - 'weights': Dataframe of neuron weights. Only available if return_weights is True.
    """

    # check input
    if isinstance(spike_rate_cols, str):
        spike_rate_cols = [spike_rate_cols]
    for col in spike_rate_cols:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data.")
    if variable_col not in data.columns:
        raise ValueError(f"Column '{variable_col}' not found in data.")
    if neuron_col not in data.columns:
        raise ValueError(f"Column '{neuron_col}' not found in data.")

    # set n_jobs
    if n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs < 1:
        raise ValueError("n_jobs must be -1 or a positive integer.")
    n_jobs = min(n_samples, n_jobs, cpu_count())
    chunksize = int(np.ceil(n_samples / n_jobs))

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
    if n_permute > 0:
        null_accuracies = defaultdict(list)
    if return_weights:
        weights = defaultdict(list)

    # start analysis
    with mp.Pool(n_jobs) as pool:
        pbar = tqdm(total=n_samples, disable=not show_progress)
        all_results = []
        func = partial(
            _pop_decode_var_helper,
            data,
            spike_rate_cols,
            variable_col,
            min_trials,
            subsample_ratio,
            n_splits,
            n_permute,
            return_weights,
        )
        for results in pool.imap_unordered(func, range(n_samples), chunksize=chunksize):
            all_results.append(results)
            if show_progress:
                pbar.update()
        pbar.close()

    # gather results
    for results in all_results:
        for k, v in results["accuracies"].items():
            accuracies[k].extend(v)
        if n_permute > 0:
            for k, v in results["null_accuracies"].items():
                null_accuracies[k].extend(v)
        if return_weights:
            for k, v in results["weights"].items():
                weights[k].extend(v)

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
    results = {"accuracies": accuracies}
    if n_permute > 0:
        results["null_accuracies"] = null_accuracies
    if return_weights:
        results["weights"] = weights
    return results
