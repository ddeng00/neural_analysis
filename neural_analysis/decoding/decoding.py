from collections import defaultdict
from functools import partial
import multiprocessing as mp
from os import cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

from ._decoding import (
    _decode_cross_cond_and_time_helper,
    _decode_cross_cond_helper,
    _decode_cross_time_helper,
    _decode_helper,
)
from ..utils import group_df_by


def decode_cross_cond_and_time(
    data: pd.DataFrame,
    spike_rate_cols: list[str],
    variable_col: str,
    condition_col: str,
    neuron_col: str,
    min_trials: int | None = None,
    n_samples: int = 250,
    subsample_ratio: float = 1.0,
    n_permute: int = 10,
    same_cond_only: bool = True,
    skip_same_time: bool = False,
    show_progress: bool = True,
    return_weights: bool = False,
    n_jobs: int = -1,
) -> dict[str : pd.DataFrame]:
    """
    Estimate cross-conditional and cross-temporal variable decodability.

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
    n_splits : int, default=5
        Number of cross-validation splits to use.
    n_permute : int, default=10
        Number of permutation tests to perform for each pseudo-population.
        If 0, no permuatation test will be performed.
    same_cond_only : bool, default=True
        Whether to allow same-condition test conditions for variables.
    skip_same_time : bool, default=False
        Whether to skip auto-comparisons.
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
        - 'accuracies': list of decoding accuracies.
        - 'null_accuracies': list of null distribution accuracies. Only available if permutation tests are performed.
        - 'weights': list of neuron weights. Only available if return_weights is True.
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
    accuracies = []
    if n_permute > 0:
        null_accuracies = []
    if return_weights:
        weights = []

    # start analysis
    with mp.Pool(n_jobs) as pool:
        pbar = tqdm(total=n_samples, disable=not show_progress)
        func = partial(
            _decode_cross_cond_and_time_helper,
            data,
            spike_rate_cols,
            variable_col,
            condition_col,
            min_trials,
            subsample_ratio,
            n_permute,
            cond_grp_1,
            cond_grp_2,
            same_cond_only,
            skip_same_time,
            return_weights,
        )
        for results in pool.imap_unordered(func, range(n_samples), chunksize=chunksize):
            accuracies.extend(results["accuracies"])
            if n_permute > 0:
                null_accuracies.extend(results["null_accuracies"])
            if return_weights:
                weights.extend(results["weights"])
            if show_progress:
                pbar.update()
        pbar.close()

    # return results
    results = {"accuracies": accuracies}
    if n_permute > 0:
        results["null_accuracies"] = null_accuracies
    if return_weights:
        results["weights"] = weights
    return results


def decode_cross_cond(
    data: pd.DataFrame,
    spike_rate_cols: str | list[str],
    variable_col: str,
    condition_col: str,
    neuron_col: str,
    min_trials: int | None = None,
    n_samples: int = 250,
    subsample_ratio: float = 1.0,
    n_permute: int = 10,
    same_cond_only: bool = False,
    show_progress: bool = True,
    return_weights: bool = False,
    n_jobs: int = -1,
) -> dict[str : pd.DataFrame]:
    """
    Estimate cross-conditional variable decodability.

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
    same_cond_only : bool, default=False
        Whether to allow same-condition test conditions for variables.
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
        - 'accuracies': list of decoding accuracies.
        - 'null_accuracies': list of null distribution accuracies. Only available if permutation tests are performed.
        - 'weights': list of neuron weights. Only available if return_weights is True.
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
    accuracies = []
    if n_permute > 0:
        null_accuracies = []
    if return_weights:
        weights = []

    # start analysis
    with mp.Pool(n_jobs) as pool:
        pbar = tqdm(total=n_samples, disable=not show_progress)
        func = partial(
            _decode_cross_cond_helper,
            data,
            spike_rate_cols,
            variable_col,
            condition_col,
            min_trials,
            subsample_ratio,
            n_permute,
            cond_grp_1,
            cond_grp_2,
            same_cond_only,
            return_weights,
        )
        for results in pool.imap_unordered(func, range(n_samples), chunksize=chunksize):
            accuracies.extend(results["accuracies"])
            if n_permute > 0:
                null_accuracies.extend(results["null_accuracies"])
            if return_weights:
                weights.extend(results["weights"])
            if show_progress:
                pbar.update()
        pbar.close()

    # return results
    results = {"accuracies": accuracies}
    if n_permute > 0:
        results["null_accuracies"] = null_accuracies
    if return_weights:
        results["weights"] = weights
    return results


def decode_cross_time(
    data: pd.DataFrame,
    spike_rate_cols: list[str],
    variable_col: str,
    neuron_col: str,
    min_trials: int | None = None,
    n_splits: int = 5,
    n_samples: int = 250,
    subsample_ratio: float = 1.0,
    n_permute: int = 10,
    skip_same_time: bool = False,
    show_progress: bool = True,
    return_weights: bool = False,
    n_jobs: int = -1,
) -> dict[str : pd.DataFrame]:
    """
    Estimate cross-temporal variable decodability.

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
    skip_same_time : bool, default=True
        Whether to skip auto-comparisons.
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
        - 'accuracies': list of decoding accuracies.
        - 'null_accuracies': list of null distribution accuracies. Only available if permutation tests are performed.
        - 'weights': list of neuron weights. Only available if return_weights is True.
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
    accuracies = []
    if n_permute > 0:
        null_accuracies = []
    if return_weights:
        weights = []

    # start analysis
    with mp.Pool(n_jobs) as pool:
        pbar = tqdm(total=n_samples, disable=not show_progress)
        func = partial(
            _decode_cross_time_helper,
            data,
            spike_rate_cols,
            variable_col,
            min_trials,
            subsample_ratio,
            n_splits,
            n_permute,
            skip_same_time,
            return_weights,
        )
        for results in pool.imap_unordered(func, range(n_samples), chunksize=chunksize):
            accuracies.extend(results["accuracies"])
            if n_permute > 0:
                null_accuracies.extend(results["null_accuracies"])
            if return_weights:
                weights.extend(results["weights"])
            if show_progress:
                pbar.update()
        pbar.close()

    # return results
    results = {"accuracies": accuracies}
    if n_permute > 0:
        results["null_accuracies"] = null_accuracies
    if return_weights:
        results["weights"] = weights
    return results


def decode(
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
    Estimate variable decodability.

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
        - 'accuracies': list of decoding accuracies.
        - 'null_accuracies': list of null distribution accuracies. Only available if permutation tests are performed.
        - 'weights': list of neuron weights. Only available if return_weights is True.
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
    accuracies = []
    if n_permute > 0:
        null_accuracies = []
    if return_weights:
        weights = []

    # start analysis
    with mp.Pool(n_jobs) as pool:
        pbar = tqdm(total=n_samples, disable=not show_progress)
        func = partial(
            _decode_helper,
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
            accuracies.extend(results["accuracies"])
            if n_permute > 0:
                null_accuracies.extend(results["null_accuracies"])
            if return_weights:
                weights.extend(results["weights"])
            if show_progress:
                pbar.update()
        pbar.close()

    # return results
    results = {"accuracies": accuracies}
    if n_permute > 0:
        results["null_accuracies"] = null_accuracies
    if return_weights:
        results["weights"] = weights
    return results
