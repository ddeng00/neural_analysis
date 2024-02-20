from os import cpu_count
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

from ._geometry_helper import _coding_similarity_cross_cond_helper
from ..utils import group_df_by


def coding_similarity_cross_cond(
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
    Estimate variable coding vector similarity across conditions.

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
    similarities = []
    if n_permute > 0:
        null_similarities = []

    # start analysis
    with mp.Pool(n_jobs) as pool:
        pbar = tqdm(total=n_samples, disable=not show_progress)
        func = partial(
            _coding_similarity_cross_cond_helper,
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
            similarities.extend(results["similarities"])
            if n_permute > 0:
                null_similarities.extend(results["null_similarities"])
            if show_progress:
                pbar.update()
        pbar.close()

    # convert results to dataframe
    similarities = pd.DataFrame(similarities)
    if n_permute > 0:
        null_similarities = pd.DataFrame(null_similarities)

    # return results
    results = {"similarities": similarities}
    if n_permute > 0:
        results["null_similarities"] = null_similarities
    return results
