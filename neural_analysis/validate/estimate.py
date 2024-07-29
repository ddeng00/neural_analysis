from typing import Callable

import numpy as np
import pandas as pd

from ..partition import make_balanced_dichotomies
from ..preprocess import construct_pseudopopulation, remove_groups_missing_conditions
from ..utils import isin_2d


def resampled_estimate_over_dichotomies(
    data: pd.DataFrame,
    var: str,
    group: str,
    conditions: list[str],
    func: Callable,
    *,
    n_resamples: int = 24,
    n_samples_per_cond: int | None = None,
    permute: bool = False,
    random_state: int | np.random.RandomState | None = None,
    **kwargs,
):

    output = {}

    # remove groups missing conditonal trials
    if n_samples_per_cond is None:
        n_samples_per_cond = data.groupby(conditions).size().min()
    output["n_groups_total"] = data[group].nunique()
    data = remove_groups_missing_conditions(
        data, group, conditions, n_samples_per_cond=n_samples_per_cond
    )
    output["n_samples_per_cond"] = n_samples_per_cond
    output["n_groups_removed"] = output["n_groups_total"] - data[group].nunique()

    # define dichotomies
    unique_conditions = data[conditions].drop_duplicates().values
    dichotomies, dich_names, dich_diffs = make_balanced_dichotomies(
        unique_conditions, cond_names=conditions, return_one_sided=True
    )
    output["dichotomies"] = dichotomies
    output["dichotomy_names"] = dich_names
    output["dichotomy_difficulties"] = dich_diffs

    def helper(permute):
        res, res_boot = [], []
        X, conds = construct_pseudopopulation(
            data,
            var,
            group,
            conditions,
            n_samples_per_cond=n_samples_per_cond,
            random_state=random_state,
        )
        for split, name in zip(dichotomies, dich_names):
            y = isin_2d(conds, split).astype(int)
            

    pass
