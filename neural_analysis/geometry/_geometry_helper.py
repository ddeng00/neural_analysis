from collections import defaultdict
from itertools import product, permutations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine


def _coding_similarity_cross_cond_helper(
    data: pd.DataFrame,
    spike_rate_cols: list[str],
    variable_col: str,
    condition_col: str,
    min_trials: int,
    subsample_ratio: float,
    n_permute: int,
    cond_grp_1: list[Any],
    cond_grp_2: list[Any],
    *args,
) -> dict[str:dict]:

    # initialize variables
    similarities = defaultdict(list)
    if n_permute > 0:
        null_similarities = defaultdict(list)

    # generate random pseudo-population
    pop = {
        neuron: df.groupby([variable_col, condition_col])
        .sample(n=min_trials)
        .reset_index(drop=True)
        for neuron, df in data.items()
    }

    # select random subset of neurons
    neurons = list(pop.keys())
    if subsample_ratio < 1:
        to_remove = np.random.choice(
            neurons,
            size=int((1 - subsample_ratio) * len(neurons)),
            replace=False,
        )
        pop = {neuron: df for neuron, df in pop.items() if neuron not in to_remove}

    for period in spike_rate_cols:
        # gather base data
        X, y, cond = {}, None, None
        for neuron, df in pop.items():
            X[neuron] = np.asarray(df[period])
            if y is None:
                y = df[variable_col]
            if cond is None:
                cond = df[condition_col]
        X = pd.DataFrame(X)
        X.loc[:, :] = StandardScaler().fit_transform(X)

        # estimate cosine similarity for each pair of cross-conditional variable coding vectors
        left, right = permutations(cond_grp_1, 2), permutations(cond_grp_2, 2)
        for (l1, l2), (r1, r2) in product(left, right):
            X_l1, X_l2 = X[cond == l1].mean(), X[cond == l2].mean()
            X_r1, X_r2 = X[cond == r1].mean(), X[cond == r2].mean()
            v1 = X_r1.to_numpy() - X_l1.to_numpy()
            v2 = X_r2.to_numpy() - X_l2.to_numpy()
            similarities[period].append(1 - cosine(v1, v2))

            # perform geometric permutation tests
            for _ in range(n_permute):
                v1 = X_r1.sample(frac=1).to_numpy() - X_l1.sample(frac=1).to_numpy()
                v2 = X_r2.sample(frac=1).to_numpy() - X_l2.sample(frac=1).to_numpy()
                null_similarities[period].append(1 - cosine(v1, v2))

    # return results
    results = {"similarities": similarities}
    if n_permute > 0:
        results["null_similarities"] = null_similarities
    return results
