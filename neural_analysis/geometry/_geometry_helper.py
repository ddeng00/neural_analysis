from itertools import product, permutations, combinations
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
    similarities = []
    if n_permute > 0:
        null_similarities = []

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
    neurons = list(pop.keys())

    for period in spike_rate_cols:
        X, y, cond = [], None, None
        for _, df in pop.items():
            X.append(np.asarray(df[period]).reshape(-1, 1))
            if y is None:
                y = np.asarray(df[variable_col])
            if cond is None:
                cond = np.asarray(df[condition_col])
        X = np.concatenate(X, axis=1)
        cond_masks = {cond_val: cond == cond_val for cond_val in np.unique(cond)}
        n_neurons = X.shape[1]

        # estimate cosine similarity for each pairing of cross-conditional variable coding vectors
        scaler = StandardScaler()
        sims, nulls = [], [[] for _ in range(n_permute)]
        for cond_grp_2_i in permutations(cond_grp_2):
            X_fit = scaler.fit_transform(X)
            vecs = [
                np.mean(X_fit[cond_masks[r]], axis=0)
                - np.mean(X_fit[cond_masks[l]], axis=0)
                for l, r in zip(cond_grp_1, cond_grp_2_i)
            ]
            sims.append(
                np.mean([1 - cosine(v1, v2) for v1, v2 in combinations(vecs, 2)])
            )

            # perform geometric permutation tests
            for i in range(n_permute):
                X_perm = X.copy()
                for mask in cond_masks.values():
                    r_order = np.random.permutation(n_neurons)
                    X_perm[mask] = X_perm[mask][:, r_order]
                X_perm = scaler.fit_transform(X_perm)
                vecs = [
                    np.mean(X_perm[cond_masks[r]], axis=0)
                    - np.mean(X_perm[cond_masks[l]], axis=0)
                    for l, r in zip(cond_grp_1, cond_grp_2_i)
                ]
                nulls[i].append(
                    np.mean([1 - cosine(v1, v2) for v1, v2 in combinations(vecs, 2)])
                )

        # store results
        similarities.append({"period": period, "similarity": max(sims)})
        null_similarities.extend(
            [{"period": period, "similarity": max(null)} for null in nulls]
        )

    # return results
    results = {"similarities": similarities}
    if n_permute > 0:
        results["null_similarities"] = null_similarities
    return results
