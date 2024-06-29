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


def _coding_similarity_cross_cond_ref_to_gen_helper(
    data_ref: pd.DataFrame,
    data_gen: pd.DataFrame,
    spike_rate_cols: list[str],
    variable_col: str,
    condition_col: str,
    min_trials_ref: int,
    min_trials_gen: int,
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
    pop_ref = {
        neuron: df.groupby([variable_col, condition_col])
        .sample(n=min_trials_ref)
        .reset_index(drop=True)
        for neuron, df in data_ref.items()
    }
    pop_gen = {
        neuron: df.groupby([variable_col, condition_col])
        .sample(n=min_trials_gen)
        .reset_index(drop=True)
        for neuron, df in data_gen.items()
    }

    # select random subset of neurons
    neurons = list(pop_ref.keys())
    if subsample_ratio < 1:
        to_remove = np.random.choice(
            neurons,
            size=int((1 - subsample_ratio) * len(neurons)),
            replace=False,
        )
        pop_ref = {
            neuron: df for neuron, df in pop_ref.items() if neuron not in to_remove
        }
        pop_gen = {
            neuron: df for neuron, df in pop_gen.items() if neuron not in to_remove
        }
    neurons = list(pop_ref.keys())

    for period in spike_rate_cols:
        # gather base data
        X_ref, y_ref, cond_ref = [], None, None
        for _, df in pop_ref.items():
            X_ref.append(np.asarray(df[period]).reshape(-1, 1))
            if y_ref is None:
                y_ref = np.asarray(df[variable_col])
            if cond_ref is None:
                cond_ref = np.asarray(df[condition_col])
        X_ref = np.concatenate(X_ref, axis=1)
        cond_masks_ref = {
            cond_val: cond_ref == cond_val for cond_val in np.unique(cond_ref)
        }

        X_gen, y_gen, cond_gen = [], None, None
        for _, df in pop_gen.items():
            X_gen.append(np.asarray(df[period]).reshape(-1, 1))
            if y_gen is None:
                y_gen = np.asarray(df[variable_col])
            if cond_gen is None:
                cond_gen = np.asarray(df[condition_col])
        X_gen = np.concatenate(X_gen, axis=1)
        cond_masks_gen = {
            cond_val: cond_gen == cond_val for cond_val in np.unique(cond_gen)
        }
        n_neurons = X_ref.shape[1]

        # estimate cosine similarity for each pairing of cross-conditional variable coding vectors
        scaler = StandardScaler()
        sims, nulls = [], [[] for _ in range(n_permute)]
        for c1, c2 in product(cond_grp_1, cond_grp_2):
            X_fit_ref = scaler.fit_transform(X_ref)
            X_fit_gen = scaler.transform(X_gen)
            vec_ref = np.mean(X_fit_ref[cond_masks_ref[c2]], axis=0) - np.mean(
                X_fit_ref[cond_masks_ref[c1]], axis=0
            )
            vec_gen = np.mean(X_fit_gen[cond_masks_gen[c2]], axis=0) - np.mean(
                X_fit_gen[cond_masks_gen[c1]], axis=0
            )
            sims.append(1 - cosine(vec_ref, vec_gen))

            # perform geometric permutation tests
            for i in range(n_permute):
                X_perm_ref = X_ref.copy()
                X_perm_gen = X_gen.copy()
                for mask in cond_masks_ref.values():
                    r_order = np.random.permutation(n_neurons)
                    X_perm_ref[mask] = X_perm_ref[mask][:, r_order]
                for mask in cond_masks_gen.values():
                    r_order = np.random.permutation(n_neurons)
                    X_perm_gen[mask] = X_perm_gen[mask][:, r_order]
                X_perm_ref = scaler.fit_transform(X_perm_ref)
                X_perm_gen = scaler.transform(X_perm_gen)
                vec_ref = np.mean(X_perm_ref[cond_masks_ref[c2]], axis=0) - np.mean(
                    X_perm_ref[cond_masks_ref[c1]], axis=0
                )
                vec_gen = np.mean(X_perm_gen[cond_masks_gen[c2]], axis=0) - np.mean(
                    X_perm_gen[cond_masks_gen[c1]], axis=0
                )
                nulls[i].append(1 - cosine(vec_ref, vec_gen))

        # store results
        similarities.append({"period": period, "similarity": np.mean(sims)})
        null_similarities.extend(
            [{"period": period, "similarity": np.mean(null)} for null in nulls]
        )

    # return results
    results = {"similarities": similarities}
    if n_permute > 0:
        results["null_similarities"] = null_similarities
    return results
