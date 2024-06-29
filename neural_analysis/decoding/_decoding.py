from collections import defaultdict
from itertools import product, combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def _decode_cross_cond_and_time_helper(
    data: pd.DataFrame,
    spike_rate_cols: list[str],
    variable_col: str,
    condition_col: str,
    min_trials: int,
    subsample_ratio: float,
    n_permute: int,
    cond_grp_1: list[Any],
    cond_grp_2: list[Any],
    same_cond_only: bool,
    skip_same_time: bool,
    return_weights: bool,
    *args,
) -> dict[str:dict]:

    # initialize variables
    accuracies = []
    if n_permute > 0:
        null_accuracies = []
    if return_weights:
        weights = []

    # define decoding model
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(dual="auto")),
        ]
    )
    null_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(dual="auto")),
        ]
    )

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

    # preprocessing
    df = next(iter(pop.values()))
    y, cond = df[variable_col], df[condition_col]

    for c1, c2 in product(cond_grp_1, cond_grp_2):
        if same_cond_only and c1 != c2:
            continue

        # train/test split
        train_mask = (cond != c1) & (cond != c2)
        test_mask = (cond == c1) | (cond == c2)

        # estimate accuracies across spike time periods
        for train_period in spike_rate_cols:
            # gather base data
            X_base = {
                neuron: np.asarray(df[train_period]) for neuron, df in pop.items()
            }
            X_base = pd.DataFrame(X_base)
            model.fit(X_base[train_mask], y[train_mask])
            if return_weights:
                curr_weights = {
                    neuron: coef for neuron, coef in zip(neurons, model["clf"].coef_[0])
                }
                curr_weights["train_period"] = train_period
                curr_weights["test_cond_v1"] = c1
                curr_weights["test_cond_v2"] = c2
                weights.append(curr_weights)

            for test_period in spike_rate_cols:
                if skip_same_time and train_period == test_period:
                    continue

                # gather comparison data
                X_other = {
                    neuron: np.asarray(df[test_period]) for neuron, df in pop.items()
                }
                X_other = pd.DataFrame(X_other)
                accuracies.append(
                    {
                        "accuracy": model.score(X_other[test_mask], y[test_mask]),
                        "train_period": train_period,
                        "test_period": test_period,
                        "test_cond_v1": c1,
                        "test_cond_v2": c2,
                    }
                )

                # perform permutation tests
                for _ in range(n_permute):
                    y_perm = np.random.permutation(y)
                    null_model.fit(X_base[train_mask], y_perm[train_mask])
                    null_accuracies.append(
                        {
                            "accuracy": null_model.score(
                                X_other[test_mask], y_perm[test_mask]
                            ),
                            "train_period": train_period,
                            "test_period": test_period,
                            "test_cond_v1": c1,
                            "test_cond_v2": c2,
                        }
                    )

    # return results
    results = {"accuracies": accuracies}
    if n_permute > 0:
        results["null_accuracies"] = null_accuracies
    if return_weights:
        results["weights"] = weights
    return results


def _decode_cross_cond_helper(
    data: pd.DataFrame,
    spike_rate_cols: list[str],
    variable_col: str,
    condition_col: str,
    min_trials: int,
    subsample_ratio: float,
    n_permute: int,
    cond_grp_1: list[Any],
    cond_grp_2: list[Any],
    same_cond_only: bool,
    return_weights: bool,
    *args,
) -> dict[str:dict]:

    # initialize variables
    accuracies = []
    if n_permute > 0:
        null_accuracies = []
    if return_weights:
        weights = []

    # define decoding model
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(dual="auto")),
        ]
    )

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
        # gather base data
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

        # estimate accuracies for each cross-condition split
        # for c1, c2 in product(cond_grp_1, cond_grp_2):
        #     if same_cond_only and c1 != c2:
        #         continue
        for c11, c12 in combinations(cond_grp_1, 2):
            for c21, c22 in combinations(cond_grp_2, 2):


            # train/test split
            # train_mask = ~cond_masks[c1] & ~cond_masks[c2]
            # test_mask = cond_masks[c1] | cond_masks[c2]
                train_mask = ~cond_masks[c11] & ~cond_masks[c12] & ~cond_masks[c21] & ~cond_masks[c22]
                test_mask = cond_masks[c11] | cond_masks[c12] | cond_masks[c21] | cond_masks[c22]
                model.fit(X[train_mask], y[train_mask])
                accuracies.append(
                    {
                        "accuracy": model.score(X[test_mask], y[test_mask]),
                        "period": period,
                        # "test_cond_v1": c1,
                        # "test_cond_v2": c2,
                    }
                )
                if return_weights:
                    curr_weights = {
                        neuron: coef for neuron, coef in zip(neurons, model["clf"].coef_[0])
                    }
                    curr_weights["period"] = period
                    # curr_weights["test_cond_v1"] = c1
                    # curr_weights["test_cond_v2"] = c2
                    weights.append(curr_weights)

                # perform permutation tests
                for _ in range(n_permute):

                    # # geometric
                    # X_perm = X.copy()
                    # for mask in cond_masks.values():
                    #     r_order = np.random.permutation(n_neurons)
                    #     X_perm[mask] = X_perm[mask][:, r_order]
                    # model.fit(X_perm[train_mask], y[train_mask])
                    # null_accuracies.append(
                    #     {
                    #         "accuracy": model.score(X_perm[test_mask], y[test_mask]),
                    #         "period": period,
                    #         "test_cond_v1": c1,
                    #         "test_cond_v2": c2,
                    #     }
                    # )

                    # normal
                    y_perm = np.random.permutation(y)
                    model.fit(X[train_mask], y_perm[train_mask])
                    null_accuracies.append(
                        {
                            "accuracy": model.score(X[test_mask], y_perm[test_mask]),
                            "period": period,
                            # "test_cond_v1": c1,
                            # "test_cond_v2": c2,
                        }
                    )

    # return results
    results = {"accuracies": accuracies}
    if n_permute > 0:
        results["null_accuracies"] = null_accuracies
    if return_weights:
        results["weights"] = weights
    return results


def _decode_cross_cond_ref_to_gen_helper(
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
    same_cond_only: bool,
    return_weights: bool,
    *args,
) -> dict[str:dict]:

    # initialize variables
    accuracies = []
    if n_permute > 0:
        null_accuracies = []
    if return_weights:
        weights = []

    # define decoding model
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(dual="auto")),
        ]
    )

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

        # estimate accuracies for each cross-condition split
        for c11, c12 in combinations(cond_grp_1, 2):
            for c21, c22 in combinations(cond_grp_2, 2):
        # for c1, c2 in product(cond_grp_1, cond_grp_2):
        #     if same_cond_only and c1 != c2:
        #         continue

            # train/test split
            # train_mask = ~cond_masks_ref[c1] & ~cond_masks_ref[c2]
            # test_mask = cond_masks_gen[c1] | cond_masks_gen[c2]
                train_mask = ~cond_masks_ref[c11] & ~cond_masks_ref[c12] & ~cond_masks_ref[c21] & ~cond_masks_ref[c22]
                test_mask = cond_masks_gen[c11] | cond_masks_gen[c12] | cond_masks_gen[c21] | cond_masks_gen[c22]
                model.fit(X_ref[train_mask], y_ref[train_mask])
                accuracies.append(
                    {
                        "accuracy": model.score(X_gen[test_mask], y_gen[test_mask]),
                        "period": period,
                        # "test_cond_v1": c1,
                        # "test_cond_v2": c2,
                    }
                )
                if return_weights:
                    curr_weights = {
                        neuron: coef for neuron, coef in zip(neurons, model["clf"].coef_[0])
                    }
                    curr_weights["period"] = period
                    # curr_weights["test_cond_v1"] = c1
                    # curr_weights["test_cond_v2"] = c2
                    weights.append(curr_weights)

                # perform permutation tests
                for _ in range(n_permute):
                    model.fit(X_ref[train_mask], np.random.permutation(y_ref[train_mask]))
                    null_accuracies.append(
                        {
                            "accuracy": model.score(
                                X_gen[test_mask], np.random.permutation(y_gen[test_mask])
                            ),
                            "period": period,
                            # "test_cond_v1": c1,
                            # "test_cond_v2": c2,
                        }
                    )

    # return results
    results = {"accuracies": accuracies}
    if n_permute > 0:
        results["null_accuracies"] = null_accuracies
    if return_weights:
        results["weights"] = weights
    return results


def _decode_cross_time_helper(
    data: dict[str : pd.DataFrame],
    spike_rate_cols: list[str],
    variable_col: str,
    min_trials: int,
    subsample_ratio: float,
    n_splits: int,
    n_permute: int,
    skip_same_time: bool,
    return_weights: bool,
    *args,
) -> dict[str:dict]:

    # initialize variables
    accuracies = []
    if n_permute > 0:
        null_accuracies = []
    if return_weights:
        weights = []

    # define decoding model
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(dual="auto")),
        ]
    )
    null_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(dual="auto")),
        ]
    )
    cv = StratifiedKFold(n_splits=n_splits)

    # generate random pseudo-population
    pop = {
        neuron: df.groupby(variable_col).sample(n=min_trials).reset_index(drop=True)
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

    # estimate accuracies across each pair of spike time period
    for train_period in spike_rate_cols:
        # gather base data
        X_base, y = {}, None
        for neuron, df in pop.items():
            X_base[neuron] = np.asarray(df[train_period])
            if y is None:
                y = df[variable_col]
        X_base = pd.DataFrame(X_base)

        # cross-temporal cross-validation
        for train_idx, test_idx in cv.split(X_base, y):
            model.fit(X_base.iloc[train_idx], y.iloc[train_idx])
            if return_weights:
                curr_weights = {
                    neuron: coef for neuron, coef in zip(neurons, model["clf"].coef_[0])
                }
                curr_weights["train_period"] = train_period
                weights.append(curr_weights)

            # estimate generalization to every other spike window
            for test_period in spike_rate_cols:
                if skip_same_time and train_period == test_period:
                    continue

                # gather comparison data
                X_other = {
                    neuron: np.asarray(df[test_period]) for neuron, df in pop.items()
                }
                X_other = pd.DataFrame(X_other)
                accuracies.append(
                    {
                        "accuracy": model.score(
                            X_other.iloc[test_idx], y.iloc[test_idx]
                        ),
                        "train_period": train_period,
                        "test_period": test_period,
                    }
                )

                # perform permutation tests
                for _ in range(n_permute):
                    y_perm = np.random.permutation(y)
                    null_model.fit(X_base.iloc[train_idx], y_perm[train_idx])
                    null_accuracies.append(
                        {
                            "accuracy": null_model.score(
                                X_other.iloc[test_idx], y_perm[test_idx]
                            ),
                            "train_period": train_period,
                            "test_period": test_period,
                        }
                    )

    # return results
    results = {"accuracies": accuracies}
    if n_permute > 0:
        results["null_accuracies"] = null_accuracies
    if return_weights:
        results["weights"] = weights
    return results


def _decode_helper(
    data: pd.DataFrame,
    spike_rate_cols: list[str],
    variable_col: str,
    min_trials: int,
    subsample_ratio: float,
    n_splits: int,
    n_permute: int,
    return_weights: bool,
    *args,
) -> dict[str:dict]:

    # initialize variables
    accuracies = []
    if n_permute > 0:
        null_accuracies = []
    if return_weights:
        weights = []

    # define decoding model
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(dual="auto")),
        ]
    )
    cv = StratifiedKFold(n_splits=n_splits)

    # generate random pseudo-population
    pop = {
        neuron: df.groupby(variable_col).sample(n=min_trials).reset_index(drop=True)
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

    # estimate accuracies for each spike time period
    for period in spike_rate_cols:
        # gather data
        X, y = {}, None
        for neuron, df in pop.items():
            X[neuron] = np.asarray(df[period])
            if y is None:
                y = df[variable_col]
        X = pd.DataFrame(X)

        # cross-validate
        cv_results = cross_validate(
            model, X, y, cv=cv, n_jobs=1, return_estimator=return_weights
        )
        accuracies.extend(
            [{"accuracy": acc, "period": period} for acc in cv_results["test_score"]]
        )
        if return_weights:
            cv_weights = [
                {neuron: coef for neuron, coef in zip(neurons, pip["clf"].coef_[0])}
                for pip in cv_results["estimator"]
            ]
            for w in cv_weights:
                w["period"] = period
            weights.extend(cv_weights)

        # perform permutation tests
        for _ in range(n_permute):
            null_scores = cross_val_score(
                model,
                X,
                np.random.permutation(y),
                cv=cv,
                n_jobs=1,
            )
            null_accuracies.extend(
                [{"accuracy": acc, "period": period} for acc in null_scores]
            )

    # return results
    results = {"accuracies": accuracies}
    if n_permute > 0:
        results["null_accuracies"] = null_accuracies
    if return_weights:
        results["weights"] = weights
    return results


def _decode_ref_to_gen_helper(
    data_ref: pd.DataFrame,
    data_gen: pd.DataFrame,
    spike_rate_cols: list[str],
    variable_col: str,
    min_trials_ref: int,
    min_trials_gen: int,
    subsample_ratio: float,
    n_permute: int,
    return_weights: bool,
    *args,
) -> dict[str:dict]:

    # initialize variables
    accuracies = []
    if n_permute > 0:
        null_accuracies = []
    if return_weights:
        weights = []

    # define decoding model
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(dual="auto")),
        ]
    )

    # generate random pseudo-population
    pop_ref = {
        neuron: df.groupby(variable_col).sample(n=min_trials_ref).reset_index(drop=True)
        for neuron, df in data_ref.items()
    }
    pop_gen = {
        neuron: df.groupby(variable_col).sample(n=min_trials_gen).reset_index(drop=True)
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

    # estimate accuracies for each spike time period
    for period in spike_rate_cols:
        # gather data
        X_ref, y_ref = {}, None
        for neuron, df in pop_ref.items():
            X_ref[neuron] = np.asarray(df[period])
            if y_ref is None:
                y_ref = df[variable_col]
        X_ref = pd.DataFrame(X_ref)

        X_gen, y_gen = {}, None
        for neuron, df in pop_gen.items():
            X_gen[neuron] = np.asarray(df[period])
            if y_gen is None:
                y_gen = df[variable_col]
        X_gen = pd.DataFrame(X_gen)

        # fit model on reference data
        model.fit(X_ref, y_ref)

        # validate on generalize data
        accuracies.append(
            {
                "accuracy": model.score(X_gen, y_gen),
                "period": period,
            }
        )
        if return_weights:
            w = {neuron: coef for neuron, coef in zip(neurons, model["clf"].coef_[0])}
            w["period"] = period
            weights.append(w)

        # perform permutation tests
        for _ in range(n_permute):
            model.fit(X_ref, np.random.permutation(y_ref))
            null_accuracies.append(
                {
                    "accuracy": model.score(X_gen, np.random.permutation(y_gen)),
                    "period": period,
                }
            )

    # return results
    results = {"accuracies": accuracies}
    if n_permute > 0:
        results["null_accuracies"] = null_accuracies
    if return_weights:
        results["weights"] = weights
    return results
