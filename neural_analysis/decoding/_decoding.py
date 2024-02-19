from collections import defaultdict
from itertools import product
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
            ("clf", LinearSVC()),
        ]
    )
    null_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LinearSVC()),
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
            ("clf", LinearSVC()),
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

        # # estimate accuracies for each cross-condition split
        # for c1, c2 in product(cond_grp_1, cond_grp_2):
        #     if skip_same_cond and c1 == c2:
        #         continue

        #     c1_ind, c2_ind = (cond == c1), (cond == c2)
        #     X_train, y_train = X[~c1_ind & ~c2_ind], y[~c1_ind & ~c2_ind]
        #     X_test, y_test = X[c1_ind | c2_ind], y[c1_ind | c2_ind]
        #     model.fit(X_train, y_train)
        #     accuracies[period].append(model.score(X_test, y_test))
        #     if return_weights:
        #         weights[period].append(
        #             pd.Series(model["clf"].coef_[0], index=X_train.columns)
        #         )

        #     # perform geometric permutation tests
        #     for _ in range(n_permute):
        #         X_c1 = X[c1_ind].sample(frac=1, axis=1)
        #         X_c2 = X[c2_ind].sample(frac=1, axis=1)
        #         X_c1.columns = X.columns
        #         X_c2.columns = X.columns
        #         X_test = pd.concat([X_c1, X_c2])
        #         null_accuracies[period].append(model.score(X_test, y_test))

        # estimate accuracies for each cross-condition split
        for c1, c2 in product(cond_grp_1, cond_grp_2):
            if same_cond_only and c1 != c2:
                continue

            # train/test split
            train_mask = (cond != c1) & (cond != c2)
            test_mask = (cond == c1) | (cond == c2)
            model.fit(X[train_mask], y[train_mask])
            accuracies.append(
                {
                    "accuracy": model.score(X[test_mask], y[test_mask]),
                    "period": period,
                    "test_cond_v1": c1,
                    "test_cond_v2": c2,
                }
            )
            if return_weights:
                curr_weights = {
                    neuron: coef for neuron, coef in zip(neurons, model["clf"].coef_[0])
                }
                curr_weights["period"] = period
                curr_weights["test_cond_v1"] = c1
                curr_weights["test_cond_v2"] = c2
                weights.append(curr_weights)

            # perform permutation tests
            for _ in range(n_permute):
                y_perm = np.random.permutation(y)
                model.fit(X[train_mask], y_perm[train_mask])
                null_accuracies.append(
                    {
                        "accuracy": model.score(X[test_mask], y_perm[test_mask]),
                        "period": period,
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


def _decode_cross_time_helper(
    data: pd.DataFrame,
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
            ("clf", LinearSVC()),
        ]
    )
    null_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LinearSVC()),
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
            ("clf", LinearSVC()),
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
