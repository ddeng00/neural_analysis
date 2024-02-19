from collections import defaultdict
from itertools import product, permutations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import cosine


def _pop_vec_sim_var_cross_cond_helper(
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


def _pop_decode_var_cross_cond_helper(
    data: pd.DataFrame,
    spike_rate_cols: list[str],
    variable_col: str,
    condition_col: str,
    min_trials: int,
    subsample_ratio: float,
    n_permute: int,
    cond_grp_1: list[Any],
    cond_grp_2: list[Any],
    return_weights: bool,
    *args,
) -> dict[str:dict]:

    # initialize variables
    accuracies = defaultdict(list)
    if n_permute > 0:
        null_accuracies = defaultdict(list)
    if return_weights:
        weights = defaultdict(list)

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

        # estimate accuracies for each cross-condition split
        for c1, c2 in product(cond_grp_1, cond_grp_2):
            c1_ind, c2_ind = (cond == c1), (cond == c2)
            X_train, y_train = X[~c1_ind & ~c2_ind], y[~c1_ind & ~c2_ind]
            X_test, y_test = X[c1_ind | c2_ind], y[c1_ind | c2_ind]
            model.fit(X_train, y_train)
            accuracies[period].append(model.score(X_test, y_test))
            if return_weights:
                weights[period].append(
                    pd.Series(model["clf"].coef_[0], index=X_train.columns)
                )

            # perform geometric permutation tests
            for _ in range(n_permute):
                X_c1 = X[c1_ind].sample(frac=1, axis=1)
                X_c2 = X[c2_ind].sample(frac=1, axis=1)
                X_c1.columns = X.columns
                X_c2.columns = X.columns
                X_test = pd.concat([X_c1, X_c2])
                null_accuracies[period].append(model.score(X_test, y_test))

    # return results
    results = {"accuracies": accuracies}
    if n_permute > 0:
        results["null_accuracies"] = null_accuracies
    if return_weights:
        results["weights"] = weights
    return results


def _pop_decode_var_cross_temp_helper(
    data: pd.DataFrame,
    spike_rate_cols: list[str],
    variable_col: str,
    min_trials: int,
    subsample_ratio: float,
    n_splits: int,
    n_permute: int,
    skip_self: bool,
    return_weights: bool,
    *args,
) -> dict[str:dict]:

    # initialize variables
    accuracies = defaultdict(list)
    if n_permute > 0:
        null_accuracies = defaultdict(list)
    if return_weights:
        weights = defaultdict(list)

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

    # estimate accuracies across each pair of spike windows
    for window1 in spike_rate_cols:
        # gather base data
        X_base, y = {}, None
        for neuron, df in pop.items():
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
            X_other = {neuron: np.asarray(df[window2]) for neuron, df in pop.items()}
            X_other = pd.DataFrame(X_other)

            # cross-temporal cross-validate
            for train_idx, test_idx in cv.split(X_base, y):
                X_train = X_base.iloc[train_idx]
                X_test = X_other.iloc[test_idx]
                model.fit(X_train, y.iloc[train_idx])
                accuracy = model.score(X_test, y.iloc[test_idx])
                accuracies[name].append(accuracy)
                if return_weights:
                    weights[name].append(
                        pd.Series(model["clf"].coef_[0], index=X_train.columns)
                    )

            # perform permutation tests
            for _ in range(n_permute):
                y_perm = np.random.permutation(y)
                for train_idx, test_idx in cv.split(X_base, y_perm):
                    X_train = X_base.iloc[train_idx]
                    X_test = X_other.iloc[test_idx]
                    model.fit(X_train, y_perm[train_idx])
                    null_accuracy = model.score(X_test, y_perm[test_idx])
                    null_accuracies[name].append(null_accuracy)

    # return results
    results = {"accuracies": accuracies}
    if n_permute > 0:
        results["null_accuracies"] = null_accuracies
    if return_weights:
        results["weights"] = weights
    return results


def _pop_decode_var_helper(
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
    accuracies = defaultdict(list)
    if n_permute > 0:
        null_accuracies = defaultdict(list)
    if return_weights:
        weights = defaultdict(list)

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

    # estimate accuracies for each spike window
    for window in spike_rate_cols:
        # gather data
        X, y = {}, None
        for neuron, df in pop.items():
            X[neuron] = np.asarray(df[window])
            if y is None:
                y = df[variable_col]
        X = pd.DataFrame(X)

        # cross-validate
        cv_results = cross_validate(
            model, X, y, cv=cv, n_jobs=1, return_estimator=return_weights
        )
        accuracies[window].extend(cv_results["test_score"])
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
                n_jobs=1,
            )
            null_accuracies[window].extend(null_scores)

    # return results
    results = {"accuracies": accuracies}
    if n_permute > 0:
        results["null_accuracies"] = null_accuracies
    if return_weights:
        results["weights"] = weights
    return results
