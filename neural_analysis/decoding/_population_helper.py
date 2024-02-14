from collections import defaultdict
from itertools import product, permutations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import cosine


def _pop_vec_sim_var_cross_cond_helper(
    data,
    spike_rate_cols,
    variable_col,
    condition_col,
    min_trials,
    subsample_ratio,
    n_permute,
    vc1,
    vc2,
    *args,
):

    # initialize variables
    similarities = defaultdict(list)
    null_similarities = defaultdict(list)

    # generate random pseudo-population
    pseudo = {
        neuron: df.groupby([variable_col, condition_col])
        .sample(n=min_trials)
        .reset_index(drop=True)
        for neuron, df in data.items()
    }

    # select random subset of neurons
    neurons = list(pseudo.keys())
    if subsample_ratio < 1:
        to_remove = np.random.choice(
            neurons,
            size=int((1 - subsample_ratio) * len(neurons)),
            replace=False,
        )
        pseudo = {
            neuron: df for neuron, df in pseudo.items() if neuron not in to_remove
        }

    for period in spike_rate_cols:
        # gather base data
        X, y, cond = {}, None, None
        for neuron, df in pseudo.items():
            X[neuron] = np.asarray(df[period])
            if y is None:
                y = df[variable_col]
            if cond is None:
                cond = df[condition_col]
        X = pd.DataFrame(X)
        X.loc[:, :] = StandardScaler().fit_transform(X)

        # estimate cosine similarity for each pair of cross-conditional variable coding vectors
        left, right = permutations(vc1, 2), permutations(vc2, 2)
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

    return similarities, null_similarities


def _pop_decode_var_cross_cond_helper(
    data,
    spike_rate_cols,
    variable_col,
    condition_col,
    min_trials,
    subsample_ratio,
    n_permute,
    vc1,
    vc2,
    *args,
):

    # initialize variables
    accuracies = defaultdict(list)
    null_accuracies = defaultdict(list)

    # define decoding model
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LinearSVC()),
        ]
    )

    # generate random pseudo-population
    pseudo = {
        neuron: df.groupby([variable_col, condition_col])
        .sample(n=min_trials)
        .reset_index(drop=True)
        for neuron, df in data.items()
    }

    # select random subset of neurons
    neurons = list(pseudo.keys())
    if subsample_ratio < 1:
        to_remove = np.random.choice(
            neurons,
            size=int((1 - subsample_ratio) * len(neurons)),
            replace=False,
        )
        pseudo = {
            neuron: df for neuron, df in pseudo.items() if neuron not in to_remove
        }

    for period in spike_rate_cols:
        # gather base data
        X, y, cond = {}, None, None
        for neuron, df in pseudo.items():
            X[neuron] = np.asarray(df[period])
            if y is None:
                y = df[variable_col]
            if cond is None:
                cond = df[condition_col]
        X = pd.DataFrame(X)

        # estimate accuracies for each cross-condition split
        for c1, c2 in product(vc1, vc2):
            c1_ind, c2_ind = (cond == c1), (cond == c2)
            X_train, y_train = X[~c1_ind & ~c2_ind], y[~c1_ind & ~c2_ind]
            X_test, y_test = X[c1_ind | c2_ind], y[c1_ind | c2_ind]
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            accuracies[period].append(accuracy)

            # perform geometric permutation tests
            for _ in range(n_permute):
                X_c1 = X[c1_ind].sample(frac=1, axis=1)
                X_c2 = X[c2_ind].sample(frac=1, axis=1)
                X_c1.columns = X.columns
                X_c2.columns = X.columns
                X_test = pd.concat([X_c1, X_c2])
                null_accuracy = model.score(X_test, y_test)
                null_accuracies[period].append(null_accuracy)

    return accuracies, null_accuracies


def _pop_decode_var_cross_temp_helper(
    data,
    spike_rate_cols,
    variable_col,
    min_trials,
    subsample_ratio,
    n_permute,
    skip_self,
    *args,
):

    # initialize variables
    accuracies = defaultdict(list)
    null_accuracies = defaultdict(list)

    # define decoding model
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LinearSVC()),
        ]
    )
    cv = StratifiedKFold()

    # generate random pseudo-population
    pseudo = {
        neuron: df.groupby(variable_col).sample(n=min_trials).reset_index(drop=True)
        for neuron, df in data.items()
    }

    # select random subset of neurons
    neurons = list(pseudo.keys())
    if subsample_ratio < 1:
        to_remove = np.random.choice(
            neurons,
            size=int((1 - subsample_ratio) * len(neurons)),
            replace=False,
        )
        pseudo = {
            neuron: df for neuron, df in pseudo.items() if neuron not in to_remove
        }

    # estimate accuracies across each pair of spike windows
    for window1 in spike_rate_cols:
        # gather base data
        X_base, y = {}, None
        for neuron, df in pseudo.items():
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
            X_other = {}
            for neuron, df in pseudo.items():
                X_other[neuron] = np.asarray(df[window2])
            X_other = pd.DataFrame(X_other)

            # cross-temporal cross-validate
            for train_idx, test_idx in cv.split(X_base, y):
                X_train = X_base.iloc[train_idx]
                X_test = X_other.iloc[test_idx]
                model.fit(X_train, y.iloc[train_idx])
                accuracy = model.score(X_test, y.iloc[test_idx])
                accuracies[name].append(accuracy)

                # perform permutation tests
                for _ in range(n_permute):
                    y_perm = np.random.permutation(y)
                    for train_idx, test_idx in cv.split(X_base, y_perm):
                        X_train = X_base.iloc[train_idx]
                        X_test = X_other.iloc[test_idx]
                        model.fit(X_train, y_perm.iloc[train_idx])
                        null_accuracy = model.score(X_test, y_perm.iloc[test_idx])
                        null_accuracies[name].append(null_accuracy)

    return accuracies, null_accuracies
