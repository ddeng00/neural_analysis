import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from tqdm import trange

from .utils import group_df_by


def decode_condition(
    df: pd.DataFrame,
    firing_rate: str,
    condition: str,
    unit: str = "cluster_id",
    n_psuedo_pops: int = 250,
    subunit_ratio: float = 0.75,
    n_permutes: int = 1000,
    n_splits: int = 5,
    n_repeats: int = 10,
    min_units: int = 10,
    show_progress: bool = True,
    n_jobs: int = -1,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Population decoding of variable of interest.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing neural firing rate and variable of interest.
    firing_rate : str
        Column name of neural firing rate.
    condition : str
        Column name of condition variable of interest.
    unit : str, default="cluster_id"
        Column name of unit identifier, by default "cluster_id".
    n_psuedo_pops : int, default=250
        Number of psuedo-populations to generate, by default 250.
    subunit_ratio : float, default=0.75
        Ratio of units to include in psuedo-population, by default 0.75.
    n_permutes : int, default=1000
        Number of permutations to perform, by default 1000.
    n_splits : int, default=5
        Number of splits for cross-validation, by default 5.
    n_repeats : int, default=10
        Number of repeats for cross-validation, by default 10.
    min_units : int, default=10
        Number of units minimum to decode. If less than min_units, return None.
    show_progress : bool, default=True
        Show progress bar, by default True.
    n_jobs : int, default=-1
        Number of jobs to run in parallel, by default -1.

    Returns
    -------
    acc : array-like
        Array of accuracies for each psuedo-population.
    acc_permute : array-like
        Array of permuted accuracies for each psuedo-population.
    """

    # check if enough units
    n_units = df[unit].nunique()
    if n_units < min_units:
        return None, None

    # get features for each condition
    df = group_df_by(df, by=condition)
    df = {
        c: [vv[firing_rate].rename(u) for u, vv in group_df_by(v, by=unit).items()]
        for c, v in df.items()
    }

    # create pipeline
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    model = Pipeline(
        [
            ("sampler", RandomUnderSampler()),
            ("scaler", StandardScaler()),
            ("clf", LinearSVC()),
        ]
    )

    # estimate decoding accuracy
    acc, acc_permute = [], []
    reps = trange(n_repeats) if show_progress else range(n_repeats)
    for _ in reps:
        # generate pseuo-population
        X, y = [], []
        for c, v in df.items():
            XX = pd.concat(
                [vv.sample(frac=1, ignore_index=True) for vv in v], axis=1
            ).dropna()
            yy = np.full(len(XX), c)
            X.append(XX)
            y.append(yy)
        X, y = pd.concat(X), np.concatenate(y)

        # select random subset of units
        tot_units = len(X.columns)
        selected = np.random.choice(
            tot_units, int(tot_units * subunit_ratio), replace=False
        )
        X = X.iloc[:, selected]

        # cross-validate
        acc_pp = cross_val_score(model, X, y, cv=cv, n_jobs=n_jobs)
        acc.extend(acc_pp)

        # permutation test
        for _ in range(n_permutes):
            acc_permute.extend(
                cross_val_score(
                    model,
                    X,
                    np.random.permutation(y),
                    cv=cv,
                    n_jobs=n_jobs,
                )
            )

        # show stats
        if show_progress:
            reps.set_postfix({"n_selected": len(selected), "acc": acc_pp.mean()})

    acc, acc_permute = np.asarray(acc), np.asarray(acc_permute)
    return acc, acc_permute
