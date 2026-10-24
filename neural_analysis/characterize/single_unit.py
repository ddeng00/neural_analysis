import numpy as np
from scipy.stats import norm


def longest_run(arr):
    a = np.asarray(arr, dtype=bool)
    L = a.shape[-1]

    if L == 0:
        return np.zeros(a.shape[:-1], dtype=int)

    idx = np.arange(L)
    last_false = np.where(~a, idx, -1)
    last_false = np.maximum.accumulate(last_false, axis=-1)
    run_length = np.where(a, idx - last_false, 0)

    return run_length.max(axis=-1)


def sig_onset(pvalues, null_max_runs, alpha=0.05, cluster_alpha=0.05):
    """
    Return the onset index of the first cluster that survives
    permutation-based cluster correction.

    null_max_runs must contain one maximum run length per permutation.
    """
    significant = np.asarray(pvalues) < alpha
    null_max_runs = np.asarray(null_max_runs)

    padded = np.pad(significant.astype(np.int8), (1, 1))
    transitions = np.diff(padded)

    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)

    for start, end in zip(starts, ends):
        run_length = end - start

        permutation_p = (1 + np.count_nonzero(null_max_runs >= run_length)) / (
            null_max_runs.size + 1
        )

        if permutation_p <= cluster_alpha:
            return start

    return np.nan


def poisson_score_test(Y, X, axis=0, overdispersion=True):
    """
    Score test for a Poisson/quasi-Poisson model with an intercept
    under the null and one centered predictor.
    """
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float).reshape(-1)
    Y = np.moveaxis(Y, axis, 0)

    if Y.shape[0] != X.size:
        raise ValueError("X length must equal Y.shape[axis].")

    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values.")

    if np.any(~np.isfinite(Y)) or np.any(Y < 0):
        raise ValueError("Y must contain finite, nonnegative values.")

    X = X - X.mean()
    x_ss = np.sum(X**2)

    if x_ss == 0:
        raise ValueError("X has no variation.")

    n = Y.shape[0]
    mu = Y.mean(axis=0)
    residual = Y - mu

    score = np.tensordot(X, residual, axes=(0, 0))

    if overdispersion:
        pearson_terms = np.divide(
            residual**2,
            mu,
            out=np.zeros_like(residual),
            where=mu > 0,
        )

        # One parameter—the intercept—is estimated under the null.
        phi = pearson_terms.sum(axis=0) / max(n - 1, 1)

        # Correct for overdispersion without making the test
        # anticonservative when estimated dispersion is below one.
        phi = np.maximum(phi, 1.0)
    else:
        phi = np.ones_like(mu)

    variance = x_ss * mu * phi

    # A zero-mean count bin necessarily contains no spikes and therefore
    # provides no evidence against the null.
    Z = np.zeros_like(mu, dtype=float)
    P = np.ones_like(mu, dtype=float)

    valid = variance > 0
    Z[valid] = score[valid] / np.sqrt(variance[valid])
    P[valid] = 2 * norm.sf(np.abs(Z[valid]))

    return Z, P
