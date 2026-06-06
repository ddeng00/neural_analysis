import numpy as np

def _longest_run(arr):
    "Find the length of the longest run of True values along the last axis of `arr`."

    a = np.asarray(arr, dtype=bool)
    L = a.shape[-1]
    if L == 0:
        return np.zeros(a.shape[:-1], dtype=int)
    idx = np.arange(L)
    last_false = np.where(~a, idx, -1)
    last_false = np.maximum.accumulate(last_false, axis=-1)
    run_len = np.where(a, idx - last_false, 0)
    return run_len.max(axis=-1)


def _sig_onset(pvalues, min_samples, alpha=0.1):
    "Find the first index of a run of significant p-values of at least `min_samples` length."

    sig = pvalues < alpha
    i = 0
    while i < len(sig):
        if sig[i]:
            j = i
            while j < len(sig) and sig[j]:
                j += 1
            if j - i >= min_samples:
                return i
            i = j
        else:
            i += 1
    return np.nan