from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from scipy.stats import binomtest
import statsmodels.api as sm


def _absolute_distance(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray:
    """Return the absolute distance between x and y."""
    return np.abs(x - y)


def fit_gamma(
    feature: npt.ArrayLike, return_se: bool = False
) -> tuple[float, float] | tuple[float, float, float, float]:
    """
    Fits a gamma distribution to feature.

    Parameters
    ----------
        feature : array-like of shape (n_samples,)
            The feature-of-interest to fit.
        return_se : bool, default=False
            Whether to return standard errors of the parameters.

    Returns
    -------
        k : float
            The shape parameter of the gamma distribution.
        theta : float
            The scale parameter of the gamma distribution.
        k_se : float
            The standard error of the shape parameter. Only returned if return_se is True.
        theta_se : float
            The standard error of the scale parameter. Only returned if return_se is True.
    """
    feature = np.asarray(feature)
    if np.any(feature <= 0.0):
        raise ValueError("feature must be positive.")

    gamma = sm.GLM(feature, np.ones(feature.shape), family=sm.families.Gamma())
    gamma_res = gamma.fit()
    if return_se:
        return (
            gamma_res.params[0],
            gamma_res.params[1],
            gamma_res.bse[0],
            gamma_res.bse[1],
        )
    else:
        return gamma_res.params[0], gamma_res.params[1]


def match_features(
    feature1: npt.ArrayLike,
    feature2: npt.ArrayLike,
    dist_fn: Callable = _absolute_distance,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Matches features from two groups using the Hungarian algorithm.

    Parameters
    ----------
        feature1 : array-like of shape (n_samples,)
            Features from the first group.
        feature2 : array-like of shape (m_samples,)
            Features from the second group.
        dist_fn : callable, default=_absolute_distance
            A function that takes two arrays and returns a distance matrix.

    Returns
    -------
        ind1 : array-like of shape (min(n_samples, m_samples),)
            The indices of the matched features from the first group.
        ind2 : array-like of shape (min(n_samples, m_samples),)
            The indices of the matched features from the second group.
    """
    feature1, feature2 = np.asarray(feature1), np.asarray(feature2)
    cost_mat = dist_fn(feature1[:, None], feature2[None, :])
    ind1, ind2 = linear_sum_assignment(cost_mat)
    return ind1, ind2


def memory_performance(
    assert_seen: npt.ArrayLike, has_seen: npt.ArrayLike
) -> tuple[int, int, int, int]:
    """
    Compute memory performance metrics.

    Parameters
    ----------
        assert_seen : array-like of shape (n_samples, )
            Whether the subject asserted to have seen the stimulus. Must be boolean.
        has_seen : array-like of shape (n_samples, )
            Whether the subject has seen the stimulus. Must be boolean.

    Returns
    -------
        tp : int
            The number of true positives.
        tn : int
            The number of true negatives.
        fp : int
            The number of false positives.
        fn : int
            The number of false negatives.
    """
    assert_seen = np.asarray(assert_seen, dtype=bool)
    has_seen = np.asarray(has_seen, dtype=bool)
    if assert_seen.shape != has_seen.shape:
        raise ValueError("assert_seen and has_seen must have the same shape.")

    tp = np.sum(assert_seen & has_seen)
    tn = np.sum(~assert_seen & ~has_seen)
    fp = np.sum(assert_seen & ~has_seen)
    fn = np.sum(~assert_seen & has_seen)

    return tp, tn, fp, fn


def test_memory(assert_seen: npt.ArrayLike, has_seen: npt.ArrayLike) -> float:
    """Test by binomial test if the subject remembers the stimulus.

    Parameters
    ----------
        assert_seen : array-like of shape (n_samples,)
            Whether the subject asserted to have seen the stimulus. Must be boolean.
        has_seen : array-like of shape (n_samples,)
            Whether the subject has seen the stimulus. Must be boolean.

    Returns
    -------
        pvalue : float
            The p-value of the binomial test.
    """
    tp, tn, fp, fn = memory_performance(assert_seen, has_seen)
    pvalue = binomtest(tp + tn, tp + tn + fp + fn).pvalue
    return pvalue


def bin_feature(feature: npt.ArrayLike, n_bins: int) -> npt.NDArray:
    """
    Split features into n_bins.

    Parameters
    ----------
        feature : array-like of shape (n_samples,)
            The feature to bin.
        n_bins : int
            The number of bins.

    Returns
    -------
        bin_ind : array-like of shape (n_samples,)
            The bin indices.
    """
    feature = np.asarray(feature)
    bins = np.quantile(feature, np.linspace(0, 1, n_bins + 1))
    bin_ind = np.digitize(feature, bins)
    return bin_ind
