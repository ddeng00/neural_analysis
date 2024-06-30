import numpy as np
import numpy.typing as npt
from scipy.stats import sem, t, binom


def compute_confidence_interval(
    data: npt.ArrayLike, *, confidence: float = 0.95, axis: int | None = None
) -> np.ndarray:
    """
    Compute the confidence interval for a dataset.

    Parameters
    ----------
    data : array-like
        An array of data points.
    confidence : float, default=0.95
        The confidence level for the interval. Default is 0.95 (95% confidence).
    axis : int or None, default=None
        The axis along which to compute the confidence interval. If None, the interval is computed over the flattened
        array.

    Returns
    -------
    interval : numpy.ndarray
        An array containing the confidence interval for the data.
    """

    data = np.asarray(data)
    n = data.shape[axis] if axis is not None else data.size
    stderr = sem(data, axis=axis)
    interval = stderr * t.ppf((1 + confidence) / 2, n - 1)
    return interval


def binomial_test(k: int, n: int, p: float, alternative: str = "greater") -> float:
    """
    Perform a binomial test for the probability of success in a Bernoulli trial.

    Parameters
    ----------
    k : int
        The number of successes in the trial.
    n : int
        The number of trials.
    p : float
        The hypothesized probability of success.
    alternative : {'two-sided', 'greater', 'less'}, default='greater'
        The alternative hypothesis to test.

    Returns
    -------
    pvalue : float
        The p-value of the test.
    """

    if alternative == "two-sided":
        pvalue = 2 * min(binom.cdf(k, n, p), 1 - binom.cdf(k - 1, n, p))
    elif alternative == "greater":
        pvalue = 1 - binom.cdf(k - 1, n, p)
    elif alternative == "less":
        pvalue = binom.cdf(k, n, p)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    return pvalue


def pvalue_to_decimal(pvalue: float, levels: list[float] = [0.05, 0.01, 0.001]) -> str:
    """
    Convert p-value to a string representation using asterisks.

    Parameters
    ----------
    pvalue : float
        The p-value to convert.
    levels : list of float, default=[0.05, 0.01, 0.001]
        The significance levels for conversion. The number of asterisks corresponds to the number of levels the p-value crosses.
        For example, if levels are [0.05, 0.01, 0.001] and p-value is 0.007, it would return '**' because 0.007 < 0.05 and 0.007 < 0.01.

    Returns
    -------
    str
        A string of asterisks representing the significance level of the p-value.
    """
    levels = sorted(levels, reverse=True)
    for i, level in enumerate(levels):
        if pvalue <= level:
            return "*" * (i + 1)
    return "ns"  # 'ns' stands for not significant
