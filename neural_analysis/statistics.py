import numpy as np
import numpy.typing as npt
from scipy.stats import sem, t, chi2


def compute_confidence_interval(
    samples: npt.ArrayLike, *, confidence: float = 0.95, axis: int | None = None
) -> np.ndarray:
    """
    Compute the confidence interval for a dataset.

    Parameters
    ----------
    samples : array-like
        An array of sample data points.
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

    samples = np.asarray(samples)
    n = samples.shape[axis] if axis is not None else samples.size
    stderr = sem(samples, axis=axis)
    interval = stderr * t.ppf((1 + confidence) / 2, n - 1)
    return interval


def likelihood_ratio_test(
    llf_full: float, llf_restr: float, df_full: int, df_restr: int
) -> float:
    """
    Compute the likelihood ratio test between two models.

    Parameters
    ----------
    llf_full : float
        The log-likelihood of the full model.
    llf_restr : float
        The log-likelihood of the restricted model.
    df_full : int
        The degrees of freedom of the full model.
    df_restr : int
        The degrees of freedom of the restricted model.

    Returns
    -------
    dict[str, float]
        A dictionary containing the test statistic and p-value.
            - statistic : float
                The likelihood ratio test statistic.
            - pvalue : float
                The p-value for the test.
            - df_constraint : int
                The degrees of freedom constraint
    """

    df = df_full - df_restr
    statistic = -2 * (llf_restr - llf_full)
    pvalue = chi2.sf(statistic, df)
    return {"statistic": statistic, "pvalue": pvalue, "df_constraint": df}
