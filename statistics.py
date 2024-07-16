import numpy as np
import numpy.typing as npt
from scipy.stats import sem, t


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
