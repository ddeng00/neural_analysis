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
