from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment


def _absolute_distance(x: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray:
    """Return the absolute distance between x and y."""
    return np.abs(x - y)


def match_closest_pairs(
    x1: npt.ArrayLike,
    x2: npt.ArrayLike,
    distance_fn: Callable = _absolute_distance,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Matches values between two arrays using the Hungarian algorithm.

    Parameters
    ----------
        x1 : array-like of shape (n_samples,)
            Values from the first group.
        x2 : array-like of shape (m_samples,)
            Values from the second group.
        distance_fn : callable, default=_absolute_distance
            A function that takes two values and returns a distance measure.
            By default, this is the absolute distance.

    Returns
    -------
        ind1, ind2: array-likes of shape (min(n_samples, m_samples),)
            Indices of the matched values from the first and second groups.
    """
    x1, x2 = np.asarray(x1), np.asarray(x2)
    cost_mat = distance_fn(x1[:, None], x2[None, :])
    ind1, ind2 = linear_sum_assignment(cost_mat)
    return ind1, ind2
