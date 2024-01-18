from typing import overload

import numpy as np
import numpy.typing as npt
import scipy.stats as stats


def margin_of_error(
    x: npt.ArrayLike,
    confidence_level: float = 0.95,
) -> float:
    """
    Return the margin of the sample data.

    Parameters
    ----------
        x : array-like
            Sample data.
        confidence_level : float, default = 0.95
            Confidence level of the confidence interval.

    Returns
    -------
        moe: float
            Margin of error of the sample data.
    """

    t_score = stats.t.ppf((1 + confidence_level) / 2, df=len(x) - 1)
    moe = stats.sem(x) * t_score
    return moe


def confidence_interval(
    x: npt.ArrayLike,
    confidence_level: float = 0.95,
    return_mean: bool = False,
) -> tuple[float, float] | tuple[float, float, float]:
    """
    Return the confidence interval of the sample data.

    Parameters
    ----------
        x : array-like
            Sample data.
        confidence_level : float, default = 0.95
            Confidence level of the confidence interval.
        return_mean : bool, default = False
            Whether to return the mean of the sample data.

    Returns
    -------
        lower : float
            Lower bound of the confidence interval.
        upper : float
            Upper bound of the confidence interval.
        mean : float
            Mean of the sample data. Only returned if return_mean is True.
    """
    moe = margin_of_error(x, confidence_level)
    mean = np.mean(x)
    lower, upper = mean - moe, mean + moe

    if return_mean:
        return lower, upper, mean
    return lower, upper


def confusion_matrix_binary(
    predicted: npt.ArrayLike,
    actual: npt.ArrayLike,
    return_frequency: bool = False,
) -> npt.NDArray:
    """
    Compute the confusion matrix for a binary classificaiton problem (e.g., memory task).

    Parameters
    ----------
        predicted : array-like of shape (n_samples,)
            The predicted labels. Must be boolean.
        actual : array-like of shape (n_samples,)
            The actual labels. Must be boolean.
        return_frequency : bool, default=False
            Whether to return the confusion matrix as a frequency matrix.

    Returns
    -------
        confusion_matrix : ndarray of shape (2, 2)
            The confusion matrix where the cells are:

    |                 | Actually True | Actually False |
    |-----------------|---------------|-----------------|
    | Predicted True  |      TP       |      FP         |
    | Predicted False |      FN       |      TN         |
    """
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    if predicted.shape != actual.shape:
        raise ValueError("predicted and actual must have the same shape.")

    confusion_matrix = np.zeros((2, 2), dtype=int)
    confusion_matrix[0, 0] = np.sum(predicted & actual)  # TP
    confusion_matrix[0, 1] = np.sum(predicted & ~actual)  # FP
    confusion_matrix[1, 0] = np.sum(~predicted & actual)  # FN
    confusion_matrix[1, 1] = np.sum(~predicted & ~actual)  # TN
    if return_frequency:
        confusion_matrix = confusion_matrix / confusion_matrix.sum()

    return confusion_matrix


def check_interpretable_binary(
    predicted: npt.ArrayLike, actual: npt.ArrayLike
) -> float:
    """
    Check whether results from a binary classification task are safe to interpret.

    Parameters
    ----------
        predicted : array-like of shape (n_samples,)
            The predicted labels. Must be boolean.
        actual : array-like of shape (n_samples,)
            The actual labels. Must be boolean.

    Returns
    -------
        pvalue : float
    """

    conf_mat = confusion_matrix_binary(predicted, actual)
    tp, fp, fn, tn = conf_mat.flatten()
    pvalue = stats.binomtest(tp + tn, tp + tn + fp + fn).pvalue
    return pvalue


@overload
def mutual_information(
    confusion_matrix: npt.NDArray,
) -> float:
    """
    Estimate mutual information from a confusion matrix.

    Parameters
    ----------
    confusion_matrix : ndarray of shape (2, 2)
        Confusion matrix.

    Returns
    -------
    mi : float
        Estimated mutual information in bits.
    """

    confusion_matrix = np.asarray(confusion_matrix)
    if confusion_matrix.ndim != 2:
        raise ValueError("confusion_matrix must be a 2D array.")
    if confusion_matrix.shape[0] != confusion_matrix.shape[1]:
        raise ValueError("confusion_matrix must be a square array.")

    # calculate marginal probabilities
    p_x = confusion_matrix.sum(axis=1) / confusion_matrix.sum()
    p_y = confusion_matrix.sum(axis=0) / confusion_matrix.sum()
    p_xy = confusion_matrix / confusion_matrix.sum()

    # calculate mutual information
    H_x = -np.sum(p_x * np.log2(p_x + np.finfo(float).eps))
    H_y = -np.sum(p_y * np.log2(p_y + np.finfo(float).eps))
    H_xy = -np.sum(p_xy * np.log2(p_xy + np.finfo(float).eps))
    mi = H_x + H_y - H_xy

    return mi


@overload
def mutual_information(x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    """
    Estimate mutual information between two random variables.

    Parameters
    ----------
    x : array-like of shape (n_samples,)
        The first random variable.
    y : array-like of shape (n_samples,)
        The second random variable.

    Returns
    -------
    mi : float
        Estimated mutual information in bits.
    """

    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    xy = np.vstack((x, y))

    # kernel-density estimate of marginal probabilities
    p_x = stats.gaussian_kde(x)
    p_y = stats.gaussian_kde(y)
    p_xy = stats.gaussian_kde(xy)

    # calculate mutual information
    H_x = -np.sum(p_x(x) * np.log2(p_x(x) + np.finfo(float).eps))
    H_y = -np.sum(p_y(y) * np.log2(p_y(y) + np.finfo(float).eps))
    H_xy = -np.sum(p_xy(xy) * np.log2(p_xy(xy) + np.finfo(float).eps))
    mi = H_x + H_y - H_xy

    return mi