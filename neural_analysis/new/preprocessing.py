import numpy as np
import numpy.typing as npt
from sklearn.utils import resample


def quantize(x: npt.ArrayLike, n_bins: int) -> npt.NDArray:
    """
    Quantize x into n_bins of equal size.

    Parameters
    ----------
        x : array-like of shape (n_samples,)
            The data to quantize.
        n_bins : int
            The number of bins.

    Returns
    -------
        quantized : array-like of shape (n_samples,)
            The quantized data.
    """

    x = np.asarray(x)
    bins = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    quantized = np.digitize(x, bins)
    return quantized


def balanced_resample(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    upsample: bool = False,
    with_replacement: bool | None = None,
    random_state: int | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Resample the data to balance the number of samples in each class.

    Parameters
    ----------
        X : array-like of shape (n_samples, n_features)
            The data to resample.
        y : array-like of shape (n_samples,)
            The class labels. Must be binary.
        upsample : bool, default=False
            Whether to upsample the minority class or downsample the majority class.
        with_replacement : bool, default=None
            Whether to sample with replacement. If None, this is set to True if upsampling
            and False if downsampling.
        random_state : int, default=None
            The random state to use for sampling.

    Returns
    -------
        X_resampled : ndarray of shape (n_samples, n_features)
            The resampled data.
        y_resampled : ndarray of shape (n_samples,)
            The resampled class labels.
    """

    X, y = np.asarray(X), np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if with_replacement is None:
        with_replacement = upsample

    # Get the minority and majority classes
    classes, counts = np.unique(y, return_counts=True)
    minority_class, majority_class = (
        classes[np.argmin(counts)],
        classes[np.argmax(counts)],
    )
    X_minority, X_majority = X[y == minority_class], X[y == majority_class]
    y_minority, y_majority = y[y == minority_class], y[y == majority_class]

    # Resample the data
    if upsample:
        n_samples = np.max(counts)
        X_minority, y_minority = resample(
            X_minority,
            y_minority,
            n_samples=n_samples,
            replace=with_replacement,
            random_state=random_state,
        )
    else:
        n_samples = np.min(counts)
        X_majority, y_majority = resample(
            X_majority,
            y_majority,
            n_samples=n_samples,
            replace=with_replacement,
            random_state=random_state,
        )

    # Recombine the data
    X_resampled = np.vstack((X_minority, X_majority))
    y_resampled = np.hstack((y_minority, y_majority))

    return X_resampled, y_resampled
