import numpy as np
import numpy.typing as npt
from sklearn.utils import resample


def bin_data(
    x: npt.ArrayLike,
    n_bins: int,
    xmin: float | None = None,
    xmax: float | None = None,
    quantize: bool = True,
) -> npt.NDArray:
    """
    Quantize x into n_bins.

    Parameters
    ----------
        x : array-like of shape (n_samples,)
            The data to quantize.
        n_bins : int
            The number of bins.
        xmin : float, default=None
            The minimum value of the bins.
            If None, this is set to the minimum value of x.
        xmax : float, default=None
            The maximum value of the bins.
            If None, this is set to the maximum value of x.
        quantize : bool, default=True
            If true, quantize x into quantiles. Otherwise, bin x into uniform bins.

    Returns
    -------
        binned : array-like of shape (n_samples,)
            The binned data.
    """

    x = np.asarray(x)
    if xmin is not None:
        x[x >= xmin] = xmin
    if xmax is not None:
        x[x >= xmax] = xmax

    if quantize:
        bins = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    else:
        bins = np.linspace(np.min(x), np.max(x), n_bins + 1)
    binned = np.digitize(x, bins)

    return binned


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
