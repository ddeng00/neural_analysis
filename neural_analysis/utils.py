from itertools import chain, combinations, groupby
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
import re

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, optimal_leaf_ordering


def remove_patsy_subterms(terms: list[str]) -> list[str]:
    """
    Remove subterms from a list of Patsy terms.

    Parameters
    ----------
    terms : list of str
        List of Patsy terms to be processed.

    Returns
    -------
    list of str
        List of Patsy terms with subterms removed.
    """

    rem_terms, final_terms = list(terms), []
    while len(rem_terms) > 0:
        term = rem_terms[-1]
        final_terms.append(term)
        subterms = [":".join(subterm) for subterm in powerset(term.split(":"))]
        rem_terms = [t for t in rem_terms if t not in subterms]
    return final_terms[::-1]


def sanitize_patsy_terms(terms: list[str]) -> list[str]:
    """
    Sanitize Patsy terms by removing the 'C()' prefix and any additional information.

    Parameters
    ----------
    terms : list of str
        List of Patsy terms to be sanitized.

    Returns
    -------
    list of str
        Sanitized Patsy terms.
    """

    terms = [t.split("[")[0] for t in terms]
    return [
        t.split(",")[0].strip().replace("C(", "").replace("I(", "").replace(")", "")
        for t in terms
    ]


def anscombe_transform(x: npt.ArrayLike) -> np.ndarray:
    """
    Apply the Anscombe transform to a set of values.

    Parameters
    ----------
    x : numpy.ndarray
        Array of values to be transformed.

    Returns
    -------
    numpy.ndarray
        Transformed values.
    """
    return 2 * np.sqrt(np.asarray(x) + 3 / 8)


def seriate(
    mat: np.ndarray | pd.DataFrame,
    order_only: bool = False,
) -> np.ndarray | pd.DataFrame:
    """
    Seriate a square symmetric matrix using Optimal Leaf Ordering.

    This will accept either:
      - a distance matrix (non-negative, zero diagonal), or
      - a similarity matrix (any real values, symmetry enforced).

    If you pass a similarity matrix, it will be converted to a
    distance by: D = max(S) - S, with zeros on the diagonal.

    Parameters
    ----------
    mat : numpy.ndarray or pandas.DataFrame
        A square symmetric matrix (distance or similarity).
    order_only : bool, default False
        If True, return only the leaf ordering (list of indices).

    Returns
    -------
    order : List[int], if order_only=True
        The seriation order.
    reordered_mat : numpy.ndarray or pandas.DataFrame
        The matrix reordered according to the optimal leaf order.
        If input was a DataFrame, indices and columns are preserved.
    """
    # Pull out labels if it's a DataFrame
    if isinstance(mat, pd.DataFrame):
        labels = (mat.index.tolist(), mat.columns.tolist())
        arr = mat.values
    else:
        labels = None
        arr = np.asarray(mat)

    # Basic sanity checks
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Input must be a square matrix.")
    if not np.allclose(arr, arr.T, atol=1e-8):
        raise ValueError("Input matrix must be symmetric.")

    # Determine if this is already a distance matrix
    is_distance = np.all(arr >= 0) and np.allclose(np.diag(arr), 0, atol=1e-8)

    if is_distance:
        dist = arr.copy()
    else:
        # Treat as similarity: convert to distance
        max_val = np.nanmax(arr)
        dist = max_val - arr
        np.fill_diagonal(dist, 0.0)

    # Condense and cluster
    condensed = squareform(dist)
    Z = linkage(condensed, method="average")
    Z = optimal_leaf_ordering(Z, condensed)
    order = dendrogram(Z, no_plot=True)["leaves"]

    if order_only:
        return order

    # Reorder the matrix
    reordered = arr[np.ix_(order, order)]

    # If DataFrame, reattach labels
    if labels is not None:
        row_labels, col_labels = labels
        new_idx = [row_labels[i] for i in order]
        new_col = [col_labels[j] for j in order]
        return pd.DataFrame(reordered, index=new_idx, columns=new_col)

    return reordered


def powerset(iterable: Iterable) -> Iterable:
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def min_adjacency_shuffle(
    arr: list, adj_rate: float = 0.0, disc_rate: float = 1.0
) -> np.ndarray:

    # set up random number generator
    rng = np.random.default_rng()

    # define discount function
    discount_fn = np.vectorize(
        lambda x: (1 - np.exp(-disc_rate * x) + adj_rate) / (1 + adj_rate)
    )

    # generate pseudorandom sequence
    remaining = Counter(arr)
    recency = {k: np.inf for k in remaining.keys()}
    new_arr = []
    for _ in range(len(arr)):

        # calculate probabilities
        probs = np.array([p * discount_fn(recency[k]) for k, p in remaining.items()])
        if all(probs == 0):
            return min_adjacency_shuffle(arr, adj_rate, disc_rate)

        # select next item
        sel = rng.choice(list(remaining.keys()), p=probs / np.sum(probs))

        # update remaining item counts and track recency
        remaining[sel] -= 1
        for k in recency.keys():
            if k == sel:
                recency[k] = 0
            else:
                recency[k] += 1
        new_arr.append(sel)

    return np.asarray(new_arr)


def max_diversity_subset(
    arr: npt.ArrayLike, n: int, metric: str = "hamming"
) -> np.ndarray:

    # Check dimensions
    arr = np.asarray(arr)
    if n > arr.shape[0]:
        raise ValueError("n must be less than or equal to the number of items")

    # Compute distance matrix
    dist_matrix = squareform(pdist(arr, metric=metric))

    # Start with the item that has the highest total distance (i.e., most "dissimilar" overall).
    total_distances = np.sum(dist_matrix, axis=1)
    first = np.argmax(total_distances)
    selected = [first]

    # Create a set of candidate indices (those not yet selected).
    candidates = set(range(arr.shape[0]))
    candidates.remove(first)

    # Iteratively add the candidate that maximizes the total distance to already selected items.
    while len(selected) < n:
        best_candidate = None
        best_increase = -np.inf

        for candidate in candidates:
            # Calculate the sum of distances from candidate to each already selected item.
            increase = sum(dist_matrix[candidate][j] for j in selected)
            if increase > best_increase:
                best_increase = increase
                best_candidate = candidate

        selected.append(best_candidate)
        candidates.remove(best_candidate)

    return np.take(arr, selected, axis=0)


def repetitiveness(arr: npt.ArrayLike) -> float:
    occuraces = defaultdict(list)
    for i, ele in enumerate(arr):
        occuraces[ele].append(i)
    distances = []
    for ele in occuraces.values():
        if len(ele) > 1:
            distances.extend([ele[i + 1] - ele[i] for i in range(len(ele) - 1)])
    if distances:
        return sum(distances) / len(distances)
    return 0


def max_rep_length(arr: npt.ArrayLike) -> int:
    return max(len(list(group)) for _, group in groupby(arr))


def remove_subsets(lst: list[list]) -> list[list]:
    lst = sorted(lst, key=len)
    result = []
    for i, x in enumerate(lst):
        if not any(set(x).issubset(set(lst[j])) for j in range(i + 1, len(lst))):
            result.append(x)
    return result


def remove_if_exists(path: Path | str) -> None:
    """
    Remove a file or directory if it exists.

    Parameters
    ----------
    path : `pathlib.Path` or str
        File or directory to be removed.
    """

    path = Path(path)
    if path.exists():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()


def validate_file(path: Path | str) -> Path:
    """
    Validate a file path and return a `pathlib.Path` object.

    Parameters
    ----------
    path : `pathlib.Path` or str
        File path to be validated.

    Returns
    -------
    path : `pathlib.Path`
        Validated file path.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File '{path}' does not exist.")
    if not path.is_file():
        raise FileNotFoundError(f"'{path}' is not a file.")
    return path


def validate_dir(path: Path | str) -> Path:
    """
    Validate a directory path and return a `pathlib.Path` object.

    Parameters
    ----------
    path : `pathlib.Path` or str
        Directory path to be validated.

    Returns
    -------
    path : `pathlib.Path`
        Validated directory path.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Directory '{path}' does not exist.")
    if not path.is_dir():
        raise FileNotFoundError(f"'{path}' is not a directory.")
    return path


def natsort_path_key(path: Path | str) -> list[object]:
    """
    Return key for natural sort order.

    Parameters
    ----------
    path : `pathlib.Path` or str
        Path to be sorted by natural sort order.

    Returns
    -------
    list of object
        Key for natural sort order.
    """

    path = Path(path)
    ns = re.compile("([0-9]+)")
    return [int(s) if s.isdigit() else s.lower() for s in ns.split(path.name)]


def sanitize_filename(filename: str) -> str:
    """Return a sanitized version of a filename."""
    return "".join(c for c in filename if (c.isalnum() or c in "._- "))


def read_mat(path: Path | str) -> dict[str, list]:
    """
    Read a `.mat` file into a Python dictionary.

    Metadata from the original `.mat` file is dropped.

    Parameters
    ----------
    path : `pathlib.Path` or str
        Path to `.mat` file to be read.

    Returns
    -------
    dict of {str : list}
        Dictionary containing relevant data from the specified `.mat` file.
    """

    mat = loadmat(str(path), simplify_cells=True)
    return {k: v for k, v in mat.items() if not k.startswith("__")}


def create_rolling_window(
    start: float,
    stop: float,
    step: float,
    width: float,
    exclude_oob: bool = False,
    alignment: str = "center",
):
    """
    Generate rolling windows based on the given start and end times.

    Parameters
    ----------
    start : float
        Start time.
    stop : float
        End time.
    step : float
        Step size between consecutive windows.
    width : float
        Size of each rolling window.
    exclude_oob : bool, default=False
        If True, adjust windows to stay within the bounds of the trials.
    alignment : {"center", "left", "right"}, default="center"
        Defines how the window is positioned relative to each step:
        - "center": window is centered at the step.
        - "left": window starts at the step.
        - "right": window ends at the step.

    Returns
    -------
    tuple of `numpy.ndarray`
        A tuple containing:
        - Array of start times for each rolling window.
        - Array of end times for each rolling window.
        - Array of labeled times for each rolling window.
    """

    if alignment not in {"center", "left", "right"}:
        raise ValueError("alignment must be one of {'center', 'left', 'right'}")

    pos = start
    starts, ends, labels = [], [], []

    stop += 1e-6  # To ensure the last window is included

    while pos <= stop:
        if alignment == "center":
            w_start = pos - width / 2
            w_end = pos + width / 2
            ref = pos
        elif alignment == "left":
            w_start = pos
            w_end = pos + width
            ref = w_start
        elif alignment == "right":
            w_start = pos - width
            w_end = pos
            ref = w_end

        if exclude_oob:
            w_start = max(w_start, start)
            w_end = min(w_end, stop)

        starts.append(w_start)
        ends.append(w_end)
        labels.append(ref)

        pos += step

    return np.array(starts), np.array(ends), np.array(labels)


def isin_2d(x1, x2):
    """
    Check if any of the rows in a 2D array `x1` are present in a 2D array `x2`.

    Parameters:
    -----------
    x1 : numpy.ndarray
        A 2D numpy array where each row is a vector to check for presence in `x2`.
    x2 : numpy.ndarray
        A 2D numpy array where each row is a vector to check against.

    Returns:
    --------
    bool
        True if any row in `x1` is present in `x2`, False otherwise.

    Example:
    --------
    >>> import numpy as np
    >>> x1 = np.array([[1, 2, 3], [4, 5, 6]])
    >>> x2 = np.array([[7, 8, 9], [1, 2, 3]])
    >>> isin_2d(x1, x2)
    True
    """

    return (x1[:, None] == x2).all(-1).any(-1)


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
    ret_val = "ns"
    for i, level in enumerate(levels):
        if pvalue <= level:
            ret_val = "*" * (i + 1)
    return ret_val


def polar_median_split(x: npt.ArrayLike) -> tuple[np.array, float]:
    """
    Split the input array into two groups based on the median angle after
    minimizing the circular range.

    Parameters
    ----------
    x : npt.ArrayLike
        Input array of angles in radians.

    Returns
    -------
    groups: numpy.ndarray
        A boolean array indicating the group assignment.
    median: float
        The median angle of the split.
    """

    # Process input
    x = np.asarray(x)

    # Select offset minimizing polar range
    best_offset, min_range = None, np.inf
    for off in np.linspace(0, 2 * np.pi, 360):
        shifted = (x - off) % (2 * np.pi)
        r = shifted.max() - shifted.min()
        if r < min_range:
            min_range = r
            best_offset = off

    # Rotate by best offset
    shifted = (x - best_offset) % (2 * np.pi)
    med = np.median(shifted)
    return shifted >= med, med
