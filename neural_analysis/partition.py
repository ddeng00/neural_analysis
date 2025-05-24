from typing import Any
from itertools import combinations, product

import numpy as np
import numpy.typing as npt

from .utils import isin_2d


def make_balanced_dichotomies(
    conditions: npt.ArrayLike,
    cond_names: npt.ArrayLike,
    return_one_sided: bool = False,
    dichot_map: dict[str, Any] | None = None,
    named_only: bool = False,
):
    # process input
    conditions = np.asarray(conditions)

    # trivial case where there is only one possible dichotomy
    if len(conditions) == 2:
        return (
            (
                [conditions[0]]
                if return_one_sided
                else [([conditions[0]], [conditions[1]])]
            ),
            cond_names,
            np.array([0.0]),
        )

    # generate dichotomies
    dichotomies = []
    if named_only:
        # Variable dichotomies
        for i, c in enumerate(conditions[0]):
            mask = conditions[:, i] == c
            dichotomies.append((conditions[mask], conditions[~mask]))

        # XOR dichotomy
        xor_mask = np.array([False] * len(conditions))
        xor_mask[0] = True
        comparison = conditions == conditions[0]
        for n in range(2, len(conditions[0]) + 1, 2):
            for inds in combinations(range(len(conditions[0])), n):
                matches = np.ones(len(conditions[0]), dtype=bool)
                matches[list(inds)] = False
                xor_mask[np.argwhere(np.all(comparison == matches, axis=1))] = True
        dichotomies.append((conditions[xor_mask], conditions[~xor_mask]))

        # custom dichotomies
        if dichot_map is not None:
            for dichot_name, dichot in dichot_map.items():
                mask = isin_2d(conditions, dichot)
                dichotomies.append((conditions[mask], conditions[~mask]))
    else:
        # all possible dichotomies
        cond_inds = np.arange(len(conditions))
        first_ind, rest_inds = cond_inds[0], cond_inds[1:]
        for set1 in combinations(rest_inds, len(rest_inds) // 2):
            set1 = [first_ind, *set1]
            set2 = np.setdiff1d(cond_inds, set1)
            dichotomies.append((conditions[set1], conditions[set2]))

    # define one-sided
    one_sided = [d[0] for d in dichotomies]

    # compute difficulty measure
    difficulties = []
    for set1, set2 in dichotomies:
        is_adjacent = [sum(c1 != c2) == 1 for c1, c2 in product(set1, set2)]
        difficulties.append(np.sum(is_adjacent))

    # name dichotomies
    dich_names = []
    min_diff, max_diff = np.min(difficulties), np.max(difficulties)
    for i, (split, diff) in enumerate(zip(one_sided, difficulties)):
        if diff == max_diff:
            dich_names.append("XOR")
        elif diff == min_diff:
            for j, cond in enumerate(cond_names):
                if all(split[:, j] == split[0, j]):
                    dich_names.append(cond)
                    break
        else:
            dich_names.append(f"unnamed_{i}")

    # assign additional dichotomies
    if dichot_map is not None:
        for dichot_name, dichot in dichot_map.items():
            for i, d in enumerate(dichotomies):
                if all(isin_2d(d, dichot)) or not any(isin_2d(d, dichot)):
                    dich_names[i] = dichot_name

    return one_sided if return_one_sided else dichotomies, dich_names, difficulties


# def make_balanced_dichotomies(
#     conditions: npt.ArrayLike,
#     cond_names: str | list[str] | None = None,
#     return_one_sided: bool = False,
# ):
#     # process input
#     if cond_names is not None and not isinstance(cond_names, list):
#         cond_names = [cond_names]

#     # generate dichotomies
#     conditions = np.asarray(conditions)
#     dichotomies = []
#     cond_inds = np.arange(len(conditions))
#     first_ind, rest_inds = cond_inds[0], cond_inds[1:]
#     for set1 in combinations(rest_inds, len(rest_inds) // 2):
#         set1 = [first_ind, *set1]
#         set2 = np.setdiff1d(cond_inds, set1)
#         dichotomies.append((conditions[set1], conditions[set2]))
#     one_sided = [d[0] for d in dichotomies]

#     # trivial case where there is only one dichotomy
#     if len(dichotomies) == 1:
#         return (
#             one_sided if return_one_sided else dichotomies,
#             cond_names,
#             np.array([0.0]),
#         )

#     # compute number of adjacencies
#     difficulties = []
#     for set1, set2 in dichotomies:
#         is_adjacent = [sum(c1 != c2) == 1 for c1, c2 in product(set1, set2)]
#         difficulties.append(np.sum(is_adjacent))

#     # name dichotomies
#     if cond_names is None:
#         dich_names = [f"unnamed_{i}" for i in range(len(conditions))]
#     else:
#         min_diff, max_diff = np.min(difficulties), np.max(difficulties)
#         dich_names, seq_ind = [], 0
#         for split, diff in zip(one_sided, difficulties):
#             if diff == max_diff:
#                 dich_names.append("XOR")
#             elif diff == min_diff:
#                 for i, cond in enumerate(cond_names):
#                     if all(split[:, i] == split[0, i]):
#                         dich_names.append(cond)
#                         break
#             else:
#                 dich_names.append(f"unnamed_{seq_ind}")
#                 seq_ind += 1

#     return one_sided if return_one_sided else dichotomies, dich_names, difficulties
