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

    # shortcut if named only
    if named_only:
        one_sided = [
            conditions[conditions[:, j] == conditions[0, j]]
            for j in range(conditions.shape[1])
        ]
        dich_names = list(cond_names)

        # Preserve dichot_map overrides / additional named dichotomies.
        if dichot_map is not None:
            for dichot_name, dichot in dichot_map.items():
                mask = isin_2d(conditions, dichot)
                if not mask[0]:
                    mask = ~mask

                split = conditions[mask]
                for i, named_split in enumerate(one_sided):
                    if np.array_equal(named_split, split):
                        dich_names[i] = dichot_name
                        break
                else:
                    one_sided.append(split)
                    dich_names.append(dichot_name)

        dichotomies = [
            (split, conditions[~isin_2d(conditions, split)])
            for split in one_sided
        ]
        difficulties = [
            np.sum([
                np.sum(c1 != c2) == 1
                for c1, c2 in product(set1, set2)
            ])
            for set1, set2 in dichotomies
        ]

        return (
            one_sided if return_one_sided else dichotomies,
            dich_names,
            difficulties,
        )

    # generate dichotomies
    dichotomies = []
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
    xor_inds = []
    min_diff, max_diff = np.min(difficulties), np.max(difficulties)
    for i, (split, diff) in enumerate(zip(one_sided, difficulties)):
        variable = next(
            (
                cond
                for j, cond in enumerate(cond_names)
                if all(split[:, j] == split[0, j])
            ),
            None,
        )
        if variable is not None:
            dich_names.append(variable)
        elif diff == max_diff:
            xor_inds.append(i)
            dich_names.append("XOR")
        else:
            dich_names.append(f"unnamed_{i}")

    if len(xor_inds) > 1:
        for n, i in enumerate(xor_inds, 1):
            dich_names[i] = f"XOR_{n}"

    # assign additional dichotomies
    if dichot_map is not None:
        for dichot_name, dichot in dichot_map.items():
            for i, d in enumerate(one_sided):
                if all(isin_2d(d, dichot)) or not any(isin_2d(d, dichot)):
                    dich_names[i] = dichot_name
    return one_sided if return_one_sided else dichotomies, dich_names, difficulties
