from itertools import combinations, product

import numpy as np
import numpy.typing as npt


def make_balanced_dichotomies(
    conditions: npt.ArrayLike,
    compute_difficulties: bool = False,
    return_one_sided: bool = False,
):
    conditions = np.asarray(conditions)
    dichotomies = []
    cond_inds = np.arange(len(conditions))
    first_ind, rest_inds = cond_inds[0], cond_inds[1:]
    for set1 in combinations(rest_inds, len(rest_inds) // 2):
        set1 = [first_ind, *set1]
        set2 = np.setdiff1d(cond_inds, set1)
        dichotomies.append((conditions[set1], conditions[set2]))
    output = [d[0] for d in dichotomies] if return_one_sided else dichotomies

    if compute_difficulties:
        difficulties = []
        for set1, set2 in dichotomies:
            is_adjacent = [sum(c1 != c2) == 1 for c1, c2 in product(set1, set2)]
            difficulties.append(np.sum(is_adjacent))
        difficulties = np.array(difficulties)
        min_diff, max_diff = np.min(difficulties), np.max(difficulties)
        difficulties = (difficulties - min_diff) / (max_diff - min_diff)
        return output, difficulties
    return output
