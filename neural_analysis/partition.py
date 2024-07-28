from itertools import combinations, product

import numpy as np
import numpy.typing as npt


def make_balanced_dichotomies(
    conditions: npt.ArrayLike,
    condition_names: list[str] | None = None,
    return_one_sided: bool = False,
):
    # generate dichotomies
    conditions = np.asarray(conditions)
    dichotomies = []
    cond_inds = np.arange(len(conditions))
    first_ind, rest_inds = cond_inds[0], cond_inds[1:]
    for set1 in combinations(rest_inds, len(rest_inds) // 2):
        set1 = [first_ind, *set1]
        set2 = np.setdiff1d(cond_inds, set1)
        dichotomies.append((conditions[set1], conditions[set2]))
    one_sided = [d[0] for d in dichotomies]
    # results.append(one_sided if return_one_sided else dichotomies)

    # compute number of adjacencies
    difficulties = []
    for set1, set2 in dichotomies:
        is_adjacent = [sum(c1 != c2) == 1 for c1, c2 in product(set1, set2)]
        difficulties.append(np.sum(is_adjacent))
    difficulties = np.array(difficulties)
    min_diff, max_diff = np.min(difficulties), np.max(difficulties)
    difficulties = (difficulties - min_diff) / (max_diff - min_diff)

    # name dichotomies
    if condition_names is None:
        dich_names = [f"unnamed_{i}" for i in range(len(conditions))]
    else:
        dich_names, seq_ind = [], 0
        for split, diff in zip(one_sided, difficulties):
            if diff == 1:
                dich_names.append("XOR")
            elif diff == 0:
                for i, cond in enumerate(conditions):
                    if all(split[:, i] == split[0:i]):
                        dich_names.append(cond)
                        break
            else:
                dich_names.append(f"unnamed_{seq_ind}")
                seq_ind += 1

    return one_sided if return_one_sided else dichotomies, dich_names, difficulties
