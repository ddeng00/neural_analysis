from typing import Callable

import numpy as np
import pandas as pd


def perform_anova(
    data: pd.DataFrame,
    target: str,
    factors: list[str],
    *,
    target_transform: Callable | None = np.sqrt,
    include_interactions: bool = True,
) -> pd.Series:
    """
    Perform a n-way ANOVA on the data. The data should be in long format.

    Parameters
    ----------
    data : pd.DataFrame
        The data in long format.
    target : str
        The name of the target variable.
    factors : list[str]
        The names of the factors.
    target_transform : Callable or None, default=`numpy.sqrt`
        The transformation to apply to the target variable.
    include_interactions : bool, default=True
        Whether to include interactions, by default True.

    Returns
    -------
    pd.Series
        The p-values of the factors. If applicable, interaction terms are delimited by `:`.
    """

    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    data = data[[target] + factors].copy()
    if target_transform is not None:
        data[target] = data[target].apply(target_transform)
    data[factors] = data[factors].astype("category")
    if include_interactions:
        formula = f"{target} ~ {' * '.join(factors)}"
    else:
        formula = f"{target} ~ {' + '.join(factors)}"
    model = ols(formula, data).fit()
    anova_table = anova_lm(model, typ=3 if include_interactions else 2)
    return anova_table["PR(>F)"].iloc[1:-1]


def estimate_latency():
    pass
