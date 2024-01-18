import numpy as np
import numpy.typing as npt
import statsmodels.api as sm


def fit_gamma(
    x: npt.ArrayLike, return_se: bool = False
) -> tuple[float, float] | tuple[float, float, float, float]:
    """
    Fits a gamma distribution to x.

    Parameters
    ----------
        x : array-like of shape (n_samples,)
            The data to fit.
        return_se : bool, default=False
            Whether to return standard errors of the parameters.

    Returns
    -------
        k : float
            The shape parameter of the gamma distribution.
        theta : float
            The scale parameter of the gamma distribution.
        k_se : float
            The standard error of the shape parameter. Only returned if return_se is True.
        theta_se : float
            The standard error of the scale parameter. Only returned if return_se is True.
    """
    x = np.asarray(x)
    if np.any(x <= 0.0):
        raise ValueError("x must be positive.")

    gamma = sm.GLM(x, np.ones(x.shape), family=sm.families.Gamma())
    gamma_res = gamma.fit()
    if return_se:
        return (
            gamma_res.params[0],
            gamma_res.params[1],
            gamma_res.bse[0],
            gamma_res.bse[1],
        )
    else:
        return gamma_res.params[0], gamma_res.params[1]
