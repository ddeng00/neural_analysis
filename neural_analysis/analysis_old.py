# from collections.abc import Iterable

# import numpy as np
# from numpy.typing import ArrayLike
# from statsmodels.stats.rates import test_poisson
# from tqdm.contrib.concurrent import process_map
# from statsmodels.discrete.discrete_model import PoissonResults
# from statsmodels.discrete.count_model import (
#     Poisson,
#     ZeroInflatedPoisson,
#     NegativeBinomialP,
#     ZeroInflatedNegativeBinomialP,
#     ZeroInflatedPoissonResults,
#     CountResults,
# )
# from statsmodels.tools.tools import add_constant
# from scipy.stats import norm, chi2

# from .utils_old import spike_counts, get_spikes


# def poisson_overdispersion_test(results: PoissonResults, alpha: float = 0.05) -> bool:
#     """
#     Significance test for overdispersion in Poisson GLM.

#     Parameters
#     ----------
#     results : PoissonResults
#         Results of Poisson GLM.
#     alpha : float
#         Desired significance level.

#     Returns
#     -------
#     bool
#         True if overdispersion is significant, False otherwise.

#     Reference
#     ---------
#     [1] Lee, S., Park, C., & Kim, B.S. (1995). Tests for detecting overdispersion in poisson models. Communications in Statistics-theory and Methods, 24, 2405-2420.
#     [2] Blasco-Moreno, A., Pérez-Casany, M., Puig, P., Morante, M.D., & Castells, E. (2019). What does a zero mean? Understanding false, random and structural zeros in ecology. Methods in Ecology and Evolution, 10, 949 - 959. doi: 10.1111/2041-210X.13175
#     """

#     # get variables
#     endog = results.model.endog
#     pred = results.predict()

#     # compute test statistic
#     f1 = np.sum((endog - pred) ** 2 / pred)
#     f2 = np.sqrt(2 * np.sum(pred**2))
#     test_statistic = f1 / f2

#     # compute p-value
#     pvalue = norm.sf(test_statistic)
#     return pvalue <= alpha


# def zip_overdispersion_test(
#     results: ZeroInflatedPoissonResults, alpha: float = 0.05
# ) -> bool:
#     """
#     Significance test for overdispersion in zero-inflated Poisson GLM.

#     Parameters
#     ----------
#     results : ZeroInflatedPoissonResults
#         Results of zero-inflated Poisson GLM.
#     alpha : float
#         Desired significance level.

#     Returns
#     -------
#     bool
#         True if overdispersion is significant, False otherwise.

#     References
#     ----------
#     [1] Ridout, M.S., Hinde, J.P., & Demétrio, C.G. (2001). A Score Test for Testing a Zero-Inflated Poisson Regression Model Against Zero-Inflated Negative Binomial Alternatives. Biometrics, 57. doi: 10.1111/j.0006-341X.2001.00800.x
#     [2] Blasco-Moreno, A., Pérez-Casany, M., Puig, P., Morante, M.D., & Castells, E. (2019). What does a zero mean? Understanding false, random and structural zeros in ecology. Methods in Ecology and Evolution, 10, 949 - 959. doi: 10.1111/2041-210X.13175
#     """

#     # get variables
#     exog, endog = results.model.exog, results.model.endog
#     infl_ratio = np.exp(results.params["inflate_const"])
#     infl_ratio = infl_ratio / (1 + infl_ratio)
#     pred = np.exp(exog @ results.params.drop("inflate_const"))
#     i0 = (endog == 0).astype(int)
#     p0 = infl_ratio + (1 - infl_ratio) * np.exp(-pred)

#     # compute test statistic
#     f1 = (endog - pred) ** 2 - endog
#     f2 = -i0 * pred**2 * infl_ratio / p0
#     test_statistic = 0.5 * np.sum(f1 + f2)

#     # compute p-value
#     pvalue = norm.sf(test_statistic)
#     return pvalue <= alpha


# def poisson_zero_inflation_test(results: PoissonResults, alpha: float = 0.05) -> bool:
#     """
#     Significance test for zero-inflation in Poisson GLM.

#     Parameters
#     ----------
#     results : PoissonResults
#         Results of Poisson GLM.
#     alpha : float
#         Desired significance level.

#     Returns
#     -------
#     bool
#         True if zero-inflation is significant, False otherwise.

#     References
#     ---------
#     [1] van den Broek, J.H. (1995). A score test for zero inflation in a Poisson distribution. Biometrics, 51 2, 738-43 . doi: 10.2307/2532940. PMID: 7786998.
#     [2] Blasco-Moreno, A., Pérez-Casany, M., Puig, P., Morante, M.D., & Castells, E. (2019). What does a zero mean? Understanding false, random and structural zeros in ecology. Methods in Ecology and Evolution, 10, 949 - 959. doi: 10.1111/2041-210X.13175
#     """

#     # get variables
#     exog, endog = results.model.exog, results.model.endog
#     pred = results.predict()
#     i0 = (endog == 0).astype(int)
#     cov = np.asarray(exog)

#     # compute test statistic
#     e = np.exp(pred)
#     f1 = np.sum(i0 * e - 1) ** 2
#     f2 = np.sum(e - 1)
#     f3 = (
#         -np.reshape(pred, (1, -1))
#         @ cov
#         @ np.linalg.inv(cov.T @ np.diag(pred) @ cov)
#         @ cov.T
#         @ np.reshape(pred, (-1, 1))
#     ).item()
#     test_statistic = f1 / (f2 + f3)

#     # compute p-value
#     pvalue = chi2.sf(test_statistic, 1)
#     return pvalue <= alpha


# def fit_best_glm(
#     endog: ArrayLike,
#     exog: ArrayLike,
#     exposure: ArrayLike = None,
#     fit_intercept: bool = True,
#     check_overdispersion: bool = False,
#     check_zero_inflation: bool = False,
#     alpha: float = 0.05,
# ) -> CountResults:
#     """
#     Fit the best discrete GLM for the given count data, acounting for overdispersion and zero-inflation.

#     Parameters
#     ----------
#     endog : array_like
#         Endogenous variable.
#     exog : array_like
#         Exogenous variable.
#     exposure : array_like
#         Exposure variable.
#     fit_intercept : bool
#         Whether to fit an intercept.
#     check_overdispersion : bool
#         Whether to check for overdispersion.
#     check_zero_inflation : bool
#         Whether to check for zero-inflation.
#     alpha : float
#         Desired significance level for overdispersion and zero-inflation tests.

#     Returns
#     -------
#     results : "statsmodels.discrete.count_model.CountResults"
#         Results of the best GLM.
#     """
#     if fit_intercept:
#         exog = add_constant(exog)
#     ele_cnt = exog.shape[1]

#     results = Poisson(endog, exog, exposure=exposure).fit(
#         method="bfgs", maxiter=5000, disp=0
#     )
#     overdispersed, zero_inflated = False, False
#     if check_overdispersion:
#         overdispersed = poisson_overdispersion_test(results, alpha=alpha)
#     if check_zero_inflation:
#         zero_inflated = poisson_zero_inflation_test(results, alpha=alpha)

#     if overdispersed and zero_inflated:
#         results = ZeroInflatedPoisson(endog, exog, exposure=exposure).fit(
#             method="bfgs", maxiter=5000, disp=0
#         )
#         if zip_overdispersion_test(results, alpha=alpha):
#             bounds = [(None, None)] * (ele_cnt + 1) + [(0, None)]
#             results = ZeroInflatedNegativeBinomialP(endog, exog, exposure=exposure).fit(
#                 method="lbfgs", maxiter=5000, disp=0, bounds=bounds
#             )
#     elif overdispersed:
#         bounds = [(None, None)] * ele_cnt + [(0, None)]
#         results = NegativeBinomialP(endog, exog, exposure=exposure).fit(
#             method="lbfgs", maxiter=5000, disp=0, bounds=bounds
#         )
#     elif zero_inflated:
#         results = ZeroInflatedPoisson(endog, exog, exposure=exposure).fit(
#             method="bfgs", maxiter=5000, disp=0
#         )
#     return results


# def _permutation_test_helper(results: CountResults) -> float:
#     """
#     Helper function for permutation test parallel computation.

#     Parameters
#     ----------
#     results : CountResults
#         Results of discrete GLM.

#     Returns
#     -------
#     permute_pvalue : float
#         The p-value associated with the permutation test.
#     """
#     exog, endog = results.model.exog, results.model.endog.copy()
#     exposure = results.model.exposure
#     ele_cnt = exog.shape[1]
#     np.random.shuffle(endog)
#     cls_name = results.model.__class__.__name__
#     try:
#         if cls_name == "Poisson":
#             permute_results = Poisson(endog, exog, exposure=exposure).fit(
#                 method="bfgs", maxiter=5000, disp=0
#             )
#         elif cls_name == "ZeroInflatedPoisson":
#             permute_results = ZeroInflatedPoisson(endog, exog, exposure=exposure).fit(
#                 method="bfgs", maxiter=5000, disp=0
#             )
#         elif cls_name == "NegativeBinomialP":
#             bounds = [(None, None)] * ele_cnt + [(0, None)]
#             permute_results = NegativeBinomialP(endog, exog, exposure=exposure).fit(
#                 method="lbfgs", maxiter=5000, disp=0, bounds=bounds
#             )
#         elif cls_name == "ZeroInflatedNegativeBinomialP":
#             bounds = [(None, None)] * (ele_cnt + 1) + [(0, None)]
#             permute_results = ZeroInflatedNegativeBinomialP(
#                 endog, exog, exposure=exposure
#             ).fit(method="lbfgs", maxiter=5000, disp=0, bounds=bounds)
#         else:
#             raise ValueError(f"Unknown model class: {cls_name}")
#     except:  # hack for resolving singular matrix error
#         return 0
#     llr = permute_results.llr
#     return llr if llr is not None else 0


# def llr_permutation_test(
#     results: CountResults,
#     alpha: float = 0.05,
#     n_permute: int = 10000,
#     chunksize: int = 1,
# ) -> tuple[list[float], float]:
#     """
#     Compute the one-way permutation test for the likelihood ratio test statistic.

#     Parametersss
#     ----------
#     results : CountResults
#         Results of discrete GLM.
#     alpha : float
#         Desired significance level.
#     n_permute : int
#         Number of permutations to perform for the permutation test.
#     chunksize : int
#         Size of chunks sent to worker processes.

#     Returns
#     -------
#     null : list of float
#         The null statistics from the permutation test.
#     pvalue : float [0, 1]
#         The p-value associated with the permutation test.
#     """

#     # check if permutation test is possible or necessary
#     if not results.converged or not results.llr_pvalue or results.llr_pvalue > alpha:
#         print("Permutation test: not possible or necessary.")
#         return None, None

#     # compute p-values
#     null = process_map(
#         _permutation_test_helper,
#         [results] * n_permute,
#         chunksize=chunksize,
#         desc=f"Permutation Test",
#     )
#     pvalue = np.greater_equal(null, results.llr).mean()
#     return null, pvalue


# def latency_analysis(
#     spike_timings: Iterable[float],
#     start_times: Iterable[float],
#     end_times: Iterable[float],
#     event_times: Iterable[float],
#     alpha: float = 0.05,
# ) -> tuple[float, float]:
#     """
#     Compute the estimated start and end times of the first significant neural activities across time windows.

#     Parameters
#     ----------
#     spike_timings : iterable of float
#         Timings of all spikes of interest.
#     start_times : iterable of float
#         Start times of each time window.
#     end_times : iterable of float
#         End times of each time window.
#     event_times : iterable of float
#         Timings of the event of interest.
#     alpha : float
#         Desired significance level.

#     Returns
#     -------
#     activity_start : float
#         Estimated start time of first significant activity across all window.
#     activity_end : float
#         Estimated end time of first significant activity across all window.

#     References
#     ----------
#     [1] Hanes DP, Thompson KG, Schall JD. Relationship of presaccadic activity in frontal eye field and supplementary eye field to saccade initiation in macaque: Poisson spike train analysis. Exp Brain Res. 1995;103(1):85-96. doi: 10.1007/BF00241967. PMID: 7615040.
#     """

#     spike_timings = np.asarray(spike_timings)
#     start_times = np.asarray(start_times)
#     end_times = np.asarray(end_times)
#     event_times = np.asarray(event_times)

#     base_cnts = spike_counts(spike_timings, start_times, event_times)
#     base_rates = base_cnts / (event_times - start_times)
#     spikes_in_window = get_spikes(spike_timings, start_times, end_times)
#     activity_start, activity_end = [], []

#     def _failure():
#         activity_start.append(np.nan)
#         activity_end.append(np.nan)

#     for i, spikes in enumerate(spikes_in_window):
#         spikes = np.asarray(spikes)
#         # check if analysis is possible
#         base_rate = base_rates[i]
#         if base_rate == 0 or len(spikes) == 3:
#             _failure()
#             continue

#         # set initial search head to first spike after event onset, if possible
#         head = np.argmax(spikes >= event_times[i])

#         # forward search
#         candidates = np.arange(head + 1, len(spikes))
#         if len(candidates) == 0:
#             _failure()
#             continue
#         cnts = candidates - head + 1
#         t_int = np.cumsum(spikes[candidates] - spikes[head])
#         pvalues = test_poisson(cnts, t_int, base_rate, method="wald").pvalue
#         tail = candidates[np.argmin(pvalues)]

#         # backward search
#         candidates = np.arange(head, tail)
#         if len(candidates) == 0:
#             _failure()
#             continue
#         cnts = tail - candidates + 1
#         t_int = np.cumsum((spikes[tail] - spikes[candidates])[::-1])[::-1]
#         pvalues = test_poisson(cnts, t_int, base_rate, method="wald").pvalue
#         head = candidates[np.argmin(pvalues)]

#         # check for significance
#         cnts = tail - head
#         t_int = spikes[tail] - spikes[head]
#         pvalue = test_poisson(cnts, t_int, base_rate, method="wald").pvalue
#         if pvalue > alpha:
#             activity_start.append(np.nan)
#             activity_end.append(np.nan)
#         else:
#             activity_start.append(spikes[head])
#             activity_end.append(spikes[tail])

#     return activity_start, activity_end
