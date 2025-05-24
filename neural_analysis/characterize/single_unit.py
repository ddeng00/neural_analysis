from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from patsy import dmatrices

from ..statistics import likelihood_ratio_test
from ..utils import sanitize_patsy_terms


def glm_test(
    data: pd.DataFrame,
    formula: str,
    alpha: float = 0.1,
    maxiter: int = 100,
    n_permutes: int = 0,
    pairwise: bool = True,
    n_jobs: int = -1,
):

    # preprocess input
    data = data.reset_index(drop=True)
    y, X = dmatrices(formula, data, return_type="dataframe")
    endog = y.columns[0]

    # define restricted models
    terms = sanitize_patsy_terms(X.design_info.term_name_slices.keys())
    slices = list(X.design_info.term_name_slices.values())
    term_sets = np.array([set(t.split(":")) for t in terms])
    restr = {}
    for term, term_set in zip(terms, term_sets):
        if "Intercept" in term_set:
            continue
        res_mask = np.ones(X.shape[1], dtype=bool)
        full_mask = np.ones(X.shape[1], dtype=bool)
        for s in np.take(slices, np.where(term_sets >= term_set)[0]):
            res_mask[s] = False
        for s in np.take(slices, np.where(term_sets > term_set)[0]):
            full_mask[s] = False
        restr[term] = res_mask, full_mask

    # define helper function
    glm = lambda x, y: sm.Poisson(x, y).fit(
        disp=0, maxiter=maxiter, warn_convergence=False, cov_type="HC0", use_t=True
    )

    def glm_helper(permute):

        # permute data if necessary
        yy = y.sample(frac=1, ignore_index=True) if permute else y

        # compute total effects of predictors
        model = glm(yy, X)
        total_effects = []
        for term, (res_mask, full_mask) in restr.items():
            try:
                model_res = glm(yy, X.loc[:, res_mask])
                model_full = model if all(full_mask) else glm(yy, X.loc[:, full_mask])
                results = likelihood_ratio_test(
                    model_full.llf,
                    model_res.llf,
                    model_full.df_model,
                    model_res.df_model,
                )
                total_effects.append({"predictor": term, "pvalue": results["pvalue"]})
            except Exception as e:
                print(f"Error in {term}: {e}")
                total_effects.append({"predictor": term})

        # finalize results
        if permute:
            return pd.DataFrame(total_effects)
        return pd.DataFrame(total_effects), model

    # get effets of full model
    total_effects, model = glm_helper(False)
    total_effects["model_pvalue"] = model.llr_pvalue
    if np.isnan(model.llr_pvalue):
        return {"total_effects": total_effects}

    # perform permutation testing
    if n_permutes > 0:
        null = pd.concat(
            Parallel(n_jobs=n_jobs)(
                delayed(glm_helper)(True) for _ in range(n_permutes)
            )
        )
        total_effects["pvalue_orig"] = total_effects["pvalue"]
        total_effects["pvalue"] = total_effects[["predictor", "pvalue"]].apply(
            lambda x: (
                null.loc[null["predictor"] == x["predictor"], "pvalue"] <= x["pvalue"]
            ).mean(),
            axis=1,
        )
    total_effects = total_effects.set_index("predictor")

    # check if further testing is necessary
    if not pairwise:
        return total_effects
    if np.isnan(model.llr_pvalue) or model.llr_pvalue > alpha:
        return {"total_effects": total_effects, "pairwise_effects": None}
    if total_effects["pvalue"].min() > alpha:
        return {"total_effects": total_effects, "pairwise_effects": None}

    # pairwise comparisons for significant predictors
    pairwise_effects = []
    cond_aves = X.mean(axis=0)
    for term in total_effects.index[total_effects["pvalue"] <= alpha]:

        # identify covariates
        subterms = term.split(":")
        cov_mask = np.ones(X.shape[1], dtype=bool)
        for t in subterms:
            cov_mask[slices[terms.index(t)]] = False

        # get per group averages
        grps, yy, sel_inds = [], [], []
        for grp, inds in data.groupby(subterms).groups.items():
            grps.append(grp)
            ym = np.mean(y.iloc[inds])
            yy.append(np.log(ym) if ym > 0 else np.nan)
            sel_inds.append(inds[0])
        yy = np.array(yy)
        XX = X.iloc[sel_inds].reset_index(drop=True)
        XX.loc[:, cov_mask] = cond_aves[cov_mask].values
        for col in XX:
            if ":" in col:
                XX[col] = XX[col.split(":")].prod(axis=1)

        # perform t-tests on all pairwise contrasts
        left_inds, right_inds = np.array(list(combinations(range(len(grps)), 2))).T
        contrasts = XX.iloc[left_inds] - XX.iloc[right_inds].values
        contrasts = model.t_test(contrasts).summary_frame()

        # process results
        contrasts = contrasts[["coef", "std err", "P>|t|"]].reset_index(drop=True)
        contrasts.columns = ["diff", "se", "pvalue"]
        contrasts["predictor"] = term
        contrasts["c0"] = [grps[i] for i in left_inds]
        contrasts["c1"] = [grps[i] for i in right_inds]
        contrasts["diff_gt"] = yy[left_inds] - yy[right_inds]
        pairwise_effects.append(contrasts)

    pairwise_effects = pd.concat(pairwise_effects).reset_index(drop=True)
    return {
        "total_effects": total_effects.reset_index(),
        "pairwise_effects": pairwise_effects,
    }


# def get_single_trial_latencies(
#     spikes: npt.ArrayLike,
#     search_starts: npt.ArrayLike,
#     search_ends: npt.ArrayLike,
#     alignments: npt.ArrayLike | None = None,
#     baseline_starts: npt.ArrayLike | None = None,
#     baseline_ends: npt.ArrayLike | None = None,
#     baseline_fr: float | list[float] | None = None,
#     *,
#     alpha: float = 0.05,
#     sorted: bool = True,
#     unit_conversion: float = 1.0,
# ) -> list[int]:

#     # process inputs
#     spikes = np.asarray(spikes) * unit_conversion
#     if alignments is None:
#         alignments = search_starts

#     # compute baseline firing rates if not provided
#     if baseline_fr is None:
#         if baseline_starts is None or baseline_ends is None:
#             baseline_fr = len(spikes) / (spikes[-1] - spikes[0])
#             baseline_fr = np.repeat(baseline_fr, len(search_starts))
#         else:
#             baseline_fr = compute_spike_rates(
#                 spikes, baseline_starts, baseline_ends, sorted=sorted
#             )
#     elif not isinstance(baseline_fr, list):
#         baseline_fr = [baseline_fr] * len(search_starts)

#     # process trial spikes
#     spikes = get_spikes(spikes, search_starts, search_ends, alignments, sorted=sorted)
#     sig_activities = []
#     for spike_train, ref_fr in zip(spikes, baseline_fr):

#         # check if analysis is possible
#         if len(spike_train) < 2:
#             sig_activities.append([np.nan, np.nan])
#             continue

#         # define all search windows
#         windows = np.array(
#             [[x, y] for x, y in product(np.arange(spike_train), repeat=2) if x < y]
#         )
#         cnts = np.diff(windows, axis=1).ravel()
#         isis = np.diff(spike_train[windows], axis=1).ravel()

#         # check for significant activities
#         pvals = test_poisson(cnts, isis, ref_fr, method="score").pvalue
#         if not any(pvals < alpha):
#             sig_activities.append([np.nan, np.nan])
#         else:
#             sig_activities.append(spike_train[windows[np.argmin(pvals)]])

#     return np.asarray(sig_activities)


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
