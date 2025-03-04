import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from joblib import Parallel, delayed


def poisson_wald(
    data: pd.DataFrame,
    formula: str,
    n_permutes: int = 1024,
    n_jobs: int = -1,
):

    # define GLM and helper function
    response = formula.split("~")[0].strip()

    def _helper(permute):
        df = data.copy()
        if permute:
            df[response] = np.random.permutation(df[response].values)
        model = smf.glm(formula=formula, data=df, family=sm.families.Poisson())
        return model.fit().wald_test_terms(scalar=True).table

    # get baseline statistics
    results = _helper(False)

    # compute permutation statistics
    if n_permutes > 0:
        null = list(
            Parallel(n_jobs=n_jobs)(delayed(_helper)(True) for _ in range(n_permutes))
        )
        null = pd.concat(null)
        results["pvalue"] = [
            (null.loc[term, "statistic"] <= results.loc[term, "statistic"]).mean()
            for term in results.index
        ]

    # return results
    results = results.reset_index(names="predictor")
    return results[results["predictor"] != "Intercept"]


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
