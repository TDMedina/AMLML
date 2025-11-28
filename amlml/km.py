from itertools import combinations, pairwise
from string import ascii_uppercase
from typing import Literal

import pandas as pd
from lifelines.statistics import logrank_test, multivariate_logrank_test
# import statsmodels.stats.multitest as smm
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import numpy as np
from scipy.optimize import differential_evolution, brute


def optimize_survival_splits(data, n_groups=2, minimum_group_size=0.05,
                             criterion: Literal["risk", "durations"] = "risk",
                             method: Literal["DE", "Brute"] = "Brute"):
    def multivar_test(cutoffs: list[float]):
        cutoffs = np.cumulative_sum(cutoffs)
        if max(cutoffs) >= 1 or len(set(cutoffs)) != len(cutoffs):
            return 1e6
        cutoffs = [data[criterion].quantile(x) for x in cutoffs]

        groups = ascii_uppercase[:len(cutoffs)+1]
        data["group"] = groups[0]
        for group, cutoff in zip(groups[1:], cutoffs):
            data.loc[data[criterion] > cutoff, "group"] = group

        if ((vcounts := data.group.value_counts()).shape[0] != len(cutoffs)+1
                or vcounts.min() < int(minimum_group_size*data.shape[0])):
            return 1

        multivar_results = multivariate_logrank_test(data["durations"],
                                                     data["group"],
                                                     data["events"])
        # pairwise_results = iterate_logrank_tests(data)
        # pairwise_sum = max(x.p_value for x in pairwise_results.values())
        # return pairwise_sum
        return multivar_results.p_value

    if method == "DE":
        optimal = differential_evolution(multivar_test, [(0, 1)]*(n_groups-1),
                                         strategy="best1bin", maxiter=1000, tol=1e-8)
    elif method == "Brute":
        optimal = brute(multivar_test, [(0, 1)]*(n_groups-1), Ns=100, full_output=False,
                        finish=None)
    return optimal


def plot_survival_curves(data):
    kmf = KaplanMeierFitter()
    for group in sorted(data.group.unique()):
        group_data = data.loc[data.group == group]
        kmf.fit(group_data.durations, event_observed=group_data.events, label=group)
        kmf.plot_survival_function(ci_show=True)

    plt.title('Kaplan-Meier Survival Curves')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.grid(True)
    plt.show()


def iterate_logrank_tests(data, neighbors_only=False):
    tests = dict()
    if neighbors_only:
        pairs = pairwise(sorted(data.group.unique()))
    else:
        pairs = sorted([tuple(sorted(x)) for x in combinations(data.group.unique(), 2)])
    for pair in pairs:
        group1 = data.loc[data.group == pair[0]]
        group2 = data.loc[data.group == pair[1]]
        tests[pair] = logrank_test(group1.durations, group2.durations,
                                   group1.events, group2.events)
    return tests
    # return

    # logrank_results = logrank_test(
    #     group_A['duration'],
    #     group_B['duration'],
    #     event_observed_A=group_A['event'],
    #     event_observed_B=group_B['event']
    # )

    # multivar_results = multivariate_logrank_test(
    #     data['duration'],
    #     data['group'],
    #     data['event']
    #     )
    #
    # return multivar_results
    # groups = data['group'].unique()
    # p_values = []
    # pairs = []
    #
    # for g1, g2 in combinations(groups, 2):
    #     group1 = data[data['group'] == g1]
    #     group2 = data[data['group'] == g2]
    #
    #     results = logrank_test(
    #         group1['duration'], group2['duration'],
    #         event_observed_A=group1['event'],
    #         event_observed_B=group2['event']
    #         )
    #
    #     p_values.append(results.p_value)
    #     pairs.append((g1, g2))
    #
    # # Show raw results
    # for pair, p in zip(pairs, p_values):
    #     print(f"{pair}: raw p-value = {p:.4f}")
    #
    # # Bonferroni correction
    # reject_bonf, pvals_bonf, _, _ = smm.multipletests(p_values, alpha=0.05, method='bonferroni')
    #
    # # FDR correction (less conservative)
    # reject_fdr, pvals_fdr, _, _ = smm.multipletests(p_values, alpha=0.05, method='fdr_bh')
    #
    # # Display results
    # print("\nCorrected pairwise comparisons:")
    # for i, pair in enumerate(pairs):
    #     print(f"{pair}: Bonferroni p = {pvals_bonf[i]:.4f}, "
    #           f"FDR p = {pvals_fdr[i]:.4f}, "
    #           f"Significant (Bonf) = {reject_bonf[i]}, "
    #           f"Significant (FDR) = {reject_fdr[i]}")


# tests = dict()
# for i in tqdm(range(1, 100), position=0, leave=True):
#     for j in tqdm(range(i, 100), position=1, leave=False):
#         split_point1 = km_test_df.risk.quantile(i/100)
#         split_point2 = km_test_df.risk.quantile(j/100)
#         km_test_df["group"] = "low"
#         km_test_df.loc[split_point1 < km_test_df.risk, "group"] = "mid"
#         km_test_df.loc[split_point2 < km_test_df.risk, "group"] = "high"
#         if (km_test_df.group.value_counts().shape[0] < 3 or
#                 km_test_df.group.value_counts().min() < 16):
#             continue
#         tests[(i, j)] = multivariate_logrank_test(km_test_df.duration, km_test_df.group,
#                                                   km_test_df.event)
#
# go.Figure(go.Scatter3d(x=[x[0] for x in tests.keys()], y=[x[1] for x in tests.keys()],
#                        z=[np.log10(x.p_value) for x in tests.values()],
#                        mode="markers",
#                        marker_size=2))