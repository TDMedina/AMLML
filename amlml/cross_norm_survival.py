
import os
from string import ascii_uppercase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import IndexSlice as idx
import plotly.graph_objects as go
import plotly.express as px
from scipy.integrate import simpson
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import torchtuples as tt

from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

from amlml.parallel_modelling import CrossNormalizedModel
from amlml.lasso import test_lasso_penalties, get_non_zero_genes
from amlml.km import plot_survival_curves, optimize_survival_splits, iterate_logrank_tests
from amlml.data_loader import normalization_generator, prepare_log2_expression
from amlml.cross_normalization import zscore_normalize_genes_by_group



def cross_validation_run(data_generator, covariate_cardinality,
                         l1_ratio=1, iterate_alphas=True,
                         alpha_min_ratio=0.01, n_alphas=21, alphas=None):
    norm_method_results = dict()
    for norm_method, data in data_generator:
        print(f"Running normalization method: {norm_method}")
        # (expression, outcomes, categoricals, non_categoricals,
        #  _, group_labels, expression_table) = data

        # expression_data = expression["train"]
        # categorical_data = categoricals["train"]
        # non_categorical_data = non_categoricals["train"]
        # outcome_data = outcomes["train"]
        # group_data = group_labels["train"]
        # original_data = expression_table["train"]

        kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        results = dict()
        for i, (train_fold, val_fold) in enumerate(kf.split(data.expression, data.reference.tech)):
            print(f"Running fold {i}...")
            results[i] = dict()
            print(f"    Preparing input data...")
            expression = data.expression[train_fold]
            categoricals = data.categoricals[train_fold]
            non_categoricals = data.non_categoricals[train_fold]
            outcomes_dur = data.outcomes[0][train_fold]
            outcomes_event = data.outcomes[1][train_fold]
            tech = data.reference.tech.iloc[train_fold]

            alpha_data = pd.DataFrame(expression,
                                      index=data.reference.index[train_fold],
                                      columns=data.reference.columns).join(tech)
            alpha_data = zscore_normalize_genes_by_group(alpha_data)

            print(f"    Preparing validation data...")
            expression_val = data.expression[val_fold]
            categoricals_val = data.categoricals[val_fold]
            non_categoricals_val = data.non_categoricals[val_fold]
            outcomes_val_dur = data.outcomes[0][val_fold]
            outcomes_val_event = data.outcomes[1][val_fold]

            if iterate_alphas:
                print(f"    Calculating lasso penalties...")
                alpha_table = test_lasso_penalties(alpha_data,
                                                   (outcomes_dur, outcomes_event),
                                                   l1_ratio=l1_ratio,
                                                   alpha_min_ratio=alpha_min_ratio,
                                                   n_alphas=n_alphas,
                                                   alphas=alphas)
                geneset = get_non_zero_genes(alpha_table)
            else:
                geneset = {0: (slice(None), list(alpha_data.columns))}
            print("Running validation...")
            for alpha, (index, genes) in tqdm(geneset.items()):
                results[i][alpha] = dict()
                results[i][alpha]["genes"] = ",".join(genes)

                print(f"Preparing training fold for alpha={alpha}...")
                outcomes_train = (outcomes_dur, outcomes_event)
                outcomes_val = (outcomes_val_dur, outcomes_val_event)

                expression_train_subset = expression[:, index]
                expression_val_subset = expression_val[:, index]
                train_args = (expression_train_subset, categoricals, non_categoricals)
                val_args = (expression_val_subset, categoricals_val, non_categoricals_val)

                network = CrossNormalizedModel(
                    n_genes=expression_train_subset.shape[-1],
                    shrinkage_factor=10,
                    minimum_penultimate_size=10,
                    final_size=1,
                    n_clinical=categoricals.shape[-1] +
                               non_categoricals.shape[-1],
                    covariate_cardinality=covariate_cardinality,
                    embedding_dims={"race": 3, "ethnicity": 3,
                                    "interaction": 3,
                                    "protocol": 3},
                    )
                model = CoxPH(network, tt.optim.Adam)

                batch_size = 95
                print("Calculating learning rate...")
                lrfinder = model.lr_finder(train_args, outcomes_train, batch_size,
                                           tolerance=10)
                lr_optim = lrfinder.get_best_lr()
                model.optimizer.set_lr(lr_optim)

                epochs = 360
                eps = 0.005
                verbose = True
                losses = []

                print("Running epochs...")
                for epoch in range(epochs):
                    log = model.fit(train_args, outcomes_train, batch_size,
                                    epochs=1,
                                    verbose=verbose)
                    log = log.to_pandas()
                    losses.append(log.train_loss.iloc[-1])
                    if len(losses) < 10:
                        continue
                    if np.var(losses[-10:]) < eps:
                        break

                results[i][alpha]["epochs"] = epoch

                print("Evaluating performance...")
                results[i][alpha]["pll"] = model.partial_log_likelihood(val_args,
                                                                        outcomes_val).mean()
                # TODO:
                results[i][alpha]["pll_train"] = model.partial_log_likelihood(train_args,
                                                                              outcomes_train).mean()
                model.compute_baseline_hazards()

                surv = model.predict_surv_df(val_args)

                outcomes_val = np.array(outcomes_val)

                ev = EvalSurv(surv, outcomes_val[0], outcomes_val[1], censor_surv="km")
                results[i][alpha]["ctd"] = ev.concordance_td()

                time_grid = np.linspace(outcomes_val[0].min(), outcomes_val[0].max(),
                                        100)
                brier_scores = ev.brier_score(time_grid)
                results[i][alpha]["ibs"] = (simpson(y=brier_scores.values, x=brier_scores.index)
                                            / (brier_scores.index[-1] - brier_scores.index[0]))

                # ev.integrated_nbll(time_grid)

                predictions = [float(x[0]) for x in model.predict(val_args)]
                km_test_df = pd.DataFrame(zip(outcomes_val[0], outcomes_val[1], predictions),
                                          columns=["duration", "event", "risk"])
                print("Calculating survival split...")
                optimal_splits = optimize_survival_splits(km_test_df, n_groups=3)
                risk_splits = np.cumulative_sum(optimal_splits.x)
                # plot_survival_curves(km_test_df)
                logranks = iterate_logrank_tests(km_test_df)
                results[i][alpha]["risk_splits"] = risk_splits
                results[i][alpha]["km_logrank"] = logranks
                print("Done")
        norm_method_results[norm_method] = results
    return norm_method_results, km_test_df


def make_results_table(results):
    fields = ["epochs", "pll", "ctd", "ibs"]
    parsed = []
    for fold, alpha_dict in results.items():
        for alpha, result_dict in alpha_dict.items():
            entry = ([fold, alpha, len(result_dict["genes"].split(","))]
                     + [result_dict[x] for x in fields])
            entry += [tuple(sorted(result_dict["risk_splits"]))]
            # print(fold, alpha, result_dict["km_logrank"].keys())
            entry += [result_dict["km_logrank"][tuple(x)].p_value
                      if tuple(x) in result_dict["km_logrank"] else None
                      for x in ["AB", "BC", "AC"]]
            parsed.append(entry)
    cols = ["fold", "alpha", "gene_count", *fields, "risk_splits", "km_logrank_AB",
            "km_logrank_BC", "km_logrank_AC"]
    table = pd.DataFrame(parsed, columns=cols)
    table["group"] = [i for _ in range(len(results.keys()))
                      for i in ascii_uppercase[:len(list(results.values())[0].keys())]]
    table.set_index(["fold", "group", "alpha"], inplace=True)
    return table


def make_mean_results_table(results_table):
    table = results_table[["gene_count", "pll", "epochs", "ctd", "ibs"]].groupby("group").mean()
    return table


def make_median_results_table(results_table):
    table = results_table[["gene_count", "pll", "epochs", "ctd", "ibs"]].groupby("group").median()
    return table


def plot_survival_splits(results_table):
    splits = results_table[["gene_count"]].copy()
    splits["split1"] = [x[0] for x in results_table.risk_splits]
    splits["split2"] = [x[1] for x in results_table.risk_splits]
    fig = go.Figure()
    for fold in splits.index.unique(level="fold"):
        data = splits.loc[idx[fold, :, :],]
        fig.add_trace(go.Bar(x=data.reset_index()["group"],
                             y=data.split1,
                             offsetgroup=fold,
                             marker_color="red",
                             marker_line_color="red"))
        fig.add_trace(go.Bar(x=data.reset_index()["group"],
                             y=data.split2-data.split1,
                             offsetgroup=fold,
                             marker_color="blue",
                             marker_line_color="blue"))
        fig.add_trace(go.Bar(x=data.reset_index()["group"],
                             y=1-data.split2,
                             offsetgroup=fold,
                             marker_color="green",
                             marker_line_color="green"))
    fig.update_layout(barmode="stack")

    # splits = splits.reset_index().sort_values(["group", "gene_count"])
    # fig2 = go.Figure()
    # fig2.add_trace(go.Scatter(x=splits.gene_count, y=splits.split1))
    # fig2.add_trace(go.Scatter(x=splits.gene_count, y=splits.split2))
    logranks = ["km_logrank_AB", "km_logrank_BC", "km_logrank_AC"]
    splits = splits.join(results_table[logranks])
    fig3 = go.Figure()
    for logrank in logranks:
        fig3.add_trace(go.Scatter(x=splits.gene_count, y=splits[logrank],
                                  mode="markers"))
    melted = pd.melt(splits[["gene_count"]+logranks], id_vars="gene_count")
    melted["value"] = np.log10(melted["value"])
    fig4 = px.scatter(melted, x="gene_count", y="value", color="variable")
    fig4.add_trace(go.Scatter(x=[0, 1600], y=[np.log10(0.05)]*2, mode="lines",
                              marker_line_color="red"))
    return fig, fig4


def plot_performance_metrics(results_table, plot_mean=False, plot_median=False):
    metrics = ["epochs", "pll", "ctd", "ibs"]
    trimmed = results_table.copy()
    trimmed["pll"] = [max(x, -20) for x in results_table["pll"]]
    title = "Gene counts vs metric"
    if plot_mean:
        trimmed = make_mean_results_table(trimmed)
        title += " means"
    elif plot_median:
        trimmed = make_median_results_table(trimmed)
        title += " medians"
    figs = px.scatter(pd.melt(trimmed[["gene_count"] + metrics], id_vars=["gene_count"]),
                      x="gene_count", y="value", trendline="lowess",
                      facet_col="variable", facet_col_wrap=2)
    figs.update_layout(title=title)
    figs.update_yaxes(matches=None)
    figs.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
    return figs


cv_results = cross_validation_run(
    data_generator=normalization_generator(methods=None, verbose=True),
    covariate_cardinality={"race": 7, "ethnicity": 3, "protocol": 7},
    iterate_alphas=False
    )
