
from collections import deque, namedtuple
import math
from string import ascii_uppercase
from typing import Callable

import numpy as np
from numpy.polynomial import Polynomial
import pandas as pd
from pandas import IndexSlice as idx
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import variation
from scipy.integrate import simpson
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import torch
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau
from torch.nn import BCEWithLogitsLoss
import torchtuples as tt

from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from pycox.models.loss import CoxPHLoss

from amlml.parallel_modelling import CrossNormalizedModel, SuperModel
from amlml.lasso import test_lasso_penalties, get_non_zero_genes
from amlml.km import plot_survival_curves, optimize_survival_splits, iterate_logrank_tests
from amlml.data_loader import (NetworkDataset, main_loader,
                               prepare_zscore_expression)
from amlml.cross_normalization import zscore_normalize_genes_by_group
from amlml.coxph_eval import (partial_log_likelihood, compute_baseline_hazards, predict_survival_table,
                              CV_Result, CV_ResultsCollection)


ConvergeTest = namedtuple("ConvergeTest", ["passed", "score", "threshold"])


def generate_loss_slope_test(rel_slope_threshold=0.01, n_losses=10):
    def loss_slope_test(losses):
        polyfit = Polynomial.fit(list(range(n_losses)), losses,
                                 deg=1, window=[0, n_losses-1])
        relative_slope = np.abs(polyfit.coef[1] / np.mean(losses))
        return ConvergeTest(relative_slope < rel_slope_threshold,
                            relative_slope, rel_slope_threshold)
    return loss_slope_test


def generate_loss_var_test(metric="cov", var_threshold=0.005, cov_threshold=0.01):
    if metric == "cov":
        test = variation
        threshold = cov_threshold
    elif metric == "var":
        test = np.var
        threshold = var_threshold
    else:
        raise ValueError(f"Unrecognized metric: {metric}")
    def var_test(losses):
        score = test(losses, ddof=1)
        return ConvergeTest(score < threshold,
                            score, threshold)
    return var_test


def generate_loss_convergence_test(metric="cov", test_loss_slope=True,
                                   var_threshold=0.005, cov_threshold=0.01,
                                   rel_slope_threshold=0.01, n_losses=10):
    slope_test = (generate_loss_slope_test(rel_slope_threshold, n_losses)
                  if test_loss_slope else lambda losses: ConvergeTest(True, None, None))
    var_test = generate_loss_var_test(metric=metric, cov_threshold=cov_threshold,
                                      var_threshold=var_threshold)
    def loss_convergence_test(losses):
        slope_result = slope_test(losses)
        var_result = var_test(losses)
        result = ConvergeTest(var_result.passed and slope_result.passed,
                              [var_result.score, slope_result.score],
                              [var_result.threshold, slope_result.threshold])
        return result
    return loss_convergence_test


def lr_cyclic_step_calculator(sample_size, batch_size, epochs_per_cycle):
    """Calculate the steps required to synchronize CyclicLR steps with epochs."""
    steps = ((sample_size // batch_size) + math.ceil(sample_size % batch_size)) * epochs_per_cycle
    step_up = math.floor(steps/2)
    step_down = math.ceil(steps/2)
    return step_up, step_down


# %% CV run.
def cross_validation_run(dataset: NetworkDataset,
                         network_type: Callable = CrossNormalizedModel,
                         include_clinical_variables=True, covariate_cardinality=None,
                         l1_ratio=1, iterate_alphas=True,
                         alpha_min_ratio=0.01, n_alphas=21, alphas=None,
                         survival_splits=3, cov_threshold=0.01,
                         rel_slope_threshold=0.01,
                         batch_size=100, epochs=360, epochs_per_cycle=6,
                         save_network=False, end_with_lr_cycle=False,
                         classify=False, classification_threshold=1460):
    verbose = True
    len_loss_convergence = epochs_per_cycle*3
    convergence_test = generate_loss_convergence_test(
        metric="cov",
        test_loss_slope=True,
        cov_threshold=cov_threshold,
        rel_slope_threshold=rel_slope_threshold,
        n_losses=len_loss_convergence
        )
    if classify:
        dataset = dataset.filter_low_censorship_and_classify(classification_threshold)
    print(f"Running dataset: {dataset.name}")
    # kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)  # Used seed=0 for development.
    kf = StratifiedKFold(n_splits=5, random_state=10, shuffle=True)
    # kf = RepeatedStratifiedKFold(n_splits=5, random_state=10, n_repeats=10)
    # results = dict()
    results = []
    # for i, (train_fold, val_fold) in enumerate(kf.split(data.expression, data.reference.tech)):
    for i, (train_fold, val_fold) in enumerate(kf.split(dataset.expression, dataset.tech)):
        print(f"Running fold {i}...")
        # results[i] = dict()
        # print(f"    Preparing input data...")
        fold_data = dataset[train_fold]
        fold_data_val = dataset[val_fold]
        lr_cycle_steps = len(fold_data)

        # expression = data.expression[train_fold]
        # if include_clinical_variables:
        #     categoricals = data.categoricals[train_fold]
        #     non_categoricals = data.non_categoricals[train_fold]
        # else:
        #     categoricals = tensor([], dtype=float32)
        #     non_categoricals = tensor([], dtype=float32)
        # outcomes_dur = data.outcomes[0][train_fold]
        # outcomes_event = data.outcomes[1][train_fold]
        # tech = data.reference.tech.iloc[train_fold]

        # print(f"    Preparing validation data...")
        # expression_val = data.expression[val_fold]
        # if include_clinical_variables:
        #     categoricals_val = data.categoricals[val_fold]
        #     non_categoricals_val = data.non_categoricals[val_fold]
        # else:
        #     categoricals_val = tensor([], dtype=float32)
        #     non_categoricals_val = tensor([], dtype=float32)
        # outcomes_val_dur = data.outcomes[0][val_fold]
        # outcomes_val_event = data.outcomes[1][val_fold]

        if iterate_alphas:
            print(f"    Calculating lasso penalties...")
            # alpha_data = pd.DataFrame(expression,
            #                           index=data.reference.index[train_fold],
            #                           columns=data.reference.columns).join(tech)
            lasso_data = fold_data.make_expression_table()
            lasso_data = zscore_normalize_genes_by_group(lasso_data)
            alpha_table = test_lasso_penalties(lasso_data,
                                               # (outcomes_dur, outcomes_event),
                                               fold_data.outcomes,
                                               l1_ratio=l1_ratio,
                                               alpha_min_ratio=alpha_min_ratio,
                                               n_alphas=n_alphas,
                                               alphas=alphas)
            geneset = get_non_zero_genes(alpha_table)
        else:
            # geneset = {0: (slice(None), list(data.reference.columns))}
            geneset = {0: (slice(None), list(dataset.genes))}
        # print("Running validation...")
        for j, (alpha, (index, genes)) in enumerate(geneset.items()):
            print(f"    Preparing training fold for alpha={alpha}...")
            alpha_data = fold_data.subset_genes(index)
            alpha_val = fold_data_val.subset_genes(index)

            # outcomes_train = (outcomes_dur, outcomes_event)
            # outcomes_val = (outcomes_val_dur, outcomes_val_event)

            # if network_type == SuperModel:
            #     expression_train_subset = expression[:, :, index]
            #     expression_val_subset = expression_val[:, :, index]
            # else:
            #     expression_train_subset = expression[:, index]
            #     expression_val_subset = expression_val[:, index]

            # if include_clinical_variables:
            #     train_args = (expression_train_subset, categoricals, non_categoricals)
            #     val_args = (expression_val_subset, categoricals_val, non_categoricals_val)
            #     n_clinical = categoricals.shape[-1] + non_categoricals.shape[-1]
            # else:
            #     train_args = expression_train_subset
            #     val_args = expression_val_subset
            #     n_clinical = None

            network_parameters = dict(
                # n_genes=expression_train_subset.shape[-1],
                n_genes=alpha_data.n_genes,
                shrinkage_factor=10,
                minimum_penultimate_size=10,
                final_size=1,
                include_clinical_variables=include_clinical_variables,
                n_clinical=alpha_data.n_clinical,
                covariate_cardinality=covariate_cardinality,
                embedding_dims={"race": 3, "ethnicity": 3,
                                "interaction": 3,
                                "protocol": 3},
                )
            if network_type == SuperModel:
                network_parameters.update(dict(
                    n_tech=2,
                    n_expansion=4
                    ))

            network = network_type(**network_parameters)

            # Start by using PyCox's lr_finder.
            if classify:
                lr_init = 1e-5
            else:
                print("    Calculating learning rate...")
                model = CoxPH(network, tt.optim.Adam)
                lrfinder = model.lr_finder(
                    alpha_data.network_args,
                    alpha_data.outcomes,
                    batch_size,
                    #tolerance=10,
                    # verbose=verbose
                    )
                lr_init = lrfinder.get_best_lr(lr_min=1e-6)
                del model
                print(f"    Calculated learning rate = {lr_init}")

            # Continue with base pytorch components to use more advanced LR tools.
            print("    Initializing model...")
            optimizer = torch.optim.Adam(network.parameters(), lr=lr_init)
            # optimizer = torch.optim.Adam(network.parameters(), lr=0.005)
            steps = lr_cyclic_step_calculator(len(alpha_data), batch_size, epochs_per_cycle)
            steps = dict(zip(["step_size_up", "step_size_down"], steps))
            lr_scheduler = CyclicLR(
                optimizer, base_lr=lr_init/10, max_lr=lr_init*10,
                # mode="exp_range", gamma=0.8, **steps, scale_mode="cycle",
                mode="triangular2", **steps, scale_mode="iterations"
                # mode="triangular", scale_fn=lambda cycle: 0.8**(cycle-1), **steps,
                # scale_mode="cycle",
                )

            loss_fn = BCEWithLogitsLoss() if classify else CoxPHLoss()
            run_loss = (lambda predicted, batch_: loss_fn(predicted, batch_.classes) if classify
            else loss_fn(predicted, *batch_.outcomes))
            # coxph_loss = CoxPHLoss()
            losses = []
            losses_val = []
            learning_rates = []
            print("    Training...")
            progress = tqdm(range(1, epochs+1), desc="Epochs",
                            postfix={"Loss": "0", "Var": "--", "Slope": "--"},
                            dynamic_ncols=True, position=0, leave=True)
            for epoch in progress:
                for batch in alpha_data.generate_batches(batch_size, shuffle_=True):
                    learning_rates.append(lr_scheduler.get_last_lr())
                    optimizer.zero_grad()
                    predicted_hazards = network(*batch.network_args)
                    loss = run_loss(predicted_hazards, batch)
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()

                # Calculate validation loss.
                predicted_hazards_val = network(*alpha_val.network_args)
                loss_val = run_loss(predicted_hazards_val, alpha_val)
                losses_val.append(float(loss_val))

                # Track epoch loss and check loss convergence.
                losses.append(float(loss))
                if len(losses) >= len_loss_convergence:
                    # converge_check = convergence_test(losses)
                    converge_check = convergence_test(losses[-len_loss_convergence:])
                    progress.set_postfix({"Loss": f"{float(loss):.3f}",
                                          "Var": f"{converge_check.score[0]:.4f}",
                                          "Slope": f"{converge_check.score[1]:.4f}"})
                    if (converge_check.passed and
                            (epoch % epochs_per_cycle == 0 or not end_with_lr_cycle)):
                        break
                else:
                    progress.set_postfix({"Loss": f"{float(loss):.3f}",
                                          "Var": "--",
                                          "Slope": "--"})



            # Evaluate.
            predicted_hazards = network(*alpha_data.network_args).detach().numpy()
            predicted_hazards_val = network(*alpha_val.network_args).detach().numpy()

            if classify:
                classes_train = pd.DataFrame(zip(alpha_data.classes.view([1, -1]),
                                                 predicted_hazards),
                                             columns=["Training", "Predicted"])
                classes_val = pd.DataFrame(zip(alpha_val.classes.view([1, -1]),
                                               predicted_hazards),
                                           columns=["Validation", "Predicted"])
                results.append(CV_Result(
                    fold=i,
                    alpha=alpha,
                    genes=genes,
                    n_epochs=epoch,
                    lrs=[float(x[0]) for x in learning_rates],
                    losses_train=losses,
                    losses_val=losses_val,
                    name=dataset.name,
                    classes_train=classes_train,
                    classes_val=classes_val,
                    network=network if save_network else None,
                    pll_train=None,
                    pll_val=None,
                    hazards=None,
                    ibs=None,
                    ctd=None,
                    risk_splits=None,
                    logranks=None
                    ))
                continue

            baseline_hazards = compute_baseline_hazards(alpha_data.outcome_target_table,
                                                        predicted_hazards)
            survival = predict_survival_table(predicted_hazards_val,
                                              baseline_hazards.cumulative)

            # outcomes_val = np.array(outcomes_val)

            ev = EvalSurv(survival,
                          alpha_val.durations.detach().numpy(),
                          alpha_val.events.detach().numpy(),
                          censor_surv="km")
            ctd = ev.concordance_td()

            time_grid = np.linspace(float(alpha_val.durations.min()),
                                    float(alpha_val.durations.max()),
                                    100)
            brier_scores = ev.brier_score(time_grid)
            ibs = (simpson(y=brier_scores.values, x=brier_scores.index)
                   / (brier_scores.index[-1] - brier_scores.index[0]))

            # ev.integrated_nbll(time_grid)

            km_val_df = alpha_val.outcome_target_table
            km_val_df = km_val_df.assign(risk=predicted_hazards_val)

            print("    Calculating survival split...")
            optimal_splits = optimize_survival_splits(km_val_df, n_groups=survival_splits)
            risk_splits = np.cumulative_sum(optimal_splits.x)
            logranks = iterate_logrank_tests(km_val_df)


            # results[i][j] = CV_Results(
            results.append(CV_Result(
                fold=i,
                alpha=alpha,
                genes=genes,
                n_epochs=epoch,
                lrs=[float(x[0]) for x in learning_rates],
                losses_train=losses,
                losses_val=losses_val,
                pll_train=partial_log_likelihood(alpha_data.outcome_target_table,
                                                 predicted_hazards),
                pll_val=partial_log_likelihood(alpha_val.outcome_target_table,
                                               predicted_hazards_val),
                network=network if save_network else None,
                name=dataset.name,
                hazards=baseline_hazards,
                ibs=ibs,
                ctd=ctd,
                risk_splits=risk_splits,
                logranks=logranks
                ))

            # # optimizer = tt.optim.Adam(network.parameters())
            # # model = CoxPH(network, tt.optim.Adam(lr=0.01))
            # # optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
            # # model = CoxPH(network, optimizer)
            #
            # # Set initial learning rate with temporary fake model.
            # # This has to use the "fake" model with tt.optim.Adam, because the
            # # lr_finder thing is only compatible this way, but tt.optim.Adam is not
            # # compatible with torch's LR schedulers.
            # print("Calculating learning rate...")
            # model = CoxPH(network, tt.optim.Adam)
            # print("A")
            # lrfinder = model.lr_finder(train_args, outcomes_train, batch_size,
            #                            tolerance=10)
            # lr_optim = lrfinder.get_best_lr()
            #
            # # Start the real model with a torch LR scheduler.
            # # optimizer = torch.optim.Adam(network.parameters(), lr=lr_optim)
            # # model = CoxPH(network, optimizer)
            # model.optimizer.set_lr(lr_optim)
            # lr_scheduler = CyclicLR(model.optimizer.optimizer,
            #                         base_lr=lr_optim/5,
            #                         max_lr=lr_optim*5)
            # # lr_scheduler = ReduceLROnPlateau(model.optimizer.optimizer, patience=8)
            # print("B")
            # callbacks = [tt.callbacks.LRScheduler(lr_scheduler, model.optimizer)]
            #
            # losses = []
            # print("Running epochs...")
            # for epoch in range(epochs):
            #     log = model.fit(train_args, outcomes_train, batch_size,
            #                     epochs=1,
            #                     verbose=verbose,
            #                     callbacks=callbacks)
            #     log = log.to_pandas()
            #     losses.append(log.train_loss.iloc[-1])
            #     if len(losses) < len_loss_convergence:
            #         continue
            #     if convergence_test(losses[-len_loss_convergence:]):
            #         break
            #     # if variation(losses[-10:], ddof=1) < convergence_threshold:
            #     #     break
            #     # if np.var(losses[-10:]) < eps:
            #     #     break

            # # results[i][alpha]["epochs"] = epoch

            # print("Evaluating performance...")
            # results[i][alpha]["pll"] = model.partial_log_likelihood(val_args,
            #                                                         outcomes_val).mean()
            # results[i][alpha]["pll_train"] = model.partial_log_likelihood(train_args,
            #                                                               outcomes_train).mean()
            # TODO: Implement baseline hazards, CTD, and IBS.
            # model.compute_baseline_hazards()
            #
            # surv = model.predict_surv_df(val_args)
            #
            # outcomes_val = np.array(outcomes_val)
            #
            # ev = EvalSurv(surv, outcomes_val[0], outcomes_val[1], censor_surv="km")
            # results[i][alpha]["ctd"] = ev.concordance_td()
            #
            # time_grid = np.linspace(outcomes_val[0].min(), outcomes_val[0].max(),
            #                         100)
            # brier_scores = ev.brier_score(time_grid)
            # results[i][alpha]["ibs"] = (simpson(y=brier_scores.values, x=brier_scores.index)
            #                             / (brier_scores.index[-1] - brier_scores.index[0]))
            #
            # # ev.integrated_nbll(time_grid)
            #
            # predictions = [float(x[0]) for x in model.predict(val_args)]
            # km_test_df = pd.DataFrame(zip(outcomes_val[0], outcomes_val[1], predictions),
            #                           columns=["duration", "event", "risk"])
            # print("Calculating survival split...")
            # optimal_splits = optimize_survival_splits(km_test_df, n_groups=survival_splits)
            # risk_splits = np.cumulative_sum(optimal_splits.x)
            # # plot_survival_curves(km_test_df)
            # logranks = iterate_logrank_tests(km_test_df)
            # results[i][alpha]["risk_splits"] = risk_splits
            # results[i][alpha]["km_logrank"] = logranks
            # print("Done")
    results = CV_ResultsCollection(results, None, 5)
    return results
    # norm_method_results[data.name] = results
    # return norm_method_results


def cv_multiple(datasets: list[tuple[Callable, NetworkDataset]], **kwargs):
    results = []
    names = []
    for network_type, (train, _) in datasets:
        names.append(train.name)
        sub_results = cross_validation_run(dataset=train, network_type=network_type,
                                           **kwargs)
        results.extend(sub_results.results)
    print("Collating results...")
    results = CV_ResultsCollection(results, names, 5)
    return results


# %% Results parsing.
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


# def make_results_table2(results):
#     parsed = []
#     for fold, alpha_dict in results.items():
#         for alpha_dex, result_dict in alpha_dict.items():
#             entry = dict(
#                 alpha=result_dict["alpha"],
#                 n_genes=len(result_dict["genes"]),
#                 n_epochs=result_dict["epochs"],
#                 # lrs=result_dict["lrs"],
#                 losses_train=result_dict["losses"],
#                 losses_val=result_dict["losses_val"],
#                 pll_train=result_dict["pll_train"],
#                 pll_val=result_dict["pll_val"],
#                 )
#             parsed.append(entry)
#     # TODO: Continue here.
#     table = pd.DataFrame(parsed, columns=["alpha", "n_genes", "n_epochs", "lrs",
#                                           "losses_train", "losses_val", "pll_train",
#                                           "pll_val"])
#     return table
#     fields = ["alpha", "epochs", "lrs", "losses", "losses_val", "pll_train",
#               "pll_val"]
#
# def make_mean_results_table(results_table):
#     table = results_table[["gene_count", "pll", "epochs", "ctd", "ibs"]].groupby("group").mean()
#     return table
#
#
# def make_median_results_table(results_table):
#     table = results_table[["gene_count", "pll", "epochs", "ctd", "ibs"]].groupby("group").median()
#     return table


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


# %% Runs.
# cv_results = cross_validation_run(
#     data_generator=normalization_generator(methods=None, verbose=True),
#     include_clinical_variables=True,
#     covariate_cardinality={"race": 7, "ethnicity": 3, "protocol": 7},
#     iterate_alphas=True,
#     n_alphas=21,
#     alphas=None,
#     survival_splits=2,
#     cov_threshold=0.01,
#     rel_slope_threshold=1
#     )


# TODO: May need to implement slope convergence test.
# cv_results = cross_validation_run(
#     data_generator=[("SuperModel",
#                      main_loader(normalization=prepare_supermodel_expression)[0])],
#     network_type=SuperModel,
#     include_clinical_variables=True,
#     covariate_cardinality={"race": 7, "ethnicity": 3, "protocol": 7},
#     iterate_alphas=False,
#     n_alphas=21,
#     alphas=None,
#     survival_splits=2,
#     cov_threshold=0.01,
#     rel_slope_threshold=1
#     )
if __name__ == "__main__":
    train, test = main_loader(prepare_zscore_expression)


# %% Single run.
    cv_results = cross_validation_run(
        dataset=train,
        include_clinical_variables=False,
        covariate_cardinality={"race": 7, "ethnicity": 3, "protocol": 7},
        iterate_alphas=False,
        survival_splits=2,
        cov_threshold=0.01,
        rel_slope_threshold=0.01,
        batch_size=100,
        epochs=360,
        epochs_per_cycle=6
        )

