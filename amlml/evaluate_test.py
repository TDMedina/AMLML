
from collections import namedtuple
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
from torch.optim.lr_scheduler import CyclicLR, ConstantLR
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch import sigmoid
import torchtuples as tt

from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from pycox.models.loss import CoxPHLoss

from amlml.parallel_modelling import CrossNormalizedModel, SuperModel
from amlml.lasso import test_lasso_penalties, get_non_zero_genes
from amlml.km import optimize_survival_splits, iterate_logrank_tests
from amlml.data_loader import NetworkDataset
from amlml.cross_normalization import zscore_normalize_genes_by_group
from amlml.coxph_eval import (partial_log_likelihood, compute_baseline_hazards, predict_survival_table,
                              TestResult, TestResultsCollection, classify_by_hazard_at_threshold)


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


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


def generate_loss_function(classify: bool):
    loss_fn = BCEWithLogitsLoss() if classify else CoxPHLoss()
    if classify:
        def run_loss(predicted, batch):
            loss = loss_fn(predicted, batch.classes)
            return loss
    else:
        def run_loss(predicted, batch):
            loss = loss_fn(predicted, *batch.outcomes)
            return loss
    return run_loss


def lr_cyclic_step_calculator(sample_size, batch_size, epochs_per_cycle):
    """Calculate the steps required to synchronize CyclicLR steps with epochs."""
    steps = math.ceil(sample_size / batch_size) * epochs_per_cycle
    step_up = math.floor(steps/2)
    step_down = math.ceil(steps/2)
    return step_up, step_down


# %% CV run.
def test_run(train: NetworkDataset, test: NetworkDataset,
             network_type: Callable = CrossNormalizedModel,
             include_clinical_variables=True, covariate_cardinality=None,
             use_coxnet_alpha=True, coxnet_alpha=None, coxnet_l1_ratio=1,
             network_l1_reg=False, network_l1_alpha=None, network_weight_decay=1e-4,
             survival_splits=2, cov_threshold=0.01,
             rel_slope_threshold=0.01,
             batch_size=100, epochs=360, epochs_per_cycle=6,
             save_network=False,
             lr_init=None, constant_lr=False, end_with_lr_cycle=False,
             lr_cycle_mode="triangular",
             classify=False, hazard_classify=False,
             use_rmst=True, classification_threshold=1460,
             minimum_penultimate_size=10, shrinkage_factor=10,
             rmst_max_time=None, rmst_tukey_factor=None,
             zero_params=False, kaiming_weights=False,
             bellows_normalization=True, n_tech=2, n_expansion=4,
             remove_age_over=None, restrict_tech=None, use_shallow=False,
             dropout=0.2
             ):
    network_l1_alpha = 0 if network_l1_alpha is None else network_l1_alpha
    len_loss_convergence = epochs_per_cycle
    convergence_test = generate_loss_convergence_test(
        metric="cov",
        test_loss_slope=True,
        cov_threshold=cov_threshold,
        rel_slope_threshold=rel_slope_threshold,
        n_losses=len_loss_convergence
        )

    if remove_age_over is not None:
        train = train.filter_by_age_at_diagnosis(remove_age_over)
        test = test.filter_by_age_at_diagnosis(remove_age_over)
    if restrict_tech is not None:
        train = train.filter_by_tech(restrict_tech)
        test = test.filter_by_tech(restrict_tech)
    if classify or hazard_classify:
        if use_rmst:
            train.estimate_durations_with_rmst(max_time=rmst_max_time,
                                                       tukey_factor=rmst_tukey_factor)
            train.classify_by_rmst_ci_interval(save=True)
            test.estimate_durations_with_rmst(max_time=rmst_max_time,
                                                      tukey_factor=rmst_tukey_factor)
            test.classify_by_rmst_ci_interval(save=True)
        else:
            train = train.filter_low_censorship_and_classify_by_duration(
                classification_threshold
                )
            test = test.filter_low_censorship_and_classify_by_duration(
                classification_threshold
                )
        if hazard_classify:
            bce_loss = BCELoss()

    print(f"Running dataset: {train.name}")

    if use_coxnet_alpha:
        print(f"    Calculating lasso gene set...")
        lasso_data = train.make_expression_table()
        lasso_data = zscore_normalize_genes_by_group(lasso_data)
        alpha_table = test_lasso_penalties(lasso_data,
                                           train.outcomes,
                                           l1_ratio=coxnet_l1_ratio,
                                           alphas=[coxnet_alpha])
        geneset = get_non_zero_genes(alpha_table)
    elif network_l1_reg:
        geneset = {network_l1_alpha: (slice(None), list(train.genes))}
    else:
        geneset = {0: (slice(None), list(train.genes))}
    alpha, (index, genes) = list(geneset.items())[0]
    train = train.subset_genes(index)
    test = test.subset_genes(index)

    network_parameters = dict(
        n_genes=train.n_genes,
        shrinkage_factor=shrinkage_factor,
        minimum_penultimate_size=minimum_penultimate_size,
        final_size=1,
        include_clinical_variables=include_clinical_variables,
        n_clinical=train.n_clinical,
        covariate_cardinality=covariate_cardinality,
        embedding_dims={"race": 3, "ethnicity": 3,
                        "interaction": 3,
                        "protocol": 3},
        zero_params=zero_params,
        kaiming_weights=kaiming_weights,
        output_xavier=classify,
        use_shallow=use_shallow,
        dropout=dropout
        )
    if network_type == SuperModel:
        network_parameters.update(dict(
            n_tech=n_tech,
            n_expansion=n_expansion,
            bellows_normalization=bellows_normalization
            ))

    network = network_type(**network_parameters)
    network = network.to(DEVICE)
    network = torch.compile(network)

    # Start by using PyCox's lr_finder.
    if lr_init is not None:
        lr_init_ = lr_init
    else:
        print("    Calculating learning rate...")
        model = CoxPH(network, tt.optim.Adam)
        lrfinder = model.lr_finder(
            train.network_args,
            train.outcomes,
            batch_size,
            #tolerance=10,
            # verbose=verbose
            )
        lr_init_ = lrfinder.get_best_lr(lr_min=1e-4, lr_max=1e-2)/10
        del model
        print(f"    Calculated learning rate = {lr_init_}")

    # Continue with base pytorch components to use more advanced LR tools.
    print("    Initializing model...")
    optimizer = torch.optim.Adam(network.parameters(), lr=lr_init_, weight_decay=network_weight_decay)
    if constant_lr:
        lr_scheduler = ConstantLR(optimizer, factor=1)
    else:
        steps = lr_cyclic_step_calculator(len(train), batch_size, epochs_per_cycle)
        steps = dict(zip(["step_size_up", "step_size_down"], steps))
        lr_scheduler = CyclicLR(
            optimizer, base_lr=lr_init_/10, max_lr=lr_init_*10,
            mode=lr_cycle_mode, **steps, scale_mode="iterations",
            )

    run_loss = generate_loss_function(classify)
    losses_train = []
    learning_rates = []
    print("    Training...")
    progress = tqdm(range(1, epochs+1), desc="Epochs",
                    postfix={"Loss": "0", "Var": "--", "Slope": "--"},
                    dynamic_ncols=True, position=0, leave=True)
    for epoch in progress:
        epoch_loss = torch.tensor(0.0, device=DEVICE)
        for batch in train.generate_batches(batch_size, shuffle_=True):
            learning_rates.append(lr_scheduler.get_last_lr())
            optimizer.zero_grad()
            predictions_train = network(*batch.network_args)
            loss = run_loss(predictions_train, batch)
            epoch_loss += loss * len(batch)
            if network_l1_reg:
                loss = loss + alpha * network.connected_layers.layers[0].weight.abs().sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
            optimizer.step()
            lr_scheduler.step()
        epoch_loss = (epoch_loss / len(train))
        losses_train.append(epoch_loss)

        with torch.no_grad():
            # Track epoch loss and check loss convergence.
            if len(losses_train) >= len_loss_convergence:
                converge_check = convergence_test(torch.stack(losses_train[-len_loss_convergence:]).cpu().numpy())
                progress.set_postfix({"Loss": f"{losses_train[-1]:.3f}",
                                      "Var": f"{converge_check.score[0]:.4f}",
                                      "Slope": f"{converge_check.score[1]:.4f}"})
                if (converge_check.passed and
                        (epoch % epochs_per_cycle == 0 or not end_with_lr_cycle)):
                    break
            else:
                progress.set_postfix({"Loss": f"{losses_train[-1]:.3f}",
                                      "Var": "--",
                                      "Slope": "--"})

    # Evaluate.
    with torch.no_grad():
        predictions_train = network(*train.network_args)
        losses_train = [x.item() for x in losses_train]
        first_weights = network.connected_layers.layers[0].weight.cpu().numpy()
        predictions_test = network(*test.network_args)
        loss_test = run_loss(predictions_test, test).item()

        if classify:
            classes_train = pd.DataFrame(zip(train.classes.squeeze().tolist(),
                                             sigmoid(predictions_train).squeeze().tolist()),
                                         columns=["Training", "Predicted"])
            classes_test = pd.DataFrame(zip(test.classes.squeeze().tolist(),
                                            sigmoid(predictions_test).squeeze().tolist()),
                                        columns=["Validation", "Predicted"])
            result = TestResult(
                alpha=alpha,
                genes=genes,
                n_epochs=epoch,
                lrs=[float(x[0]) for x in learning_rates],
                losses_train=losses_train,
                loss_test=None,
                name=train.name,
                classes_train=classes_train,
                classes_test=classes_test,
                network=network if save_network else None,
                classify_loss_train = losses_train[-1],
                classify_loss_test = loss_test,
                first_weights=first_weights
                )
            return result

        predictions_train = predictions_train.cpu().numpy()
        predictions_test = predictions_test.cpu().numpy()
        baseline_hazards = compute_baseline_hazards(train.outcome_target_table,
                                                    predictions_train)
        survival_train = predict_survival_table(predictions_train,
                                                baseline_hazards.cumulative)
        survival_test = predict_survival_table(predictions_test,
                                               baseline_hazards.cumulative)
        if hazard_classify:
            classes_train = classify_by_hazard_at_threshold(survival_train,
                                                            train.class_threshold)
            classes_test = classify_by_hazard_at_threshold(survival_test,
                                                           train.class_threshold)
            classify_loss_train = bce_loss(torch.tensor(classes_train, dtype=torch.float32, device=DEVICE).view(-1, 1),
                                           train.classes)
            classify_loss_test = bce_loss(torch.tensor(classes_test, dtype=torch.float32, device=DEVICE).view(-1, 1),
                                         test.classes)
        else:
            classes_train, classes_test = None, None
            classify_loss_train, classify_loss_test = None, None

        ev = EvalSurv(survival_test,
                      test.durations.cpu().numpy(),
                      test.events.cpu().numpy(),
                      censor_surv="km")
        ctd = ev.concordance_td()

        time_grid = np.linspace(float(test.durations.min()),
                                float(test.durations.max()),
                                100)
        brier_scores = ev.brier_score(time_grid)
        ibs = (simpson(y=brier_scores.values, x=brier_scores.index)
               / (brier_scores.index[-1] - brier_scores.index[0]))

        # ev.integrated_nbll(time_grid)

        km_df = train.outcome_target_table
        km_df = km_df.assign(risk=predictions_train)

        print("    Calculating survival split...")
        optimal_splits = optimize_survival_splits(km_df, n_groups=survival_splits,
                                                  method="Brute")
        risk_splits = np.cumulative_sum(optimal_splits)
        km_test_df = test.outcome_target_table
        km_test_df["risk"] = predictions_test
        groups = ascii_uppercase[:len(risk_splits) + 1]
        km_test_df["group"] = groups[0]
        for group, cutoff in zip(groups[1:], risk_splits):
            km_test_df.loc[km_test_df["risk"] > cutoff, "group"] = group
        logranks = iterate_logrank_tests(km_test_df)

        results = TestResult(
            alpha=alpha,
            genes=genes,
            n_epochs=epoch,
            lrs=[float(x[0]) for x in learning_rates],
            losses_train=losses_train,
            loss_test=loss_test,
            pll_train=partial_log_likelihood(train.outcome_target_table,
                                             predictions_train),
            pll_test=partial_log_likelihood(test.outcome_target_table,
                                            predictions_test),
            network=network if save_network else None,
            name=train.name,
            hazards_train=predictions_train,
            hazards_test=predictions_test,
            hazards_baseline=baseline_hazards,
            ibs=ibs,
            ctd=ctd,
            risk_splits=risk_splits,
            logranks=logranks,
            survival_train=survival_train,
            survival_test=survival_test,
            classes_train=classes_train,
            classes_test=classes_test,
            classify_loss_train=classify_loss_train,
            classify_loss_test=classify_loss_test,
            first_weights=first_weights
            )
        return results


def test_multiple(datasets: list[tuple[Callable, NetworkDataset]], alphas, **kwargs):
    results = []
    names = []
    for (network_type, (train, test)), alpha in zip(datasets, alphas):
        kwargs["coxnet_alpha"] = alpha
        kwargs["network_l1_alpha"] = alpha
        names.append(train.name)
        sub_results = test_run(train=train, test=test, network_type=network_type,
                               **kwargs)
        results.append(sub_results)
    print("Collating results...")
    results = TestResultsCollection(results, names)
    return results
