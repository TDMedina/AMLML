
from collections import namedtuple
import math
from string import ascii_uppercase
from typing import Callable, Literal

import numpy as np
from numpy.polynomial import Polynomial
import pandas as pd
from scipy.stats import variation
from scipy.integrate import simpson
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
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
from amlml.data_loader import NetworkDataset, tensify
from amlml.evaluation import (partial_log_likelihood, compute_baseline_hazards, predict_survival_table,
                              TestResult, TestResultCollection,
                              classify_by_hazard_at_threshold)


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


def generate_loss_function(classify: bool, **kwargs):
    loss_fn = BCEWithLogitsLoss(**kwargs) if classify else CoxPHLoss(**kwargs)
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


def preclassify_dataset(dataset, use_rmst, rmst_max_time=None, rmst_tukey_factor=None, classification_threshold=None,
                        filter_ambiguous=None):
    if use_rmst:
        dataset.estimate_durations_with_rmst(max_time=rmst_max_time, tukey_factor=rmst_tukey_factor)
        if classification_threshold is not None:
            dataset.classify_by_rmst_threshold(threshold=classification_threshold, save=True)
        else:
            dataset.classify_by_rmst_ci_interval(save=True)
        if filter_ambiguous is not None:
            dataset = dataset.filter_ambiguous(filter_ambiguous)
    else:
        dataset = dataset.filter_low_censorship_and_classify_by_duration(classification_threshold)
    print(f"Class balance (low/high): {dataset.class_balance():.2f}")
    return dataset


def prepare_gene_feature_selection(dataset, feature_selector: Literal["coxnet", "network_l1", None] = "coxnet",
                                   coxnet_alpha_min_ratio=None, coxnet_l1_ratio=1, coxnet_n_alphas=None, coxnet_alphas=None,
                                   network_l1_alphas=None
                                   ):
    if feature_selector == "coxnet":
        print(f"    Calculating lasso penalties...")
        alpha_table = test_lasso_penalties(dataset.z_expression, dataset.outcomes, l1_ratio=coxnet_l1_ratio,
                                           alpha_min_ratio=coxnet_alpha_min_ratio,
                                           n_alphas=coxnet_n_alphas, alphas=coxnet_alphas)
        geneset = get_non_zero_genes(alpha_table)
    elif feature_selector == "network_l1":
        geneset = {alpha: (slice(None), list(dataset.genes)) for alpha in network_l1_alphas}
    else:
        geneset = {0: (slice(None), list(dataset.genes))}
    return geneset


def initialize_optimizer(network, dataset, batch_size, epochs_per_cycle, lr_cycle_mode,
                         constant_lr=False, lr_init=None, network_weight_decay=1e-4):
    # Start by using PyCox's lr_finder.
    if lr_init is None:
        print("    Calculating learning rate...")
        model = CoxPH(network, tt.optim.Adam)
        lrfinder = model.lr_finder(dataset.network_args, dataset.outcomes, batch_size,)  # tolerance=10, verbose=verbose)
        lr_init = lrfinder.get_best_lr(lr_min=1e-4, lr_max=1e-2) / 10
        del model
        print(f"    Calculated learning rate = {lr_init}")

    # Continue with base pytorch components to use more advanced LR tools.
    print("    Initializing optimizer...")
    optimizer = torch.optim.Adam(network.parameters(), lr=lr_init, weight_decay=network_weight_decay)
    if constant_lr:
        lr_scheduler = ConstantLR(optimizer, factor=1)
    else:
        steps = lr_cyclic_step_calculator(len(dataset), batch_size, epochs_per_cycle)
        steps = dict(zip(["step_size_up", "step_size_down"], steps))
        lr_scheduler = CyclicLR(optimizer, base_lr=lr_init / 10, max_lr=lr_init * 10,
                                mode=lr_cycle_mode, **steps, scale_mode="iterations")
    return optimizer, lr_scheduler


def train_network(network, optimizer, lr_scheduler, loss_fn, train, test,
                  epochs, min_epochs, epochs_per_cycle,
                  convergence_test, len_loss_convergence, end_with_lr_cycle,
                  batch_size, network_l1_alpha=None):
    losses_train, losses_test, learning_rates = [], [], []
    progress = tqdm(range(1, epochs + 1), desc="Epochs",
                    postfix={"Loss": "0", "Test Loss": "0", "Var": "--", "Slope": "--"},
                    dynamic_ncols=True, position=0, leave=True)
    epoch = 0
    for epoch in progress:
        epoch_loss = torch.tensor(0.0, device=DEVICE)
        for batch in train.generate_batches(batch_size, shuffle_=True):
            learning_rates.append(lr_scheduler.get_last_lr())
            optimizer.zero_grad()
            predictions_train = network(*batch.network_args)
            loss = loss_fn(predictions_train, batch)
            epoch_loss += loss.item() * len(batch)
            if network_l1_alpha is not None:
                loss = loss + network_l1_alpha * network.connected_layers.layers[0].weight.abs().sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
            optimizer.step()
            lr_scheduler.step()
        epoch_loss = (epoch_loss / len(train)).item()
        losses_train.append(epoch_loss)

        with torch.no_grad():
            # Calculate test loss.
            predictions_test = network(*test.network_args)
            loss_test = loss_fn(predictions_test, test).item()
            losses_test.append(loss_test)

            # Track epoch loss and check loss convergence.
            if len(losses_train) < len_loss_convergence:
                progress.set_postfix({"Loss": f"{losses_train[-1]:.3f}", "Test Loss:": f"{losses_test[-1]:.3f}",
                                      "Var": "--", "Slope": "--"})
                continue
            converge_check = convergence_test(losses_train[-len_loss_convergence:])
            progress.set_postfix({"Loss": f"{losses_train[-1]:.3f}", "Test Loss": f"{losses_test[-1]:.3f}",
                                  "Var": f"{converge_check.score[0]:.4f}", "Slope": f"{converge_check.score[1]:.4f}"})
            if (converge_check.passed and epoch >= min_epochs
                    and (epoch % epochs_per_cycle == 0 or not end_with_lr_cycle)):
                break

    return losses_train, losses_test, learning_rates, epoch


def evaluate_classifier(network, metrics, train, test,
                        fold, alpha, alpha_index, genes,
                        save_network=False):
    losses_train, losses_test, learning_rates, epoch = metrics

    network.eval()
    with torch.no_grad():
        first_weights = network.connected_layers.layers[0].weight.cpu().numpy()

        classes = []
        for dataset in [train, test]:
            logits = network(*dataset.network_args)
            sigmoided = sigmoid(logits)
            class_table = pd.DataFrame(zip(dataset.events.squeeze().tolist(),
                                           dataset.classes.squeeze().tolist(),
                                           # sigmoid(network(*dataset.network_args)).squeeze().tolist()),
                                           logits.squeeze().tolist(),
                                           sigmoided.squeeze().tolist()),
                                       columns=["Event", "Truth", "Raw", "Predicted"])
            class_table["Classification"] = (class_table.Predicted >= 0.5).astype(int)
            classes.append(class_table)
        # calibrate_predictions(*classes)

        result = TestResult(
            fold=fold,
            alpha=alpha,
            alpha_index=alpha_index,
            genes=genes,
            n_epochs=epoch,
            lrs=[float(x[0]) for x in learning_rates],
            losses_train=losses_train,
            losses_test=losses_test,
            name=train.name,
            classes_train=classes[0],
            classes_test=classes[1],
            network=network if save_network else None,
            classify_loss_train=losses_train[-1],
            classify_loss_test=losses_test[-1],
            first_weights=first_weights
            )
    return result


def evaluate_hazards(network, metrics, train, test,
                     fold, alpha, alpha_index, genes,
                     skip_diverged=True, hazard_classify=True, survival_splits=2,
                     save_network=False):
    losses_train, losses_test, learning_rates, epoch = metrics

    network.eval()
    with torch.no_grad():
        losses_train = [x.item() for x in losses_train]
        losses_test = [x.item() for x in losses_test]
        first_weights = network.connected_layers.layers[0].weight.cpu().numpy()

        predictions_train = network(*train.network_args).cpu().numpy()
        predictions_test = network(*test.network_args).cpu().numpy()

        if skip_diverged and losses_train[-1] > 50:
            print(f"Warning: model diverged (loss={losses_train[-1]}. Skipping evaluation.")
            result = TestResult(fold, alpha, alpha_index, genes, epoch, learning_rates, losses_train, losses_test,
                                hazards_train=predictions_train, hazards_test=predictions_test,
                                first_weights=first_weights)
            return result

        baseline_hazards = compute_baseline_hazards(train.outcome_target_table, predictions_train)
        survival_train = predict_survival_table(predictions_train, baseline_hazards.cumulative)
        survival_test = predict_survival_table(predictions_test, baseline_hazards.cumulative)

        # Evaluate classification by hazard.
        if hazard_classify:
            print("    Calculating hazard classes...")
            bce_loss = BCELoss()
            class_tables, class_losses = [], []
            for surv, dataset in [(survival_train, train), (survival_test, test)]:
                class_table = classify_by_hazard_at_threshold(surv, dataset.class_threshold)
                class_loss = bce_loss(tensify(class_table).view(-1, 1), dataset.classes).item()
                class_table = pd.DataFrame(zip(dataset.classes.squeeze().tolist(),
                                               class_table),
                                               columns=["Truth", "Predicted"])
                class_table["Classification"] = (class_table.Predicted >= 0.5).astype(int)
                class_tables.append(class_table)
                class_losses.append(class_loss)
        else:
            class_tables, class_losses = [None, None], [None, None]

        print("    Calculating concordance...")
        ev = EvalSurv(survival_test, test.durations.cpu().numpy(), test.events.cpu().numpy(), censor_surv="km")
        # ev.integrated_nbll(time_grid)
        time_grid = np.linspace(float(test.durations.min()), float(test.durations.max()), 100)
        brier_scores = ev.brier_score(time_grid)
        ibs = (simpson(y=brier_scores.values, x=brier_scores.index)
               / (brier_scores.index[-1] - brier_scores.index[0]))
        ctd = ev.concordance_td()

        print("    Calculating survival split...")
        # Train survival split.
        km_df = train.outcome_target_table
        km_df = km_df.assign(risk=predictions_train)
        optimal_splits = optimize_survival_splits(km_df, n_groups=survival_splits, method="Brute")
        risk_splits_q = np.cumulative_sum(optimal_splits)
        risk_splits = [km_df["risk"].quantile(x) for x in risk_splits_q]

        # Test survival split.
        km_val_df = test.outcome_target_table
        km_val_df["risk"] = predictions_test
        groups = ascii_uppercase[:len(risk_splits) + 1]
        km_val_df["group"] = groups[0]
        for group, cutoff in zip(groups[1:], risk_splits):
            km_val_df.loc[km_val_df["risk"] > cutoff, "group"] = group
        logranks = iterate_logrank_tests(km_val_df)

        result = TestResult(
            fold=fold,
            alpha=alpha,
            alpha_index=alpha_index,
            genes=genes,
            n_epochs=epoch,
            lrs=[float(x[0]) for x in learning_rates],
            losses_train=losses_train,
            losses_test=losses_test,
            pll_train=partial_log_likelihood(train.outcome_target_table, predictions_train),
            pll_test=partial_log_likelihood(test.outcome_target_table, predictions_test),
            network=network if save_network else None,
            name=dataset.name,
            hazards_train=predictions_train,
            hazards_test=predictions_test,
            hazards_baseline=baseline_hazards,
            ibs=ibs,
            ctd=ctd,
            risk_splits=risk_splits,
            risk_split_quantiles=risk_splits_q,
            risk_split_counts=km_val_df.group.value_counts(),
            logranks=logranks,
            survival_train=survival_train,
            survival_test=survival_test,
            classes_train=class_tables[0],
            classes_test=class_tables[1],
            classify_loss_train=class_losses[0],
            classify_loss_test=class_losses[1],
            first_weights=first_weights,
            km_table_train=km_df,
            km_table_test=km_val_df
            )
        return result


# %% CV run.
def cross_validation_run(#dataset: NetworkDataset,
                         train: NetworkDataset, test: NetworkDataset = None,
                         network_type: Callable = CrossNormalizedModel,
                         cv_splits=5,
                         include_clinical_variables=True, covariate_cardinality=None,
                         feature_selector: Literal["coxnet", "network_l1", None] = "coxnet",
                         coxnet_l1_ratio=1, coxnet_alpha_min_ratio=0.01, coxnet_n_alphas=20, coxnet_alphas=None, qnorm_coxnet=False,
                         network_l1_alphas=None, network_weight_decay=1e-4,
                         survival_splits=2, cov_threshold=0.01,
                         rel_slope_threshold=0.01,
                         batch_size=100, epochs=360, epochs_per_cycle=6, min_epochs=0,
                         save_network=False,
                         lr_init=None, constant_lr=False, end_with_lr_cycle=False,
                         lr_cycle_mode="triangular",
                         classify=False, hazard_classify=False,
                         use_rmst=True, classification_threshold=None,
                         minimum_penultimate_size=10, shrinkage_factor=10,
                         rmst_max_time=None, rmst_tukey_factor=None,
                         zero_params=False, kaiming_weights=False,
                         bellows_normalization=False, use_shallow=False,
                         # remove_age_over=None, restrict_tech=None, minimum_duration=None,
                         # filter_events=None,
                         filter_ambiguous=None,
                         # filter_minimum_censorship=None,
                         dropout=0.2, leakyrelu=0, skip_diverged=True,
                         _nullify_expression=False,
                         _debug_run=False
                         ):
    coxnet_n_alphas = coxnet_n_alphas + 1 if coxnet_n_alphas is not None else None
    network_l1_alphas = [0] if network_l1_alphas is None else network_l1_alphas

    len_loss_convergence = epochs_per_cycle // 2
    convergence_test = generate_loss_convergence_test(metric="cov", test_loss_slope=True, cov_threshold=cov_threshold,
                                                      rel_slope_threshold=rel_slope_threshold, n_losses=len_loss_convergence)
    classify_params = {"use_rmst": use_rmst, "rmst_max_time": rmst_max_time, "rmst_tukey_factor": rmst_tukey_factor,
                       "classification_threshold": classification_threshold, "filter_ambiguous": filter_ambiguous}
    l1_params = {"feature_selector": feature_selector, "coxnet_alpha_min_ratio": coxnet_alpha_min_ratio,
                 "coxnet_l1_ratio": coxnet_l1_ratio, "coxnet_n_alphas": coxnet_n_alphas, "coxnet_alphas": coxnet_alphas,
                 "network_l1_alphas": network_l1_alphas}
    network_params = {"shrinkage_factor": shrinkage_factor, "minimum_penultimate_size": minimum_penultimate_size, "final_size": 1,
                      "include_clinical_variables": include_clinical_variables, "n_clinical": train.n_clinical,
                      "covariate_cardinality": covariate_cardinality,
                      "embedding_dims": {"race": 3, "ethnicity": 3, "interaction": 3, "protocol": 3},
                      "zero_params": zero_params, "kaiming_weights": kaiming_weights, "output_xavier": classify,
                      "use_shallow": use_shallow, "dropout": dropout, "leakyrelu": leakyrelu}
    if network_type == SuperModel:
        network_params.update({"n_tech": 2, "n_expansion": 4, "bellows_normalization": bellows_normalization})
    lr_params = {"batch_size": batch_size, "epochs_per_cycle": epochs_per_cycle, "lr_cycle_mode": lr_cycle_mode,
                 "constant_lr": constant_lr, "lr_init": lr_init, "network_weight_decay": network_weight_decay}
    train_params = {"epochs": epochs, "min_epochs": min_epochs, "epochs_per_cycle": epochs_per_cycle, "batch_size": batch_size,
                    "convergence_test": convergence_test, "len_loss_convergence": len_loss_convergence, "end_with_lr_cycle": end_with_lr_cycle}

    if test is None:
        kf = (StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=10) if cv_splits == 1
              else StratifiedKFold(n_splits=cv_splits, random_state=10, shuffle=True))
        splits = kf.split(train.expression, train.tech)
        data = ((train[train_dex], train[val_dex]) for train_dex, val_dex in splits)
    else:
        data = ((train, test),)

    results = []
    print(f"Running dataset: {train.name}")
    for i, (train_fold, test_fold) in enumerate(data):
        if classify or hazard_classify:
            train_fold = preclassify_dataset(train_fold, **classify_params)
            test_fold = preclassify_dataset(test_fold, **classify_params)
        genesets = prepare_gene_feature_selection(train_fold, **l1_params)

        for j, (alpha, (index, genes)) in enumerate(genesets.items()):
            print(f"    Preparing training fold {i} of {train_fold.name} for alpha={alpha} ({len(genes)} genes)...")
            train_alpha = train_fold.subset_genes(index)
            test_alpha = test_fold.subset_genes(index)

            network_fold_params = network_params | {"n_genes": train_alpha.n_genes}
            network = network_type(**network_fold_params)
            network = network.to(DEVICE)

            optimizer, lr_scheduler = initialize_optimizer(network, train_alpha, **lr_params)

            loss_kwargs = {"pos_weight": tensify(train_alpha.class_balance())} if classify else {}
            loss_fn = generate_loss_function(classify, **loss_kwargs)

            if _nullify_expression:
                train_alpha.expression = train_alpha.expression * 0
            print("    Training...")
            metrics = train_network(network, optimizer, lr_scheduler, loss_fn, train_alpha, test_alpha,
                                    **train_params, network_l1_alpha=alpha if feature_selector == "network_l1" else None)
            if classify:
                result = evaluate_classifier(network, metrics, train_alpha, test_alpha,
                                             i, alpha, j, genes, save_network)
            else:
                result = evaluate_hazards(network, metrics, train_alpha, test_alpha,
                                          i, alpha, j, genes,
                                          skip_diverged, hazard_classify, survival_splits, save_network)
            del network
            del optimizer
            del lr_scheduler
            results.append(result)

    results = TestResultCollection(results, None, cv_splits)
    return results

















    if _debug_run:
        dataset = dataset._debug_set()

    print(f"Running dataset: {dataset.name}")
    # if cv_splits == 1:
    #     kf = StratifiedKFold(n_splits=cv_splits+1, random_state=10, shuffle=True)
    #     splits = kf.split(dataset.expression, dataset.tech)
    #     splits = list(splits)[0:1]
    # else:
    #     kf = StratifiedKFold(n_splits=cv_splits, random_state=10, shuffle=True)
    #     splits = kf.split(dataset.expression, dataset.tech)
    # # kf = RepeatedStratifiedKFold(n_splits=5, random_state=10, n_repeats=10)
    # results = []
    # for i, (train_fold, val_fold) in enumerate(splits):
    print(f"Running fold {i}...")
    fold_data = dataset[train_fold]
    fold_data_val = dataset[val_fold]

    genesets = prepare_gene_feature_selection(dataset, feature_selector, coxnet_alpha_min_ratio, coxnet_l1_ratio,
                                              coxnet_n_alphas, coxnet_alphas, network_l1_alphas)
    for j, (alpha, (index, genes)) in enumerate(genesets.items()):
        print(f"    Preparing training fold {i} of {dataset.name} for "
              f"alpha={alpha} ({len(genes)} genes)...")
        alpha_data = fold_data.subset_genes(index)
        alpha_val = fold_data_val.subset_genes(index)

        network_parameters = dict(
            n_genes=alpha_data.n_genes,
            shrinkage_factor=shrinkage_factor,
            minimum_penultimate_size=minimum_penultimate_size,
            final_size=1,
            include_clinical_variables=include_clinical_variables,
            n_clinical=alpha_data.n_clinical,
            covariate_cardinality=covariate_cardinality,
            embedding_dims={"race": 3, "ethnicity": 3,
                            "interaction": 3,
                            "protocol": 3},
            zero_params=zero_params,
            kaiming_weights=kaiming_weights,
            output_xavier=classify,
            use_shallow=use_shallow,
            dropout=dropout,
            leakyrelu=leakyrelu
        )
        if network_type == SuperModel:
            network_parameters.update(dict(
                n_tech=2,
                n_expansion=4,
                bellows_normalization=bellows_normalization
            ))

        network = network_type(**network_parameters)
        network = network.to(DEVICE)
        # network = torch.compile(network)

        # Start by using PyCox's lr_finder.
        if lr_init is not None:
            lr_init_ = lr_init
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
            lr_init_ = lrfinder.get_best_lr(lr_min=1e-4, lr_max=1e-2)/10
            del model
            print(f"    Calculated learning rate = {lr_init_}")

        # Continue with base pytorch components to use more advanced LR tools.
        print("    Initializing model...")
        optimizer = torch.optim.Adam(network.parameters(), lr=lr_init_, weight_decay=network_weight_decay)
        if constant_lr:
            lr_scheduler = ConstantLR(optimizer, factor=1)
        else:
            steps = lr_cyclic_step_calculator(len(alpha_data), batch_size, epochs_per_cycle)
            steps = dict(zip(["step_size_up", "step_size_down"], steps))
            lr_scheduler = CyclicLR(
                optimizer, base_lr=lr_init_/10, max_lr=lr_init_*10,
                mode=lr_cycle_mode, **steps, scale_mode="iterations",
            )

        if classify:
            pos_weight = dataset.class_balance()
            pos_weight = torch.tensor(pos_weight, dtype=torch.float32, device=DEVICE)
            loss_kwargs = {"pos_weight": pos_weight}
        else:
            loss_kwargs = {}
        run_loss = generate_loss_function(classify, **loss_kwargs)
        losses_train = []
        losses_val = []
        learning_rates = []
        print("    Training...")
        progress = tqdm(range(1, epochs+1), desc="Epochs",
                        postfix={"Loss": "0", "Var": "--", "Slope": "--"},
                        dynamic_ncols=True, position=0, leave=True)
        if _nullify_expression:
            alpha_data.expression = alpha_data.expression * 0
        for epoch in progress:
            epoch_loss = torch.tensor(0.0, device=DEVICE)
            for batch in alpha_data.generate_batches(batch_size, shuffle_=True):
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
            epoch_loss = (epoch_loss / len(alpha_data))
            losses_train.append(epoch_loss)

            # Calculate validation loss.
            with torch.no_grad():
                predictions_val = network(*alpha_val.network_args)
                loss_val = run_loss(predictions_val, alpha_val)
                losses_val.append(loss_val)

                # Track epoch loss and check loss convergence.
                if len(losses_train) >= len_loss_convergence:
                    converge_check = convergence_test(torch.stack(losses_train[-len_loss_convergence:]).cpu().numpy())
                    progress.set_postfix({"Loss": f"{losses_train[-1]:.3f}",
                                          "Val Loss": f"{losses_val[-1]:.3f}",
                                          "Var": f"{converge_check.score[0]:.4f}",
                                          "Slope": f"{converge_check.score[1]:.4f}"})
                    if (converge_check.passed
                            and (epoch % epochs_per_cycle == 0 or not end_with_lr_cycle)
                            and epoch >= min_epochs):
                        break
                else:
                    progress.set_postfix({"Loss": f"{losses_train[-1]:.3f}",
                                          "Val Loss:": f"{losses_val[-1]:.3f}",
                                          "Var": "--",
                                          "Slope": "--"})

        # Evaluate.
        network.eval()
        with torch.no_grad():
            losses_train = [x.item() for x in losses_train]
            losses_val = [x.item() for x in losses_val]
            first_weights = network.connected_layers.layers[0].weight.cpu().numpy()
            predictions_train = network(*alpha_data.network_args)
            predictions_val = network(*alpha_val.network_args)

            if classify:
                classes_train = pd.DataFrame(zip(alpha_data.events.squeeze().tolist(),
                                                 alpha_data.classes.squeeze().tolist(),
                                                 sigmoid(predictions_train).squeeze().tolist()),
                                             columns=["Event", "Training", "Predicted"])
                classes_train["Classification"] = (classes_train.Predicted >= 0.5).astype(int)
                classes_val = pd.DataFrame(zip(alpha_val.events.squeeze().tolist(),
                                               alpha_val.classes.squeeze().tolist(),
                                               sigmoid(predictions_val).squeeze().tolist()),
                                           columns=["Event", "Test", "Predicted"])
                classes_val["Classification"] = (classes_val.Predicted >= 0.5).astype(int)
                results.append(TestResult(
                    fold=i,
                    alpha=alpha,
                    alpha_index=j,
                    genes=genes,
                    n_epochs=epoch,
                    lrs=[float(x[0]) for x in learning_rates],
                    losses_train=losses_train,
                    losses_test=losses_val,
                    name=dataset.name,
                    classes_train=classes_train,
                    classes_test=classes_val,
                    network=network if save_network else None,
                    classify_loss_train = losses_train[-1],
                    classify_loss_test = losses_val[-1],
                    first_weights=first_weights
                ))
                continue

            predictions_train = predictions_train.cpu().numpy()
            predictions_val = predictions_val.cpu().numpy()

            skipping = skip_diverged and losses_train[-1] > 50
            if skipping:
                baseline_hazards = "Diverged"
                survival_train = "Diverged"
                survival_val = "Diverged"
            else:
                baseline_hazards = compute_baseline_hazards(alpha_data.outcome_target_table,
                                                            predictions_train)
                survival_train = predict_survival_table(predictions_train,
                                                        baseline_hazards.cumulative)
                survival_val = predict_survival_table(predictions_val,
                                                      baseline_hazards.cumulative)
            if hazard_classify and not skipping:
                classes_train = classify_by_hazard_at_threshold(survival_train,
                                                                dataset.class_threshold)
                classes_val = classify_by_hazard_at_threshold(survival_val,
                                                              dataset.class_threshold)
                classify_loss_train = bce_loss(torch.tensor(classes_train, dtype=torch.float32, device=DEVICE).view(-1, 1),
                                               alpha_data.classes).item()
                classify_loss_val = bce_loss(torch.tensor(classes_val, dtype=torch.float32, device=DEVICE).view(-1, 1),
                                             alpha_val.classes).item()
                classes_train = pd.DataFrame(zip(alpha_data.classes.squeeze().tolist(),
                                                 classes_train),
                                             columns=["Training", "Predicted"])
                classes_train["Classification"] = (classes_train.Predicted >= 0.5).astype(int)
                classes_val = pd.DataFrame(zip(alpha_val.classes.squeeze().tolist(),
                                               classes_val),
                                           columns=["Test", "Predicted"])
                classes_val["Classification"] = (classes_val.Predicted >= 0.5).astype(int)
            else:
                classes_train, classes_val = None, None
                classify_loss_train, classify_loss_val = None, None

            ev = EvalSurv(survival_val,
                          alpha_val.durations.cpu().numpy(),
                          alpha_val.events.cpu().numpy(),
                          censor_surv="km")
            ctd = ev.concordance_td()

            time_grid = np.linspace(float(alpha_val.durations.min()),
                                    float(alpha_val.durations.max()),
                                    100)
            brier_scores = ev.brier_score(time_grid)
            ibs = (simpson(y=brier_scores.values, x=brier_scores.index)
                   / (brier_scores.index[-1] - brier_scores.index[0]))

            # ev.integrated_nbll(time_grid)

            km_df = alpha_data.outcome_target_table
            km_df = km_df.assign(risk=predictions_train)

            print("    Calculating survival split...")
            optimal_splits = optimize_survival_splits(km_df, n_groups=survival_splits,
                                                      method="Brute")
            risk_splits_q = np.cumulative_sum(optimal_splits)
            risk_splits = [km_df["risk"].quantile(x) for x in risk_splits_q]
            km_val_df = alpha_val.outcome_target_table
            km_val_df["risk"] = predictions_val
            groups = ascii_uppercase[:len(risk_splits) + 1]
            km_val_df["group"] = groups[0]
            for group, cutoff in zip(groups[1:], risk_splits):
                km_val_df.loc[km_val_df["risk"] > cutoff, "group"] = group
            logranks = iterate_logrank_tests(km_val_df)

            results.append(TestResult(
                fold=i,
                alpha=alpha,
                alpha_index=j,
                genes=genes,
                n_epochs=epoch,
                lrs=[float(x[0]) for x in learning_rates],
                losses_train=losses_train,
                losses_test=losses_val,
                pll_train=partial_log_likelihood(alpha_data.outcome_target_table,
                                                 predictions_train),
                pll_test=partial_log_likelihood(alpha_val.outcome_target_table,
                                                predictions_val),
                network=network if save_network else None,
                name=dataset.name,
                hazards_train=predictions_train,
                hazards_test=predictions_val,
                hazards_baseline=baseline_hazards,
                ibs=ibs,
                ctd=ctd,
                risk_splits=risk_splits,
                risk_split_quantiles=risk_splits_q,
                risk_split_counts=km_val_df.group.value_counts(),
                logranks=logranks,
                survival_train=survival_train,
                survival_test=survival_val,
                classes_train=classes_train,
                classes_test=classes_val,
                classify_loss_train=classify_loss_train,
                classify_loss_test=classify_loss_val,
                first_weights=first_weights,
                km_table_train=km_df,
                km_table_test=km_val_df
            ))
    results = TestResultCollection(results, None, cv_splits)
    return results


def run_multiple(datasets: list[tuple[Callable, NetworkDataset]], cv=True, **kwargs):
    results = []
    names = []
    for network_type, (train, test) in datasets:
        if cv:
            test = None
        else:
            kwargs["cv_splits"] = 1
        names.append(train.name)
        sub_results = cross_validation_run(train=train, test=test,
                                           network_type=network_type,
                                           **kwargs)
        results.extend(sub_results.results)
    print("Collating results...")
    results = TestResultCollection(results, names, kwargs["cv_splits"])
    return results


def test_multiple(datasets: list[tuple[Callable, NetworkDataset]], alphas, **kwargs):
    results = []
    names = []
    for (network_type, (train, test)), alpha in zip(datasets, alphas):
        kwargs["coxnet_alpha"] = alpha
        kwargs["network_l1_alpha"] = alpha
        names.append(train.name)
        sub_results = cross_validation_run(train=train, test=test, network_type=network_type, **kwargs)
        results.append(sub_results)
    print("Collating results...")
    results = TestResultCollection(results, names)
    return results

# # %% Results parsing.
# def make_results_table(results):
#     fields = ["epochs", "pll", "ctd", "ibs"]
#     parsed = []
#     for fold, alpha_dict in results.items():
#         for alpha, result_dict in alpha_dict.items():
#             entry = ([fold, alpha, len(result_dict["genes"].split(","))]
#                      + [result_dict[x] for x in fields])
#             entry += [tuple(sorted(result_dict["risk_splits"]))]
#             # print(fold, alpha, result_dict["km_logrank"].keys())
#             entry += [result_dict["km_logrank"][tuple(x)].p_value
#                       if tuple(x) in result_dict["km_logrank"] else None
#                       for x in ["AB", "BC", "AC"]]
#             parsed.append(entry)
#     cols = ["fold", "alpha", "gene_count", *fields, "risk_splits", "km_logrank_AB",
#             "km_logrank_BC", "km_logrank_AC"]
#     table = pd.DataFrame(parsed, columns=cols)
#     table["group"] = [i for _ in range(len(results.keys()))
#                       for i in ascii_uppercase[:len(list(results.values())[0].keys())]]
#     table.set_index(["fold", "group", "alpha"], inplace=True)
#     return table
#
#
# def plot_survival_splits(results_table):
#     splits = results_table[["gene_count"]].copy()
#     splits["split1"] = [x[0] for x in results_table.risk_splits]
#     splits["split2"] = [x[1] for x in results_table.risk_splits]
#     fig = go.Figure()
#     for fold in splits.index.unique(level="fold"):
#         data = splits.loc[idx[fold, :, :],]
#         fig.add_trace(go.Bar(x=data.reset_index()["group"],
#                              y=data.split1,
#                              offsetgroup=fold,
#                              marker_color="red",
#                              marker_line_color="red"))
#         fig.add_trace(go.Bar(x=data.reset_index()["group"],
#                              y=data.split2-data.split1,
#                              offsetgroup=fold,
#                              marker_color="blue",
#                              marker_line_color="blue"))
#         fig.add_trace(go.Bar(x=data.reset_index()["group"],
#                              y=1-data.split2,
#                              offsetgroup=fold,
#                              marker_color="green",
#                              marker_line_color="green"))
#     fig.update_layout(barmode="stack")
#
#     logranks = ["km_logrank_AB", "km_logrank_BC", "km_logrank_AC"]
#     splits = splits.join(results_table[logranks])
#     fig3 = go.Figure()
#     for logrank in logranks:
#         fig3.add_trace(go.Scatter(x=splits.gene_count, y=splits[logrank],
#                                   mode="markers"))
#     melted = pd.melt(splits[["gene_count"]+logranks], id_vars="gene_count")
#     melted["value"] = np.log10(melted["value"])
#     fig4 = px.scatter(melted, x="gene_count", y="value", color="variable")
#     fig4.add_trace(go.Scatter(x=[0, 1600], y=[np.log10(0.05)]*2, mode="lines",
#                               marker_line_color="red"))
#     return fig, fig4
#
#
# def plot_performance_metrics(results_table, plot_mean=False, plot_median=False):
#     metrics = ["epochs", "pll", "ctd", "ibs"]
#     trimmed = results_table.copy()
#     trimmed["pll"] = [max(x, -20) for x in results_table["pll"]]
#     title = "Gene counts vs metric"
#     if plot_mean:
#         trimmed = make_mean_results_table(trimmed)
#         title += " means"
#     elif plot_median:
#         trimmed = make_median_results_table(trimmed)
#         title += " medians"
#     figs = px.scatter(pd.melt(trimmed[["gene_count"] + metrics], id_vars=["gene_count"]),
#                       x="gene_count", y="value", trendline="lowess",
#                       facet_col="variable", facet_col_wrap=2)
#     figs.update_layout(title=title)
#     figs.update_yaxes(matches=None)
#     figs.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
#     return figs
