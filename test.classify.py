
from datetime import datetime
import os
import pickle
from pathlib import Path

import pandas as pd

from amlml.analysis import run_multiple
from amlml.evaluation import calibrate_predictions
from amlml.data_loader import (normalization_generator, prepare_log2_expression,
                               prepare_zscore_expression, prepare_qn_expression,
                               prepare_qnz_expression, prepare_npn_expression,
                               prepare_supermodel_expression, prepare_zupermodel_expression)


args = dict(
    filter_ambiguous=30,
    include_clinical_variables=False,
    covariate_cardinality={"race": 7, "ethnicity": 3, "protocol": 7},

    # Regularization.
    feature_selector="coxnet",

    # Coxnet demands numerical values for some params even when not used.
    coxnet_n_alphas=1,
    coxnet_alpha_min_ratio=0.01,
    coxnet_alphas=None,  # Handled by arg iterator below.
    qnorm_coxnet=False,

    network_l1_alphas=None,
    network_weight_decay=1e-4,

    # Training.
    cov_threshold=1e-2,
    rel_slope_threshold=1e-3,
    # batch_size=350,
    batch_size=2000,
    # epochs=360,
    epochs=2500,
    min_epochs=1000,
    dropout=0.2,
    leakyrelu=0.1,

    # Learning rate.
    lr_init=0.001,
    constant_lr=False,
    epochs_per_cycle=100,
    end_with_lr_cycle=False,
    lr_cycle_mode="triangular",

    # Model architecture.
    bellows_normalization=False,
    use_shallow=False,
    minimum_penultimate_size=4,
    shrinkage_factor=2,
    kaiming_weights=True,

    # Classifier model.
    classify=True,
    hazard_classify=False,
    use_rmst=True, rmst_max_time=1*365, rmst_tukey_factor=None,
    classification_threshold=1*365,

    # Outputs.
    save_network=False,
    skip_diverged=True,

    # Debugging
    _nullify_expression=False,
    _debug_run=False
)

prefilter_args = dict(
    keep_minimum_survival=1,
    keep_tech=None,
    keep_event=None,
    keep_minimum_censorship=30,
    filter_duration=None,
    filter_age=None
    )

methods = [
    prepare_log2_expression,
    prepare_log2_expression,
    # prepare_zscore_expression,
    # prepare_npn_expression,
    # prepare_qn_expression,
    # prepare_qnz_expression,
    # prepare_supermodel_expression,
    # prepare_zupermodel_expression
]

iter_args = dict(
    methods=methods,
    coxnet_alphas=[[0.0805], [0.026554]],
    # include_clinical_variables=[False, False],
    use_shallow=[False, False],
    calibration_oof_path=[
        "/home/tyler/Documents/Projects/ML/Data/2026-Apr-01/results_classify.1.deep.without_clinical.with_rmst.coxnet_without_qnorm.leaky/oof_tables/LOG2.train.subset.ambiguity_filtered.gene_subset.alpha_0.oof_table.tsv",
        "/home/tyler/Documents/Projects/ML/Data/2026-Apr-01/results_classify.1.deep.without_clinical.with_rmst.coxnet_without_qnorm.leaky/oof_tables/LOG2.train.subset.ambiguity_filtered.gene_subset.alpha_2.oof_table.tsv"]
    )

iter_args = [dict(zip(iter_args.keys(), iter_vals)) for iter_vals in zip(*iter_args.values())]

today = datetime.today().strftime("%Y-%b-%d")
save = True

all_results = []
for run_args in iter_args:
    print("Args:", run_args)
    oof_path = run_args["calibration_oof_path"]
    del run_args["calibration_oof_path"]
    args["datasets"] = normalization_generator([run_args["methods"]],
                                               verbose=True,
                                               **prefilter_args)
    args.update(run_args)
    del args["methods"]

    cv_results = run_multiple(**args, cv=False)
    cv_results.parameters = str(args)
    all_results.append(cv_results)
    table = cv_results.tabulate(classify=True)
    if oof_path is not None:
        oof_table = pd.read_csv(oof_path, index_col=0, sep="\t")
        for result in cv_results.results:
            calibrated, calibrator = calibrate_predictions(oof_table, result.classes_test)
            result.classes_test["Calibrated"] = calibrated
            with open(f"{oof_path.replace('.tsv', '.calibrator.pkl')}", "wb") as outfile:
                pickle.dump(calibrator, outfile)

    if save:
        clinical = "with" if args["include_clinical_variables"] else "without"
        l1_reg = args["feature_selector"]
        depth = "shallow" if args["use_shallow"] else "deep"
        rmst = "with" if args["use_rmst"] else "without"
        qnorm = "with" if args["qnorm_coxnet"] else "without"
        thresh = f"{args["classification_threshold"]/365:.1f}"
        leaky = ".leaky" if args["leakyrelu"] > 0 else ""

        outname = f"test_classify.{thresh}.{depth}.{clinical}_clinical.{rmst}_rmst.{l1_reg}_{qnorm}_qnorm{leaky}"
        outpath = f"./Data/{today}/{outname}/"
        Path(outpath).mkdir(parents=True, exist_ok=True)
        with open(f"{outpath}/{outname}.pickle", "wb") as outfile:
            pickle.dump(cv_results, outfile)
        with open(f"{outpath}/{outname}.args", "w") as outfile:
            outfile.write(str(args) + "\n")
        table.to_csv(f"{outpath}/{outname}.tsv", sep="\t")
