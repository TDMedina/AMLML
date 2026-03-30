
from datetime import datetime
from itertools import product
import os
import pickle

import numpy as np

from amlml.cross_norm_survival import cv_multiple
from amlml.data_loader import (
    normalization_generator,
    prepare_log2_expression,
    prepare_zscore_expression,
    prepare_qn_expression,
    prepare_qnz_expression,
    prepare_npn_expression,
    prepare_supermodel_expression,
    prepare_zupermodel_expression
    )


args = dict(
    filter_ambiguous=30,
    include_clinical_variables=False,
    covariate_cardinality={"race": 7, "ethnicity": 3, "protocol": 7},

    # Regularization.
    feature_selector="coxnet",

    coxnet_n_alphas=5,  # Note: Reset this.
    coxnet_alpha_min_ratio=1/32,  # Note: Reset this to 1/64.
    coxnet_alphas=None,
    qnorm_coxnet=False,  # Note: New.

    network_l1_alphas=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
    network_weight_decay=1e-4,

    # Cross-Validation.
    cv_splits=5,

    # Training.
    cov_threshold=0.00105,
    rel_slope_threshold=0.00105,
    # batch_size=350,
    batch_size=2000,
    # epochs=360,
    epochs=2500,
    min_epochs=1000,
    dropout=0.2,
    leakyrelu=0,

    # Learning rate.
    # lr_init=None,
    lr_init=0.001,
    constant_lr=False,
    epochs_per_cycle=100,
    end_with_lr_cycle=False,
    lr_cycle_mode="triangular",

    # Model architecture.
    bellows_normalization=False,
    use_shallow=True,
    minimum_penultimate_size=4,
    shrinkage_factor=2,
    kaiming_weights=True,

    # Classifier model.
    classify=False,
    hazard_classify=True,
    use_rmst=True, rmst_max_time=365, rmst_tukey_factor=None,
    classification_threshold=365,

    # Outputs.
    save_network=False,
    skip_diverged=True,

    # Debugging
    _nullify_expression=False,
    _debug_run=False
)

methods = [
    prepare_log2_expression,
    # prepare_zscore_expression,
    # prepare_npn_expression,
    # prepare_qn_expression,
    # prepare_qnz_expression,
    # prepare_supermodel_expression,
    # prepare_zupermodel_expression
]

prefilter_args = dict(
    keep_minimum_survival=1,
    keep_tech=None,
    keep_event=None,
    keep_minimum_censorship=30,
    filter_duration=None,
    filter_age=None
    )

iter_args = dict(
    # include_clinical_variables=[True, False],
    # use_shallow=[True, False],
    # leakyrelu=[0, 0.1],
    # rmst_max_time=[1*365, 2*365, 7*365],
    # classification_threshold=[3*365, 2038, 7*365]
    )

iter_args = [dict(zip(iter_args.keys(), iter_vals)) for iter_vals in product(*iter_args.values())]

today = datetime.today().strftime("%Y-%b-%d")
save = False

for run_args in iter_args:
    print("Args:", run_args)
    args["datasets"] = normalization_generator(methods, verbose=True, **prefilter_args)
    args.update(run_args)

    cv_results = cv_multiple(**args)
    cv_results.parameters = str(args)
    table = cv_results.tabulate()
    aggregate = cv_results.make_agg_table()

    if save:
        clinical = "with" if args["include_clinical_variables"] else "without"
        l1_reg = args["feature_selector"]
        depth = "shallow" if args["use_shallow"] else "deep"
        rmst = "with" if args["use_rmst"] else "without"
        qnorm = "with" if args["qnorm_coxnet"] else "without"
        thresh = f"{args["classification_threshold"]/365:.1f}"
        leaky = ".leaky" if args["leakyrelu"] > 0 else ""

        outname = f"results_hazard.{thresh}.{depth}.{clinical}_clinical.{rmst}_rmst.{l1_reg}_{qnorm}_qnorm{leaky}"
        outpath = f"./Data/{today}/{outname}/"
        os.makedirs(outpath)
        with open(f"{outpath}/{outname}.pickle", "wb") as outfile:
            pickle.dump(cv_results, outfile)
        with open(f"{outpath}/{outname}.args", "w") as outfile:
            outfile.write(str(args) + "\n")
        table.to_csv(f"{outpath}/{outname}.tsv", sep="\t")
        aggregate.to_csv(f"{outpath}/{outname}.agg.tsv", sep="\t")
