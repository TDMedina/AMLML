
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

methods = [
    prepare_log2_expression,
    prepare_zscore_expression,
    prepare_npn_expression,
    # prepare_qn_expression,
    # prepare_qnz_expression,
    prepare_supermodel_expression,
    prepare_zupermodel_expression
]

args = dict(
    datasets=normalization_generator(methods=[methods], verbose=True),
    remove_age_over=None,
    restrict_tech=None,
    include_clinical_variables=False,  # Note: Reset this.
    covariate_cardinality={"race": 7, "ethnicity": 3, "protocol": 7},

    # Regularization.
    use_coxnet_alphas=True,
    coxnet_n_alphas=10,
    coxnet_alpha_min_ratio=1/64, # Default = 0.01, classify = 0.05
    coxnet_alphas=None,
    qnorm_coxnet=False,  # Note: New.

    network_l1_reg=False,
    network_l1_alphas=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
    network_weight_decay=1e-4,

    # Cross-Validation.
    cv_splits=5,  # Default = 5

    # Training.
    cov_threshold=0.00105,  # Default = 0.01
    rel_slope_threshold=0.00105,  # Default = 0.01
    # batch_size=350,
    batch_size=2000,
    # epochs=360,
    epochs=2500,
    dropout=0.2,

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
    shrinkage_factor=10,
    kaiming_weights=True,

    # Classifier model.
    classify=False,
    hazard_classify=True,
    use_rmst=True, rmst_max_time=None, rmst_tukey_factor=None,
    classification_threshold=None,

    # Outputs.
    save_network=False,
    skip_diverged=True,

    # Debugging
    _nullify_expression=False,
)

# methods=[
#     # prepare_log2_expression,
#     prepare_zscore_expression,
#     # prepare_npn_expression,
#     # prepare_supermodel_expression,
#     # prepare_zupermodel_expression
#     ]
# argsets = [
#     # dict(datasets=normalization_generator(methods), use_rmst=True, use_shallow=True),
#     # dict(datasets=normalization_generator(methods), use_rmst=True, use_shallow=False),
#     dict(datasets=normalization_generator(methods), use_rmst=False, use_shallow=True, network_l1_alpha=0.04),
#     # dict(datasets=normalization_generator(methods), use_rmst=False, use_shallow=False),
# ]
# argsets = list(product(
#     [False],  # use_rmst
#     [True],  # use_shallow
# ))
# argsets = [dict(zip(["use_rmst", "use_shallow", "network_l1_alpha"], argset)) for argset in argsets]
#
# all_results = []
# for argset in argsets:
#     args["datasets"] = normalization_generator(methods)
#     args.update(argset)

today = datetime.today().strftime("%Y-%b-%d")
iter_args = product([True, False], repeat=0)
for run_arg in iter_args:
    args["datasets"] = normalization_generator(methods, verbose=True)
    # args["include_clinical_variables"] = run_arg[0]
    # args["use_coxnet_alphas"] = run_arg[1]
    # args["network_l1_args"] = not run_arg[1]
    # args["use_shallow"] = run_arg[0]
    # args["use_rmst"] = run_arg[2]

    cv_results = cv_multiple(**args)
    cv_results.parameters = str(args)
    table = cv_results.tabulate()
    aggregate = cv_results.make_agg_table()

    clinical = "with" if args["include_clinical_variables"] else "without"
    l1_reg = "network_l1" if args["network_l1_reg"] else "coxnet"
    depth = "shallow" if args["use_shallow"] else "deep"
    rmst = "with" if args["use_rmst"] else "without"
    qnorm = "with" if args["qnorm_coxnet"] else "without"

    outname = f"results_hazard.{depth}.{clinical}_clinical.{rmst}_rmst.{l1_reg}_{qnorm}_qnorm"
    outpath = f"./Data/{today}/{outname}/"
    os.makedirs(outpath)
    with open(f"{outpath}/{outname}.pickle", "wb") as outfile:
        pickle.dump(cv_results, outfile)
    with open(f"{outpath}/{outname}.args", "w") as outfile:
        outfile.write(str(args) + "\n")
    table.to_csv(f"{outpath}/{outname}.tsv", sep="\t")
    aggregate.to_csv(f"{outpath}/{outname}.agg.tsv", sep="\t")
