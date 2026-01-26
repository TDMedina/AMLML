
from itertools import product
import os
import pickle

import numpy as np

from amlml.cross_norm_survival import cv_multiple
from amlml.data_loader import (
    normalization_generator,
    prepare_log2_expression,
    prepare_zscore_expression,
    prepare_npn_expression,
    prepare_supermodel_expression,
    prepare_zupermodel_expression
    )


args = dict(
    datasets=normalization_generator(
        methods=[
            # prepare_log2_expression,
            # prepare_zscore_expression,
            # prepare_npn_expression,
            prepare_supermodel_expression,
            # prepare_zupermodel_expression
            ],
        verbose=True),
    remove_age_over=None,
    restrict_tech=None,
    include_clinical_variables=True,
    covariate_cardinality={"race": 7, "ethnicity": 3, "protocol": 7},

    # Regularization.
    use_coxnet_alphas=False,
    coxnet_n_alphas=10,  # Default = 21
    coxnet_alpha_min_ratio=1/64, # Default = 0.01, classify = 0.05
    coxnet_alphas=None,
    network_l1_reg=True,
    network_l1_alphas=np.linspace(0.01, 0.1, 2),
    network_weight_decay=1e-4,

    # Cross-Validation.
    # TODO: Testing.
    cv_splits=1,  # Default = 5

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
    classification_threshold=365*4,
    use_rmst=False, rmst_max_time=None, rmst_tukey_factor=None,

    # Outputs.
    save_network=False,
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

cv_results = cv_multiple(**args)
cv_results.parameters = str(args)
table = cv_results.tabulate()
aggregate = cv_results.make_agg_table()

# depth = "shallow" if args["use_shallow"] else "deep"
# rmst = "with" if args["use_rmst"] else "without"
# clinical = "with" if args["include_clinical_variables"] else "without"
# l1_reg = "network_l1" if args["network_l1_reg"] else "coxnet"
#
# outname = f"results_hazard.{depth}.{clinical}_clinical.{rmst}_rmst.{l1_reg}"
#
# os.mkdir(f"./Data/{outname}/")
# with open(f"./Data/{outname}/{outname}.pickle", "wb") as outfile:
#     pickle.dump(cv_results, outfile)
# with open(f"./Data/{outname}/{outname}.args", "w") as outfile:
#     outfile.write(str(args) + "\n")
# table.to_csv(f"./Data/{outname}/{outname}.tsv")
# aggregate.to_csv(f"./Data/{outname}/{outname}.agg.tsv")
