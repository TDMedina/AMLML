
from datetime import datetime
import os
import pickle

from amlml.evaluate_test import test_multiple
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
            prepare_log2_expression,
            # prepare_zscore_expression,
            # prepare_npn_expression,
            # prepare_supermodel_expression,
            # prepare_zupermodel_expression
            ],
        verbose=True),
    alphas=[0.087346],
    remove_age_over=None,
    restrict_tech=None,
    include_clinical_variables=True,
    covariate_cardinality={"race": 7, "ethnicity": 3, "protocol": 7},

    # Regularization.
    use_coxnet_alpha=True,
    coxnet_alpha=[0.087346],
    qnorm_coxnet=False,

    network_l1_reg=False,
    network_l1_alpha=0.05,
    network_weight_decay=1e-4,

    # Training.
    cov_threshold=0.00105,
    rel_slope_threshold=0.00105,
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
    use_shallow=False,
    minimum_penultimate_size=4,
    shrinkage_factor=10,
    kaiming_weights=True,

    # Classifier model.
    classify=False,
    hazard_classify=True,
    classification_threshold=None,
    use_rmst=True, rmst_max_time=None, rmst_tukey_factor=None,

    # Outputs.
    save_network=False,
    test_by_tech=None
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

test_results = test_multiple(**args)
test_results.parameters = str(args)
table = test_results.tabulate()
aggregate = test_results.make_agg_table()


# TODO:
# today = datetime.today().strftime("%Y-%b-%d")
# depth = "shallow" if args["use_shallow"] else "deep"
# rmst = "with" if args["use_rmst"] else "without"
# clinical = "with" if args["include_clinical_variables"] else "without"
# l1_reg = "network_l1" if args["network_l1_reg"] else "coxnet"
# tech = f".{args["test_by_tech"]}" if args["test_by_tech"] is not None else ""
# qnorm = "with" if args["qnorm_coxnet"] else "without"
#
# outname = f"test.results_hazard.{depth}.{clinical}_clinical.{rmst}_rmst.{l1_reg}_{qnorm}{tech}"
# outpath = f"./Data/{today}/{outname}/"
# os.makedirs(outpath)
# with open(f"{outpath}/{outname}.pickle", "wb") as outfile:
#     pickle.dump(test_results, outfile)
# with open(f"{outpath}/{outname}.args", "w") as outfile:
#     outfile.write(str(args) + "\n")
# table.to_csv(f"{outpath}/{outname}.tsv", sep="\t")
# aggregate.to_csv(f"{outpath}/{outname}.agg.tsv", sep="\t")

