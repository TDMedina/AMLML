
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
            # prepare_log2_expression,
            # prepare_zscore_expression,
            # prepare_npn_expression,
            prepare_supermodel_expression,
            # prepare_zupermodel_expression
            ],
        verbose=True),
    alphas=[0.05],
    remove_age_over=None,
    restrict_tech=None,
    include_clinical_variables=True,
    covariate_cardinality={"race": 7, "ethnicity": 3, "protocol": 7},

    # Regularization.
    use_coxnet_alpha=False,
    coxnet_alpha=None,
    network_l1_reg=True,
    network_l1_alpha=0.05,
    network_weight_decay=1e-4,

    # Training.
    cov_threshold=0.06,  # Default = 0.01
    rel_slope_threshold=0.06,  # Default = 0.01
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

test_results = test_multiple(**args)

test_results.parameters = str(args)
table = test_results.tabulate()
aggregate = test_results.make_agg_table()


# TODO:
depth = "shallow" if args["use_shallow"] else "deep"
rmst = "with" if args["use_rmst"] else "without"
clinical = "with" if args["include_clinical_variables"] else "without"
l1_reg = "network_l1" if args["network_l1_reg"] else "coxnet"

outname = f"test.results_hazard.{depth}.{clinical}_clinical.{rmst}_rmst.{l1_reg}"

os.mkdir(f"./Data/{outname}/")
with open(f"./Data/{outname}/{outname}.pickle", "wb") as outfile:
    pickle.dump(test_results, outfile)
with open(f"./Data/{outname}/{outname}.args", "w") as outfile:
    outfile.write(str(args) + "\n")
table.to_csv(f"./Data/{outname}/{outname}.tsv")
aggregate.to_csv(f"./Data/{outname}/{outname}.agg.tsv")
