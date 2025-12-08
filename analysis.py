
from amlml.cross_norm_survival import cv_multiple
from amlml.data_loader import (
    normalization_generator,
    prepare_log2_expression,
    prepare_zscore_expression,
    prepare_npn_expression,
    prepare_supermodel_expression,
    prepare_zupermodel_expression
    )

cv_results = cv_multiple(

    # Input data.
    datasets=normalization_generator(
        methods=[
            prepare_log2_expression,
            # prepare_zscore_expression,
            # prepare_npn_expression,
            # prepare_supermodel_expression,
            # prepare_zupermodel_expression
            ],
        verbose=True),
    remove_age_over=None,
    restrict_tech=None,
    include_clinical_variables=True,
    covariate_cardinality={"race": 7, "ethnicity": 3, "protocol": 7},

    # Cross-validation.
    iterate_alphas=True,
    n_alphas=10,  # Default = 21
    # alphas=[0.03],
    alphas=None,
    alpha_min_ratio=1/64, # Default = 0.01, classify = 0.05
    l1_ratio=1,
    survival_splits=2,
    cv_splits=5,  # Default = 5

    # Training.
    cov_threshold=0.0001,  # Default = 0.01
    rel_slope_threshold=0.0001,  # Default = 0.01
    # batch_size=350,
    batch_size=2000,
    # epochs=360,
    epochs=2500,

    # Learning rate.
    lr_init=None,
    # lr_init=0.001,
    constant_lr=False,
    epochs_per_cycle=100,
    end_with_lr_cycle=False,
    lr_cycle_mode="triangular",

    # Model architecture.
    bellows_normalization=False,
    minimum_penultimate_size=10,
    shrinkage_factor=10,
    kaiming_weights=True,

    # Classifier model.
    hazard_classify=True,
    classify=False,
    classification_threshold=365*4,
    use_rmst=True, rmst_max_time=None, rmst_tukey_factor=None,

    # Outputs.
    save_network=False,
    )
