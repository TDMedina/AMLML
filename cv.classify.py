
from datetime import datetime
from itertools import product
from pathlib import Path
import pickle

from amlml.analysis import run_multiple
from amlml.data_loader import (
    normalization_generator,
    prepare_log2_expression,
    prepare_zscore_expression,
    prepare_qn_expression,
    prepare_qnz_expression,
    prepare_npn_expression,
    prepare_supermodel_expression,
    prepare_superlogger_expression,
    prepare_zupermodel_expression
    )


args = dict(
    filter_ambiguous=30,
    include_clinical_variables=True,
    include_categoricals=True,
    covariate_cardinality={"race": 7, "ethnicity": 3, "protocol": 7},

    # Regularization.
    feature_selector="coxnet",

    coxnet_n_alphas=5,
    coxnet_alpha_min_ratio=1/32,
    coxnet_alphas=None,
    qnorm_coxnet=False,

    network_l1_alphas=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
    network_weight_decay=1e-4,

    # Cross-Validation.
    cv_splits=5,

    # Training.
    cov_threshold=1e-2,
    rel_slope_threshold=1e-3,
    batch_size=2000,
    epochs=250,
    min_epochs=10,
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
    # keep_clinical_variables=[
    #     "Age at Diagnosis in Days", "Bone marrow leukemic blast percentage (%)",
    #     "CEBPA mutation", "CNS disease", "MLL", "NPM mutation", "Peripheral blasts (%)",
    #     "WBC at Diagnosis", "inv(16)", "t(8;21)", "Race", "Protocol", "Ethnicity"
    #     ],
    keep_clinical_variables=[
        "Age at Diagnosis in Days", "Bone marrow leukemic blast percentage (%)",
        "CEBPA mutation", "CNS disease", "Chloroma", "FLT3/ITD positive?", "KMT2A-MLLT3",
        "MLL", "Minus X", "NPM mutation", "Peripheral blasts (%)", "WBC at Diagnosis",
        "WT1 mutation", "del9q", "inv(16)", "monosomy 7", "t(10;11)(p11.2;q23)", "t(6;9)",
        "t(8;21)", "trisomy 21", "trisomy 8",
        "Race", "Protocol", "Ethnicity"
        ],
    filter_duration=None,
    filter_age=None
    )

methods = [
    prepare_log2_expression,
    prepare_zscore_expression,
    prepare_npn_expression,
    # prepare_qn_expression,
    # prepare_qnz_expression,
    prepare_supermodel_expression,
    prepare_superlogger_expression,
    prepare_zupermodel_expression
    ]

iter_args = dict(
    # use_shallow=[True, False],
    # leakyrelu=[0, 0.1],
    # rmst_max_time=[2038, 5*365, 7*365],
    # classification_threshold=[3*365, 2038, 7*365]
    )

iter_args = [dict(zip(iter_args.keys(), iter_vals)) for iter_vals in product(*iter_args.values())]

today = datetime.today().strftime("%Y-%b-%d")
save = False

for run_args in iter_args:
    print("Args:", run_args)
    args["datasets"] = normalization_generator(methods, verbose=True, **prefilter_args)
    args.update(run_args)

    cv_results = run_multiple(**args)
    cv_results.parameters = str(args)
    table = cv_results.tabulate(classify=True)
    aggregate = cv_results.make_agg_table(classify=True)
    oof_tables = cv_results.make_oof_class_tables()

    if save:
        clinical = "with" if args["include_clinical_variables"] else "without"
        l1_reg = args["feature_selector"]
        depth = "shallow" if args["use_shallow"] else "deep"
        rmst = "with" if args["use_rmst"] else "without"
        qnorm = "with" if args["qnorm_coxnet"] else "without"
        thresh = f"{args["classification_threshold"]/365:.1f}"
        leaky = ".leaky" if args["leakyrelu"] > 0 else ""

        outname = f"cv_classify.{thresh}.{depth}.{clinical}_clinical.{rmst}_rmst.{l1_reg}_{qnorm}_qnorm{leaky}"
        outpath = f"./Data/{today}/{outname}/"
        oof_dir = f"{outpath}/oof_tables/"
        Path(oof_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{outpath}/{outname}.pickle", "wb") as outfile:
            pickle.dump(cv_results, outfile)
        with open(f"{outpath}/{outname}.args", "w") as outfile:
            outfile.write(str(args) + "\n")
        table.to_csv(f"{outpath}/{outname}.tsv", sep="\t")
        aggregate.to_csv(f"{outpath}/{outname}.agg.tsv", sep="\t")
        for (name, alpha_dex), oof_table in oof_tables.items():
            oof_path = f"{oof_dir}/{name}.alpha_{alpha_dex}.oof_table.tsv"
            oof_table.to_csv(oof_path, sep="\t", index=True)
