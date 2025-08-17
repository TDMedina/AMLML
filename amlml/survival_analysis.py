
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
from scipy.integrate import simpson

import torch
from torch import float32, tensor
import torchtuples as tt

from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

from amlml.survival_data import (read_clinical_data, prepare_outcomes,
                                 code_categoricals, add_nan_mask_stack)
from amlml.rna_data import read_count_data
from amlml.parallel_modelling import CombinedParallelModel, SuperModel
from amlml.gene_set import GeneSet
from amlml.lasso import test_lasso_penalties, get_non_zero_genes
from amlml.km import plot_survival_curves, optimize_survival_splits, iterate_logrank_tests

l1_ratio = float(sys.argv[1])
torch.manual_seed(15370764774595565096)

def read_data(expression_data, clinical_data, clinical_yaml, geneset):
    expression = read_count_data(expression_data, geneset, calculate_tpm_=True)
    expression.columns = pd.MultiIndex.from_tuples([("Expression", x[0]) for x in expression.columns])
    clinical, cols = read_clinical_data(clinical_data, clinical_yaml)
    data = (clinical
            .reset_index("case_name")
            .join(expression, how="left")
            .set_index("case_name", append=True))
    data = data.droplevel(0, axis=0)
    return data, cols


def make_splits(data, seed):
    split_gen = np.random.default_rng(seed)
    train = split_gen.choice(range(data.shape[0]),
                             size=int(0.6*data.shape[0]),
                             replace=False)
    # train = split_gen.choice(range(data.shape[0]),
    #                          size=int(0.8*data.shape[0]),
    #                          replace=False)
    remainder = set(range(data.shape[0])) - set(train)
    val = split_gen.choice(list(remainder),
                           size=int(0.5*len(remainder)),
                           replace=False)
    # val = split_gen.choice(list(remainder),
    #                        size=0,
    #                        replace=False)
    test = np.array(list(remainder - set(val)))
    return train, val, test

def split(data, seed):
    if isinstance(data, pd.DataFrame):
        data = np.array(data)
    splits = make_splits(data, seed)
    train, val, test = [data[x] for x in splits]
    return train, val, test, splits

# %% Read data.
geneset = GeneSet("Homo_sapiens.GRCh38.113.chr.gtf.gz",
                  include=["gene", "transcript", "exon"])
data, cols = read_data(expression_data="unstranded_and_tpm.tsv",
                       clinical_data="TARGET/clinical.tsv",
                       clinical_yaml="Data/Clinical/variables.yaml",
                       geneset=geneset)
drop_list = ["TARGET-20-PAPXVK"]  # Filtered due to multiple listed events
data = data.drop(drop_list)
del geneset
# %% Prepare expression.

expression = data.Expression
# expression = np.zeros((1610, 19551))  # To test when expression is left as zero.
expression = np.stack([np.array(expression), np.zeros(expression.shape)], axis=0)
expression = tensor(expression, dtype=float32).permute(1, 0, 2)

# %% Prepare covariates.
categoricals = data["Covariates"][[x for x in cols["Covariates"]
                                   if cols["Covariates"][x] == "categorical"]]
code_categoricals(categoricals)
categoricals = tensor(categoricals.to_numpy(), dtype=torch.int32)

non_categoricals = data["Covariates"][[x for x in cols["Covariates"]
                                       if cols["Covariates"][x] != "categorical"]]
non_categoricals = tensor(add_nan_mask_stack(non_categoricals), dtype=float32)
non_categoricals = non_categoricals.permute(1, 0, 2)

# %% Prepare outcomes.
outcomes = data.Outcomes
outcomes.columns = ["event", "duration"]
outcomes = outcomes[["duration", "event"]]
outcomes.loc[:, "event"] = [int(x == "Dead") for x in outcomes.event]


# %% Split data into test, train, and validation.
groups = ["train", "validation", "test", "splits"]
split_to_dict = lambda x: dict(zip(groups, split(x, 123)))
expression = split_to_dict(expression)
categoricals = split_to_dict(categoricals)
non_categoricals = split_to_dict(non_categoricals)
outcomes = split_to_dict(outcomes)

outcomes["train"] = prepare_outcomes(outcomes["train"])
outcomes["validation"] = prepare_outcomes(outcomes["validation"])
outcomes["test"] = outcomes["test"].T.astype(float)


# %% Define network and model.
# network = SuperModel(n_genes=expression["train"].shape[-1],
#                      n_tech=2,
#                      n_expansion=4,
#                      n_clinical=36,
#                      covariate_cardinality={"race": 7, "ethnicity": 3, "protocol": 5},
#                      embedding_dims={"race": 3, "ethnicity": 3, "interaction": 3, "protocol": 3},
#                      shrinkage_factor=10,
#                      minimum_size=10,
#                      final_size=1)
#
# model = CoxPH(network, tt.optim.Adam)
#
# train_args = (expression["train"], categoricals["train"], non_categoricals["train"])
# validation_args = (expression["validation"], categoricals["validation"], non_categoricals["validation"])
# test_args = (expression["test"], categoricals["test"], non_categoricals["test"])
#
#
# # %% Run the model.
# batch_size = 161
# lrfinder = model.lr_finder(train_args, outcomes["train"], batch_size, tolerance=10)
# # lrfinder.plot()
# # plt.show()
#
# lr_optim = lrfinder.get_best_lr()
# model.optimizer.set_lr(0.0001)
#
# epochs = 60
# # callbacks = [tt.callbacks.EarlyStopping()]
# callbacks = []
# verbose = True
#
# log = model.fit(train_args, outcomes["train"], batch_size, epochs, callbacks, verbose,
#                 val_data=(validation_args, outcomes["validation"]),
#                 val_batch_size=batch_size)
# # log.plot()
# # plt.show()
#
#
# # %% Evaluate the model.
# model.partial_log_likelihood(validation_args, outcomes["validation"]).mean()
# model.compute_baseline_hazards()
#
# surv = model.predict_surv_df(test_args)
# # surv.plot()
# # plt.ylabel("S(t | x)")
# # plt.xlabel("Time")
# # plt.show()
#
# ev = EvalSurv(surv, outcomes["test"][0], outcomes["test"][1], censor_surv="km")
# ev.concordance_td()
#
# time_grid = np.linspace(outcomes["test"][0].min(), outcomes["test"].max(), 100)
# # ev.brier_score(time_grid).plot()
# # plt.show()
#
# brier_scores = ev.brier_score(time_grid)
# ibs = (simpson(y=brier_scores.values, x=brier_scores.index)
#        / (brier_scores.index[-1] - brier_scores.index[0]))
#
# # ev.integrated_nbll(time_grid)
#
# predictions = [float(x[0]) for x in model.predict(test_args)]
# km_test_df = pd.DataFrame(zip(*outcomes["test"], predictions),
#                           columns=["duration", "event", "risk"])
#
#
# # %% Survival partitioning.
# optimal_splits = optimize_survival_splits(km_test_df, n_groups=3)
# risk_splits = np.cumulative_sum(optimal_splits.x)
# plot_survival_curves(km_test_df)


# %% Do Elastic Net to test gene subsets.
coefficients = test_lasso_penalties(data.Expression.iloc[expression["splits"][0]],
                                    outcomes["train"], l1_ratio=l1_ratio)
non_zero_genes = get_non_zero_genes(coefficients)

aders5 = ["ENSG00000129187", "ENSG00000131747", "ENSG00000103222",
          "ENSG00000005381", "ENSG00000159228"]
plsc6 = ["ENSG00000088305", "ENSG00000205336", "ENSG00000174059",
         "ENSG00000120833", "ENSG00000128040"] #, "ENSG00000226777"]
lsc17 = ["ENSG00000088305", "ENSG00000130584", "ENSG00000205978", "ENSG00000128805",
         "ENSG00000104341", "ENSG00000138722", "ENSG00000113657", "ENSG00000105810",
         "ENSG00000088882", "ENSG00000120833", "ENSG00000095932", "ENSG00000134531",
         "ENSG00000174059", "ENSG00000196139", "ENSG00000205336", "ENSG00000166681",]
        # "ENSG00000226777"]
gene_id_dict = {'ENSG00000104341': 'ENSG00000104341.17',
                'ENSG00000159228': 'ENSG00000159228.13',
                'ENSG00000088305': 'ENSG00000088305.18',
                'ENSG00000120833': 'ENSG00000120833.14',
                'ENSG00000129187': 'ENSG00000129187.14',
                'ENSG00000131747': 'ENSG00000131747.15',
                'ENSG00000138722': 'ENSG00000138722.10',
                'ENSG00000166681': 'ENSG00000166681.14',
                'ENSG00000205336': 'ENSG00000205336.14',
                'ENSG00000128040': 'ENSG00000128040.11',
                'ENSG00000196139': 'ENSG00000196139.14',
                'ENSG00000088882': 'ENSG00000088882.8',
                'ENSG00000103222': 'ENSG00000103222.20',
                'ENSG00000113657': 'ENSG00000113657.13',
                'ENSG00000105810': 'ENSG00000105810.10',
                'ENSG00000095932': 'ENSG00000095932.7',
                'ENSG00000130584': 'ENSG00000130584.12',
                'ENSG00000005381': 'ENSG00000005381.8',
                'ENSG00000128805': 'ENSG00000128805.15',
                'ENSG00000134531': 'ENSG00000134531.10',
                'ENSG00000205978': 'ENSG00000205978.6',
                'ENSG00000174059': 'ENSG00000174059.17'}
aders5 = [gene_id_dict[x] for x in aders5]
plsc6 = [gene_id_dict[x] for x in plsc6]
lsc17 = [gene_id_dict[x] for x in lsc17]
non_zero_genes.update({"ADE-RS5": aders5, "pLSC6": plsc6, "LSC17": lsc17})

def test_network(expression_data, alpha):
    group_name = f"{alpha:.3f}" if isinstance(alpha, float) else alpha
    outdir = f"./training_test_{group_name}_{expression_data["train"].shape[-1]}genes/"
    os.mkdir(outdir)
    network = SuperModel(n_genes=expression_data["train"].shape[-1],
                         n_tech=2,
                         n_expansion=4,
                         n_clinical=36,
                         covariate_cardinality={"race": 7, "ethnicity": 3, "protocol": 5},
                         embedding_dims={"race": 3, "ethnicity": 3, "interaction": 3,
                                         "protocol": 3},
                         shrinkage_factor=10,
                         minimum_size=10,
                         final_size=1)

    model = CoxPH(network, tt.optim.Adam)

    train_args = (expression_data["train"], categoricals["train"], non_categoricals["train"])
    validation_args = (expression_data["validation"], categoricals["validation"],
                       non_categoricals["validation"])
    test_args = (expression_data["test"], categoricals["test"], non_categoricals["test"])

    batch_size = 161
    lrfinder = model.lr_finder(train_args, outcomes["train"], batch_size, tolerance=10)
    lrfinder.plot()
    plt.savefig(f"{outdir}/learning.png")
    plt.close()

    # lr_optim = lrfinder.get_best_lr()
    model.optimizer.set_lr(0.0001)

    epochs = 120
    callbacks = [tt.callbacks.EarlyStopping(patience=60)]
    verbose = True

    log = model.fit(train_args, outcomes["train"], batch_size, epochs, callbacks, verbose,
                    val_data=(validation_args, outcomes["validation"]),
                    val_batch_size=batch_size)
    log.plot()
    plt.savefig(f"{outdir}/fitness.png")
    plt.close()

    likelihood = model.partial_log_likelihood(validation_args, outcomes["validation"]).mean()
    model.compute_baseline_hazards()

    surv = model.predict_surv_df(test_args)
    # surv.plot()
    # plt.ylabel("S(t | x)")
    # plt.xlabel("Time")
    # plt.show()

    ev = EvalSurv(surv, outcomes["test"][0], outcomes["test"][1], censor_surv="km")
    concordance = ev.concordance_td()

    time_grid = np.linspace(outcomes["test"][0].min(), outcomes["test"].max(), 100)
    ev.brier_score(time_grid).plot()
    plt.savefig(f"{outdir}/brier_plot.png")
    plt.close()

    brier_scores = ev.brier_score(time_grid)
    ibs = (simpson(y=brier_scores.values, x=brier_scores.index)
           / (brier_scores.index[-1] - brier_scores.index[0]))

    # ev.integrated_nbll(time_grid)

    predictions = [float(x[0]) for x in model.predict(test_args)]
    km_test_df = pd.DataFrame(zip(*outcomes["test"], predictions),
                              columns=["duration", "event", "risk"])

    optimal_splits = optimize_survival_splits(km_test_df, n_groups=3)
    risk_splits = np.cumulative_sum(optimal_splits.x)

    kmf = KaplanMeierFitter()
    for group in sorted(km_test_df.group.unique()):
        group_data = km_test_df.loc[km_test_df.group == group]
        kmf.fit(group_data.duration, event_observed=group_data.event, label=group)
        kmf.plot_survival_function(ci_show=True)

    plt.title('Kaplan-Meier Survival Curves')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{outdir}/kaplan_meier.png")
    plt.close()

    logranks = iterate_logrank_tests(km_test_df)
    with open(f"{outdir}/results.txt", "w") as outfile:
        outfile.write(f"Likelihood:\t{likelihood}\n")
        outfile.write(f"Concordance:\t{concordance}\n")
        outfile.write(f"IBS:\t{ibs}\n")
        outfile.write(f"Risk Splits:\t{risk_splits}\n")
        for x, y in logranks.items():
            outfile.write(f"Logrank {x}:\t{y}\n\n")
    results = {"likelihood": likelihood,
               "concordance": concordance,
               "ibs": ibs,
               "risk_splits": risk_splits,
               "logranks": logranks}
    return results

all_results = dict()
for alpha, genes in tqdm({"ADE-RS5": aders5, "pLSC6": plsc6, "LSC17": lsc17}.items()):
# for alpha, genes in tqdm(non_zero_genes.items()):
    expression = data.Expression
    expression = expression.loc[:, list(genes)]
    expression = np.stack([np.array(expression), np.zeros(expression.shape)], axis=0)
    expression = tensor(expression, dtype=float32).permute(1, 0, 2)
    expression = split_to_dict(expression)
    results = test_network(expression, alpha)
    all_results[alpha] = results
    plt.close('all')
