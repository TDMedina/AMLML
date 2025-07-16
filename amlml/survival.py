
import numpy as np
import pandas as pd
import yaml

import matplotlib.pyplot as plt
from scipy.integrate import simpson

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch
from torch import float32, tensor
import torchtuples as tt

from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

from amlml.parallel_modelling import CombinedParallelModel

def read_combined_data(data_file, filter_max_expression=None):
    data = pd.read_csv(data_file, sep="\t", index_col=[0, 1], header=[0, 1])
    if filter_max_expression is not None:
        high_expression = data.Expression.iloc[:, 1:].median() > filter_max_expression
        high_expression = high_expression.loc[list(high_expression)].index
        high_expression = [("Expression", x) for x in high_expression]
        data = data.drop(high_expression, axis=1)
    return data


def read_clinical_columns(col_yaml):
    with open(col_yaml) as infile:
        cols = yaml.safe_load(infile)
    return cols


def make_splits(data, seed):
    split_gen = np.random.default_rng(seed)
    train = split_gen.choice(range(data.shape[0]),
                             size=int(0.6*data.shape[0]),
                             replace=False)
    remainder = set(range(data.shape[0])) - set(train)
    val = split_gen.choice(list(remainder),
                           size=int(0.5*len(remainder)),
                           replace=False)
    test = np.array(list(remainder - set(val)))
    # train, val, test = data[:, train, :], data[:, val, :], data[:, test, :]
    return train, val, test


def split(data, seed):
    if isinstance(data, pd.DataFrame):
        data = np.array(data)
    splits = make_splits(data, seed)
    train, val, test = [data[x] for x in splits]
    return train, val, test, splits


def prepare_outcomes(outcome_data):
    outcome_data = tensor(outcome_data.T.astype(float), dtype=float32)
    outcome_data = (outcome_data[0], outcome_data[1])
    return outcome_data


data = read_combined_data("/home/tyler/Repositories/AMLML2/new_combo.fix.data.tsv",
                          100)
clin_cols = read_clinical_columns("/home/tyler/ML/Data/Clinical/variables.yaml")
combo = data.droplevel("file_root", axis=0)
drop_list = ["TARGET-20-PAPXVK"]  # Filtered due to multiple listed events
combo = combo.drop(drop_list)

expression = combo.Expression
expression = np.stack([np.array(expression), np.zeros(expression.shape)], axis=0)
# Need to pre-permute the data to work with CoxPH wrapper, so that patients are in the
# first dimension for batching. This was previously handled by the training function that
# I wrote, which manually batched by looping through chunks of the second dimension.
expression = tensor(expression, dtype=float32).permute(1, 0, 2)

clinical_covariates = combo.Clinical.loc[:, clin_cols["Covariates"].keys()]
clinical_outcomes = combo.Clinical.loc[:, clin_cols["Outcomes"]]
clinical_outcomes.columns = ["event", "duration"]
clinical_outcomes = clinical_outcomes[["duration", "event"]]
clinical_outcomes.loc[:, "event"] = [int(x == "Alive") for x in clinical_outcomes.event]

exp_train, exp_val, exp_test, splits = split(expression, 123)
covar_train, covar_val, covar_test, _ = split(clinical_covariates, 123)
outcome_train, outcome_val, outcome_test, _ = split(clinical_outcomes, 123)

outcome_train, outcome_val = [prepare_outcomes(x) for x in [outcome_train, outcome_val]]
outcome_test = outcome_test.T.astype(float)

network = CombinedParallelModel(n_genes=exp_train.shape[2],
                                n_tech=2,
                                n_expansion=4,
                                shrinkage_factor=10,
                                minimum_size=10,
                                final_size=1)

model = CoxPH(network, tt.optim.Adam)

batch_size = 161
lrfinder = model.lr_finder(exp_train, outcome_train, batch_size, tolerance=10)
lrfinder.plot()

lr_optim = lrfinder.get_best_lr()
model.optimizer.set_lr(0.0001)

# We include the `EarlyStopping` callback to stop training when the validation loss
# stops improving. After training, this callback will also load the best performing
# model in terms of validation loss.
epochs = 60
# callbacks = [tt.callbacks.EarlyStopping()]
callbacks = []
verbose = True

# log = model.fit(exp_train, outcome_train, batch_size, epochs, callbacks, verbose,
#                 val_data=(exp_val, outcome_val), val_batch_size=batch_size)
log = model.fit(exp_train, outcome_train, batch_size, epochs, callbacks, verbose,
                val_data=(exp_val, outcome_val), val_batch_size=batch_size)
_ = log.plot()


model.partial_log_likelihood(exp_val, outcome_val).mean()
model.compute_baseline_hazards()

surv = model.predict_surv_df(exp_test)
surv.plot()
plt.ylabel("S(t | x)")
plt.xlabel("Time")


ev = EvalSurv(surv, outcome_test[0], outcome_test[1], censor_surv="km")
ev.concordance_td()

time_grid = np.linspace(outcome_test[0].min(), outcome_test.max(), 100)
ev.brier_score(time_grid).plot()

brier_scores = ev.brier_score(time_grid)
ibs = (simpson(y=brier_scores.values, x=brier_scores.index)
       / (brier_scores.index[-1] - brier_scores.index[0]))

# ev.integrated_nbll(time_grid)
