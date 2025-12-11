
import pandas as pd
import numpy as np
from numpy import nan
import torch
from torch import tensor, float32
import yaml

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

def read_clinical_data(data_file, column_yaml):
    cols = read_clinical_columns(column_yaml)
    data = pd.read_csv(data_file, sep="\t", index_col=[0, 1], header=0)
    col_names = [(key, val) for key, vals in cols.items() for val in vals]
    data = data[[col[1] for col in col_names]]
    data.columns = pd.MultiIndex.from_tuples(col_names)
    categoricals = [("Covariates", x) for x in cols["Covariates"]
                    if cols["Covariates"][x] == "categorical"]
    binaries = [("Covariates", x) for x in cols["Covariates"]
                if cols["Covariates"][x] == "binary"]
    for col in categoricals:
        data[col] = pd.Categorical(data[col])
    binary_vals = {"No": 0, False: 0, "Female": 0, "Yes": 1, True: 1, "Male": 1}
    for col in binaries:
        data[col] = [binary_vals[x] if x in binary_vals else nan for x in data[col]]
    return data, cols


def read_clinical_columns(col_yaml):
    with open(col_yaml) as infile:
        cols = yaml.safe_load(infile)
    return cols


def prepare_outcomes(outcome_data):
    outcome_data = tensor(outcome_data.T.astype(float), dtype=float32, device=DEVICE)
    outcome_data = (outcome_data[0], outcome_data[1])
    return outcome_data


def code_categoricals(data):
    for col in data:
        if data[col].dtype.name == "category":
            data[col] = data[col].cat.codes


def add_nan_mask_stack(data):
    mask_stack = pd.isna(data).astype(int)
    masked = data.copy()
    for col in masked:
        masked.loc[pd.isna(masked[col]), col] = 0
    masked = np.stack([masked, mask_stack])
    return masked
