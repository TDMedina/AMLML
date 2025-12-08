
import numpy as np
from numpy import log2, random
import pandas as pd
from pandas import IndexSlice as idx
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from qnorm import quantile_normalize
from scipy.stats import zscore, norm


def by_sample(func):
    def flipper(*args, **kwargs):
        dataframe = func(*[arg.T for arg in args], **kwargs).T
        return dataframe
    return flipper


def log2_transform(data: pd.DataFrame, shift_sub1=False, is_log_transformed=False):
    if is_log_transformed and not shift_sub1:
        return data
    if is_log_transformed:
        data = 2**data
    if shift_sub1 and data.min().min() < 1:
        data = data+1
    data = log2(data)
    return data


def zscore_normalize(data):
    data = pd.DataFrame(zscore(data, ddof=1), columns=data.columns, index=data.index)
    return data


def zscore_normalize_genes_by_group(data):
    data = data.copy()
    data["int_dex"] = range(data.shape[0])
    data.set_index(["int_dex", "Tech"], append=True, inplace=True)
    techs = [zscore_normalize(data.loc[idx[:, :, tech]]) for tech in
             data.index.get_level_values("Tech").unique()]
        # rna = zscore_normalize(data.loc[idx[:, :, "RNAseq"]])
        # array = zscore_normalize(data.loc[idx[:, :, "Microarray"]])
    # data = pd.concat([rna, array])
    data = pd.concat(techs)
    data.sort_index(level=1, inplace=True)
    data = data.droplevel(level=1)
    return data


@by_sample
def npn(data):
    data = pd.DataFrame(norm.ppf(data.rank() / (data.shape[0]+1)),
                        columns=data.columns, index=data.index)
    data = data / data.std()
    # data = data / data.iloc[:, 0].std()
    return data


@by_sample
def quantile_normalize_2d(data, target):
    data = pd.concat([quantile_normalize(data[[sample]], target=target[sample])
                      for sample in data.columns], axis=1)
    return data


@by_sample
def tdm(data, target):
    target_quartiles = target.describe()
    data_quartiles = data.describe()

    iqr_ratio = ((data_quartiles.loc["75%"] - data_quartiles.loc["25%"])
                 / (target_quartiles.loc["75%"] - target_quartiles.loc["25%"]))

    target_arm = dict()
    target_arm["upper"] = target_quartiles.loc["max"] - target_quartiles.loc["75%"]
    target_arm["lower"] = target_quartiles.loc["25%"] - target_quartiles.loc["min"]

    rescale_arm = dict()
    rescale_arm["upper"] = data_quartiles.loc["75%"] + (target_arm["upper"]*iqr_ratio)
    rescale_arm["lower"] = data_quartiles.loc["25%"] - (target_arm["lower"]*iqr_ratio)
    rescale_arm["lower"] = rescale_arm["lower"].clip(lower=data_quartiles.loc["min"], axis=0)

    data = data.clip(**rescale_arm, axis=1)  # Windsorize.

    rescale_arm["iqr"] = rescale_arm["upper"] - rescale_arm["lower"]

    data = ((data - rescale_arm["lower"])
            / rescale_arm["iqr"]
            * (target_quartiles.loc["max"] - target_quartiles.loc["min"])
            + target_quartiles.loc["min"])
    return data


def _test_npn():
    rng = random.default_rng()
    data = pd.DataFrame({
        "normal": rng.normal(10, 3, 10000)+700,
        "poisson": rng.poisson(10, 10000)+3000,
        "binomial": rng.binomial(100, 0.1, 10000)*32,
        "random": rng.random(10000)*70+9.5,
        })
    transformed = npn(data)
    figure = make_subplots(1, 3, subplot_titles=["Data", "Multi-SD", "Single-SD"])
    for i, datum in enumerate([data, transformed[0], transformed[1]], start=1):
        for col in datum.columns:
            figure.add_trace(go.Histogram(x=datum[col], name=col), row=1, col=i)
    return data, transformed, figure


def _test_tdm():
    rng = random.default_rng()
    data = pd.DataFrame([rng.normal(i**2, i, 5000) for i in range(1, 5)]).T
    data = pd.DataFrame([
        rng.normal(10, 3, 5000)+700,
        rng.poisson(10, 5000)+3000,
        rng.binomial(100, 0.1, 5000)*32,
        rng.random(5000)*70+9.5,
        ]).T
    target = pd.DataFrame([rng.normal(i**3, i+3, 5000) for i in range(1, 5)]).T
    rescaled = tdm(data, target)
    figure = make_subplots(3, 1, shared_xaxes=True, shared_yaxes=True)
    for i, datum in enumerate([data, target, rescaled], start=1):
        for col in data.columns:
            figure.add_trace(go.Histogram(x=datum[col], name=col), row=i, col=1)
    return data, target, figure

