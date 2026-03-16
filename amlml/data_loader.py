
from collections.abc import Callable
from collections import namedtuple
from functools import wraps
import math
from random import shuffle
import pickle
from typing import Literal

import numpy as np
import pandas as pd
from pandas import IndexSlice as idx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from qnorm import quantile_normalize
from scipy.optimize import brute
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.utils import gen_batches

from lifelines import KaplanMeierFitter

import torch
from torch import float32, tensor
from torch.utils.data import Dataset

# from amlml.survival_data import prepare_outcomes, code_categoricals, add_nan_mask_stack
from amlml.rna_data import read_rna_dataset, replace_with_tpm
from amlml.microarray_data import read_microarray_dataset, read_gse37642
from amlml.gene_set import GeneSet
from amlml.cross_normalization import log2_transform, zscore_normalize, npn
from amlml.parallel_modelling import CrossNormalizedModel, SuperModel
from amlml.km import optimize_survival_splits


# Dataset = namedtuple("Dataset", ["expression", "outcomes",
#                                  "categoricals", "non_categoricals",
#                                  "reference"])

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

DataLabels = namedtuple("DataLabels", ["index", "columns", "tech"])


def tensify(x):
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    return tensor(x, dtype=torch.float32, device=DEVICE)


class NetworkDataset(Dataset):
    def __init__(self, expression, durations, events,
                 categoricals, non_categoricals,
                 raw_ages, z_expression,
                 index, tech, genes, name=None,
                 classes=None, class_threshold=None,
                 rmst=False, max_time=None, tukey_factor=3):
        self.expression = expression
        self.durations = durations
        self.events = events
        self.categoricals = categoricals
        self.non_categoricals = non_categoricals

        self.index = index
        self.tech = tech
        self.genes = genes

        self.raw_ages = raw_ages
        self.z_expression = z_expression

        self.classes = classes
        self.class_threshold = class_threshold

        self.name = name

        self.rmst = None
        if rmst:
            self.rmst = self.estimate_durations_with_rmst(max_time=max_time,
                                                          tukey_factor=tukey_factor,
                                                          keep_original_durations=False)

        assert(1 == len({x.shape[0] for x in self.data}))

    def __len__(self):
        return self.expression.shape[0]

    def __getitem__(self, index):
        classes = self.classes[index] if self.classes is not None else None
        class_threshold = self.class_threshold if self.class_threshold is not None else None
        return NetworkDataset(*[x[index] for x in self.data],
                              raw_ages=self.raw_ages.iloc[index],
                              z_expression=self.z_expression.iloc[index],
                              index=self.index[index],
                              tech=self.tech[index],
                              classes=classes,
                              class_threshold=class_threshold,
                              genes=self.genes)

    def __getitems__(self, indices):
        classes = self.classes[indices] if self.classes is not None else None
        class_threshold = self.class_threshold if self.class_threshold is not None else None
        return NetworkDataset(*[x[indices] for x in self.data],
                              raw_ages=self.raw_ages.iloc[indices],
                              z_expression=self.z_expression.iloc[indices],
                              index=self.index[indices],
                              tech=self.tech[indices],
                              classes=classes,
                              class_threshold=class_threshold,
                              genes=self.genes)

    @property
    def data(self):
        return [self.expression, self.durations, self.events,
                self.categoricals, self.non_categoricals]
    #
    # @property
    # def network_data(self):
    #     return [self.expression, (self.durations, self.events),
    #             self.categoricals, self.non_categoricals]

    @property
    def outcomes(self):
        return self.durations, self.events

    @property
    def outcome_target_table(self):
        return pd.DataFrame({"durations": self.durations.cpu().detach().numpy(),
                             "events": self.events.cpu().detach().numpy()})

    @property
    def class_table(self):
        table = self.outcome_target_table
        table["groups"] = self.classes.squeeze().tolist()
        table.index = self.index
        return table

    @property
    def n_genes(self):
        return self.expression.shape[-1]

    @property
    def n_clinical(self):
        return self.categoricals.shape[-1] + self.non_categoricals.shape[-1]

    def make_expression_table(self):
        if len(self.expression.shape) == 3:
            table = pd.DataFrame(self.expression.cpu().sum(axis=1), index=self.index,
                                 columns=self.genes)
        else:
            table = pd.DataFrame(self.expression.cpu(), index=self.index, columns=self.genes)
        return table

    @property
    def network_args(self):
        return self.expression, self.categoricals, self.non_categoricals

    def subset_genes(self, genes):
        data = NetworkDataset(expression=self.expression[..., genes],
                              durations=self.durations, events=self.events,
                              categoricals=self.categoricals, non_categoricals=self.non_categoricals,
                              raw_ages=self.raw_ages,
                              z_expression=self.z_expression.iloc[:, list(genes)],
                              index=self.index, tech=self.tech, genes=self.genes[list(genes)],
                              classes=self.classes, class_threshold=self.class_threshold)
        return data

    def generate_batches(self, batch_size, shuffle_=True):
        index = list(range(len(self)))
        if shuffle_:
            shuffle(index)
        for batch_dex in gen_batches(n=len(self), batch_size=batch_size):
            yield self[index[batch_dex]]

    def estimate_durations_with_rmst(self, max_time=None, tukey_factor=None,
                                     keep_original_durations=False, save=True):
        table = self.outcome_target_table
        if tukey_factor is not None:
            times = table.loc[table.events == 1, "durations"].describe()
            max_time = times["75%"] + tukey_factor * (times["75%"] - times["25%"])
        elif max_time is not None:
            max_time = max_time
        else:
            max_time = table.loc[table.events == 1, "durations"].max()

        km = KaplanMeierFitter()
        km.fit(table.durations, table.events)
        rmst_cumsum = np.cumsum(km.survival_function_.KM_estimate.iloc[1:]
                                * np.diff(km.timeline)).loc[km.timeline[1:] <= max_time]
        rmst_tau = rmst_cumsum.iloc[-1]
        table["rmst_predictions"] = table.durations
        censor_times = table.loc[(table.events == 0) & (table.durations <= max_time),
                                 "durations"]
        table.loc[censor_times.index, "rmst_predictions"] = np.array(
            censor_times.values
            + ((rmst_tau - rmst_cumsum.loc[censor_times])
            / km.survival_function_.loc[censor_times, "KM_estimate"]),
            dtype=np.float32)
        if keep_original_durations:
            return table
        table = (table[["rmst_predictions", "events"]]
                 .rename(columns={"rmst_predictions": "durations"}))
        table["groups"] = False
        if save:
            self.rmst = table
        return table

    def plot_rmst_exploration(self):
        table = self.estimate_durations_with_rmst(keep_original_durations=True, save=False)
        tables = {i: self.estimate_durations_with_rmst(max_time=365*i, save=False)
                  for i in range(10, 0, -1)}
        tables = [x["durations"].rename(f"durations_{i}") for i, x in tables.items()]
        table = pd.concat([table]+tables, axis=1).melt(id_vars="events")
        plot = px.histogram(table, x="value", facet_col="variable", color="events",
                            facet_col_wrap=4)
        return plot

    @staticmethod
    def rmst_method(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            if self.rmst is None:
                raise NotImplementedError("Must run estimate_durations_with_rmst first.")
            return fn(self, *args, **kwargs)
        return wrapper
    # @property
    # def rmst_estimates_table(self):
    #     if self._rmst is not None:
    #         return self._rmst
    #     table = self.estimate_durations_with_rmst()
    #     table = (table[["rmst_predictions", "events"]]
    #              .rename(columns={"rmst_predictions": "durations"}))
    #     table["groups"] = False
    #     self._rmst = table
    #     return table

    @rmst_method
    def calculate_rmst_class_cis(self):
        error = (self.rmst.groupby("groups").durations
                 .agg(error=lambda x: 1.96 * x.std() / np.sqrt(x.count()),
                      mean=lambda x: x.mean())
                 .assign(ci_upper=lambda x: x["mean"] + x["error"],
                         ci_lower=lambda x: x["mean"] - x["error"]))
        return error

    @rmst_method
    def optimize_rmst_class_cis(self, plot=False):
        errors = []
        def _loss(threshold):
            self.rmst["groups"] = self.rmst["durations"] >= self.rmst.durations.quantile(*threshold)
            error = self.calculate_rmst_class_cis()
            distance = error.loc[True, "ci_lower"] - error.loc[False, "ci_upper"]
            error["distance"] = distance
            error["threshold"] = threshold[0]
            errors.append(error.reset_index())
            return -distance
        optimal = brute(func=_loss, ranges=[(0.05, 0.95)], Ns=91,
                        full_output=True, finish=None)
        errors = pd.concat(errors).set_index(["threshold", "groups"])
        optimal_table = pd.DataFrame(optimal[-2:]).T
        optimal_table["durations"] = self.rmst.durations.quantile(optimal_table[0]).values
        optimal_table.columns = ["quantile", "ci_interval", "durations"]
        optimal_table["ci_interval"] = -optimal_table.ci_interval
        optimum = optimal_table.iloc[optimal_table.ci_interval.argmax()]
        if plot:
            fig = make_subplots(2, 1)
            fig .add_trace(go.Scatter(x=optimal_table.durations,
                                      y=optimal_table.ci_interval),
                           row=1, col=1)
            fig.add_trace(go.Scatter(x=[optimum.durations], y=[optimum.ci_interval],
                                     text=[f"{optimum.durations / 365:.1f} years"],
                                     mode="markers+text", textposition="top center"),
                          row=1, col=1)
            for thresh in errors.index.get_level_values("threshold").unique():
                for group in [False, True]:
                    subtable = errors.loc[thresh, group]
                    fig.add_trace(go.Scatter(x=subtable[["ci_lower", "mean", "ci_upper"]],
                                             y=[thresh]*3, mode="markers+lines",),
                                  row=2, col=1)
            return optimum, optimal_table, fig
        return optimum, optimal_table, errors

    @rmst_method
    def classify_by_rmst_ci_interval(self, save=False):
        optimum, _, _ = self.optimize_rmst_class_cis(plot=False)
        self.rmst["groups"] = self.rmst.durations >= optimum.durations
        if save:
            self.classes = tensify(self.rmst["groups"]).view([-1, 1])
            self.class_threshold = optimum.durations
        return self.rmst

    @rmst_method
    def classify_by_rmst_threshold(self, threshold, save=False):
        self.rmst["groups"] = self.rmst.durations >= threshold
        if save:
            self.classes = tensify(self.rmst["groups"]).view([-1, 1])
            self.class_threshold = threshold

    def classify_by_duration(self, threshold=1460, save=False):
        table = self.outcome_target_table
        table = table.loc[(table.durations >= threshold) | (table.events == 1)]
        table["groups"] = (table.durations >= threshold).astype(int)
        if save:
            self.classes = tensify(table["groups"]).view([-1, 1])
        return table

    def filter_low_censorship_and_classify_by_duration(self, threshold=1460):
        classes = self.classify_by_duration(threshold)
        data = self[classes.index]
        data.name = self.name + "_no_low_censor"
        data.classes = tensify(classes["groups"]).view([-1, 1])
        data.class_threshold = threshold
        return data

    def filter_minimum_duration(self, threshold):
        index = self.durations >= threshold
        index = np.where(index.cpu())[0]
        dataset = self[index]
        dataset.name = self.name + "_duration_filtered"
        return dataset

    def filter_minimum_censorship(self, threshold):
        index = (self.events == 1) | (self.durations >= threshold)
        index = np.where(index.cpu())[0]
        dataset = self[index]
        dataset.name = self.name + "_min_censor_filtered"
        return dataset

    def filter_by_event(self, event):
        index = self.events == event
        index = np.where(index.cpu())[0]
        dataset = self[index]
        dataset.name = self.name + "_event_filtered"
        return dataset

    def filter_by_age_at_diagnosis(self, age_in_days, keep_less_than=True):
        if keep_less_than:
            index = self.raw_ages < age_in_days
        else:
            index = self.raw_ages >= age_in_days
        index = np.where(index)[0]
        dataset = self[index]
        dataset.name = self.name + "_age_filtered"
        return dataset

    def filter_by_tech(self, tech):
        index = np.where(self.tech == tech)
        dataset = self[index]
        dataset.name = self.name + "_tech_filtered"
        return dataset

    def filter_ambiguous(self, distance):
        thresh = self.class_threshold
        non_ambiguous = ((self.rmst.events == 1)
                         | (self.rmst.durations < thresh-distance)
                         | (self.rmst.durations > thresh+distance))
        index = np.where(non_ambiguous)
        dataset = self[index]
        dataset.name = self.name + "_ambiguity_filtered"
        dataset.rmst = self.rmst.loc[non_ambiguous]
        return dataset

    @rmst_method
    def plot_class_duration_distribution(self, rmst=False):
        table = self.rmst if rmst else self.class_table
        fig = go.Figure()
        for name, x in zip(["High", "Low"], [True, False]):
            for i, name2 in enumerate(["Censored", "Death"]):
                subtable = table.loc[(table["groups"] == x) & (table.events == i)]
                fig.add_trace(go.Histogram(x=subtable.durations, name=f"{name2}-{name}"))
        return fig

    def plot_survival_times(self):
        table = self.outcome_target_table.sort_values("durations")
        table["x"] = list(range(1, table.shape[0]+1))
        figure = go.Figure()
        for i, name in enumerate(["Censored", "Death"]):
            subtable = table.loc[table.events == i]
            figure.add_trace(go.Scatter(x=subtable.x, y=subtable.durations,
                                        marker_color=i, name=name, mode="markers"))
        return figure

    def plot_survival_histogram(self):
        table = self.outcome_target_table
        figure = go.Figure()
        for i, name in enumerate(["Censored", "Death"]):
            figure.add_trace(go.Histogram(x=table.loc[table.events == i], name=name))
        return figure

    def plot_kaplan_meier(self):
        km = KaplanMeierFitter()
        km.fit(self.durations, self.events)
        figure = go.Figure(go.Scatter(x=km.timeline, y=km.survival_function_.KM_estimate))
        upper_ci = km.confidence_interval_["KM_estimate_upper_0.95"]
        lower_ci = km.confidence_interval_["KM_estimate_lower_0.95"]
        ci_params = dict(
            mode="lines",
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
            )
        figure.add_trace(go.Scatter(x=km.timeline, y=upper_ci, **ci_params))
        figure.add_trace(go.Scatter(x=km.timeline, y=lower_ci, **ci_params,
                                    fill="tonexty", fillcolor="rgba(68, 68, 68, 0.3)"))
        return figure

    def _debug_set(self):
        return self[:10]

    def class_balance(self):
        counts = self.class_table.groups.value_counts()
        balance = counts[0] / counts[1]
        return balance

def set_intersect_of_genes(*args):
    genesets = [set(dataset.Expression.columns) for dataset in args]
    genes = set.intersection(*genesets)
    for dataset in args:
        removers = set(dataset.Expression.columns) - genes
        removers = [("Expression", gene) for gene in removers]
        dataset.drop(removers, axis=1, inplace=True)
    return


def read_model_data():
    geneset = GeneSet("Homo_sapiens.GRCh38.113.chr.gtf.gz",
                      include=["gene", "transcript", "exon"])

    target_aml, cols = read_rna_dataset(
        expression_data="Data/TARGET_AML_gene_counts/second_stranded_counts.tsv",
        clinical_data="Data/TARGET_AML_gene_counts/clinical.tsv",
        clinical_yaml="Data/TARGET_AML_gene_counts/Clinical/variables.yaml",
        geneset=geneset,
        median_tpm_above_quantile=.99,
        minimum_expression=(10, .10),
        variance_cutoff=None,
        return_tpm=False
    )
    target_aml[("Tech", "")] = "RNAseq"
    drop_list = ["TARGET-20-PAPXVK"]  # Filtered due to multiple listed events
    target_aml = target_aml.drop(drop_list)
    # all_group_labels = [0]*target_aml.shape[0]

    ## Read TCGA-AML microarray data.
    tcga_aml, _ = read_microarray_dataset(
        expression_data="Data/TCGA_LAML_arrays/tcga_laml.microarray_data.tsv",
        clinical_data="Data/TCGA_LAML_arrays/tcga-aml.clinical.homogenized.incomplete.tsv",
        clinical_yaml="Data/TARGET_AML_gene_counts/Clinical/variables.yaml",
        geneset=geneset
        )
    tcga_aml[("Tech", "")] = "Microarray"
    # all_group_labels += [1]*tcga_aml.shape[0]

    # Read GSE37642 HGU133plus2 data.
    gse37642_hgu133plus2, _ = read_gse37642(
        input_file="Data/GSE37642/GSE37642.hgu133plus2.expression.tsv",
        clinical_data="/home/tyler/Documents/Projects/ML/Data/GSE37642/GSE37642_Homogenized_Survival_data.tsv",
        clinical_yaml="Data/TARGET_AML_gene_counts/Clinical/variables.yaml"
        )
    gse37642_hgu133plus2.dropna(subset=[("Outcomes", "Vital Status"),
                                        ("Outcomes", "Overall Survival Time in Days")],
                                inplace=True)
    gse37642_hgu133plus2[("Tech", "")] = "Microarray"
    # all_group_labels += [1]*gse37642_hgu133plus2.shape[0]

    # Read GSE37642 HGU133plusA data.
    gse37642_hgu133a, _ = read_gse37642(
        input_file="Data/GSE37642/GSE37642.hgu133A.expression.tsv",
        clinical_data="/home/tyler/Documents/Projects/ML/Data/GSE37642/GSE37642_Homogenized_Survival_data.tsv",
        clinical_yaml="Data/TARGET_AML_gene_counts/Clinical/variables.yaml"
    )
    gse37642_hgu133a.dropna(subset=[("Outcomes", "Vital Status"),
                                    ("Outcomes", "Overall Survival Time in Days")],
                            inplace=True)
    gse37642_hgu133a[("Tech", "")] = "Microarray"
    # all_group_labels += [1]*gse37642_hgu133a.shape[0]

    set_intersect_of_genes(target_aml, tcga_aml, gse37642_hgu133plus2, gse37642_hgu133a)
    target_aml = replace_with_tpm(target_aml, geneset)

    data = pd.concat([target_aml, tcga_aml, gse37642_hgu133plus2, gse37642_hgu133a],
                     axis=0)
    return data, cols
    rna_data = pd.concat([target_aml], axis=0)
    ma_data = pd.concat([tcga_aml, gse37642_hgu133plus2, gse37642_hgu133a], axis=0)
    data = pd.concat([rna_data, ma_data], axis=0)
    ids = list(data.index)
    return data, rna_data, ma_data, cols, ids, all_group_labels


def read_model_data_pickle():
    with open("Data/big_pickle.pickle", "rb") as infile:
        data, cols = pickle.load(infile)
    return data, cols


def separate_tech(data):
    rna, array = data.loc[idx[:, "RNAseq"],], data.loc[idx[:, "Microarray"],]
    return rna, array


def prepare_log2_expression(data, as_tensor=True, splits=None):
    data = data.copy()
    data.loc[idx[:, "RNAseq"], :] = log2_transform(data.loc[idx[:, "RNAseq"], :], True, False)
    data.loc[idx[:, "Microarray"], :] = log2_transform(data.loc[idx[:, "Microarray"], :], True, True)
    if as_tensor:
        expression = tensify(data)
        if splits is not None:
            train, test = expression[splits[0]], expression[splits[1]]
            return train, test
        return expression
    if splits is not None:
        train, test = data.iloc[splits[0]], data.iloc[splits[1]]
        return train, test
    return data


# TODO: Update to new methods.
def prepare_qn_expression(data, as_tensor=True, with_zscore=False, **kwargs):
    expression = prepare_log2_expression(data, concatenate=True, as_tensor=False)
    expression = quantile_normalize(data["Expression"].astype("float64").T).T
    if with_zscore:
        expression = zscore_normalize(expression)
    if as_tensor:
        expression = tensor(expression.to_numpy(), dtype=float32)
    return expression


# TODO: Update to new methods.
def prepare_qnz_expression(data, as_tensor=True, **kwargs):
    return prepare_qn_expression(data, as_tensor=as_tensor, with_zscore=True)


def _prepare_scaled_expression(scale_method, data, splits, as_tensor=True, **kwargs):
    scalers = {"RNAseq": scale_method(**kwargs), "Microarray": scale_method(**kwargs)}

    train, test = prepare_log2_expression(data, as_tensor=False, splits=splits)
    for tech in ["RNAseq", "Microarray"]:
        train.loc[idx[:, tech], :] = scalers[tech].fit_transform(train.loc[idx[:, tech], :])
        test.loc[idx[:, tech], :] = scalers[tech].transform(test.loc[idx[:, tech], :])

    if as_tensor:
        train, test = tensify(train), tensify(test)
    return train, test


def prepare_zscore_expression(data, splits, as_tensor=True):
    train, test = _prepare_scaled_expression(StandardScaler, data, splits, as_tensor=as_tensor)
    return train, test


def prepare_npn_expression(data, splits, as_tensor=True):
    train, test = _prepare_scaled_expression(QuantileTransformer, data, splits, as_tensor=as_tensor,
                                             output_distribution="normal", random_state=0)
    return train, test


# def prepare_npn_expression(data, **kwargs):
#     expression = prepare_log2_expression(data, concatenate=False, as_tensor=False)
#     expression = npn(expression)
#     expression = tensify(expression.to_numpy())
#     return expression


def prepare_supermodel_expression(data, splits, with_zscore=False, as_tensor=True):
    if with_zscore:
        train, test = prepare_zscore_expression(data, splits, as_tensor=False)
    else:
        train, test = data.iloc[splits[0]], data.iloc[splits[1]]

    stacks = []
    for dataset in [train, test]:
        stack = np.zeros((2, *dataset.shape))
        for i, tech in enumerate(["RNAseq", "Microarray"]):
            mask = dataset.index.get_level_values("Tech") == tech
            stack[i, mask, :] = dataset.loc[mask]
        stacks.append(stack.transpose(1, 0, 2))
    train, test = stacks

    if as_tensor:
        train, test = tensify(train), tensify(test)
    return train, test


def prepare_zupermodel_expression(data, splits, as_tensor=True,):
    return prepare_supermodel_expression(data, splits, with_zscore=True, as_tensor=as_tensor)


def _prepare_expression(data, normalization, splits, **kwargs):
    data = normalization(data.Expression, splits=splits, **kwargs)
    return data


def _prepare_categorical(data, cols, splits):
    categoricals = [x for x in data.Covariates if cols["Covariates"][x] == "categorical"]
    categoricals = data.Covariates[categoricals]
    for x in categoricals:
        categoricals[x] = pd.Categorical(categoricals[x])
        categoricals[x] = categoricals[x].cat.codes
    categoricals = categoricals.to_numpy()
    train, test = categoricals[splits[0]], categoricals[splits[1]]
    train, test = tensor(train, dtype=torch.int32, device=DEVICE), tensor(test, dtype=torch.int32, device=DEVICE)
    return train, test


def _prepare_non_categorical(data, cols, splits):
    continuous = data.Covariates[[x for x in data.Covariates if cols["Covariates"][x] == "continuous"]]
    binary = data.Covariates[[x for x in data.Covariates if cols["Covariates"][x] == "binary"]]
    cont_mask, bin_mask = pd.isna(continuous).astype(int), pd.isna(binary).astype(int)

    continuous_train, continuous_test = continuous.iloc[splits[0]], continuous.iloc[splits[1]]
    means = continuous_train.mean().to_dict()
    continuous_train.fillna(means, inplace=True)
    continuous_test.fillna(means, inplace=True)
    scaler = StandardScaler()
    continuous_train = scaler.fit_transform(continuous_train)
    continuous_test = scaler.transform(continuous_test)
    cont_mask_train, cont_mask_test = cont_mask.iloc[splits[0]], cont_mask.iloc[splits[1]]
    cont_mask_train, cont_mask_test = (np.stack([continuous_train, cont_mask_train]),
                                       np.stack([continuous_test, cont_mask_test]))

    bin_train, bin_test = binary.iloc[splits[0]], binary.iloc[splits[1]]
    modes = bin_train.mode().T[0].to_dict()
    bin_train.fillna(modes, inplace=True)
    bin_test.fillna(modes, inplace=True)
    bin_mask_train, bin_mask_test = bin_mask.iloc[splits[0]], bin_mask.iloc[splits[1]]
    bin_mask_train, bin_mask_test = (np.stack([bin_train, bin_mask_train]),
                                     np.stack([bin_test, bin_mask_test]))

    train = tensify(np.concat([cont_mask_train, bin_mask_train], axis=-1)).permute(1, 0, 2)
    test = tensify(np.concat([cont_mask_test, bin_mask_test], axis=-1)).permute(1, 0, 2)

    # non_cat = data.Covariates[continuous+binary]
    # train, test = non_cat.iloc[splits[0]], non_cat.iloc[splits[1]]
    #
    # # Use training values for both train and test imputation.
    # imputation = train[continuous].mean().to_dict()
    # imputation.update(train[binary].mode().T[0].to_dict())
    #
    # non_cat = [tensor(add_nan_mask_stack(dataset, impute_value=imputation), dtype=float32, device=DEVICE).permute(1, 0, 2)
    #            for dataset in [train, test]]
    # train, test = non_cat
    return train, test


def _prepare_outcomes(data, splits):
    outcomes = data.Outcomes
    outcomes.columns = ["event", "duration"]
    outcomes = outcomes[["duration", "event"]]
    outcomes.loc[:, "event"] = [int(x == "Dead") for x in outcomes.event]

    train, test = outcomes.iloc[splits[0]], outcomes.iloc[splits[1]]
    train, test = tensify(train.to_numpy().T.astype(float)), tensify(test.to_numpy().T.astype(float))
    train, test = (train[0], train[1]), (test[0], test[1])
    return train, test


def prepare_data2(data, cols, normalization: Callable = prepare_supermodel_expression,
                 drop_zero_survivors=True):
    torch.set_grad_enabled(False)

    if drop_zero_survivors:
        data = data.loc[data.Outcomes["Overall Survival Time in Days"] > 0]

    data.set_index("Tech", inplace=True, append=True)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    splits = list(splitter.split(data, data.index.get_level_values("Tech")))[0]

    train_cat, test_cat = _prepare_categorical(data, cols, splits)
    train_non_cat, test_non_cat = _prepare_non_categorical(data, cols, splits)
    train_exp, test_exp = _prepare_expression(data, normalization, splits)
    train_z_exp, test_z_exp = _prepare_expression(data, prepare_zscore_expression, splits, as_tensor=False)
    train_outcomes, test_outcomes = _prepare_outcomes(data, splits)
    # print(train_outcomes, "===", test_outcomes)
    age = "Age at Diagnosis in Days"
    train_age, test_age = data.Covariates[[age]].iloc[splits[0]], data.Covariates[[age]].iloc[splits[1]]
    train_dex, test_dex = data.iloc[splits[0]].index, data.iloc[splits[1]].index
    norm_name = normalization.__name__.split("_")[1]

    train = NetworkDataset(
        expression=train_exp, durations=train_outcomes[0], events=train_outcomes[1],
        categoricals=train_cat, non_categoricals=train_non_cat,
        raw_ages=train_age, z_expression=train_z_exp,
        index=train_dex, tech=train_dex.get_level_values("Tech"), genes=data.Expression.columns,
        name=f"{norm_name.upper()}-train")

    test = NetworkDataset(
        expression=test_exp, durations=test_outcomes[0], events=test_outcomes[1],
        categoricals=test_cat, non_categoricals=test_non_cat,
        raw_ages=test_age, z_expression=test_z_exp,
        index=test_dex, tech=test_dex.get_level_values("Tech"), genes=data.Expression.columns,
        name=f"{norm_name.upper()}-test")

    torch.set_grad_enabled(True)
    return train, test


# def prepare_data(data, cols, normalization: Callable = prepare_supermodel_expression,
#                  drop_zero_survivors=True):
#     torch.set_grad_enabled(False)
#     if drop_zero_survivors:
#         data = data.loc[data.Outcomes["Overall Survival Time in Days"] > 0]
#     train, test = train_test_split(data, test_size=0.2,
#                                    stratify=data.Tech,
#                                    random_state=0)
#     train = train.sort_values("Tech", ascending=False)
#     test = test.sort_values("Tech", ascending=False)
#     norm_name = normalization.__name__.split("_")[1]
#
#     datasets = []
#     for name, data_split in zip(["train", "test"], [train, test]):
#         # Data labels.
#         # data_labels = DataLabels(data_split.index,
#         #                          data_split.Expression.columns,
#         #                          data_split.Tech)
#
#         # Expression.
#         if normalization is None:
#             expression = tensor(data_split.Expression, dtype=float32)
#         else:
#             expression = normalization(data_split)
#         expression = expression.to(DEVICE)
#         # Outcomes
#         outcomes = data_split.Outcomes
#         outcomes.columns = ["event", "duration"]
#         outcomes = outcomes[["duration", "event"]]
#         outcomes.loc[:, "event"] = [int(x == "Dead") for x in outcomes.event]
#
#         # if name == "train":
#         #     outcomes = prepare_outcomes(np.array(outcomes))
#         # else:
#         #     outcomes = outcomes.T.astype(float)
#         outcomes = prepare_outcomes(np.array(outcomes))
#
#         # Categorical covariates.
#         cats = [x for x in data_split.Covariates if cols["Covariates"][x] == "categorical"]
#         categoricals = data_split.Covariates[cats]
#         for x in categoricals:
#             categoricals[x] = pd.Categorical(categoricals[x])
#         code_categoricals(categoricals)
#         categoricals = tensor(categoricals.to_numpy(), dtype=torch.int32, device=DEVICE)
#
#         # Non-categorical covariates.
#         raw_age = data_split.Covariates[["Age at Diagnosis in Days"]].copy()
#         noncats = []
#         for var_type in ["binary", "continuous"]:
#             var_cols = [x for x in data_split.Covariates if cols["Covariates"][x] == var_type]
#             var_cols = data_split.Covariates[var_cols]
#             var_cols = tensor(add_nan_mask_stack(var_cols), dtype=float32, device=DEVICE)
#             noncats.append(var_cols)
#         noncats = torch.concat(noncats, dim=2)
#         noncats = noncats.permute(1, 0, 2)
#
#         noncats = [x for x in data_split.Covariates if cols["Covariates"][x] != "categorical"]
#         non_categoricals = data_split.Covariates[noncats]
#         non_categoricals = tensor(add_nan_mask_stack(non_categoricals), dtype=float32, device=DEVICE)
#         non_categoricals = non_categoricals.permute(1, 0, 2)
#
#         # dataset = Dataset(expression, outcomes, categoricals, non_categoricals, data_labels)
#         dataset = NetworkDataset(expression, *outcomes, categoricals, non_categoricals, raw_age,
#                                  data_split.index, data_split.Tech,
#                                  data_split.Expression.columns,
#                                  name=f"{norm_name.upper()}-{name}")
#         datasets.append(dataset)
#     torch.set_grad_enabled(True)
#     return datasets
    #
    #
    #
    #
    # # Prepare expression.
    # all_expression = normalization(rna_data, ma_data)
    #
    # # Prepare covariates.
    # all_categoricals = data["Covariates"][[x for x in cols["Covariates"]
    #                                        if cols["Covariates"][x] == "categorical"]]
    # for x in all_categoricals:
    #     all_categoricals[x] = pd.Categorical(all_categoricals[x])
    # code_categoricals(all_categoricals)
    # all_categoricals = tensor(all_categoricals.to_numpy(), dtype=torch.int32)
    #
    # all_non_categoricals = data["Covariates"][[x for x in cols["Covariates"]
    #                                            if cols["Covariates"][x] != "categorical"]]
    # all_non_categoricals = tensor(add_nan_mask_stack(all_non_categoricals), dtype=float32)
    # all_non_categoricals = all_non_categoricals.permute(1, 0, 2)
    #
    # # Prepare outcomes.
    # all_outcomes = data.Outcomes
    # all_outcomes.columns = ["event", "duration"]
    # all_outcomes = all_outcomes[["duration", "event"]]
    # all_outcomes.loc[:, "event"] = [int(x == "Dead") for x in all_outcomes.event]
    #
    # return all_expression, all_outcomes, all_categoricals, all_non_categoricals


# def split_test_data(data, all_expression, all_outcomes, all_categoricals,
#                     all_non_categoricals, ids, all_group_labels):
#     expression = dict()
#     categoricals = dict()
#     non_categoricals = dict()
#     outcomes = dict()
#     set_ids = dict()
#     group_labels = dict()
#     expression_table = dict()
#
#     (expression["train"], expression["test"],
#      categoricals["train"], categoricals["test"],
#      non_categoricals["train"], non_categoricals["test"],
#      outcomes["train"], outcomes["test"],
#      set_ids["train"], set_ids["test"],
#      group_labels["train"], group_labels["test"],
#      expression_table["train"], expression_table["test"]) = (
#         train_test_split(all_expression, all_categoricals, all_non_categoricals, all_outcomes,
#                          ids, all_group_labels, data.Expression,
#                          test_size=0.2, random_state=0, stratify=all_group_labels)
#         )
#
#     outcomes["train"] = prepare_outcomes(np.array(outcomes["train"]))
#     outcomes["test"] = outcomes["test"].T.astype(float)
#     return (expression, outcomes, categoricals, non_categoricals,
#             set_ids, group_labels,
#             expression_table)

def main_loader(normalization: Callable, verbose=True):
    if verbose:
        print("Reading data...")
    data, cols = read_model_data_pickle()
    if verbose:
        print(f"Preparing method {normalization.__name__}")
    train, test = prepare_data2(data, cols, normalization=normalization)
    return train, test

#     prepared = prepare_data(*data[:4], normalization=normalization)
#     (expression, outcomes,
#      categoricals, non_categoricals,
#      set_ids, group_labels,
#      expression_table) = split_test_data(data[0], *prepared, *data[-2:])
#
#
# def main_loader(normalization: Callable):
#     data = read_model_data()
#     prepared = prepare_data(*data[:4], normalization=normalization)
#     (expression, outcomes,
#      categoricals, non_categoricals,
#      set_ids, group_labels,
#      expression_table) = split_test_data(data[0], *prepared, *data[-2:])
#     return (expression, outcomes,
#             categoricals, non_categoricals,
#             set_ids, group_labels,
#             expression_table)


def normalization_generator(methods=None, verbose=False):
    if methods is None:
        methods = [prepare_log2_expression, prepare_zscore_expression,
                   prepare_npn_expression, prepare_supermodel_expression,
                   prepare_zupermodel_expression]
    if verbose:
        print("Reading data...")
    data, cols = read_model_data_pickle()
    for norm_method in methods:
        if (norm_method == prepare_supermodel_expression
                or norm_method == prepare_zupermodel_expression):
            network_type = SuperModel
        else:
            network_type = CrossNormalizedModel
        if verbose:
            print(f"Preparing method {norm_method.__name__}")
        train, test = prepare_data2(data, cols, norm_method)
        yield network_type, (train, test)
        # prepared = prepare_data(*data[:4], normalization=norm_method)
        # split = split_test_data(data[0], *prepared, *data[-2:])
        # yield norm_method.__name__, split
