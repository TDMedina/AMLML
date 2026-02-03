
from collections import defaultdict
from typing import Literal

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd


def partial_log_likelihood(outcome_table, predictions, eps=1e-7):
    """Adapted from pycox."""
    outcome_table["predictions"] = predictions
    outcome_table = outcome_table.sort_values("durations", ascending=False)
    outcome_table["_cum_exp_g"] = (outcome_table["predictions"]
                                   .pipe(np.exp)
                                   .cumsum()
                                   .groupby(outcome_table["durations"])
                                   .transform("max"))
    outcome_table = outcome_table.loc[outcome_table["events"] == 1]
    outcome_table["pll"] = (outcome_table["predictions"]
                            - np.log(outcome_table['_cum_exp_g'] + eps))
    return outcome_table["pll"]


def compute_baseline_hazards(outcome_table, predictions):
    """Adapted from pycox."""
    baseline_hazards = (outcome_table
                        .assign(expg=np.exp(predictions))
                        .sort_values("durations")
                        .groupby(outcome_table["durations"])
                        .agg({"expg": "sum", "events": "sum"})
                        .sort_index(ascending=False)
                        .assign(expg=lambda x: x["expg"].cumsum())
                        .pipe(lambda x: x["events"]/x["expg"])
                        .fillna(0.)
                        .iloc[::-1]
                        .rename("baseline_hazards"))
    baseline_hazards = pd.DataFrame(baseline_hazards)
    baseline_hazards["cumulative"] = baseline_hazards.baseline_hazards.cumsum()
    return baseline_hazards


def predict_survival_table(predictions, baseline_cumulative_hazards):
    """Adapted from pycox."""
    expg = np.exp(predictions).reshape(1, -1)
    bch = baseline_cumulative_hazards.to_numpy().reshape(-1, 1)
    survival_table = pd.DataFrame(bch.dot(expg),
                                  index=baseline_cumulative_hazards.index)
    survival_table = np.exp(-survival_table)
    return survival_table


def classify_by_hazard_at_threshold(survival, threshold):
    classes = [np.interp(threshold, survival.index.values, survival[x])
               for x in survival]
    return classes


# Note: Currently using pycox.models.loss.CoxPHLoss directly, from which this was adapted.
# def coxph_loss(predictions, durations, events, eps=1e-7):
#     idx = durations.sort(descending=True)[1]
#     events = events[idx]
#     log_h = predictions[idx]
#     events = events.view(-1)
#     log_h = log_h.view(-1)
#     gamma = log_h.max()
#     log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
#     return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())


class CV_Result:
    def __init__(self, fold, alpha, alpha_index, genes,
                 n_epochs, lrs, losses_train, losses_val,
                 pll_train=None, pll_val=None, network=None,
                 hazards_train=None, hazards_val=None, hazards_baseline=None,
                 ctd=None, ibs=None,
                 risk_splits=None, logranks=None,
                 survival_train=None, survival_val=None, first_weights=None,
                 classes_train=None, classes_val=None,
                 classify_loss_train=None, classify_loss_val=None,
                 parameters=None, name=None):
        self.fold = fold
        self.alpha = alpha
        self.alpha_index = alpha_index
        self.genes = genes
        self.n_epochs = n_epochs
        self.lrs = lrs
        self.losses_train = losses_train
        self.losses_val = losses_val
        self.pll_train = pll_train
        self.pll_val = pll_val

        self.network = network

        self.hazards_train = hazards_train
        self.hazards_val = hazards_val
        self.hazards_baseline = hazards_baseline
        self.ctd = ctd
        self.ibs = ibs
        self.risk_splits = risk_splits
        self.logranks = logranks

        self.survival_train = survival_train
        self.survival_val = survival_val

        self.classes_train = classes_train
        self.classes_val = classes_val

        self.classify_loss_train = classify_loss_train
        self.classify_loss_val = classify_loss_val

        self.first_weights = first_weights
        self.parameters = parameters

        self.name = name

    def tabulate(self, include_name=True):
        risk_splits = (self.risk_splits
                       if (self.risk_splits is None or len(self.risk_splits) > 1)
                       else self.risk_splits[0])
        if self.logranks is None:
            logranks = None
        else:
            logranks = [round(x.p_value, 3) for x in self.logranks.values()]
            if len(logranks) == 1:
                logranks = logranks[0]
        table = pd.DataFrame([dict(
            alpha_index=self.alpha_index,
            fold=self.fold,
            alpha=self.alpha,
            n_genes=len(self.genes),
            n_epochs=self.n_epochs,
            loss_train=self.losses_train[-1],
            loss_val=self.losses_val[-1],

            pll_train_mean=self.pll_train.mean() if self.pll_train is not None else None,
            pll_val_mean=self.pll_val.mean() if self.pll_val is not None else None,
            ctd=self.ctd,
            ibs=self.ibs,
            risk_splits=risk_splits,
            logranks=logranks,
            classify_loss_train = self.classify_loss_train,
            classify_loss_val = self.classify_loss_val,
            )])
        if include_name:
            table["name"] = self.name
            table.set_index(["name", "alpha_index", "fold"], inplace=True)
        return table

    def plot_loss(self, _as_subplot=False):
        subplots = []
        data = [
            ("training", "blue", self.losses_train),
            ("validation", "red", self.losses_val)
            ]
        for name, color, loss in data:
                subplot = go.Scatter(x=list(range(1, len(loss)+1)), y=loss,
                                     mode="lines", name=name,
                                     marker_color=color, line_color=color,
                                     legendgroup=name)
                subplots.append(subplot)
        if _as_subplot:
            return subplots
        figure = go.Figure(subplots)
        return figure

    def plot_pll(self):
        fig = go.Figure()
        for name, data in [("Train", self.pll_train), ("Val", self.pll_val)]:
            fig.add_trace(go.Histogram(x=data.abs(), name=name))
        return fig

    def plot_lr(self, _as_subplot=False):
        subplot = go.Scatter(x=list(range(1, len(self.lrs)+1)), y=self.lrs,
                             mode="markers+lines", name="Learning Rate",
                             marker_color="blue", line_color="blue")
        if _as_subplot:
            return [subplot]
        figure = go.Figure(subplot)
        return figure

    def tabulate_predictions(self, data=Literal["train", "val"]):
        data = self.__getattribute__(f"classes_{data}")
        table = (data.groupby("Training")["Predicted"]
                 .agg(group0=lambda x: (x < 0.5).sum(),
                      group1=lambda x: (x >= 0.5).sum()))
        return table

    def plot_predictions(self):
        fig = make_subplots(1, 2)
        for i, (name, data) in enumerate(zip(["Train", "Val"],
                                         [self.classes_train, self.classes_val]),
                                         start=1):
            fig.add_trace(go.Scatter(x=data.loc[data.Predicted < 0.5, "Training"],
                                     y=data.loc[data.Predicted < 0.5, "Predicted"],
                                     mode="markers", name=name), col=i)
            fig.add_trace(go.Scatter(x=data.loc[data.Predicted >= 0.5, "Training"],
                                     y=data.loc[data.Predicted >= 0.5, "Predicted"],
                                     mode="markers", name=name), col=i)
        return fig

    def rank_covariates(self, dataset, cols, embedded_dims=None):
        if embedded_dims is None:
            embedded_dims = {"race": 3, "ethnicity": 3, "interaction": 3, "protocol": 3}
        covariates = [y for x, dims in embedded_dims.items() for y in [x]*dims]
        covariates = covariates + [var for var, type_ in cols["Covariates"].items() if type_ != "categorical"]
        covariates = list(dataset.genes) + covariates
        regs = np.abs(self.first_weights).sum(axis=0)
        ranked = pd.DataFrame(zip(covariates, regs), columns=["covariate", "reg_weight"]).sort_values("reg_weight")
        return ranked


class CV_ResultsCollection:
    def __init__(self, results: list[CV_Result], methods=None, folds_per=None):
        self.results = results
        self.methods = methods
        self.folds_per = folds_per

    def tabulate(self):
        tables = [x.tabulate() for x in self.results]
        table = pd.concat(tables, axis=0)
        return table

    def make_agg_table(self):
        table = self.tabulate()
        table = (table[["alpha", "n_genes", "n_epochs",
                        "pll_train_mean", "pll_val_mean",
                        "loss_train", "loss_val",
                        "ctd", "ibs",
                        "classify_loss_train", "classify_loss_val"]]
                 .groupby(["name", "alpha_index"])
                 .mean())
        return table

    def _plot(self, plot_fn):
        rows = len(self.methods)
        cols = self.folds_per
        figure = make_subplots(rows=rows, cols=cols)
        for i in range(rows):
            for j in range(cols):
                results = self.results[j + cols * (i - 1)]
                subplots = results.__getattribute__(plot_fn)(_as_subplot=True)
                for subplot in subplots:
                    figure.add_trace(subplot, row=i + 1, col=j + 1)
        return figure

    def plot_method_loss(self):
        figs = []
        for method in self.methods:
            results = [result for result in self.results if method in result.name]
            fig = make_subplots(rows=self.folds_per, cols=len(results) // self.folds_per,
                                x_title="Alpha", y_title="Fold")
            for result in results:
                traces = result.plot_loss(_as_subplot=True)
                for trace in traces:
                    fig.add_trace(trace, row=result.fold+1, col=result.alpha_index+1)
            fig.update_layout(title=method)
            fig.update_yaxes(range=[0, 1])
            figs.append(fig)
        return figs

    def plot_loss(self):
        figure = self._plot("plot_loss")
        return figure

    def plot_pll(self):
        figure = self._plot("plot_pll")
        return figure

    def plot_lr(self):
        figure = self._plot("plot_lr")
        return figure



class TestResult:
    def __init__(self, alpha, genes,
                 n_epochs, lrs, losses_train, loss_test,
                 pll_train=None, pll_test=None, network=None,
                 hazards_train=None, hazards_test=None, hazards_baseline=None,
                 ctd=None, ibs=None,
                 risk_splits=None, logranks=None,
                 survival_train=None, survival_test=None, first_weights=None,
                 classes_train=None, classes_test=None,
                 classify_loss_train=None, classify_loss_test=None,
                 parameters=None, name=None):
        self.alpha = alpha
        self.genes = genes
        self.n_epochs = n_epochs
        self.lrs = lrs
        self.losses_train = losses_train
        self.loss_test = loss_test
        self.pll_train = pll_train
        self.pll_test = pll_test

        self.network = network

        self.hazards_train = hazards_train
        self.hazards_test = hazards_test
        self.hazards_baseline = hazards_baseline
        self.ctd = ctd
        self.ibs = ibs
        self.risk_splits = risk_splits
        self.logranks = logranks

        self.survival_train = survival_train
        self.survival_test = survival_test

        self.classes_train = classes_train
        self.classes_test = classes_test

        self.classify_loss_train = classify_loss_train
        self.classify_loss_test = classify_loss_test

        self.first_weights = first_weights
        self.parameters = parameters

        self.name = name

    def tabulate(self, include_name=True):
        risk_splits = (self.risk_splits
                       if (self.risk_splits is None or len(self.risk_splits) > 1)
                       else self.risk_splits[0])
        if self.logranks is None:
            logranks = None
        else:
            logranks = [round(x.p_value, 3) for x in self.logranks.values()]
            if len(logranks) == 1:
                logranks = logranks[0]
        table = pd.DataFrame([dict(
            alpha=self.alpha,
            n_genes=len(self.genes),
            n_epochs=self.n_epochs,
            loss_train=self.losses_train[-1],
            loss_test=self.loss_test,

            pll_train_mean=self.pll_train.mean() if self.pll_train is not None else None,
            pll_test_mean=self.pll_test.mean() if self.pll_test is not None else None,
            ctd=self.ctd,
            ibs=self.ibs,
            risk_splits=risk_splits,
            logranks=logranks,
            classify_loss_train = self.classify_loss_train,
            classify_loss_test = self.classify_loss_test,
            )])
        if include_name:
            table["name"] = self.name
            table.set_index(["name"], inplace=True)
        return table

    # TODO: Plots.
    def plot_loss(self, _as_subplot=False):
        subplots = []
        data = [
            ("training", "blue", self.losses_train),
            ("validation", "red", self.losses_val)
            ]
        for name, color, loss in data:
                subplot = go.Scatter(x=list(range(1, len(loss)+1)), y=loss,
                                     mode="lines", name=name,
                                     marker_color=color, line_color=color,
                                     legendgroup=name)
                subplots.append(subplot)
        if _as_subplot:
            return subplots
        figure = go.Figure(subplots)
        return figure

    def plot_pll(self):
        fig = go.Figure()
        for name, data in [("Train", self.pll_train), ("Val", self.pll_val)]:
            fig.add_trace(go.Histogram(x=data.abs(), name=name))
        return fig

    def plot_lr(self, _as_subplot=False):
        subplot = go.Scatter(x=list(range(1, len(self.lrs)+1)), y=self.lrs,
                             mode="markers+lines", name="Learning Rate",
                             marker_color="blue", line_color="blue")
        if _as_subplot:
            return [subplot]
        figure = go.Figure(subplot)
        return figure

    def tabulate_predictions(self, data=Literal["train", "val"]):
        data = self.__getattribute__(f"classes_{data}")
        table = (data.groupby("Training")["Predicted"]
                 .agg(group0=lambda x: (x < 0.5).sum(),
                      group1=lambda x: (x >= 0.5).sum()))
        return table

    def plot_predictions(self):
        fig = make_subplots(1, 2)
        for i, (name, data) in enumerate(zip(["Train", "Val"],
                                         [self.classes_train, self.classes_val]),
                                         start=1):
            fig.add_trace(go.Scatter(x=data.loc[data.Predicted < 0.5, "Training"],
                                     y=data.loc[data.Predicted < 0.5, "Predicted"],
                                     mode="markers", name=name), col=i)
            fig.add_trace(go.Scatter(x=data.loc[data.Predicted >= 0.5, "Training"],
                                     y=data.loc[data.Predicted >= 0.5, "Predicted"],
                                     mode="markers", name=name), col=i)
        return fig

    def rank_covariates(self, dataset, cols, embedded_dims=None):
        if embedded_dims is None:
            embedded_dims = {"race": 3, "ethnicity": 3, "interaction": 3, "protocol": 3}
        covariates = [y for x, dims in embedded_dims.items() for y in [x]*dims]
        covariates = covariates + [var for var, type_ in cols["Covariates"].items() if type_ != "categorical"]
        covariates = list(dataset.genes) + covariates
        regs = np.abs(self.first_weights).sum(axis=0)
        ranked = pd.DataFrame(zip(covariates, regs), columns=["covariate", "reg_weight"]).sort_values("reg_weight")
        return ranked


# TODO:
class TestResultsCollection:
    def __init__(self, results: list[TestResult], methods=None):
        self.results = results
        self.methods = methods

    def tabulate(self):
        tables = [x.tabulate() for x in self.results]
        table = pd.concat(tables, axis=0)
        return table

    def make_agg_table(self):
        table = self.tabulate()
        table = (table[["alpha", "n_genes", "n_epochs",
                        "pll_train_mean", "pll_test_mean",
                        "loss_train", "loss_test",
                        "ctd", "ibs",
                        "classify_loss_train", "classify_loss_test"]]
                 .groupby(["name"])
                 .mean())
        return table

    def _plot(self, plot_fn):
        rows = len(self.methods)
        figure = make_subplots(rows=rows, cols=1)
        for i in range(rows):
            results = self.results[i]
            subplots = results.__getattribute__(plot_fn)(_as_subplot=True)
            for subplot in subplots:
                figure.add_trace(subplot, row=i + 1, col=1)
        return figure

    def plot_method_loss(self):
        figs = []
        for method in self.methods:
            results = [result for result in self.results if method in result.name]
            fig = make_subplots(rows=1,
                                cols=len(results),
                                x_title="Alpha", y_title="Fold")
            for i, result in enumerate(results, start=1):
                traces = result.plot_loss(_as_subplot=True)
                for trace in traces:
                    fig.add_trace(trace, row=1, col=i)
            fig.update_layout(title=method)
            fig.update_yaxes(range=[0, 1])
            figs.append(fig)
        return figs

    def plot_loss(self):
        figure = self._plot("plot_loss")
        return figure

    def plot_pll(self):
        figure = self._plot("plot_pll")
        return figure

    def plot_lr(self):
        figure = self._plot("plot_lr")
        return figure
