
from itertools import product, groupby
from typing import Literal

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from sklearn.metrics import (precision_recall_fscore_support, classification_report,
                             accuracy_score, roc_curve, precision_recall_curve,
                             auc, average_precision_score)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression


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


def calibrate_predictions(train_classes, test_classes):
    smoosh = lambda x: x.to_numpy().reshape(-1, 1)
    calibrator = LogisticRegression(C=1e5)  # High C for minimal regularization
    calibrator.fit(smoosh(train_classes.Raw), train_classes.Truth)
    calibrated = calibrator.predict_proba(smoosh(test_classes.Raw))[:, -1]
    return calibrated, calibrator


class TestResult:
    def __init__(self, fold, alpha, alpha_index, genes,
                 n_epochs, lrs, losses_train, losses_test,
                 pll_train=None, pll_test=None, network=None,
                 hazards_train=None, hazards_test=None, hazards_baseline=None,
                 ctd=None, ibs=None,
                 risk_splits=None, risk_split_quantiles=None, risk_split_counts=None, logranks=None,
                 survival_train=None, survival_test=None, first_weights=None,
                 classes_train=None, classes_test=None,
                 classify_loss_train=None, classify_loss_test=None,
                 parameters=None, name=None, km_table_train=None, km_table_test=None,
                 classifier=None, norm_method=None, clinical=None, leaky=None,
                 l1_method=None, shallow=None, rmst=None, qnorm=None, threshold=None):
        self.fold = fold
        self.alpha = alpha
        self.alpha_index = alpha_index
        self.genes = genes
        self.n_epochs = n_epochs
        self.lrs = lrs
        self.losses_train = losses_train
        self.losses_test = losses_test
        self.pll_train = pll_train
        self.pll_test = pll_test

        self.network = network

        self.hazards_train = hazards_train
        self.hazards_test = hazards_test
        self.hazards_baseline = hazards_baseline
        self.ctd = ctd
        self.ibs = ibs
        self.risk_splits = risk_splits
        self.risk_split_quantiles = risk_split_quantiles
        self.risk_split_counts = risk_split_counts
        self.logranks = logranks

        self.survival_train = survival_train
        self.survival_test = survival_test
        self.km_table_train = km_table_train
        self.km_table_test = km_table_test

        self.classes_train = classes_train
        self.classes_test = classes_test

        self.classify_loss_train = classify_loss_train
        self.classify_loss_test = classify_loss_test

        self.first_weights = first_weights
        self.parameters = parameters

        self.name = name
        # self.classifier = classifier
        # self.norm_method = norm_method
        # self.clinical = clinical
        # self.leaky = leaky
        # self.l1_method = l1_method
        # self.shallow = shallow
        # self.rmst = rmst
        # self.qnorm = qnorm
        # self.threshold = threshold

    # def build_path_name(self):
    #     network_type = "classify" if self.classifier else "hazard"
    #     clinical = "with" if self.clinical else "without"
    #     l1_method = self.l1_method
    #     depth = "shallow" if self.shallow else "deep"
    #     rmst = "with" if self.rmst else "without"
    #     qnorm = "with" if self.qnorm else "without"
    #     thresh = f"{self.threshold/365:.1f}"
    #     leaky = "leaky_relu" if self.leaky else "relu"
    #     outname = (f"results_{network_type}.{thresh}.{depth}.{clinical}_clinical.{rmst}_rmst"
    #                f".{l1_method}_{qnorm}_qnorm.{leaky}")
    #     return outname

    def tabulate(self, include_name=True, classify=False):
        best_epoch = self.n_epochs-1
        flatten = lambda thing: (thing if (thing is None or len(thing) > 1) else thing[0])
        risk_splits = flatten(self.risk_splits)
        risk_split_quantiles = flatten(self.risk_split_quantiles)
        risk_counts = self.risk_split_counts.to_dict() if self.risk_split_counts is not None else None
        accuracy, macro = self.classification_accuracy()
        roc_auc, youden, euclid = self.roc()[1:]
        if self.logranks is None:
            logranks = None
        else:
            logranks = [round(x.p_value, 3) for x in self.logranks.values()]
            if len(logranks) == 1:
                logranks = logranks[0]
        if self.classes_test is None:
            table = pd.DataFrame([{
                ("model", "alpha_index"): self.alpha_index,
                ("model", "fold"): self.fold,
                ("model", "alpha"): self.alpha,
                ("model", "n_genes"): len(self.genes),
                ("model", "n_epochs"): self.n_epochs,
                ("model", "loss_train"): round(self.losses_train[best_epoch], 4),
                ("model", "loss_test"): round(self.losses_test[best_epoch], 4),

                ("hazard", "pll_train_mean"): None,
                ("hazard", "pll_test_mean"): None,
                ("hazard", "ctd"): None,
                ("hazard", "ibs"): None,
                ("hazard", "risk_splits"): None,
                ("hazard", "risk_split_quantiles"): None,
                ("hazard", "risk_counts"): None,
                ("hazard", "logranks"): None,

                ("classify", "loss_train"): None,
                ("classify", "loss_test"): None,
                ("classify", "accuracy"): None,
                ("classify", "macro"): None,
                ("classify", "pr_auc"): None,
                ("classify", "roc_auc"): None,
                ("classify", "roc_youden"): None,
                ("classify", "roc_euclid"): None,

                }])
        else:
            table = pd.DataFrame([{
                ("model", "alpha_index"): self.alpha_index,
                ("model", "fold"): self.fold,
                ("model", "alpha"): self.alpha,
                ("model", "n_genes"): len(self.genes),
                ("model", "n_epochs"): self.n_epochs,
                ("model", "loss_train"): round(self.losses_train[best_epoch], 4),
                ("model", "loss_test"): round(self.losses_test[best_epoch], 4),

                ("hazard", "pll_train_mean"): self.pll_train.mean() if self.pll_train is not None else None,
                ("hazard", "pll_test_mean"): self.pll_test.mean() if self.pll_test is not None else None,
                ("hazard", "ctd"): self.ctd,
                ("hazard", "ibs"): self.ibs,
                ("hazard", "risk_splits"): risk_splits,
                ("hazard", "risk_split_quantiles"): risk_split_quantiles,
                ("hazard", "risk_counts"): risk_counts,
                ("hazard", "logranks"): logranks,

                ("classify", "loss_train"): round(self.classify_loss_train, 4),
                ("classify", "loss_test"): round(self.classify_loss_test, 4),
                ("classify", "accuracy"): round(accuracy, 4),
                ("classify", "macro"): round(macro[2], 4) if macro else None,
                ("classify", "pr_auc"): round(self.precision_recall()[1], 4),
                ("classify", "roc_auc"): round(roc_auc, 4) if roc_auc else None,
                ("classify", "roc_youden"): str([float(round(x, 2)) for x in youden]),
                ("classify", "roc_euclid"): str([float(round(x, 2)) for x in euclid]),

                }])
        if include_name:
            table[("model", "name")] = self.name
            table.set_index([("model", "name"), ("model", "alpha_index"), ("model", "fold")],
                            inplace=True)
            table.index.names = [x[1] for x in table.index.names]
        table.columns = pd.MultiIndex.from_tuples(table.columns)
        if classify:
            table = table[["model", "classify"]]
            table = table.drop([("classify", "loss_train"), ("classify", "loss_test")],
                               axis=1)
        return table

    def plot_loss(self, _as_subplot=False):
        subplots = []
        data = [
            ("Train", "blue", self.losses_train),
            ("Test", "red", self.losses_test)
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
        figure.add_vline(x=self.n_epochs)
        return figure

    def plot_pll(self):
        fig = go.Figure()
        for name, data in [("Train", self.pll_train), ("Test", self.pll_test)]:
            fig.add_trace(go.Histogram(x=data.abs(), name=name))
        return fig

    def plot_lr(self, _as_subplot=False):
        subplot = go.Scatter(x=list(range(1, len(self.lrs)+1)), y=self.lrs,
                             mode="markers+lines", name="Learning Rate",
                             marker_color="blue", line_color="blue")
        if _as_subplot:
            return [subplot]
        figure = go.Figure(subplot)
        figure.add_vline(x=self.n_epochs)
        return figure

    def tabulate_predictions(self, data=Literal["train", "test"]):
        data = self.__getattribute__(f"classes_{data}")
        table = (data.groupby(data.title())["Predicted"]
                 .agg(group0=lambda x: (x < 0.5).sum(),
                      group1=lambda x: (x >= 0.5).sum()))
        return table

    def plot_predictions(self):
        names = ["Train", "Test"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=names)
        for i, (name, data) in enumerate(zip(names, [self.classes_train, self.classes_test]), start=1):
            fig.add_trace(go.Scatter(x=data.Truth, y=data.Predicted,
                                     marker_color=data.Event, mode="markers"),
                          row=1, col=i)
        return fig

    def plot_calibration_curve(self, calibrated=False):
        predicted = "Calibrated" if calibrated else "Predicted"
        caltrue, calpred = calibration_curve(self.classes_test.Truth,
                                             self.classes_test[predicted], n_bins=10)
        fig = go.Figure(go.Scatter(x=calpred, y=caltrue, mode="lines"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                 line_dash="dash", line_color="red"))
        return fig

    def roc(self, calibrated=False):
        if self.classes_test is None:
            return None, None, None, None
        predicted = "Calibrated" if calibrated else "Predicted"
        data = (self.classes_test.Truth, self.classes_test[predicted])
        roc = pd.DataFrame(dict(zip(["FPR", "TPR", "Threshold"], roc_curve(*data))))
        roc_auc = auc(roc.FPR, roc.TPR)
        youden = roc.iloc[np.argmax(roc.TPR - roc.FPR)]
        euclid = roc.iloc[np.argmin(np.sqrt(roc.FPR ** 2 + (roc.TPR - 1) ** 2))]
        return roc, roc_auc, youden, euclid

    def precision_recall(self, calibrated=False):
        if self.classes_test is None:
            return None, None
        predicted = "Calibrated" if calibrated else "Predicted"
        data = (self.classes_test.Truth, self.classes_test[predicted])
        pr = list(precision_recall_curve(*data))
        pr[-1] = np.append(pr[-1], 1)
        pr = pd.DataFrame(dict(zip(["Precision", "Recall", "Threshold"], pr)))
        pr_auc = average_precision_score(*data)
        return pr, pr_auc

    def plot_roc_and_pr_curve(self, calibrated=False):
        roc, roc_auc, youden, euclid = self.roc(calibrated=calibrated)
        pr, pr_auc = self.precision_recall(calibrated=calibrated)

        fig = make_subplots(rows=1, cols=2, subplot_titles=["ROC Curve", "PR Curve"])
        fig.add_trace(
            trace=go.Scatter(x=roc.FPR, y=roc.TPR, mode="lines",
                             customdata=pr.Threshold,
                             hovertemplate=("<b>Threshold:</b> %{customdata:.4f}<br>"
                                            "<b>FPR:</b> %{x:.4f}<br>"
                                            "<b>TPR:</b> %{y:.4f}<br>"
                                            "<extra></extra>")),
            row=1, col=1)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                                 line_dash="dash", line_color="red"), row=1, col=1)
        fig.update_xaxes(title_text="FPR", row=1, col=1)
        fig.update_yaxes(title_text="TPR", row=1, col=1)
        fig.add_annotation(text=f"ROC AUC={roc_auc:.3f}",
                           xref="x1", yref="y1",
                           x=0.75, y=0.25, showarrow=False)
        fig.add_annotation(text=f"Youden J={youden.Threshold:.3f}",
                           xref="x1", yref="y1",
                           x=youden.FPR, y=youden.TPR, showarrow=True)
        fig.add_annotation(text=f"Euclidean={euclid.Threshold:.3f}",
                           xref="x1", yref="y1",
                           x=euclid.FPR, y=euclid.TPR, showarrow=True)

        fig.add_trace(
            trace=go.Scatter(x=pr.Recall, y=pr.Precision, mode="lines",
                             customdata=pr.Threshold,
                             hovertemplate=("<b>Threshold:</b> %{customdata:.4f}<br>"
                                            "<b>Recall:</b> %{x:.4f}<br>"
                                            "<b>Precision:</b> %{y:.4f}<br>"
                                            "<extra></extra>")),
            row=1, col=2)
        baseline = self.classes_test.Truth.mean()
        fig.add_hline(y=baseline, row=1, col=2, line_dash="dash", line_color="red",
                      showlegend=False)
        fig.update_xaxes(title_text="Recall", row=1, col=2)
        fig.update_yaxes(title_text="Precision", row=1, col=2)
        fig.add_annotation(text=f"PR AUC = {pr_auc:.2f}", xref="x2", yref="y2",
                           x=0.5, y=baseline+0.1, showarrow=False)
        fig.update_layout(showlegend=False)

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

    def classification_report(self):
        report = classification_report(self.classes_test.Truth,
                                       self.classes_test.Classification)
        return report

    def classification_accuracy(self):
        if self.classes_test is None:
            return None, None
        accuracy = accuracy_score(self.classes_test.Truth,
                                  self.classes_test.Classification)
        macro = precision_recall_fscore_support(self.classes_test.Truth,
                                                self.classes_test.Classification,
                                                average="macro")
        return accuracy, macro


class TestResultCollection:
    def __init__(self, results: list[TestResult], methods=None, folds_per=None):
        self.results = results
        self.methods = methods
        self.folds_per = folds_per

    def group_by_model(self):
        key = lambda x: (x.name, x.alpha_index)
        results = sorted(self.results, key=key)
        results = {id_: TestResultCollection(list(group))
                   for id_, group in groupby(results, key=key)}
        return results

    def tabulate(self, classify=False):
        tables = [x.tabulate(classify=classify) for x in self.results]
        table = pd.concat(tables, axis=0)
        return table

    def make_agg_table(self, classify=False):
        if classify:
            drop = list(product(["classify"], ["roc_youden", "roc_euclid"]))
        else:
            drop = [*product(["hazard"], ["risk_splits", "risk_split_quantiles",
                                          "risk_counts", "logranks"]),
                    *product(["classify"], ["roc_youden", "roc_euclid", "loss_train", "loss_test"])]

        table = self.tabulate(classify=classify).drop(drop, axis=1)
        table = table.groupby(["name", "alpha_index"]).mean()
        return table

    def make_oof_class_tables(self):
        models = self.group_by_model()
        model_tables = {id_: pd.concat([result.classes_test for result in model.results])
                        for id_, model in models.items()}
        return model_tables

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

    def plot_method_loss(self, y_range=None):
        if y_range is None:
            y_range = [0, 1]
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
            fig.update_yaxes(range=y_range)
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

# class TestResult(CV_Result):
#     def __init__(self, alpha, genes,
#                  n_epochs, lrs, losses_train, losses_test,
#                  pll_train=None, pll_test=None, network=None,
#                  hazards_train=None, hazards_test=None, hazards_baseline=None,
#                  ctd=None, ibs=None,
#                  risk_splits=None, risk_split_quantiles=None, risk_split_counts=None, logranks=None,
#                  survival_train=None, survival_test=None, first_weights=None,
#                  classes_train=None, classes_test=None,
#                  classify_loss_train=None, classify_loss_test=None,
#                  parameters=None, name=None, km_table_train=None, km_table_test=None):
#         super().__init__(fold=0, alpha=alpha, alpha_index=0, genes=genes,
#               n_epochs=n_epochs, lrs=lrs,
#               losses_train=losses_train, losses_val=losses_test,
#               pll_train=pll_train, pll_val=pll_test,
#               network=network,
#               hazards_train=hazards_train, hazards_val=hazards_test, hazards_baseline=hazards_baseline,
#               ctd=ctd, ibs=ibs,
#               risk_splits=risk_splits, risk_split_quantiles=risk_split_quantiles,
#               risk_split_counts=risk_split_counts, logranks=logranks,
#               survival_train=survival_train, survival_val=survival_test,
#               first_weights=first_weights,
#               classes_train=classes_train, classes_val=classes_test,
#               classify_loss_train=classify_loss_train, classify_loss_val=classify_loss_test,
#               parameters=parameters,
#               name=name,
#               km_table_train=km_table_train, km_table_val=km_table_test)
#
# class TestResult:
#     def __init__(self, alpha, genes,
#                  n_epochs, lrs, losses_train, loss_test,
#                  pll_train=None, pll_test=None, network=None,
#                  hazards_train=None, hazards_test=None, hazards_baseline=None,
#                  ctd=None, ibs=None,
#                  risk_splits=None, risk_split_quantiles=None, risk_split_counts=None, logranks=None,
#                  survival_train=None, survival_test=None, first_weights=None,
#                  classes_train=None, classes_test=None,
#                  classify_loss_train=None, classify_loss_test=None,
#                  parameters=None, name=None, km_table_train=None, km_table_test=None):
#         self.alpha = alpha
#         self.genes = genes
#         self.n_epochs = n_epochs
#         self.lrs = lrs
#         self.losses_train = losses_train
#         self.loss_test = loss_test
#         self.pll_train = pll_train
#         self.pll_test = pll_test
#
#         self.network = network
#
#         self.hazards_train = hazards_train
#         self.hazards_test = hazards_test
#         self.hazards_baseline = hazards_baseline
#         self.ctd = ctd
#         self.ibs = ibs
#         self.risk_splits = risk_splits
#         self.risk_split_quantiles = risk_split_quantiles
#         self.risk_split_counts = risk_split_counts
#         self.logranks = logranks
#
#         self.survival_train = survival_train
#         self.survival_test = survival_test
#         self.km_table_train = km_table_train
#         self.km_table_test = km_table_test
#
#         self.classes_train = classes_train
#         self.classes_test = classes_test
#
#         self.classify_loss_train = classify_loss_train
#         self.classify_loss_test = classify_loss_test
#
#         self.first_weights = first_weights
#         self.parameters = parameters
#
#         self.name = name
#
#     def tabulate(self, include_name=True):
#         risk_splits = (self.risk_splits
#                        if (self.risk_splits is None or len(self.risk_splits) > 1)
#                        else self.risk_splits[0])
#         if self.logranks is None:
#             logranks = None
#         else:
#             logranks = [round(x.p_value, 3) for x in self.logranks.values()]
#             if len(logranks) == 1:
#                 logranks = logranks[0]
#         table = pd.DataFrame([dict(
#             alpha=self.alpha,
#             n_genes=len(self.genes),
#             n_epochs=self.n_epochs,
#             loss_train=self.losses_train[-1],
#             loss_test=self.loss_test,
#
#             pll_train_mean=self.pll_train.mean() if self.pll_train is not None else None,
#             pll_test_mean=self.pll_test.mean() if self.pll_test is not None else None,
#             ctd=self.ctd,
#             ibs=self.ibs,
#             risk_splits=risk_splits,
#             logranks=logranks,
#             classify_loss_train = self.classify_loss_train,
#             classify_loss_test = self.classify_loss_test,
#             )])
#         if include_name:
#             table["name"] = self.name
#             table.set_index(["name"], inplace=True)
#         return table
#
#     # TODO: Plots.
#     def plot_loss(self, _as_subplot=False):
#         subplots = []
#         data = [
#             ("training", "blue", self.losses_train),
#             ("validation", "red", self.losses_val)
#             ]
#         for name, color, loss in data:
#                 subplot = go.Scatter(x=list(range(1, len(loss)+1)), y=loss,
#                                      mode="lines", name=name,
#                                      marker_color=color, line_color=color,
#                                      legendgroup=name)
#                 subplots.append(subplot)
#         if _as_subplot:
#             return subplots
#         figure = go.Figure(subplots)
#         return figure
#
#     def plot_pll(self):
#         fig = go.Figure()
#         for name, data in [("Train", self.pll_train), ("Val", self.pll_val)]:
#             fig.add_trace(go.Histogram(x=data.abs(), name=name))
#         return fig
#
#     def plot_lr(self, _as_subplot=False):
#         subplot = go.Scatter(x=list(range(1, len(self.lrs)+1)), y=self.lrs,
#                              mode="markers+lines", name="Learning Rate",
#                              marker_color="blue", line_color="blue")
#         if _as_subplot:
#             return [subplot]
#         figure = go.Figure(subplot)
#         return figure
#
#     def tabulate_predictions(self, data=Literal["train", "val"]):
#         data = self.__getattribute__(f"classes_{data}")
#         table = (data.groupby("Training")["Predicted"]
#                  .agg(group0=lambda x: (x < 0.5).sum(),
#                       group1=lambda x: (x >= 0.5).sum()))
#         return table
#
#     def plot_predictions(self):
#         fig = make_subplots(1, 2)
#         for i, (name, data) in enumerate(zip(["Train", "Val"],
#                                          [self.classes_train, self.classes_val]),
#                                          start=1):
#             fig.add_trace(go.Scatter(x=data.loc[data.Predicted < 0.5, "Training"],
#                                      y=data.loc[data.Predicted < 0.5, "Predicted"],
#                                      mode="markers", name=name), col=i)
#             fig.add_trace(go.Scatter(x=data.loc[data.Predicted >= 0.5, "Training"],
#                                      y=data.loc[data.Predicted >= 0.5, "Predicted"],
#                                      mode="markers", name=name), col=i)
#         return fig
#
#     def rank_covariates(self, dataset, cols, embedded_dims=None):
#         if embedded_dims is None:
#             embedded_dims = {"race": 3, "ethnicity": 3, "interaction": 3, "protocol": 3}
#         covariates = [y for x, dims in embedded_dims.items() for y in [x]*dims]
#         covariates = covariates + [var for var, type_ in cols["Covariates"].items() if type_ != "categorical"]
#         covariates = list(dataset.genes) + covariates
#         regs = np.abs(self.first_weights).sum(axis=0)
#         ranked = pd.DataFrame(zip(covariates, regs), columns=["covariate", "reg_weight"]).sort_values("reg_weight")
#         return ranked
#
#
# # TODO:
# class TestResultsCollection:
#     def __init__(self, results: list[TestResult], methods=None):
#         self.results = results
#         self.methods = methods
#
#     def tabulate(self):
#         tables = [x.tabulate() for x in self.results]
#         table = pd.concat(tables, axis=0)
#         return table
#
#     def make_agg_table(self):
#         table = self.tabulate()
#         table = (table[["alpha", "n_genes", "n_epochs",
#                         "pll_train_mean", "pll_test_mean",
#                         "loss_train", "loss_test",
#                         "ctd", "ibs",
#                         "classify_loss_train", "classify_loss_test"]]
#                  .groupby(["name"])
#                  .mean())
#         return table
#
#     def _plot(self, plot_fn):
#         rows = len(self.methods)
#         figure = make_subplots(rows=rows, cols=1)
#         for i in range(rows):
#             results = self.results[i]
#             subplots = results.__getattribute__(plot_fn)(_as_subplot=True)
#             for subplot in subplots:
#                 figure.add_trace(subplot, row=i + 1, col=1)
#         return figure
#
#     def plot_method_loss(self):
#         figs = []
#         for method in self.methods:
#             results = [result for result in self.results if method in result.name]
#             fig = make_subplots(rows=1,
#                                 cols=len(results),
#                                 x_title="Alpha", y_title="Fold")
#             for i, result in enumerate(results, start=1):
#                 traces = result.plot_loss(_as_subplot=True)
#                 for trace in traces:
#                     fig.add_trace(trace, row=1, col=i)
#             fig.update_layout(title=method)
#             fig.update_yaxes(range=[0, 1])
#             figs.append(fig)
#         return figs
#
#     def plot_loss(self):
#         figure = self._plot("plot_loss")
#         return figure
#
#     def plot_pll(self):
#         figure = self._plot("plot_pll")
#         return figure
#
#     def plot_lr(self):
#         figure = self._plot("plot_lr")
#         return figure
