
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings

from sklearn.exceptions import FitFailedWarning
from sklearn import set_config
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sksurv.datasets import load_breast_cancer
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder

from torch import tensor, float32

set_config(display="text")  # displays text representation of estimators

def plot_coefficients(coefs):
    non_zero_means = coefs.iloc[:, -50:].abs().mean(axis=1) > 0
    coefs = coefs.loc[non_zero_means]
    fig = go.Figure()
    data = coefs.T
    data.index.names = ["alpha"]
    data.reset_index(inplace=True)
    x = data["alpha"]
    data = data.drop("alpha", axis=1)
    for gene in data.columns:
        fig.add_trace(go.Scatter(x=x, y=data[gene], mode="lines+markers", name=gene))
    fig.update_xaxes(type="log")
    fig.update_layout(xaxis_title="alpha", yaxis_title="coefficient")
    return fig

# TODO: Need to do something to get the training set of the expression data in a
#  pd.DataFrame with the column names. Worked earlier by doing the split,
#  then iloc subsetting the original expression DataFrame using the splits, which are
#  stored in the splits dict (expression["splits"][0]).


def make_lasso_data(original_data, training_splits):
    data = original_data.iloc[training_splits]
    return data


def test_lasso_penalties(data, outcomes, l1_ratio=1):
    outcomes = np.array(list(zip([x for x in outcomes[1]],
                                 [x for x in outcomes[0]])),
                        dtype="bool,f")
    # cox_lasso = CoxnetSurvivalAnalysis(l1_ratio=1.0, alpha_min_ratio=0.0001)
    # cox_lasso = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=0.00001,
    #                                    n_alphas=20, verbose=True)
    cox_lasso = CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, alpha_min_ratio=0.01,
                                       n_alphas=20, verbose=True)
    cox_lasso.fit(data, outcomes)
    coefficients_lasso = pd.DataFrame(cox_lasso.coef_, index=data.columns,
                                      columns=np.round(cox_lasso.alphas_, 5))
    return coefficients_lasso


def get_non_zero_genes(coefficients):
    coefficients = coefficients.reset_index()
    coefficients = coefficients.set_index("index", append=True)
    coefficients.index.names = ["index", "gene"]
    non_zeros = {x: list(zip(*series.index)) for x in coefficients
                 if not (series := coefficients[x].loc[coefficients[x] != 0]).empty}
    return non_zeros


def alpha_estimate_cross_validation(data, outcomes):
    outcomes = np.array(list(zip([x for x in outcomes[1]],
                                 [x for x in outcomes[0]])),
                        dtype="bool,f")

    coxnet_pipe = make_pipeline(StandardScaler(),
                                CoxnetSurvivalAnalysis(l1_ratio=0.9,
                                                       alpha_min_ratio=0.01,
                                                       n_alphas=100,
                                                       max_iter=100))
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FitFailedWarning)
    coxnet_pipe.fit(data, outcomes)

    estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
    print(estimated_alphas)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    print("Starting grid search...")
    gcv = GridSearchCV(make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=1)),
                       param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in map(float, estimated_alphas)]},
                       cv=cv,
                       error_score=0.5,
                       n_jobs=-1,
                       verbose=3).fit(data, outcomes)
    return gcv

    cv_results = pd.DataFrame(gcv.cv_results_)


    alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score


    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(alphas, mean)
    ax.fill_between(alphas, mean - std, mean + std, alpha=0.15)
    ax.set_xscale("log")
    ax.set_ylabel("concordance index")
    ax.set_xlabel("alpha")
    ax.axvline(gcv.best_params_["coxnetsurvivalanalysis__alphas"][0], c="C1")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.grid(True)

    best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
    best_coefs = pd.DataFrame(best_model.coef_, index=data.columns, columns=[
        "coefficient"])

    non_zero = np.sum(best_coefs.iloc[:, 0] != 0)
    print(f"Number of non-zero coefficients: {non_zero}")

    non_zero_coefs = best_coefs.query("coefficient != 0")
    coef_order = non_zero_coefs.abs().sort_values("coefficient").index

    _, ax = plt.subplots(figsize=(6, 8))
    non_zero_coefs.loc[coef_order].plot.barh(ax=ax, legend=False)
    ax.set_xlabel("coefficient")
    ax.grid(True)
