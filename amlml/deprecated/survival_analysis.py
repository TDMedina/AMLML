
import os
from string import ascii_uppercase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import IndexSlice as idx
import plotly.graph_objects as go
import plotly.express as px
from scipy.integrate import simpson
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from torch import float32, tensor
import torchtuples as tt

from lifelines import KaplanMeierFitter
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

from amlml.parallel_modelling import SuperModel
from amlml.lasso import test_lasso_penalties, get_non_zero_genes
from amlml.km import plot_survival_curves, optimize_survival_splits, iterate_logrank_tests
from amlml.data_loader import main_loader, prepare_supermodel_expression


(expression, outcomes, categoricals, non_categoricals, set_ids, group_labels,
 expression_table) = main_loader(prepare_supermodel_expression)


# %% Cross validation.
def calculate_zscores(data):
    data = (data - data.mean()) / data.std()
    return data


def normalize_alpha_inputs(input_data, labels):
    data = input_data.copy()
    data["int_dex"] = range(data.shape[0])
    data.set_index("int_dex", append=True, inplace=True)
    labels = np.array(labels, dtype=bool)
    rna = calculate_zscores(data.loc[~ labels])
    arrays = calculate_zscores(data.loc[labels])
    data = pd.concat([rna, arrays])
    data.sort_index(level=1, inplace=True)
    data = data.droplevel(level=1)
    return data


def cross_validation_run(expression_data, categorical_data, non_categorical_data,
                         outcome_data, group_data, covariate_cardinality,
                         original_data, l1_ratio=1, iterate_alphas=True,
                         alpha_min_ratio=0.01, n_alphas=21, alphas=None):

    kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    results = dict()

    for i, (train_fold, val_fold) in enumerate(kf.split(expression_data, group_data)):
        # TESTING
        # if i != 4:
        #     continue
        print(f"Running fold {i}...")
        results[i] = dict()
        print(f"    Preparing input data...")
        expression_train = expression_data[train_fold]
        categoricals_train = categorical_data[train_fold]
        non_categoricals_train = non_categorical_data[train_fold]
        outcomes_train_dur = outcome_data[0][train_fold]
        outcomes_train_event = outcome_data[1][train_fold]
        group_train = [group_data[i] for i in train_fold]
        alpha_data = normalize_alpha_inputs(original_data.iloc[train_fold], group_train)

        print(f"    Preparing validation data...")
        expression_val = expression_data[val_fold]
        categoricals_val = categorical_data[val_fold]
        non_categoricals_val = non_categorical_data[val_fold]
        outcomes_val_dur = outcome_data[0][val_fold]
        outcomes_val_event = outcome_data[1][val_fold]

        if iterate_alphas:
            print(f"    Calculating lasso penalties...")
            alpha_table = test_lasso_penalties(alpha_data,
                                               (outcomes_train_dur, outcomes_train_event),
                                               l1_ratio=l1_ratio,
                                               alpha_min_ratio=alpha_min_ratio,
                                               n_alphas=n_alphas,
                                               alphas=alphas)
            geneset = get_non_zero_genes(alpha_table)
        else:
            geneset = {0: (slice(None), list(alpha_data.columns))}
        print("Running validation...")
        for alpha, (index, genes) in tqdm(geneset.items()):
            # TESTING
            # if not np.isclose(alpha, 0.00121, atol=1e-4):
            #     continue
            results[i][alpha] = dict()
            results[i][alpha]["genes"] = ",".join(genes)

            print(f"Preparing training fold for alpha={alpha}...")
            outcomes_train = (outcomes_train_dur, outcomes_train_event)
            outcomes_val = (outcomes_val_dur, outcomes_val_event)

            expression_train_subset = expression_train[:, :, index]
            expression_val_subset = expression_val[:, :, index]
            train_args = (expression_train_subset, categoricals_train, non_categoricals_train)
            val_args = (expression_val_subset, categoricals_val, non_categoricals_val)

            network = SuperModel(n_genes=expression_train_subset.shape[-1],
                                 n_tech=2,
                                 n_expansion=4,
                                 n_clinical=categoricals_train.shape[-1] +
                                            non_categoricals_train.shape[-1],
                                 covariate_cardinality=covariate_cardinality,
                                 embedding_dims={"race": 3, "ethnicity": 3,
                                                 "interaction": 3,
                                                 "protocol": 3},
                                 shrinkage_factor=10,
                                 minimum_size=10,
                                 final_size=1)
            model = CoxPH(network, tt.optim.Adam)


            batch_size = 95
            print("Calculating learning rate...")
            lrfinder = model.lr_finder(train_args, outcomes_train, batch_size,
                                       tolerance=10)
            # lrfinder.plot()
            # plt.show()
            lr_optim = lrfinder.get_best_lr()
            model.optimizer.set_lr(lr_optim)

            epochs = 360
            callbacks = []
            # callbacks = [tt.callbacks.EarlyStopping(patience=30)]
            eps = 0.005
            verbose = True
            losses = []

            print("Running epochs...")
            for epoch in range(epochs):
                log = model.fit(train_args, outcomes_train, batch_size,
                                epochs=1,
                                verbose=verbose)
                log = log.to_pandas()
                losses.append(log.train_loss.iloc[-1])
                if len(losses) < 10:
                    continue
                if np.var(losses[-10:]) < eps:
                    break

            results[i][alpha]["epochs"] = epoch
            # log = model.fit(train_args, outcomes_train, batch_size, epochs, callbacks, verbose,)
            # val_data=(val_args, outcomes_val), val_batch_size=batch_size)

            print("Evaluating performance...")
            results[i][alpha]["pll"] = model.partial_log_likelihood(val_args,
                                                                    outcomes_val).mean()
            # TODO:
            results[i][alpha]["pll_train"] = model.partial_log_likelihood(train_args,
                                                                          outcomes_train).mean()
            model.compute_baseline_hazards()

            surv = model.predict_surv_df(val_args)

            outcomes_val = np.array(outcomes_val)

            ev = EvalSurv(surv, outcomes_val[0], outcomes_val[1], censor_surv="km")
            results[i][alpha]["ctd"] = ev.concordance_td()

            time_grid = np.linspace(outcomes_val[0].min(), outcomes_val[0].max(),
                                    100)
            brier_scores = ev.brier_score(time_grid)
            results[i][alpha]["ibs"] = (simpson(y=brier_scores.values, x=brier_scores.index)
                                        / (brier_scores.index[-1] - brier_scores.index[0]))

            # ev.integrated_nbll(time_grid)

            predictions = [float(x[0]) for x in model.predict(val_args)]
            km_test_df = pd.DataFrame(zip(outcomes_val[0], outcomes_val[1], predictions),
                                      columns=["duration", "event", "risk"])
            print("Calculating survival split...")
            optimal_splits = optimize_survival_splits(km_test_df, n_groups=3)
            risk_splits = np.cumulative_sum(optimal_splits.x)
            # plot_survival_curves(km_test_df)
            logranks = iterate_logrank_tests(km_test_df)
            results[i][alpha]["risk_splits"] = risk_splits
            results[i][alpha]["km_logrank"] = logranks
            print("Done")
    return results, km_test_df


def make_results_table(results):
    fields = ["epochs", "pll", "ctd", "ibs"]
    parsed = []
    for fold, alpha_dict in results.items():
        for alpha, result_dict in alpha_dict.items():
            entry = ([fold, alpha, len(result_dict["genes"].split(","))]
                     + [result_dict[x] for x in fields])
            entry += [tuple(sorted(result_dict["risk_splits"]))]
            # print(fold, alpha, result_dict["km_logrank"].keys())
            entry += [result_dict["km_logrank"][tuple(x)].p_value
                      if tuple(x) in result_dict["km_logrank"] else None
                      for x in ["AB", "BC", "AC"]]
            parsed.append(entry)
    cols = ["fold", "alpha", "gene_count", *fields, "risk_splits", "km_logrank_AB",
            "km_logrank_BC", "km_logrank_AC"]
    table = pd.DataFrame(parsed, columns=cols)
    table["group"] = [i for _ in range(len(results.keys()))
                      for i in ascii_uppercase[:len(list(results.values())[0].keys())]]
    table.set_index(["fold", "group", "alpha"], inplace=True)
    return table


def make_mean_results_table(results_table):
    table = results_table[["gene_count", "pll", "epochs", "ctd", "ibs"]].groupby("group").mean()
    return table


def make_median_results_table(results_table):
    table = results_table[["gene_count", "pll", "epochs", "ctd", "ibs"]].groupby("group").median()
    return table


cv_results = cross_validation_run(
    expression_data=expression["train"],
    categorical_data=categoricals["train"],
    non_categorical_data=non_categoricals["train"],
    outcome_data=outcomes["train"],
    group_data=group_labels["train"],
    covariate_cardinality={"race": 7, "ethnicity": 3, "protocol": 7},
    original_data=expression_table["train"],
    # iterate_alphas=False
    )


def plot_survival_splits(results_table):
    splits = results_table[["gene_count"]].copy()
    splits["split1"] = [x[0] for x in results_table.risk_splits]
    splits["split2"] = [x[1] for x in results_table.risk_splits]
    fig = go.Figure()
    for fold in splits.index.unique(level="fold"):
        data = splits.loc[idx[fold, :, :],]
        fig.add_trace(go.Bar(x=data.reset_index()["group"],
                             y=data.split1,
                             offsetgroup=fold,
                             marker_color="red",
                             marker_line_color="red"))
        fig.add_trace(go.Bar(x=data.reset_index()["group"],
                             y=data.split2-data.split1,
                             offsetgroup=fold,
                             marker_color="blue",
                             marker_line_color="blue"))
        fig.add_trace(go.Bar(x=data.reset_index()["group"],
                             y=1-data.split2,
                             offsetgroup=fold,
                             marker_color="green",
                             marker_line_color="green"))
    fig.update_layout(barmode="stack")

    # splits = splits.reset_index().sort_values(["group", "gene_count"])
    # fig2 = go.Figure()
    # fig2.add_trace(go.Scatter(x=splits.gene_count, y=splits.split1))
    # fig2.add_trace(go.Scatter(x=splits.gene_count, y=splits.split2))
    logranks = ["km_logrank_AB", "km_logrank_BC", "km_logrank_AC"]
    splits = splits.join(results_table[logranks])
    fig3 = go.Figure()
    for logrank in logranks:
        fig3.add_trace(go.Scatter(x=splits.gene_count, y=splits[logrank],
                                  mode="markers"))
    melted = pd.melt(splits[["gene_count"]+logranks], id_vars="gene_count")
    melted["value"] = np.log10(melted["value"])
    fig4 = px.scatter(melted, x="gene_count", y="value", color="variable")
    fig4.add_trace(go.Scatter(x=[0, 1600], y=[np.log10(0.05)]*2, mode="lines",
                              marker_line_color="red"))
    return fig, fig4


def plot_performance_metrics(results_table, plot_mean=False, plot_median=False):
    metrics = ["epochs", "pll", "ctd", "ibs"]
    trimmed = results_table.copy()
    trimmed["pll"] = [max(x, -20) for x in results_table["pll"]]
    title = "Gene counts vs metric"
    if plot_mean:
        trimmed = make_mean_results_table(trimmed)
        title += " means"
    elif plot_median:
        trimmed = make_median_results_table(trimmed)
        title += " medians"
    figs = px.scatter(pd.melt(trimmed[["gene_count"] + metrics], id_vars=["gene_count"]),
                      x="gene_count", y="value", trendline="lowess",
                      facet_col="variable", facet_col_wrap=2)
    figs.update_layout(title=title)
    figs.update_yaxes(matches=None)
    figs.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
    return figs


# %% Define network and model.
network = SuperModel(n_genes=expression["train"].shape[-1],
                     n_tech=2,
                     n_expansion=4,
                     n_clinical=35,
                     covariate_cardinality={"race": 7, "ethnicity": 3, "protocol": 6},
                     embedding_dims={"race": 3, "ethnicity": 3, "interaction": 3, "protocol": 3},
                     shrinkage_factor=10,
                     minimum_size=10,
                     final_size=1)

model = CoxPH(network, tt.optim.Adam)

train_args = (expression["train"], categoricals["train"], non_categoricals["train"])
# validation_args = (expression["validation"], categoricals["validation"], non_categoricals["validation"])
test_args = (expression["test"], categoricals["test"], non_categoricals["test"])


# %% Run the model.

batch_size = 190
lrfinder = model.lr_finder(train_args, outcomes["train"], batch_size, tolerance=10)
# lrfinder.plot()
# plt.show()

lr_optim = lrfinder.get_best_lr()
model.optimizer.set_lr(lr_optim)

epochs = 30
# callbacks = [tt.callbacks.EarlyStopping()]
callbacks = []
verbose = True

log = model.fit(train_args, outcomes["train"], batch_size, epochs, callbacks, verbose),
                # val_data=(validation_args, outcomes["validation"]),
                # val_batch_size=batch_size)
# log.plot()
# plt.show()


# %% Evaluate the model.

# model.partial_log_likelihood(validation_args, outcomes["validation"]).mean()
outcomes["test"] = [tensor(outcomes["test"].iloc[0].to_numpy(), dtype=float32),
                    tensor(outcomes["test"].iloc[1].to_numpy(), dtype=float32)]
model.partial_log_likelihood(test_args, outcomes["test"]).mean()
model.compute_baseline_hazards()

surv = model.predict_surv_df(test_args)
# surv.plot()
# plt.ylabel("S(t | x)")
# plt.xlabel("Time")
# plt.show()

outcomes["test"] = [x.numpy() for x in outcomes["test"]]
ev = EvalSurv(surv, outcomes["test"][0], outcomes["test"][1],
              censor_surv="km")
ev.concordance_td()  #

time_grid = np.linspace(outcomes["test"][0].min(),
                        outcomes["test"][0].max(),
                        100)
ev.brier_score(time_grid).plot()
plt.show()

brier_scores = ev.brier_score(time_grid)
ibs = (simpson(y=brier_scores.values, x=brier_scores.index)
       / (brier_scores.index[-1] - brier_scores.index[0]))

# ev.integrated_nbll(time_grid)

predictions = [float(x[0]) for x in model.predict(test_args)]
km_test_df = pd.DataFrame(zip(*outcomes["test"], predictions),
                          columns=["duration", "event", "risk"])


# %% Survival partitioning.
optimal_splits = optimize_survival_splits(km_test_df, n_groups=3)
risk_splits = np.cumulative_sum(optimal_splits.x)
plot_survival_curves(km_test_df)


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
gene_id_dict = {"ENSG00000104341": "ENSG00000104341.17",
                "ENSG00000159228": "ENSG00000159228.13",
                "ENSG00000088305": "ENSG00000088305.18",
                "ENSG00000120833": "ENSG00000120833.14",
                "ENSG00000129187": "ENSG00000129187.14",
                "ENSG00000131747": "ENSG00000131747.15",
                "ENSG00000138722": "ENSG00000138722.10",
                "ENSG00000166681": "ENSG00000166681.14",
                "ENSG00000205336": "ENSG00000205336.14",
                "ENSG00000128040": "ENSG00000128040.11",
                "ENSG00000196139": "ENSG00000196139.14",
                "ENSG00000088882": "ENSG00000088882.8",
                "ENSG00000103222": "ENSG00000103222.20",
                "ENSG00000113657": "ENSG00000113657.13",
                "ENSG00000105810": "ENSG00000105810.10",
                "ENSG00000095932": "ENSG00000095932.7",
                "ENSG00000130584": "ENSG00000130584.12",
                "ENSG00000005381": "ENSG00000005381.8",
                "ENSG00000128805": "ENSG00000128805.15",
                "ENSG00000134531": "ENSG00000134531.10",
                "ENSG00000205978": "ENSG00000205978.6",
                "ENSG00000174059": "ENSG00000174059.17"}
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

    plt.title("Kaplan-Meier Survival Curves")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
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
# for alpha, genes in tqdm({"ADE-RS5": aders5, "pLSC6": plsc6, "LSC17": lsc17}.items()):
for alpha, genes in tqdm(non_zero_genes.items()):
    expression = data.Expression
    expression = expression.loc[:, list(genes)]
    expression = np.stack([np.array(expression), np.zeros(expression.shape)], axis=0)
    expression = tensor(expression, dtype=float32).permute(1, 0, 2)
    expression = split_to_dict(expression)
    results = test_network(expression, alpha)
    all_results[alpha] = results
    plt.close("all")

# %%
