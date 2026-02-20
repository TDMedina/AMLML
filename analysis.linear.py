
import numpy as np
import pandas as pd
from qnorm import quantile_normalize
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored as cdex

from amlml.data_loader import prepare_zscore_expression, main_loader
from amlml.cross_normalization import zscore_normalize_genes_by_group, zscore_normalize

train, test = main_loader(prepare_zscore_expression)

# TODO: Finish this.
train_data = quantile_normalize(train.make_expression_table().drop("Tech", axis=1).astype("float64").T).T
train_data = zscore_normalize(train_data)
test_data = quantile_normalize(test.make_expression_table().drop("Tech", axis=1).astype("float64").T).T

train_data = zscore_normalize_genes_by_group(train.make_expression_table())
test_data = zscore_normalize_genes_by_group(test.make_expression_table())


train_outcomes = np.array(list(zip([x for x in train.outcomes[1]],
                                   [x for x in train.outcomes[0]])),
                          dtype="bool,f")
test_outcomes = np.array(list(zip([x for x in test.outcomes[1]],
                                  [x for x in test.outcomes[0]])),
                         dtype="bool,f")

model = CoxnetSurvivalAnalysis(l1_ratio=1, alpha_min_ratio=1/64,
                               n_alphas=10, alphas=None, verbose=True)
model.fit(train_data, train_outcomes)

coef = pd.DataFrame(model.coef_, index=train_data.columns,
                    columns=np.round(model.alphas_, 5))


def make_score_table(data, outcomes, model, name):
    scores = pd.DataFrame([cdex(*list(zip(*outcomes)), model.predict(data, x))
                       for x in model.alphas_],
                      index=np.round(model.alphas_, 5),
                      columns=["cindex", "concordant", "discordant", "tied_risk", "tied_time"])
    scores.index.names = ["alpha"]
    scores["genes"] = (coef > 0).sum(axis=0)
    scores["group"] = name
    scores = scores.reset_index().set_index(["group", "genes", "alpha"])
    return scores


train_scores = make_score_table(train_data, train_outcomes, model, "train")
test_scores = make_score_table(test_data, test_outcomes, model, "test")
scores = pd.concat([train_scores, test_scores]).sort_index()
