import pandas as pd
from pandas import IndexSlice as idx


def read_count_data(file, geneset, calculate_tpm_=True):
    exp_data = pd.read_csv(file, sep="\t",
                           index_col=[0, 1], header=[0, 1])
    exp_data = exp_data.loc[idx[:, "stranded_first"],].droplevel(1)
    exp_data = exp_data.drop([x for x in exp_data.columns if exp_data[x].sum() == 0],
                             axis=1)
    for col in exp_data.columns:
        if not geneset.lookup_ensembl_id(col[0].split(".")[0]):
            exp_data.drop(col, inplace=True, axis=1)
    if calculate_tpm_:
        exp_data = calculate_tpm(exp_data, geneset)
    return exp_data


def calculate_tpm(data, geneset, copy=False):
    if copy:
        data = data.copy()
    for x in data.columns:
        result = geneset.lookup_ensembl_id(x[0].split(".")[0])
        data.loc[:, x] = 1000 * data[x] / result.canonical_transcript_length
    data = data.T
    data = 1e6 * data / data.sum(axis=0)
    data = data.T
    return data
