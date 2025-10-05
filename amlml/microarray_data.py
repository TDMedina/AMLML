
import pandas as pd

from amlml.survival_data import read_clinical_data

def read_microarray_table(input_file):
    data = pd.read_csv(input_file, sep="\t", index_col=0)
    data = data.T
    data.index.names = ["case_name"]
    return data


def read_microarray_dataset(expression_data, clinical_data, clinical_yaml, geneset):
    expression = read_microarray_table(expression_data)
    expression.columns = pd.MultiIndex.from_tuples([("Expression", x) for x in expression.columns])
    clinical, cols = read_clinical_data(clinical_data, clinical_yaml)
    clinical = clinical.droplevel("file_root", axis=0)
    data = clinical.join(expression, how="inner")
    return data, cols


def read_gse37642(input_file, clinical_data, clinical_yaml):
    expression = pd.read_csv(input_file, sep="\t", index_col=0)
    expression.columns = [x.split("_")[0] for x in expression.columns]
    expression = expression.T
    expression.index.names = ["case_name"]
    expression.columns = pd.MultiIndex.from_tuples([("Expression", x) for x in expression.columns])

    clinical, cols = read_clinical_data(clinical_data, clinical_yaml)
    clinical = clinical.droplevel("GEO_ID", axis=0)
    data = clinical.join(expression, how="inner")
    return data, cols
