
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import IndexSlice as idx
from tqdm import tqdm

from amlml.survival_data import read_clinical_data


def read_sample(sample_file, count_col=5):
    _id = Path(sample_file).stem.split(".")[0]
    sample = pd.read_csv(sample_file, sep="\t", comment="#", usecols=[0,1,2,count_col])
    sample = sample.loc[sample.gene_type == "protein_coding"]
    # sample = sample.loc[~ sample.gene_id.str.startswith("N_")]
    # sample = sample.loc[~ sample.gene_name.str.startswith("MT-")]
    sample.drop("gene_type", axis=1, inplace=True)
    sample.set_index(["gene_id", "gene_name"], inplace=True)
    sample.columns = [_id]
    return sample


def read_samples(sample_list):
    with open(sample_list) as infile:
        sample_files = infile.readlines()
    sample_files = [sample_file.rstrip() for sample_file in sample_files]
    samples = [read_sample(sample_file) for sample_file in tqdm(sample_files)]
    samples = pd.DataFrame().join(samples, how="outer")
    samples = samples.T
    return samples


def read_count_data2(file, geneset, remove_version=True, remove_unexpressed=True):
    exp_data = pd.read_csv(file, sep="\t", index_col=0, header=[0, 1])

    # Remove unidentified genes, based on presence in GTF file.
    unknown_gene_mask = [col for col in exp_data.columns
                         if geneset.lookup_ensembl_id(col[0].split(".")[0])]
    exp_data = exp_data.loc[:, unknown_gene_mask]

    if remove_unexpressed:
        exp_data = exp_data.loc[:, exp_data.sum(axis=0) > 0]

    if remove_version:
        cols = [(x[0].split(".")[0], x[1]) for x in exp_data.columns]
        if len(cols) != len(set(cols)):
            raise ValueError("Multiple versions of genes detected.")
        exp_data.columns = pd.MultiIndex.from_tuples(cols)
    return exp_data


def read_count_data(file, geneset, calculate_tpm_=True, remove_version=True):
    exp_data = pd.read_csv(file, sep="\t",
                           index_col=[0, 1], header=[0, 1])
    exp_data = exp_data.loc[idx[:, "stranded_first"],].droplevel(1)
    exp_data = exp_data.drop([x for x in exp_data.columns if exp_data[x].sum() == 0],
                             axis=1)
    for col in exp_data.columns:
        if not geneset.lookup_ensembl_id(col[0].split(".")[0]):
            exp_data.drop(col, inplace=True, axis=1)
    if calculate_tpm_:
        exp_data = calculate_tpm(exp_data, geneset, ensembl_level=0)
    if remove_version:
        cols = [(x[0].split(".")[0], x[1]) for x in exp_data.columns]
        if len(cols) != len(set(cols)):
            raise ValueError("Multiple versions of genes detected.")
        exp_data.columns = cols
    return exp_data


# def calculate_tpm(data, geneset, ensembl_level, copy=True):
#     if copy:
#         data = data.copy()
#     for x in data.columns:
#         result = geneset.lookup_ensembl_id(x[ensembl_level].split(".")[0])
#         # data[x] = np.array(data[x], dtype=np.float64)
#         data.loc[:, x] = 1000 * data[x] / result.canonical_transcript_length
#     data = data.T
#     data = 1e6 * data / data.sum(axis=0)
#     data = data.T
#     return data


def calculate_tpm(data, geneset, ensembl_level):
    lengths = np.array(
        [geneset.lookup_ensembl_id(x[ensembl_level].split(".")[0]).canonical_transcript_length
         for x in data.columns]
        )
    tpm = 1000 * data / lengths
    tpm = tpm.T
    tpm = 1e6 * tpm / tpm.sum(axis=0)
    tpm = tpm.T
    return tpm


def replace_with_tpm(data, geneset):
    tpm = calculate_tpm(data[["Expression"]], geneset, 1).copy()
    data = data[["Outcomes", "Covariates"]].join(tpm)
    return data


def read_rna_dataset(expression_data, clinical_data, clinical_yaml, geneset,
                     median_tpm_above_quantile=None, minimum_expression=None,
                     variance_cutoff=None, return_tpm=False):
    # expression = read_count_data(expression_data, geneset, calculate_tpm_=True)
    clinical, cols = read_clinical_data(clinical_data, clinical_yaml)
    expression = read_count_data2(expression_data, geneset).loc[clinical.reset_index("case_name").index]
    expression = filter_rna_dataset(expression, geneset, median_tpm_above_quantile,
                                    minimum_expression, variance_cutoff)
    expression.columns = pd.MultiIndex.from_tuples([("Expression", x[0]) for x in
                                                    expression.columns])
    # TODO: Should this TPM calculation be removed and recalculated later after subsetting genes?
    if return_tpm:
        expression = calculate_tpm(expression, geneset, ensembl_level=0)
    # expression.columns = pd.MultiIndex.from_tuples([("Expression", x[0]) for x in expression.columns])
    data = (clinical
            .reset_index("case_name")
            .join(expression, how="left")
            .set_index("case_name", append=True))
    data = data.droplevel(0, axis=0)
    return data, cols


def filter_rna_dataset(dataset, geneset, median_tpm_above_quantile=None,
                       minimum_expression=None, variance_cutoff=None):
    tpm = calculate_tpm(dataset, geneset, 0)

    if median_tpm_above_quantile is not None:
        medians = tpm.median(axis=0)
        cutoff = medians.quantile(median_tpm_above_quantile)
        tpm = tpm.loc[:, medians < cutoff]
        filtered = dataset.loc[:, tpm.columns]
    else:
        filtered = dataset.copy()

    if minimum_expression is not None:
        min_tpm = minimum_expression[0]
        min_cases = dataset.shape[0] * minimum_expression[1]
        min_expressed = (tpm > min_tpm).sum(axis=0) > min_cases
        filtered = filtered.loc[:, min_expressed]

    # remove genes with low variance.
    if variance_cutoff is not None:
        filtered = filtered.loc[:, filtered.var() >= variance_cutoff]
    return filtered
