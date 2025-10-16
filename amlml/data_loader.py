
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import float32, tensor

from amlml.survival_data import prepare_outcomes, code_categoricals, add_nan_mask_stack
from amlml.rna_data import read_rna_dataset, replace_with_tpm
from amlml.microarray_data import read_microarray_dataset, read_gse37642
from amlml.gene_set import GeneSet
from amlml.cross_normalization import log2_transform, zscore_normalize, npn


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
    drop_list = ["TARGET-20-PAPXVK"]  # Filtered due to multiple listed events
    target_aml = target_aml.drop(drop_list)
    all_group_labels = [0]*target_aml.shape[0]

    ## Read TCGA-AML microarray data.
    tcga_aml, _ = read_microarray_dataset(
        expression_data="Data/TCGA_LAML_arrays/tcga_laml.microarray_data.tsv",
        clinical_data="Data/TCGA_LAML_arrays/tcga-aml.clinical.homogenized.incomplete.tsv",
        clinical_yaml="Data/TARGET_AML_gene_counts/Clinical/variables.yaml",
        geneset=geneset
        )
    all_group_labels += [1]*tcga_aml.shape[0]

    # Read GSE37642 HGU133plus2 data.
    gse37642_hgu133plus2, _ = read_gse37642(
        input_file="Data/GSE37642/GSE37642.hgu133plus2.expression.tsv",
        clinical_data="/home/tyler/Documents/Projects/ML/Data/GSE37642/GSE37642_Homogenized_Survival_data.tsv",
        clinical_yaml="Data/TARGET_AML_gene_counts/Clinical/variables.yaml"
        )
    gse37642_hgu133plus2.dropna(subset=[("Outcomes", "Vital Status"),
                                        ("Outcomes", "Overall Survival Time in Days")],
                                inplace=True)
    all_group_labels += [1]*gse37642_hgu133plus2.shape[0]

    # Read GSE37642 HGU133plusA data.
    gse37642_hgu133a, _ = read_gse37642(
        input_file="Data/GSE37642/GSE37642.hgu133A.expression.tsv",
        clinical_data="/home/tyler/Documents/Projects/ML/Data/GSE37642/GSE37642_Homogenized_Survival_data.tsv",
        clinical_yaml="Data/TARGET_AML_gene_counts/Clinical/variables.yaml"
    )
    gse37642_hgu133a.dropna(subset=[("Outcomes", "Vital Status"),
                                    ("Outcomes", "Overall Survival Time in Days")],
                            inplace=True)
    all_group_labels += [1]*gse37642_hgu133a.shape[0]

    set_intersect_of_genes(target_aml, tcga_aml, gse37642_hgu133plus2, gse37642_hgu133a)
    target_aml = replace_with_tpm(target_aml, geneset)

    rna_data = pd.concat([target_aml], axis=0)
    ma_data = pd.concat([tcga_aml, gse37642_hgu133plus2, gse37642_hgu133a], axis=0)
    data = pd.concat([rna_data, ma_data], axis=0)
    ids = list(data.index)
    return data, rna_data, ma_data, cols, ids, all_group_labels


def prepare_supermodel_data(rna_data, ma_data):
    rna_expression = np.stack([np.array(rna_data.Expression),
                               np.zeros(rna_data.Expression.shape)],
                              axis=0)
    ma_expression = np.stack([np.zeros(ma_data.Expression.shape),
                              np.array(ma_data.Expression)],
                             axis=0)
    all_expression = np.concat([rna_expression, ma_expression], axis=1)
    all_expression = tensor(all_expression, dtype=float32).permute(1, 0, 2)
    return all_expression


def prepare_log2_normed(rna_data, ma_data, concatenate=True):
    rna_expression = log2_transform(rna_data.Expression, True, False)
    ma_expression = log2_transform(ma_data.Expression, True, True)
    if not concatenate:
        return rna_expression, ma_expression
    all_expression = np.concat([rna_expression, ma_expression], axis=0)
    all_expression = tensor(all_expression, dtype=float32)
    return all_expression


def prepare_zscore_normed(rna_data, ma_data):
    rna_expression, ma_expression = prepare_log2_normed(rna_data, ma_data, False)
    rna_expression = zscore_normalize(rna_expression)
    ma_expression = zscore_normalize(ma_expression)
    all_expression = np.concat([rna_expression, ma_expression], axis=0)
    all_expression = tensor(all_expression, dtype=float32)
    return all_expression


def prepare_npn_normed(rna_data, ma_data):
    rna_expression, ma_expression = prepare_log2_normed(rna_data, ma_data, False)
    rna_expression = npn(rna_expression)
    ma_expression = npn(ma_expression)
    all_expression = np.concat([rna_expression, ma_expression], axis=0)
    all_expression = tensor(all_expression, dtype=float32)
    return all_expression


def prepare_data(data, rna_data, ma_data, cols,
                 normalization: Callable = prepare_supermodel_data):
    # Prepare expression.
    all_expression = normalization(rna_data, ma_data)

    # Prepare covariates.
    all_categoricals = data["Covariates"][[x for x in cols["Covariates"]
                                           if cols["Covariates"][x] == "categorical"]]
    for x in all_categoricals:
        all_categoricals[x] = pd.Categorical(all_categoricals[x])
    code_categoricals(all_categoricals)
    all_categoricals = tensor(all_categoricals.to_numpy(), dtype=torch.int32)

    all_non_categoricals = data["Covariates"][[x for x in cols["Covariates"]
                                               if cols["Covariates"][x] != "categorical"]]
    all_non_categoricals = tensor(add_nan_mask_stack(all_non_categoricals), dtype=float32)
    all_non_categoricals = all_non_categoricals.permute(1, 0, 2)

    # Prepare outcomes.
    all_outcomes = data.Outcomes
    all_outcomes.columns = ["event", "duration"]
    all_outcomes = all_outcomes[["duration", "event"]]
    all_outcomes.loc[:, "event"] = [int(x == "Dead") for x in all_outcomes.event]

    return all_expression, all_outcomes, all_categoricals, all_non_categoricals


def split_test_data(data, all_expression, all_outcomes, all_categoricals,
                    all_non_categoricals, ids, all_group_labels):
    expression = dict()
    categoricals = dict()
    non_categoricals = dict()
    outcomes = dict()
    set_ids = dict()
    group_labels = dict()
    expression_table = dict()

    (expression["train"], expression["test"],
     categoricals["train"], categoricals["test"],
     non_categoricals["train"], non_categoricals["test"],
     outcomes["train"], outcomes["test"],
     set_ids["train"], set_ids["test"],
     group_labels["train"], group_labels["test"],
     expression_table["train"], expression_table["test"]) = (
        train_test_split(all_expression, all_categoricals, all_non_categoricals, all_outcomes,
                         ids, all_group_labels, data.Expression,
                         test_size=0.2, random_state=0, stratify=all_group_labels)
        )

    outcomes["train"] = prepare_outcomes(np.array(outcomes["train"]))
    outcomes["test"] = outcomes["test"].T.astype(float)
    return (expression, outcomes, categoricals, non_categoricals,
            set_ids, group_labels,
            expression_table)


def main_loader(normalization: Callable):
    data = read_model_data()
    prepared = prepare_data(*data[:4], normalization=normalization)
    (expression, outcomes,
     categoricals, non_categoricals,
     set_ids, group_labels,
     expression_table) = split_test_data(data[0], *prepared, *data[-2:])
    return (expression, outcomes,
            categoricals, non_categoricals,
            set_ids, group_labels,
            expression_table)
