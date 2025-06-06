
import numpy as np
from numpy.random import default_rng
import pandas as pd
from torch import tensor, float32


def read_global_gene_expression(filepath, tissue=None, dropna=False) -> pd.DataFrame:
    gene_expression = tuple(pd.read_excel(filepath, None).items())
    for name, table in gene_expression:
        table.set_index("gene_id", inplace=True)
        table.columns = [(col,name.split("_")[-1]) for col in table.columns]
    gene_expression = gene_expression[0][1].join(gene_expression[1][1])
    gene_expression = gene_expression[sorted(gene_expression.columns)]
    gene_expression.columns = pd.MultiIndex.from_tuples(gene_expression.columns)
    if tissue is not None:
        gene_expression = gene_expression[tissue]
        if dropna:
            gene_expression.dropna(how="all", inplace=True, axis=0)
    return gene_expression


def make_expression_profile_from_data(n_genes, global_data: pd.DataFrame = None,
                                      gdata_tissue=None,
                                      gseed=None, tissue_seed=None):
    """

    See https://numpy.org/doc/2.2/reference/random/generator.html#distributions for
    information on the possible distributions and their arguments.

    :param n_genes: Number of sample distributions (genes) to return.
    :param global_data: Dataframe of global gene expression values (means) to sample from.
    :param gdata_tissue: Tissue to select from global_data table.
    :param global_data: Type of distribution to sample global values from, instead of
    actual
    data.
    :param gseed: Seed of the global distribution generator or global data gene sampler.
    :param tissue_seed Seed of the random tissue selector.
    :return:
    """
    if global_data.shape[1] != 2:
        if gdata_tissue is None:
            tissues = [x[0] for x in global_data.columns]
            gdata_tissue = str(default_rng(tissue_seed).choice(tissues, size=1, replace=False)[0])
        global_data = global_data[[gdata_tissue]]
    global_data = global_data.dropna(how="all", axis=0)
    profile = global_data.sample(n_genes, random_state=gseed, axis=0)
    return profile


def make_true_expression_from_profile(n_samples, profile, sd_scale_modifier=1, seed=None):
    shape = (n_samples, profile.shape[0])
    profile = profile.to_numpy().T
    rng = default_rng(seed)
    samples = rng.normal(loc=profile[0], scale=profile[1]*sd_scale_modifier, size=shape)
    return samples


def make_measurement_from_samples(true_expression,
                                  slope_dist_mean=None, slope_dist_sd=None,
                                  noise_dist_mean=None, noise_dist_sd=None,
                                  per_gene=True, add_noise=True,
                                  slope_seed=None, noise_seed=None):
    slope_rng = default_rng(slope_seed)
    noise_rng = default_rng(noise_seed)
    size = true_expression.shape
    # size = true_expression.shape[1] if per_gene else true_expression.shape
    # slopes = slope_rng.normal(slope_dist_mean, slope_dist_sd, size)
    if per_gene:
        slopes = slope_rng.normal(slope_dist_mean, slope_dist_sd, size[1])
        slopes = np.broadcast_to(slopes, size)
    else:
        slopes = slope_rng.normal(slope_dist_mean, slope_dist_sd, size)
    if add_noise:
        # noise = noise_rng.normal(noise_dist_mean, noise_dist_sd, size)
        noise = noise_rng.normal(noise_dist_mean, slope_dist_sd, size)
    else:
        noise = np.zeros(size)
    measurements = ((true_expression*slopes)+noise).clip(0)
    return measurements, slopes, noise


def add_zero_stacks(matrix, total_levels=2, non_zero_level=0):
    stacked = np.zeros((total_levels,) + matrix.shape)
    stacked[non_zero_level] = matrix
    return stacked


def make_simulation_set_from_data(filepath, n_genes, n_samples, slope_dist1, slope_dist2,
                                  tissue=None, per_gene=True, add_noise=True,
                                  sd_scale_modifier=1, dropna=True):
    global_data = read_global_gene_expression(filepath, tissue, dropna)
    expression_profile = make_expression_profile_from_data(n_genes, global_data)
    true_expression = make_true_expression_from_profile(n_samples, expression_profile,
                                                        sd_scale_modifier=sd_scale_modifier)
    half = n_samples // 2
    m1, s1, b1 = make_measurement_from_samples(true_expression[:half], slope_dist1, 4,
                                               0, 4, per_gene=per_gene, add_noise=add_noise)
    m2, s2, b2 = make_measurement_from_samples(true_expression[half:], slope_dist2, 4,
                                               0, 4, per_gene=per_gene, add_noise=add_noise)
    m1 = add_zero_stacks(m1, 2, 0)
    m2 = add_zero_stacks(m2, 2, 1)
    data = np.concat([m1, m2], axis=1)
    slopes = np.concat([s1, s2])
    noise = np.concat([b1, b2])
    data = np.stack([true_expression, *data])
    transforms = np.stack([slopes, noise])
    shuffler = np.random.default_rng()
    shuffler.shuffle(data, axis=1)
    data = tensor(data, dtype=float32)
    return data, transforms
