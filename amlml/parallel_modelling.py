import torch
from torch import nn, vmap
from amlml.modelling import ConnectedLayers


class ParallelBellowsLayers(nn.Module):
    def __init__(self, n_genes: int, n_tech: int, n_expansion: int):
        super().__init__()
        self.n_genes = n_genes
        self.n_tech = n_tech
        self.n_expansion = n_expansion

        self.weights1 = nn.Parameter(torch.randn(n_genes*n_tech, n_expansion))
        self.bias1 = nn.Parameter(torch.randn(n_genes*n_tech, n_expansion))

        self.weights2 = nn.Parameter(torch.randn(n_genes*n_tech, n_expansion))
        self.bias2 = nn.Parameter(torch.randn(n_genes*n_tech))

        self.activation = nn.ReLU()

    def __repr__(self):
        string = super().__repr__()
        string = string.split("\n")
        layer1 = f"  (expand): vmap(Linear(in_features=1, out_features={self.n_expansion}))"
        layer2 = f"  (shrink): vmap(Linear(in_features={self.n_expansion}, out_features=1))"
        string.insert(1, layer2)
        string.insert(1, layer1)
        string = "\n".join(string)
        return string

    @staticmethod
    def expand(gene_matrix, weight_matrix, bias_matrix):
        return gene_matrix.unsqueeze(1) @ weight_matrix.unsqueeze(0) + bias_matrix

    @staticmethod
    def shrink(gene_matrix, weight_matrix, bias_matrix):
        return gene_matrix @ weight_matrix.unsqueeze(1) + bias_matrix

    def forward(self, x):
        # Reshape input data to stack measurement data horizontally in a 2D matrix so
        # that each measurement of each gene is treated independently. Then, transpose to
        # make iteration faster.

        # CoxPH can only batch on the first dimension, so the data is now reshaped
        # prior to input.
        # x = x.permute(1, 0, 2).reshape([-1, self.n_genes*2]).T
        x = x.reshape([-1, self.n_genes*2]).T

        expand_parallel = vmap(self.expand, in_dims=(0, 0, 0))
        results = expand_parallel(x, self.weights1, self.bias1)
        results = self.activation(results)

        shrink_parallel = vmap(self.shrink, in_dims=(0, 0, 0))
        results = shrink_parallel(results, self.weights2, self.bias2)
        results = self.activation(results)

        results = results.squeeze().T
        results = results.view(results.shape[0], self.n_tech, self.n_genes).permute(1, 0, 2)
        return results


class ParallelGeneLayers(nn.Module):
    def __init__(self, n_genes: int, n_tech: int):
        super().__init__()
        self.n_genes = n_genes
        self.n_tech = n_tech

        self.weights = nn.Parameter(torch.randn(self.n_genes, self.n_tech))
        self.bias = nn.Parameter(torch.randn(self.n_genes))
        self.activation = nn.ReLU()

    def __repr__(self):
        string = super().__repr__()
        string = string.split("\n")
        layer = f"  (combine_gene): vmap(Linear(in_features={self.n_tech}, out_features=1))"
        string.insert(1, layer)
        string = "\n".join(string)
        return string

    @staticmethod
    def combine_gene(gene_matrix, weight_matrix, bias_matrix):
        return weight_matrix @ gene_matrix + bias_matrix

    def forward(self, x):
        x = x.permute(2, 0, 1)
        combine_parallel = vmap(self.combine_gene, in_dims=(0, 0, 0))
        results = combine_parallel(x, self.weights, self.bias).T
        results = self.activation(results)
        return results


class CombinedParallelModel(nn.Module):
    def __init__(self, n_genes: int, n_tech: int, n_expansion=4,
                 shrinkage_factor=10, minimum_size=10, final_size=1):
        super().__init__()
        self.n_genes = n_genes
        self.n_tech = n_tech
        self.n_expansion = n_expansion
        self.shrinkage_factor = shrinkage_factor
        self.minimum_size = minimum_size
        self.final_size = final_size

        self.local_layers = ParallelBellowsLayers(n_genes, n_tech, n_expansion)
        self.gene_layers = ParallelGeneLayers(n_genes, n_tech)
        self.connected_layers = ConnectedLayers(n_genes, shrinkage_factor,
                                                minimum_size, final_size)

    def forward(self, x):
        x = self.local_layers(x)
        x = self.gene_layers(x)
        x = self.connected_layers(x)
        return x
