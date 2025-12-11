import torch
from torch import nn, vmap
from amlml.modelling import ConnectedLayers, ShallowConnectedLayers


class ParallelBellowsLayers(nn.Module):
    def __init__(self, n_genes: int, n_tech: int, n_expansion: int, zero_params=False,
                 kaiming_weights=False):
        super().__init__()
        self.n_genes = n_genes
        self.n_tech = n_tech
        self.n_expansion = n_expansion

        self.weights1 = nn.Parameter(torch.randn(n_genes*n_tech, n_expansion))
        self.bias1 = nn.Parameter(torch.zeros(n_genes*n_tech, n_expansion))

        self.weights2 = nn.Parameter(torch.randn(n_genes*n_tech, n_expansion))
        self.bias2 = nn.Parameter(torch.zeros(n_genes*n_tech))

        if zero_params:
            for param in self.parameters():
                nn.init.zeros_(param)
        elif kaiming_weights:
            for param in [self.weights1, self.weights2]:
                nn.init.kaiming_uniform_(param, nonlinearity="relu")

        self.activation = nn.ReLU()

    def __repr__(self):
        string = super().__repr__()
        string = string.split("\n")
        layer1 = f"  (expand): vmap({self.n_genes}*{self.n_tech} * Linear(in_features=1, out_features={self.n_expansion}))"
        layer2 = f"  (shrink): vmap({self.n_genes}*{self.n_tech} * Linear(in_features={self.n_expansion}, out_features=1))"
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

        # CoxPH can only batch on the first dimension with its built-in training
        # routine, so the data is now reshaped prior to input.
        # x = x.permute(1, 0, 2).reshape([-1, self.n_genes*2]).T
        x = x.reshape([-1, self.n_genes*2]).T

        expand_parallel = vmap(self.expand, in_dims=(0, 0, 0))
        results = expand_parallel(x, self.weights1, self.bias1)
        results = self.activation(results)

        shrink_parallel = vmap(self.shrink, in_dims=(0, 0, 0))
        results = shrink_parallel(results, self.weights2, self.bias2)
        results = self.activation(results)

        results = results.squeeze().T
        results = results.view(results.shape[0], self.n_tech, self.n_genes)
        return results


class ParallelGeneLayers(nn.Module):
    def __init__(self, n_genes: int, n_tech: int, zero_params=False,
                 kaiming_weights=False):
        super().__init__()
        self.n_genes = n_genes
        self.n_tech = n_tech

        self.weights = nn.Parameter(torch.randn(self.n_genes, self.n_tech))
        self.bias = nn.Parameter(torch.zeros(self.n_genes))

        self.vmapped = torch.vmap(self.combine_gene, in_dims=(0, 0, 0))

        if zero_params:
            nn.init.zeros_(self.weights)
            nn.init.zeros_(self.bias)
        elif kaiming_weights:
            nn.init.kaiming_uniform_(self.weights, nonlinearity="relu")
        self.activation = nn.ReLU()

    def __repr__(self):
        string = super().__repr__()
        string = string.split("\n")
        layer = f"  (combine_gene): vmap({self.n_genes} * Linear(in_features={self.n_tech}, out_features=1))"
        string.insert(1, layer)
        string = "\n".join(string)
        return string

    @staticmethod
    def combine_gene(gene_matrix, weight_matrix, bias_matrix):
        return weight_matrix @ gene_matrix + bias_matrix

    def forward(self, x):
        x = x.permute(2, 1, 0)
        # combine_parallel = vmap(self.combine_gene, in_dims=(0, 0, 0))
        # results = combine_parallel(x, self.weights, self.bias).T
        results = self.vmapped(x, self.weights, self.bias).T
        results = self.activation(results)
        return results

    def einsum_forward(self, x):
        x = torch.einsum('btg,gt->bg', x, self.weights) + self.bias
        return x


# class CombinedParallelModel(nn.Module):
#     def __init__(self, n_genes: int, n_tech: int, n_expansion=4,
#                  shrinkage_factor=10, minimum_size=10, final_size=1, zero_weights=False):
#         super().__init__()
#         self.n_genes = n_genes
#         self.n_tech = n_tech
#         self.n_expansion = n_expansion
#         self.shrinkage_factor = shrinkage_factor
#         self.minimum_size = minimum_size
#         self.final_size = final_size
#
#         self.local_layers = ParallelBellowsLayers(n_genes, n_tech, n_expansion,
#                                                   zero_weights=zero_weights)
#         self.gene_layers = ParallelGeneLayers(n_genes, n_tech, zero_weights=zero_weights)
#         self.connected_layers = ConnectedLayers(n_genes, shrinkage_factor,
#                                                 minimum_size, final_size)
#
#     def forward(self, x):
#         x = self.local_layers(x)
#         x = self.gene_layers(x)
#         x = self.connected_layers(x)
#         return x


class CombinedGeneModel(nn.Module):
    def __init__(self, n_genes: int, n_tech: int, n_expansion=4, zero_params=False,
                 kaiming_weights=False):
        super().__init__()
        self.n_genes = n_genes
        self.n_tech = n_tech
        self.n_expansion = n_expansion
        # self.shrinkage_factor = shrinkage_factor
        # self.minimum_size = minimum_size
        # self.final_size = final_size

        self.local_layers = ParallelBellowsLayers(n_genes, n_tech, n_expansion,
                                                  zero_params=zero_params,
                                                  kaiming_weights=kaiming_weights)
        self.gene_layers = ParallelGeneLayers(n_genes, n_tech, zero_params=zero_params,
                                              kaiming_weights=kaiming_weights)
        # self.connected_layers = ConnectedLayers(n_genes, shrinkage_factor,
        #                                         minimum_size, final_size)

    def forward(self, x):
        x = self.local_layers(x)
        x = self.gene_layers(x)
        # x = self.connected_layers(x)
        return x


class EthnicityModel(nn.Module):
    def __init__(self, covariate_cardinality: dict, embedding_dims: dict):
        super().__init__()
        self.n_ethnicities = covariate_cardinality["ethnicity"]
        self.race_embedding = nn.Embedding(covariate_cardinality["race"],
                                           embedding_dims["race"])
        self.ethnicity_embedding = nn.Embedding(covariate_cardinality["ethnicity"],
                                                embedding_dims["ethnicity"])
        self.interaction_embedding = nn.Embedding(covariate_cardinality["race"]*covariate_cardinality["ethnicity"],
                                                  embedding_dims["interaction"])

    def forward(self, x):
        race = self.race_embedding(x[:, 0])
        ethnicity = self.ethnicity_embedding(x[:, 1])
        interaction_x = x[:, 0]*self.n_ethnicities + x[:, 1]
        interaction_emb = self.interaction_embedding(interaction_x)
        x = torch.cat([race, ethnicity, interaction_emb], dim=1)
        return x


class ProtocolModel(nn.Embedding):
    def __init__(self, cardinality: int, embedding_dim: int):
        super().__init__(cardinality, embedding_dim)


class ClinicalMissingMask(nn.Module):
    def __init__(self, n_variables: int):
        super().__init__()
        self.n_variables = n_variables

        self.weights = nn.Parameter(torch.randn(self.n_variables, 2))
        nn.init.kaiming_uniform_(self.weights, nonlinearity="relu")
        self.bias = nn.Parameter(torch.zeros(self.n_variables))
        self.activation = nn.ReLU()

    def __repr__(self):
        string = super().__repr__()
        string = string.split("\n")
        layer = f"  (mask_covariate): vmap({self.n_variables} * Linear(in_features=2, out_features=1))"
        string.insert(1, layer)
        string = "\n".join(string)
        return string

    @staticmethod
    def mask_covariate(variable_matrix, weight_matrix, bias_matrix):
        return weight_matrix @ variable_matrix + bias_matrix

    def forward(self, x):
        x = x.permute(2, 1, 0)
        mask_parallel = vmap(self.mask_covariate, in_dims=(0, 0, 0))
        results = mask_parallel(x, self.weights, self.bias).T
        results = self.activation(results)
        return results


class ClinicalModel(nn.Module):
    def __init__(self, covariate_cardinality: dict, embedding_dims: dict,
                 n_non_categorical: int):
        super().__init__()
        self.ethnicity = EthnicityModel(covariate_cardinality, embedding_dims)
        self.protocol = ProtocolModel(covariate_cardinality["protocol"],
                                      embedding_dims["protocol"])
        self.non_categorical = ClinicalMissingMask(n_non_categorical)

    def forward(self, categorical, non_categorical):
        ethnicity = self.ethnicity(categorical[:, :2])
        protocol = self.protocol(categorical[:, 2])
        non_categorical = self.non_categorical(non_categorical)
        x = torch.cat([ethnicity, protocol, non_categorical], dim=1)
        return x


class SuperModel(nn.Module):
    def __init__(self, n_genes: int, n_tech: int, n_expansion: int,
                 shrinkage_factor: int, minimum_penultimate_size: int, final_size: int,
                 include_clinical_variables=True, n_clinical: int = None,
                 covariate_cardinality: dict = None, embedding_dims: dict = None,
                 zero_params=False, kaiming_weights=False, output_xavier=False,
                 bellows_normalization=True, use_shallow=False, dropout=0.2):
        super().__init__()
        self.includes_clinical = include_clinical_variables
        dnn_input_dim = n_genes
        self.clinical_model = None

        if self.includes_clinical:
            dnn_input_dim = dnn_input_dim + n_clinical + sum(embedding_dims.values())-3
            self.clinical_model = ClinicalModel(covariate_cardinality, embedding_dims,
                                                n_clinical-3)
        if bellows_normalization:
            self.expression_model = CombinedGeneModel(n_genes, n_tech, n_expansion,
                                                      zero_params=zero_params,
                                                      kaiming_weights=kaiming_weights)
        else:
            self.expression_model = ParallelGeneLayers(n_genes, n_tech,
                                                       zero_params=zero_params,
                                                       kaiming_weights=kaiming_weights)
        if use_shallow:
            self.connected_layers = ShallowConnectedLayers(dnn_input_dim, kaiming_weights=kaiming_weights,
                                                           output_xavier=output_xavier, dropout=dropout)
        else:
            self.connected_layers = ConnectedLayers(dnn_input_dim, shrinkage_factor,
                                                    minimum_penultimate_size, final_size,
                                                    zero_params=zero_params,
                                                    kaiming_weights=kaiming_weights,
                                                    output_xavier=output_xavier, dropout=dropout)
        # self.output_activator = nn.Sigmoid()

    # def forward(self, input_tuple):
    def forward(self, expression, clinical_categorical=None, clinical_non_categorical=None):
        x = self.expression_model(expression)
        if self.includes_clinical:
            clinical = self.clinical_model(clinical_categorical, clinical_non_categorical)
            x = torch.cat([x, clinical], dim=1)
        x = self.connected_layers(x)
        # x = self.output_activator(x)
        return x


class CrossNormalizedModel(nn.Module):
    def __init__(self, n_genes: int,
                 shrinkage_factor: int, minimum_penultimate_size: int, final_size: int,
                 include_clinical_variables=True, n_clinical: int = None,
                 covariate_cardinality: dict = None, embedding_dims: dict = None,
                 zero_params=False, kaiming_weights=False, output_xavier=False, use_shallow=False,
                 dropout=0.2):
        super().__init__()
        self.includes_clinical = include_clinical_variables
        dnn_input_dim = n_genes
        self.clinical_model = None

        if self.includes_clinical:
            dnn_input_dim = dnn_input_dim + n_clinical + sum(embedding_dims.values())-3
            self.clinical_model = ClinicalModel(covariate_cardinality, embedding_dims,
                                                n_clinical-3)
        if use_shallow:
            self.connected_layers = ShallowConnectedLayers(dnn_input_dim, kaiming_weights=kaiming_weights,
                                                           output_xavier=output_xavier, dropout=dropout)
        else:
            self.connected_layers = ConnectedLayers(dnn_input_dim, shrinkage_factor,
                                                    minimum_penultimate_size, final_size,
                                                    zero_params=zero_params,
                                                    kaiming_weights=kaiming_weights,
                                                    output_xavier=output_xavier, dropout=dropout)

    def forward(self, expression, clinical_categorical=None, clinical_non_categorical=None):
        if self.includes_clinical:
            clinical = self.clinical_model(clinical_categorical, clinical_non_categorical)
            x = torch.cat([expression, clinical], dim=1)
        else:
            x = expression
        x = self.connected_layers(x)
        # x = self.output_activator(x)
        return x
