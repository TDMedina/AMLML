
import torch
from torch import optim, nn
# from tqdm import tqdm
from tqdm.auto import tqdm

from amlml.simulation import make_simulation_set_from_data


class SingleShallow(nn.Module):
    """Initial 2 layers for a single tech for a single gene.
    Shape is 1 > n > 1. Input should be a (samples, 1) matrix corresponding to one
    technology of one gene for all samples.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.in_layer = nn.Linear(1, hidden_size)
        self.out_layer = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.in_layer(x))
        x = self.out_layer(x)
        return x


class GeneGroup(nn.Module):
    def __init__(self, tech_layers, hidden_size):
        super().__init__()
        self.shallows = nn.ModuleList([SingleShallow(hidden_size)
                                       for _ in range(tech_layers)])
        self.combinor = nn.Linear(tech_layers, 1)

    def forward(self, x):
        x = [self.shallows[i](x[i:i+1].T) for i in range(x.shape[0])]
        x = torch.concatenate(x, axis=-1)
        x = self.combinor(x)
        return x


class LocalLayers(nn.Module):
    def __init__(self, tech_layers, n_genes, hidden_size):
        super().__init__()
        self.gene_groups = nn.ModuleList([GeneGroup(tech_layers, hidden_size)
                                          for _ in range(n_genes)])

    def forward(self, x):
        x = [self.gene_groups[i](x[:, :, i:i+1]) for i in range(x.shape[2])]
        x = torch.concatenate(x, axis=-1)
        return x


class ConnectedLayers(nn.Module):
    def __init__(self, n_genes, shrinkage_factor=10, minimum_penultimate_size=10,
                 final_size=1, zero_params=False, kaiming_weights=False,
                 output_xavier=False, dropout=0.2):
        super().__init__()

        layers = []
        n = n_genes
        while (n_out := n // shrinkage_factor) >= minimum_penultimate_size:
            layers.append(nn.Linear(n, n_out))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            n = n_out
        # If n_genes // 10 is less than the minimum penultimate size, add a hidden layer.
        if not layers:
            n_out = min(minimum_penultimate_size, n)  # e.g., 99 -> 10 if n_genes is larger than min_penultimate, or 9->9 otherwise.
            layers.append(nn.Linear(n, n_out))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            n = n_out
        layers.append(nn.Linear(n, final_size))
        self.layers = nn.Sequential(*layers)
        if kaiming_weights:
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        if output_xavier:
            nn.init.xavier_uniform_(self.layers[-1].weight)

    def forward(self, x):
        x = self.layers(x)
        # for layer in self.layers[:-1]:
        #     x = layer(x)
        #     x = self.activator(x)
        #     x = self.dropout(x)
        # x = self.layers[-1](x)
        return x


class ShallowConnectedLayers(nn.Module):
    def __init__(self, input_size, dropout=0.2,
                 kaiming_weights=True, output_xavier=False,
                 cutoff=300):
        super().__init__()

        hidden1 = self.compute_hidden_size(input_size)
        layers = [nn.Linear(input_size, hidden1), nn.ReLU(), nn.Dropout(dropout)]

        if input_size > cutoff:
            hidden2 = max(hidden1 // 2, 32)
            layers += [nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Dropout(dropout)]
            final_in = hidden2
        else:
            final_in = hidden1

        layers.append(nn.Linear(final_in, 1))
        self.layers = nn.Sequential(*layers)

        if kaiming_weights:
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                    nn.init.zeros_(layer.bias)
        if output_xavier:
            nn.init.xavier_uniform_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)

    @staticmethod
    def compute_hidden_size(n_genes):
        size = 4 * n_genes**0.5
        size = int(max(32, min(size, 1024)))
        return size

    def forward(self, x):
        x = self.layers(x)
        return x


class CombinedAMLModel(nn.Module):
    def __init__(self, tech_layers, n_genes, hidden_size=4, shrinkage_factor=10,
                 minimum_size=10, final_size=1):
        super().__init__()

        self.local_layers = LocalLayers(tech_layers, n_genes, hidden_size)
        self.connected_layers = ConnectedLayers(n_genes, shrinkage_factor,
                                                minimum_size, final_size)

    def forward(self, x):
        x = self.local_layers(x)
        x = self.connected_layers(x)
        return x


#
# class TechShallows(nn.Module):
#     """Combines all SingleShallows layers for all genes of one technology.
#
#     Input should be a (samples, genes) matrix corresponding to one technology for all
#     genes for all samples.
#     """
#     def __init__(self, n_genes, hidden_size):
#         super().__init__()
#         self.shallows = nn.ModuleList([SingleShallow(hidden_size)
#                                        for _ in range(n_genes)])
#         self.activation = nn.ReLU()
#
#     def forward(self, x):
#         x = [self.shallows[i](x[:, i:i+1]) for i in range(x.shape[1])]
#         x = self.activation(torch.concat(x, axis=1))
#         return x
#
#
# class GenePairs(nn.Module):
#     """Combines all TechShallows"""
#     def __init__(self, n_tech_layers, n_genes, hidden_size):
#         super().__init__()
#         self.gene_pairs = nn.ModuleList([TechShallows(n_genes, hidden_size)
#                                          for tech in range(n_tech_layers)])
#
#     def forward(self, x):
#         x = [self.gene_pairs[i](x[i]) for i in range(x.shape[0])]
#         return x


class LocalLinear3D(nn.Module):
    def __init__(self, n_features, n_levels):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(n_levels, 1) for _ in range(n_features)])

    def forward(self, x):
        x = [self.linears[i](x[:, :, i].T) for i in range(x.shape[2])]
        x = torch.concat(x, axis=1)
        return x


class AMLNeuralNetwork3D(nn.Module):
    def __init__(self, n_genes, n_levels, shrinkage_factor=10, minimum_size=10,
                 final_size=1):
        super().__init__()
        self.local_connected = LocalLinear3D(n_genes, n_levels)

        self.connected_layers = []
        n = n_genes
        while minimum_size <= (n_out := n // shrinkage_factor):
            self.connected_layers.append(nn.Linear(n, n_out))
            n = n_out
        self.connected_layers.append(nn.Linear(n_out, final_size))

        self.activator = nn.ReLU()
        self.final_activator = nn.ReLU()

    def forward(self, x):
        x = self.activator(self.local_connected(x))
        for layer in self.connected_layers[:-1]:
            x = self.activator(layer(x))
        x = self.final_activator(self.connected_layers[-1](x))
        return x


def prep(slope1, slope2, n_genes, n_samples, per_gene=True, sd_scale_modifier=1,
         add_noise=True):
    data = make_simulation_set_from_data("/home/tyler/Repositories/AMLML2/gdata.xlsx",
                                         n_genes, n_samples, slope1, slope2,
                                         tissue="BONE_MARROW",
                                         per_gene=per_gene,
                                         add_noise=add_noise,
                                         sd_scale_modifier=sd_scale_modifier)
    model3d = AMLNeuralNetwork3D(n_genes, 2, 1)
    return data, model3d


def train_model(data, model, n_epochs, batch_size=10):
    loss_fn = nn.MSELoss()
    # loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    data_in = data[1:]
    data_true = data[0]

    for epoch in tqdm(range(n_epochs), position=0, leave=True):
        for i in tqdm(range(0, data_in.shape[1], batch_size), leave=False, position=1):
            optimizer.zero_grad()
            batch_in = data_in[:, i:i+batch_size, :]
            batch_true = data_true[i:i+batch_size, :]
            batch_out = model(batch_in)
            loss = loss_fn(batch_out, batch_true)
            loss.backward()
            optimizer.step()
        # print(f"Latest loss {loss}\r", end="")
    print(f"Final loss: {loss}\r", end="")


if __name__ == '__main__':
    data, model3d = prep(10, 2, 10, 100)

