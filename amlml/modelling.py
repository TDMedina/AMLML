
import torch
from torch import optim, nn
# from tqdm import tqdm
from tqdm.auto import tqdm

from amlml.simulation import make_simulation_set_from_data


"""
1. Pass the data to the master LocalLayer.
    2. The LocalLayer passes every tech-gene slice into a GeneGroup.
        3. The GeneGroups each pass one dimension into a SingleShallow
            4. The SingleShallows run the first 2 layers: [m] > [m m m m] > [m]
        5. The GeneGroups concatenate the results and feed them into a fully connected layer: [x] [y] > [x y] > [z]
    6. The LocalLayer concatenates the results: [g] [g] [g] > [g g g] connected layer.
7. Pass the LocalLayer results to the fully connected layers.
"""

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
    def __init__(self, n_genes, shrinkage_factor=10, minimum_size=10,
                 final_size=1):
        super().__init__()

        connected_layers = []
        n = n_genes
        while minimum_size <= (n_out := n // shrinkage_factor):
            connected_layers.append(nn.Linear(n, n_out))
            n = n_out
        connected_layers.append(nn.Linear(n, final_size))
        self.connected_layers = nn.ModuleList(connected_layers)
        self.activator = nn.ReLU()
        self.final_activator = nn.ReLU()

    def forward(self, x):
        for layer in self.connected_layers[:-1]:
            x = self.activator(layer(x))
        x = self.final_activator(self.connected_layers[-1](x))
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
    def __init__(self, n_genes, n_levels, shrinkage=1):
        super().__init__()
        self.local_connected = LocalLinear3D(n_genes, n_levels)
        self.act1 = nn.ReLU()

        self.full1 = nn.Linear(n_genes, n_genes)
        self.fullact1 = nn.ReLU()

        self.full2 = nn.Linear(n_genes, n_genes)
        self.fullact2 = nn.ReLU()

        self.full3 = nn.Linear(n_genes, n_genes)
        self.fullact3 = nn.ReLU()

        self.output = nn.Linear(n_genes, 1)
        # self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.local_connected(x)
        x = self.act1(x)
        x = self.full1(x)
        x = self.fullact1(x)
        x = self.full2(x)
        x = self.fullact2(x)
        x = self.full3(x)
        x = self.fullact3(x)
        # x = self.output(x)
        # x = self.act_output(x)
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
            batch_in = data_in[:, i:i+batch_size, :]
            batch_true = data_true[i:i+batch_size, :]
            batch_out = model(batch_in)
            loss = loss_fn(batch_out, batch_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(f"Latest loss {loss}\r", end="")
    print(f"Final loss: {loss}\r", end="")


if __name__ == '__main__':
    data, model3d = prep(10, 2, 10, 100)

