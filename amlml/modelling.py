
import torch
from torch import optim, nn
from tqdm import tqdm

from amlml.simulation import make_simulation_set_from_data


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

    for epoch in tqdm(range(n_epochs)):
        for i in range(0, data_in.shape[1], batch_size):
            batch_in = data_in[:, i:i+batch_size, :]
            batch_true = data_true[i:i+batch_size, :]
            batch_out = model(batch_in)
            loss = loss_fn(batch_out, batch_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Latest loss {loss}\r", end="")


if __name__ == '__main__':
    data, model3d = prep(10, 2, 10, 100)

