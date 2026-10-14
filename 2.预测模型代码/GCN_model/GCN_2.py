import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GCN_GRU(nn.Module):
    def __init__(self, node_features, gcn_out_channels, gru_hidden_size, gru_num_layers):
        super(GCN_GRU, self).__init__()
        self.node_features = node_features
        self.gcn_out_channels = gcn_out_channels
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.gcn = GCN(node_features, gcn_out_channels)
        self.gru = nn.GRU(gcn_out_channels, gru_hidden_size, gru_num_layers, batch_first=True)
        self.fc = nn.Linear(gru_hidden_size, 1)

    def forward(self, x, edge_index):
        batch_size, time_steps, node_features = x.shape

        x = x.view(-1, node_features)
        gcn_outs = self.gcn(x, edge_index)
        gcn_outs = gcn_outs.view(batch_size, time_steps, -1)



        # Apply GRU to GCN outputs
        gru_out, _ = self.gru(gcn_outs)
        gru_out = gru_out[:, -1, :]
        gru_out = self.fc(gru_out)
        # print(gru_out.shape)
        return gru_out


def main():

    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters())

    x = torch.randn(32, 3, 6)

    # 示例边索引 (edge_index)，你需要根据实际情况定义
    edge_index_0 = torch.tensor([
        [0, 0, 1],
        [1, 2, 2]
    ], dtype=torch.long)

    edge_index = torch.tensor([
        [0, 0, 1],
        [1, 2, 2]
    ], dtype=torch.long)

    batch_size = 32

    for i in range(batch_size - 1):
        edge_index_add = edge_index_0 + 3 * (i + 1)
        edge_index = torch.cat((edge_index, edge_index_add), dim=1)

    model = GCN_GRU(6, 32, 128, 1)

    out = model(x, edge_index)

    # 打印参数量
    transformer_params = sum(p.numel() for p in model.fc.parameters())
    print(f"TransformerEncoder 的参数量: {transformer_params}")

    print(out.shape)


if __name__ == '__main__':
    main()
