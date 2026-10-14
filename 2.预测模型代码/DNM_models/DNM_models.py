import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from banama_load.typical_models import KANLinear


class Soma(nn.Module):
    def __init__(self, k, qs):
        super(Soma, self).__init__()
        self.params = nn.ParameterDict({'k': nn.Parameter(k)})
        self.params.update({'qs': nn.Parameter(qs)})

    def forward(self, x):
        y = 1 / (1 + torch.exp(-self.params['k'] * (x - self.params['qs'])))
        return y


class Membrane(nn.Module):
    def __init__(self):
        super(Membrane, self).__init__()

    def forward(self, x):
        x = torch.sum(x, 1)
        return x


class Dendritic(nn.Module):
    def __init__(self):
        super(Dendritic, self).__init__()

    def forward(self, x):
        x = torch.prod(x, 2)
        return x


class Synapse(nn.Module):

    def __init__(self, w, q, k):
        super(Synapse, self).__init__()
        self.params = nn.ParameterDict({'w': nn.Parameter(w)})
        self.params.update({'q': nn.Parameter(q)})
        self.params.update({'k': nn.Parameter(k)})

    def forward(self, x):
        num, _ = self.params['w'].shape
        x = torch.unsqueeze(x, 1)
        x = x.repeat((1, num, 1))
        y = 1 / (1 + torch.exp(
            torch.mul(-self.params['k'], (torch.mul(x, self.params['w']) - self.params['q']))))

        return y


class DNM_Linear1(nn.Module):
    def __init__(self, dim, M=10, k=0.1, qs=1.0):  # , device=torch.device('cuda:0')):
        super(DNM_Linear1, self).__init__()

        w = torch.rand([M, dim])  # .to(device)
        q = torch.rand([M, dim])  # .to(device)
        k = torch.tensor(k).float()
        qs = torch.tensor(qs).float()

        self.model = nn.Sequential(
            nn.Flatten(),
            Synapse(w, q, k),
            Dendritic(),
            Membrane(),
            Soma(k, qs)
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1)
        return x


class DNM_Linear2(nn.Module):
    def __init__(self, input_size, out_size, M, k=0.1, qs=1.0, activation=F.sigmoid):
        super(DNM_Linear2, self).__init__()

        DNM_W = torch.rand([out_size, M, input_size])  # .cuda() # [size_out, M, size_in]
        # dendritic_W = torch.rand([input_size])  # .cuda() # size_out, M, size_in]
        # membrane_W = torch.rand([M])  # .cuda() # size_out, M, size_in]
        q = torch.rand([out_size, M, input_size])  # .cuda()
        # torch.nn.init.constant_(q, qs)  # 设置q的初始值
        k = torch.tensor(k).float()  # .cuda()
        qs = torch.tensor(qs).float()  # .cuda()

        self.params = nn.ParameterDict({'DNM_W': nn.Parameter(DNM_W)})
        self.params.update({'q': nn.Parameter(q)})
        # self.params.update({'dendritic_W': nn.Parameter(dendritic_W)})
        # self.params.update({'membrane_W': nn.Parameter(membrane_W)})
        self.params.update({'k': nn.Parameter(k)})
        self.params.update({'qs': nn.Parameter(qs)})
        self.activation = activation
        self.flat = nn.Flatten()

    def forward(self, x):
        # Synapse
        out_size, M, _ = self.params['DNM_W'].shape
        x = self.flat(x)
        x = torch.unsqueeze(x, 1)
        x = torch.unsqueeze(x, 2)
        x = x.repeat(1, out_size, M, 1)
        x = torch.relu(torch.mul(self.params['k'], (torch.mul(x, self.params['DNM_W']) - self.params['q'])))
        x = torch.prod(x, dim=3)
        x = torch.sum(x, dim=2)

        # Soma
        if self.activation != None:
            x = self.activation(self.params['k'] * (x - self.params['qs']))

        return x


# class DNM_Linear2(nn.Module):
#     def __init__(self, input_size, M, k=0.1, qs=1.0):
#         super(DNM_Linear2, self).__init__()
#
#         DNM_W = torch.rand([M, input_size])  # .cuda()
#         q = torch.rand([M, input_size])  # .cuda()
#         k = torch.tensor(k)  # .cuda()
#         qs = torch.tensor(qs)  # .cuda()
#
#         self.params = nn.ParameterDict({'DNM_W': nn.Parameter(DNM_W)})
#         self.params.update({'q': nn.Parameter(q)})
#         self.params.update({'k': nn.Parameter(k)})
#         self.params.update({'qs': nn.Parameter(qs)})
#         self.flat = nn.Flatten()
#
#     def forward(self, x):
#         # Synapse
#         M, _ = self.params['DNM_W'].shape
#         x = self.flat(x)
#         x = torch.unsqueeze(x, 1)
#         x = x.repeat(1, M, 1)
#         x = 1 / (1 + torch.exp(
#             torch.mul(-self.params['k'], (torch.mul(x, self.params['DNM_W']) - self.params['q']))))
#         x = torch.prod(x, dim=2)
#         x = torch.sum(x, dim=1)
#
#         # Soma
#         x = 1 / (1 + torch.exp(-self.params['k'] * (x - self.params['qs'])))
#         x = x.view(-1, 1)
#
#         return x


class DNM_Linear3(nn.Module):
    def __init__(self, input_size, size_out, M, synapse_activation=torch.sigmoid, activation=None):
        super(DNM_Linear3, self).__init__()
        self.input_size = input_size
        self.size_out = size_out
        self.M = M
        self.synapse_activation = synapse_activation
        self.activation = activation
        self.dense_layer = nn.Linear(input_size, size_out * M)
        self.dnm_weight = nn.Parameter(torch.empty(size_out, M))
        self.k = nn.Parameter(torch.tensor(0.1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.dnm_weight, mean=0, std=np.sqrt(2 / (self.size_out + self.M)))

    def forward(self, x):
        # 重构输入数据的形状以匹配全连接层的输入
        x = x.view(x.size(0), -1)
        fc = self.dense_layer(x)
        k_fc = fc * self.k
        activation_fc = self.synapse_activation(k_fc)

        reshape_activation_fc = activation_fc.view(-1, self.size_out, self.M)
        dnm_fc = reshape_activation_fc * self.dnm_weight
        out = torch.sum(dnm_fc, dim=2)

        if self.activation:
            out = self.activation(out)
        return out


class CNN_DNMmodel(nn.Module):
    def __init__(self, feature_num, M=10, k=0.1, qs=1.0):
        super(CNN_DNMmodel, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, (1, feature_num), 1)
        self.conv2 = nn.Conv2d(64, 128, (3, 1), 1)
        # self.pooling = nn.MaxPool2d((1, 3))
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(128, 10)
        # self.fc2 = nn.Linear(32, 1)
        self.dnm_fc = DNM_Linear1(dim=10, M=M, k=k, qs=qs)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        tem = self.conv1(input)
        tem = self.conv2(tem)
        # tem = self.pooling(tem)
        tem = self.flat(tem)
        tem = self.fc1(tem)
        # out = self.fc2(tem)
        # tem = self.dropout(tem)
        out = self.dnm_fc(tem)

        return out


class LSTM_DNM(nn.Module):
    def __init__(self, feature_num, layer_num, nodes_num, M=10, k=0.1, qs=1.0):
        super(LSTM_DNM, self).__init__()

        self.feature_num = feature_num
        self.layer_num = layer_num
        self.nodes_num = nodes_num

        self.lstm1 = nn.LSTM(input_size=feature_num, num_layers=layer_num, hidden_size=nodes_num, batch_first=True)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(nodes_num, 10)
        # self.fc2 = nn.Linear(64, 1)
        self.dnm_fc = DNM_Linear1(dim=10, M=M, k=k, qs=qs)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        tem, _ = self.lstm1(input)
        tem = tem[:, -1, :]
        tem = self.flat(tem)
        # tem = self.dropout(tem)
        tem = F.relu(self.fc1(tem))
        # out = self.fc2(tem)
        out = self.dnm_fc(tem)

        return out


class C_L_DNMmodel(nn.Module):
    def __init__(self, feature_num, M=10, k=0.1, qs=1.0):
        super(C_L_DNMmodel, self).__init__()

        self.feature_num = feature_num

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, feature_num), stride=(1, 1)),
            nn.ReLU()
        )

        self.lstm3 = nn.LSTM(input_size=64, hidden_size=128, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 10)
        self.dnm_fc = DNM_Linear1(dim=10, M=M, k=k, qs=qs)

    def forward(self, input):
        tem = self.conv1(input)
        tem = tem.squeeze(dim=3)
        tem = tem.permute([0, 2, 1])
        tem, _ = self.lstm3(tem)
        tem = tem[:, -1, :]  # 取LSTM的最后一个时间步作为输出
        # tem = self.dropout(tem)
        tem = self.fc(tem)
        out = self.dnm_fc(tem)

        return out


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


class GCN_GRU_DNM(nn.Module):
    def __init__(self, node_features, gcn_out_channels, gru_hidden_size, gru_num_layers, M=10, k=0.1, qs=1.0):
        super(GCN_GRU_DNM, self).__init__()
        self.node_features = node_features
        self.gcn_out_channels = gcn_out_channels
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.gcn = GCN(node_features, gcn_out_channels)
        self.gru = nn.GRU(gcn_out_channels, gru_hidden_size, gru_num_layers, batch_first=True)
        self.fc = nn.Linear(128, 10)
        self.dnm_fc = DNM_Linear1(dim=10, M=M, k=k, qs=qs)

    def forward(self, x, edge_index):
        batch_size, time_steps, node_features = x.shape

        # Initialize a tensor to store GCN outputs
        gcn_outs = torch.zeros(batch_size, time_steps, self.gcn_out_channels).to(x.device)

        # Apply GCN to each time step
        for t in range(time_steps):
            gcn_outs[:, t, :] = self.gcn(x[:, t, :], edge_index)

        # Apply GRU to GCN outputs
        gru_out, _ = self.gru(gcn_outs)
        gru_out = gru_out[:, -1, :]
        gru_out = self.fc(gru_out)
        out = self.dnm_fc(gru_out)
        # print(gru_out.shape)
        return out


class GCN_GRU_DNM_2(nn.Module):
    def __init__(self, node_features, gcn_out_channels, gru_hidden_size, gru_num_layers, M=10, k=0.1, qs=1.0):
        super(GCN_GRU_DNM_2, self).__init__()
        self.node_features = node_features
        self.gcn_out_channels = gcn_out_channels
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.gcn = GCN(node_features, gcn_out_channels)
        self.gru = nn.GRU(gcn_out_channels, gru_hidden_size, gru_num_layers, batch_first=True)
        self.fc = nn.Linear(gru_hidden_size, 10)
        self.dnm_fc = DNM_Linear1(dim=10, M=M, k=k, qs=qs)

    def forward(self, x, edge_index):
        batch_size, time_steps, node_features = x.shape

        x = x.view(-1, node_features)
        gcn_outs = self.gcn(x, edge_index)
        gcn_outs = gcn_outs.view(batch_size, time_steps, -1)

        # Apply GRU to GCN outputs
        gru_out, _ = self.gru(gcn_outs)
        gru_out = gru_out[:, -1, :]
        gru_out = self.fc(gru_out)
        out = self.dnm_fc(gru_out)
        # print(gru_out.shape)
        return out


class GCN_GRU_KAN(nn.Module):
    def __init__(self, node_features, gcn_out_channels, gru_hidden_size, gru_num_layers):
        super(GCN_GRU_KAN, self).__init__()
        self.node_features = node_features
        self.gcn_out_channels = gcn_out_channels
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        self.gcn = GCN(node_features, gcn_out_channels)
        self.gru = nn.GRU(gcn_out_channels, gru_hidden_size, gru_num_layers, batch_first=True)
        self.fc = KANLinear(gru_hidden_size, 1)

    def forward(self, x, edge_index):
        batch_size, time_steps, node_features = x.shape

        x = x.view(-1, node_features)
        gcn_outs = self.gcn(x, edge_index)
        gcn_outs = gcn_outs.view(batch_size, time_steps, -1)

        # Apply GRU to GCN outputs
        gru_out, _ = self.gru(gcn_outs)
        gru_out = gru_out[:, -1, :]
        print(gru_out.shape)
        gru_out = self.fc(gru_out)
        # print(gru_out.shape)
        return gru_out


def main():

    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters())

    x = torch.rand(32, 3, 6)
    # model = DNM_Linear1(dim=21, M=10)
    # model = DNM_Linear2(input_size=128, out_size=1, M=10)
    # model = DNM_Linear3(input_size=128, size_out=1, M=5)
    # model = CNN_DNMmodel(feature_num=7)
    # model = LSTM_DNM(7, 1, 128)
    # model = C_L_DNMmodel(feature_num=6)
    # model = GCN_GRU_DNM(node_features=6, gcn_out_channels=32, gru_hidden_size=128, gru_num_layers=1)
    # model = GCN_GRU_DNM_2(node_features=6, gcn_out_channels=32, gru_hidden_size=128, gru_num_layers=1, M=10)


    model = GCN_GRU_KAN(node_features=6, gcn_out_channels=32, gru_hidden_size=128, gru_num_layers=1)
    edge_index = torch.tensor([
        [0, 1],
        [1, 2]
    ], dtype=torch.long)

    # 打印整个模型的参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"整个模型的参数量: {total_params}")

    # 打印 组件 的参数量
    transformer_params = sum(p.numel() for p in model.fc.parameters())
    print(f"参数量: {transformer_params}")

    y = model(x, edge_index)
    print(y.shape)

    # parameters_num = count_parameters(model)
    # print(parameters_num)


if __name__ == '__main__':
    main()
