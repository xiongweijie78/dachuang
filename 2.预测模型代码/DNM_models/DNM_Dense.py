import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DNMLayerDense(nn.Module):
    def __init__(self, input_size, size_out, M, synapse_activation=torch.sigmoid, activation=None):
        super(DNMLayerDense, self).__init__()
        self.input_size = input_size
        self.size_out = size_out
        self.M = M
        self.synapse_activation = synapse_activation
        self.activation = activation
        self.dense_layer = nn.Linear(input_size, size_out*M)
        self.dnm_weight = nn.Parameter(torch.empty(size_out, M))
        self.k = nn.Parameter(torch.tensor(0.1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.dnm_weight, mean=0, std=np.sqrt(2 / (self.size_out + self.M)))

    def forward(self, x):
        # 重构输入数据的形状以匹配全连接层的输入
        # x = x.view(x.size(0), -1)
        fc = self.dense_layer(x)
        k_fc = fc * self.k
        activation_fc = self.synapse_activation(k_fc)

        reshape_activation_fc = activation_fc.view(-1, self.size_out, self.M)
        dnm_fc = reshape_activation_fc * self.dnm_weight
        out = torch.sum(dnm_fc, dim=2)

        if self.activation:
            out = self.activation(out)
        return out