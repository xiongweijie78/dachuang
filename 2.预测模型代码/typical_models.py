import torch
from torch import nn
import torch.nn.functional as F
import math


class Linear_Model(nn.Module):
    def __init__(self, input_size, out_size):
        super(Linear_Model, self).__init__()

        self.input_size = input_size
        self.out_size = out_size

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_size)
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten()

    def forward(self, input):
        tem = self.flatten(input)
        tem = F.relu(self.fc1(tem))
        tem = F.relu(self.fc2(tem))
        tem = self.dropout(tem)
        out = self.fc3(tem)

        return out


class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, (3, 1), 1)
        self.conv2 = nn.Conv2d(32, 64, (1, 3), 1)
        self.pooling = nn.MaxPool2d((1, 4))
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        tem = F.relu(self.conv1(input))
        tem = F.relu(self.conv2(tem))
        tem = self.pooling(tem)
        tem = self.flat(tem)
        tem = F.relu(self.fc1(tem))
        # tem = self.dropout(tem)
        out = self.fc2(tem)

        return out


class LSTM_model(nn.Module):
    def __init__(self, feature_num, layer_num, nodes_num):
        super(LSTM_model, self).__init__()

        self.feature_num = feature_num
        self.layer_num = layer_num
        self.nodes_num = nodes_num

        self.lstm1 = nn.LSTM(input_size=feature_num, num_layers=layer_num, hidden_size=nodes_num, batch_first=True)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(nodes_num, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, input):
        tem, _ = self.lstm1(input)
        tem = tem[:, -1, :]
        tem = self.flat(tem)
        tem = self.dropout(tem)
        tem = F.relu(self.fc1(tem))
        out = self.fc2(tem)

        return out
class LSTM_KAN_model(nn.Module):
    def __init__(self, feature_num, layer_num, nodes_num):
        super(LSTM_KAN_model, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size=feature_num,
            num_layers=layer_num,
            hidden_size=nodes_num,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(nodes_num, 64)

        # 用 KAN 预测头替换原来的 Linear(64, 1)
        self.kan_head = KANLinear(
            in_features=64,
            out_features=1,
            grid_size=5,
            spline_order=3,
            grid_range=[-1, 1]
        )

    def forward(self, input):
        tem, _ = self.lstm1(input)
        tem = tem[:, -1, :]
        tem = self.dropout(tem)
        tem = F.relu(self.fc1(tem))

        # KAN 内部会自动调用样条基函数
        out = self.kan_head(tem)

        return out

class GRU_model(nn.Module):
    def __init__(self, feature_num, layer_num, nodes_num):
        super(GRU_model, self).__init__()

        self.feature_num = feature_num
        self.layer_num = layer_num
        self.nodes_num = nodes_num

        self.gru1 = nn.GRU(input_size=feature_num, num_layers=layer_num, hidden_size=nodes_num, batch_first=True)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(nodes_num, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        tem, _ = self.gru1(input)
        tem = tem[:, -1, :]
        tem = self.flat(tem)
        tem = self.dropout(tem)
        tem = F.relu(self.fc1(tem))
        out = self.fc2(tem)

        return out

class CNN_LSTM_model(nn.Module):
    def __init__(self, feature_num):
        super(CNN_LSTM_model, self).__init__()

        self.feature_num = feature_num

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, feature_num), stride=(1, 1)),
            nn.ReLU()
        )

        self.lstm3 = nn.LSTM(input_size=64, hidden_size=128, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(128, 1)

    def forward(self, input):
        tem = self.conv1(input)
        tem = tem.squeeze()
        tem = tem.permute([0, 2, 1])
        tem, _ = self.lstm3(tem)
        tem = tem[:, -1, :]  # 取LSTM的最后一个时间步作为输出
        tem = self.dropout(tem)
        out = self.fc(tem)
        return out

# 残差+LSTM
class Res_LSTM_model(nn.Module):
    def __init__(self, feature_num, res_channel=32, conv_out_channel=64, lstm_hidden=128):
        super(Res_LSTM_model, self).__init__()

        self.feature_num = feature_num
        self.res_channel = res_channel
        self.lstm_hidden = lstm_hidden
        self.conv_out_channel = conv_out_channel

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, res_channel, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(res_channel, conv_out_channel, kernel_size=(1, feature_num), stride=(1, 1)),
            nn.ReLU()
        )

        self.lstm1 = nn.LSTM(input_size=conv_out_channel, hidden_size=lstm_hidden, batch_first=True,
                             bidirectional=False)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(lstm_hidden, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, input):
        tem = self.conv1(input)
        tem = tem + input.repeat(1, self.res_channel, 1, 1)
        tem = self.conv2(tem)
        tem = tem.squeeze()
        tem = tem.permute([0, 2, 1])
        tem, _ = self.lstm1(tem)
        tem = tem[:, -1, :]  # 取LSTM的最后一个时间步作为输出
        # tem = self.dropout(tem)
        tem = F.relu(self.fc1(tem))
        out = self.fc2(tem)

        return out


class Transformer(nn.Module):
    def __init__(self, feature_size=7, num_layers=3, num_heads=1, dropout=0):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size * 3, 1)
        self.flatten = nn.Flatten()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        mask = self._generate_square_subsequent_mask(len(src))
        output = self.transformer_encoder(src, mask)
        output = self.flatten(output)
        output = self.decoder(output)
        return output



class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, enable_standalone_scale_spline=True, base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features, -1).contiguous())
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.flat = nn.Flatten()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 1 / 2) * self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_((self.scale_spline if not self.enable_standalone_scale_spline else 1.0) * self.curve2coeff(self.grid.T[self.spline_order : -self.spline_order], noise))
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]) + ((grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        assert result.size() == (self.out_features, self.in_features, self.grid_size + self.spline_order)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x: torch.Tensor):
        x = self.flat(x)
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(self.b_splines(x).view(x.size(0), -1), self.scaled_spline_weight.view(self.out_features, -1))
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)
        splines = self.b_splines(x).permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1) * uniform_step + x_sorted[0] - margin)
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.cat([grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1), grid, grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1)], dim=0)
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return regularize_activation * regularization_loss_activation + regularize_entropy * regularization_loss_entropy


def main():
    x = torch.randn(32, 1, 3, 4)
    # model = Linear_Model(18, 1)
    # model = CNN_model()
    #model = LSTM_model(6, 1, 128)
    #model = CNN_LSTM_model(6)
    # model = GRU_model(feature_num=6, layer_num=1, nodes_num=128)
    #model = Transformer(feature_size=6, num_layers=3, num_heads=6, dropout=0)
    model = KANLinear(18, 1)
    # model = Res_LSTM_model(4, res_channel=32, conv_out_channel=64, lstm_hidden=128)
    out = model(x)

    #打印出model的总参数量
    # print(sum(p.numel() for p in model.parameters()))

    print(out.shape)


if __name__ == "__main__":
    main()
