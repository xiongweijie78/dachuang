import torch.nn as nn
import torch

device = torch.device('cuda')


class Transformer(nn.Module):
    def __init__(self, feature_size=7, num_layers=3, num_heads=1, dropout=0):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size*3, 1)
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


def main():
    x = torch.rand(32, 3, 7)

    model = Transformer(feature_size=7, num_heads=7)
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    main()
