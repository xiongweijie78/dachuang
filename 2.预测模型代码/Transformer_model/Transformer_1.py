import torch
from torch import nn


class transformer_block(nn.Module):

    def __init__(self, embed_size, num_heads):
        super(transformer_block, self).__init__()

        self.attention = nn.MultiheadAttention(embed_size, num_heads, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(embed_size, 4 * embed_size),
                                nn.LeakyReLU(),
                                nn.Linear(4 * embed_size, embed_size))
        self.dropout = nn.Dropout(0.5)
        self.ln1 = nn.LayerNorm(embed_size, eps=1e-6)
        self.ln2 = nn.LayerNorm(embed_size, eps=1e-6)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x, need_weights=False)
        x = x + self.dropout(attn_out)
        x = self.ln1(x)

        fc_out = self.fc(x)
        x = x + self.dropout(fc_out)
        x = self.ln2(x)

        return x


class transformer_forecaster(nn.Module):

    def __init__(self, embed_size, num_heads, num_blocks):
        super(transformer_forecaster, self).__init__()

        num_len = len(numeric_covariates)
        self.embedding_cov = nn.ModuleList(
            [nn.Embedding(n, embed_size - num_len) for n in categorical_covariates_num_embeddings])
        self.embedding_static = nn.ModuleList(
            [nn.Embedding(n, embed_size - num_len) for n in categorical_static_num_embeddings])

        self.blocks = nn.ModuleList([transformer_block(embed_size, num_heads) for n in range(num_blocks)])

        self.forecast_head = nn.Sequential(nn.Linear(embed_size, embed_size * 2),
                                           nn.LeakyReLU(),
                                           nn.Dropout(0.5),
                                           nn.Linear(embed_size * 2, embed_size * 4),
                                           nn.LeakyReLU(),
                                           nn.Linear(embed_size * 4, forecast_length),
                                           nn.ReLU())

    def forward(self, x_numeric, x_category, x_static):

        tmp_list = []
        for i, embed_layer in enumerate(self.embedding_static):
            tmp_list.append(embed_layer(x_static[:, i]))
        categroical_static_embeddings = torch.stack(tmp_list).mean(dim=0).unsqueeze(1)

        tmp_list = []
        for i, embed_layer in enumerate(self.embedding_cov):
            tmp_list.append(embed_layer(x_category[:, :, i]))
        categroical_covariates_embeddings = torch.stack(tmp_list).mean(dim=0)
        T = categroical_covariates_embeddings.shape[1]

        embed_out = (categroical_covariates_embeddings + categroical_static_embeddings.repeat(1, T, 1)) / 2
        x = torch.concat((x_numeric, embed_out), dim=-1)

        for block in self.blocks:
            x = block(x)

        x = x.mean(dim=1)
        x = self.forecast_head(x)

        return x

def main():
    x = torch.rand(32, 3, 7)

    model = Transformer(feature_size=7, num_heads=7)
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    main()