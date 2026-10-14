import torch
from torch.utils.data import Dataset, DataLoader

from pandas import read_csv
from pandas import DataFrame
from pandas import concat

from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


class Power_ds(Dataset):
    def __init__(self, n_hours, n_features, data_type, unsqueeze=True):
        super(Power_ds, self).__init__()

        self.data_type = data_type
        self.n_hours = n_hours
        self.n_features = n_features
        self.unsqueeze=unsqueeze
        self.values = self.preprocessing()

        if self.data_type == 'train':
            n_hours = self.n_hours
            n_features = self.n_features
            n_train_samples = int(len(self.values) * 0.6)
            train = self.values[:n_train_samples, :]
            n_obs = n_hours * n_features
            train_X, train_y = train[:, :n_obs], train[:, -n_features]
            train_X = train_X.reshape(
                (train_X.shape[0], n_hours, n_features))
            if self.unsqueeze:
                 train_X = train_X.reshape(
                      (train_X.shape[0], 1, n_hours, n_features))  # [batch_size, 1, n_hours, n_features]
            train_y = train_y.reshape(-1, 1)
            train_X = torch.Tensor(train_X)
            train_y = torch.Tensor(train_y)
            self.data_x = train_X
            self.data_y = train_y

        if self.data_type == 'test':
            n_hours = self.n_hours
            n_features = self.n_features
            n_train_samples = int(len(self.values) * 0.6)
            test = self.values[n_train_samples:, :]
            n_obs = n_hours * n_features
            test_X, test_y = test[:, :n_obs], test[:, -n_features]
            test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
            if self.unsqueeze:
                test_X = test_X.reshape((test_X.shape[0], 1, n_hours, n_features))
            test_y = test_y.reshape(-1, 1)
            test_X = torch.Tensor(test_X)
            test_y = torch.Tensor(test_y)
            self.data_x = test_X
            self.data_y = test_y

    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]

        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def preprocessing(self):
        n_hours = self.n_hours
        n_features = self.n_features
        # Read power.csv from the same folder as this Python file.
        # This avoids dependence on the terminal's current working directory.
        data_path = Path(__file__).resolve().parent / 'power.csv'
        if not data_path.exists():
            raise FileNotFoundError(
                f'Cannot find data file: {data_path}. '
                'Please place power.csv in the same folder as power_dataset.py.'
            )
        dataset = read_csv(data_path, header=0, index_col=0)
        values = dataset.values
        values = values.astype('float32')

        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(values)
        scaled = scaler.fit_transform(values)

        # frame as supervised learning
        reframed = self.series_to_supervised(scaled, n_hours, 1)

        values = reframed.values

        return values

    def __len__(self):

        data = self.data_y
        return len(data)

    def __getitem__(self, item):

        data_x, data_y = self.data_x[item], self.data_y[item]

        return data_x, data_y


def main():
    train_ds = Power_ds(3, 6, data_type='train', unsqueeze=False)
    test_ds = Power_ds(3, 6, data_type='test', unsqueeze=False)


    train_loader = DataLoader(train_ds, batch_size=72, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

    x, y = next(iter(train_loader))
    print(x.shape, y.shape)

    print(len(train_ds), len(test_ds))


if __name__ == '__main__':
    main()
