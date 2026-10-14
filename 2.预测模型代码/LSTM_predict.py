import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from numpy import concatenate
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import numpy as np
from matplotlib import pyplot as plt
from typical_models import LSTM_model
import time
from power_dataset import Power_ds


# calculate MAPE
def mean_absolute_percentage_error(real, predict):
    res = 0
    count = 0
    for i in range(len(real)):
        if real[i] != 0:
            res += abs((predict[i] - real[i]) / real[i])
            count += 1
    if count == 0:
        return 0  # Avoid division by zero
    return res / count

def plot_predict(pre, test_labels):
    plt.plot(test_labels, label='real')
    plt.plot(pre, label='predict')
    plt.title('real-predict')
    plt.xlabel('hour')
    plt.ylabel('power')
    plt.legend()  # 显示图例，与label搭配使用
    plt.show()


n_hours = 3
n_features = 6

batchsize = 32
lr = 1e-3
epochs = 100

mape_results = []
rmse_results = []
R2_results = []
mae_results = []
runtime_results = []

max_r2 = 0
best_predict = 0

n_runs = 10

dataset = read_csv(
    'power.csv',
    header=0, index_col=0)
values = dataset.values
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1)).fit(values)

device = torch.device('cuda')
train_ds = Power_ds(3, 6, 'train', unsqueeze=False)
test_ds = Power_ds(3, 6, 'test', unsqueeze=False)

train_loader = DataLoader(train_ds, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

for run in range(n_runs):
    start_time = time.time()

    model = LSTM_model(feature_num=6, layer_num=1, nodes_num=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.L1Loss()

    loss_list = []

    for epoch in range(epochs):
        total_loss = []

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criteon(logits, y)
            total_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # if (epoch+1)%5 == 0:

        print('MAE loss of run{} ,epoch{}:{:.4}'.format(run+1, epoch+1, np.mean(total_loss)))
        loss_list.append(np.mean(total_loss))

    for test_X, test_y in test_loader:
        test_X, test_y = test_X.to(device), test_y.to(device)

        with torch.no_grad():

            yhat = model(test_X)

        yhat = yhat.cpu().detach().numpy()
        if yhat.shape[1] != 1:
            raise ValueError("Prediction result shape is incorrect")

        # print(yhat.shape)
        test_X = test_X.cpu().detach().numpy()
        test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))

        # invert scaling for forecast
        inv_yhat = concatenate((yhat, test_X[:, -n_features+1:]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:, 0]
        # invert scaling for actual
        test_y = test_y.cpu().detach().numpy()
        test_y = test_y.reshape(len(test_y), 1)
        inv_y = concatenate((test_y, test_X[:, -n_features+1:]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, 0]

        # calculate RMSE
        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        print('Test RMSE: %.3f' % rmse)
        rmse_results.append(rmse)

        # calculate MAPE
        mape = mean_absolute_percentage_error(inv_y, inv_yhat)
        print('Test mape: %.3f' % mape)
        mape_results.append(mape)

        # calculate R^2
        R2 = r2_score(inv_y, inv_yhat)
        print(f"R-squared (R^2): {R2:.3f}")
        R2_results.append(R2)

        # calculate MAE
        mae = mean_absolute_error(inv_y, inv_yhat)
        print('Test MAE: %.3f' % mae)
        mae_results.append(mae)

        # calculate TIME
        end_time = time.time()
        runtime = end_time - start_time
        runtime_results.append(runtime)

        print(
            f"Run {run + 1}: Runtime = {runtime:.3f} seconds, MAE = {mae:.3f}, MAPE = {mape:.3f}, RMSE = {rmse:.3f}, R2 = {R2:.3f}")

        if R2 > max_r2:
            best_predict = inv_yhat
            loss_trend = loss_list


# 计算n次运行的均值
mean_runtime = np.mean(runtime_results)
mean_mae = np.mean(mae_results)
mean_mape = np.mean(mape_results)
mean_rmse = np.mean(rmse_results)
mean_R2 = np.mean(R2_results)

print(f"Average Runtime over {n_runs} runs: {mean_runtime:.3f} seconds")
print(f"Average MAE over {n_runs} runs: {mean_mae:.3f}")
print(f"Average MAPE over {n_runs} runs: {mean_mape:.3f}")
print(f"Average RMSE over {n_runs} runs: {mean_rmse:.3f}")
print(f"Average R2 over {n_runs} runs: {mean_R2:.3f}")

# plot_predict(best_predict, inv_y)

# # 性能记录
# csv_file1 = 'C:\\Users\\ASUS\\Desktop\\小论文\\实验内容\\performance_excel\\performance_compare.csv'
# # test_labels = inv_y.reshape(-1)
# best_predict = best_predict.reshape(-1)
# df = pd.read_csv(csv_file1)
# df['LSTM'] = best_predict
# df.to_csv(csv_file1, index=False)
#
# # loss跟踪
# csv_file2 = 'C:\\Users\\ASUS\\Desktop\\小论文\\实验内容\\performance_excel\\loss_trail.csv'
# c = pd.Series(loss_trend, name='LSTM')
# df = pd.read_csv(csv_file2)
# df['LSTM'] = c
# df.to_csv(csv_file2, index=False)
