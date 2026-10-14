import numpy as np
import pandas as pd
import matplotlib
import time  # 新增：用于计时

matplotlib.use('Agg')#代码生成图片时直接生成本地文件
import matplotlib.pyplot as plt#导入绘图模块
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score#机器学习工具库，均方误差，平均绝对误差，决定系数
from tensorflow.keras.models import Sequential, Model#深度学习框架，高级接口，models模型。一层接一层的顺序堆叠模型。函数式通用模型
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten, Reshape#搭建模型的积木。长短期记忆。全连接层。随即关掉一部分神经元。输入数据形状。一维卷积，提取时序特征。压缩特征，降低数据量。多维变一维。一维变多维
from tensorflow.keras.optimizers import Adam#优化器工具包，自动更新神经网络权重，减小mae和mse损失。自适应学习率优化器。
import warnings

warnings.filterwarnings('ignore')#忽略所有烦人警告，只输出希望输出的

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']#将字体改为黑体
plt.rcParams['axes.unicode_minus'] = False#关闭unicode格式负号，使负号正常输出


# ------------------------------------------------------------------------------
# CNN特征提取
# ------------------------------------------------------------------------------
def cnn_feature_extraction(df, target_col='Active_Power', seq_len=12, filters=32, kernel_size=3, epochs=50):
    #df为pandas表格，原始数据。target_col标签列，要预测的有功功率。seq_len时间步长，用连续12条历史数据当作一段时序样本。
    # filters卷积核数量，最终输出32维特征。kernel_size一次卷积读取连续 3 个时间点的数据。epochesCNN 自编码器一共循环训练 50 轮
    """
    使用1D CNN进行时序特征提取
    """
    # 记录CNN开始时间
    cnn_start_time = time.time()#获取当下时间

    # 1. 准备数据

    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()#取出列表中的数字列，取这些列的名字，转换成普通python列表
    if target_col in feature_cols:
        feature_cols.remove(target_col)#不喂入功率列
    X = df[feature_cols].values#将挑选过的表格转换为numpy二维数组

    # 2. 构建序列数据
    n_samples = len(X) - seq_len
    X_seq = np.array([X[i:i + seq_len] for i in range(n_samples)])#构成三维数组

    # 3. 构建CNN自动编码器
    input_seq = Input(shape=(seq_len, len(feature_cols)))#定义输入数据类型，样本数,12，特征数(None, 12, 3)

    # 编码器
    x = Conv1D(filters, kernel_size, activation='relu', padding='same')(input_seq)#输出特征32，一次卷积三个时间点数据，计算结果变化，两边补零(None, 12, 32)
    x = MaxPooling1D(2)(x)#每相邻两个时间点取较大数据，序列长度减半(None, 6, 32)
    x = Conv1D(filters * 2, kernel_size, activation='relu', padding='same')(x)#输出特征64(None, 6, 64)
    x = Flatten()(x)#二维转一维(None,384)
    encoded = Dense(filters, activation='relu', name='encoded_features')(x)#输出32个数据的浓缩特征(None, 32)

    # 解码器
    x = Dense(filters * 2, activation='relu')(encoded)#(None, 64)
    x = Dense(seq_len * len(feature_cols), activation='linear')(x)#(None, 32)
    decoded = Reshape((seq_len, len(feature_cols)))(x)#(None,12,3)

    # 构建模型
    cnn_autoencoder = Model(input_seq, decoded)#训练模型得到前到后的路线参数
    encoder = Model(input_seq, encoded)#截取上边路线

    # 编译和训练
    cnn_autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')#Adam调整权重参数，学习率为0.001。按均方误差mse调整decoded

    print("🔄 训练CNN自动编码器中...")
    cnn_training_start = time.time()
    history_cnn = cnn_autoencoder.fit(X_seq, X_seq,#训练输入，训练标签y
                                      epochs=epochs,#训练轮数50轮
                                      batch_size=32,#一次喂32条样本
                                      shuffle=True,#打乱样本顺序
                                      validation_split=0.1,#拿0.1当验证集，算验证loss
                                      verbose=1)#打印训练日志，显示进度
    cnn_training_time = time.time() - cnn_training_start

    # 4. 提取特征
    encoded_features = encoder.predict(X_seq, verbose=0)#每条样本 32 维压缩抽象特征

    # 5. 创建特征DataFrame
    encoded_cols = [f'CNN_Feature_{i}' for i in range(filters)]
    df_encoded = pd.DataFrame(encoded_features, columns=encoded_cols,#讲numpy数组转为DataFrame，N行32列，加上特征标签
                              index=df.index[seq_len:seq_len + len(encoded_features)])#复用原始df的时间索引，保证行和原始数据时间对齐

    # 6. 合并数据
    df_final = pd.concat([df_encoded, df[[target_col]].iloc[seq_len:]], axis=1) # 将CNN提取的32维压缩特征，和真实目标标签按行拼接。从第12行开始拼接，按列拼接
    df_final = df_final.dropna()  # 删除存在缺失值的行，清洗数据

    cnn_total_time = time.time() - cnn_start_time  # 计算完整流程总耗时（构建网络+训练+特征提取全部时间）
    print(f"✅ CNN特征提取完成，耗时：{cnn_total_time:.2f}秒")
    print(f"CNN自动编码器训练时间：{cnn_training_time:.2f}秒")
    print(f"滤波器数：{filters}，卷积核大小：{kernel_size}")  # 打印CNN超参数
    print(f"处理后数据形状：{df_final.shape}")  # 打印拼接完成后的表格行列数

    return df_final, encoded_cols, {
        'cnn_training_time': cnn_training_time,  # 返回自编码器训练耗时
        'cnn_total_time': cnn_total_time  # 返回整套CNN特征工程总耗时
    }

# ------------------------------------------------------------------------------
# 步骤1：数据加载+CNN特征提取+归一化
# ------------------------------------------------------------------------------
def load_preprocess_cnn(file_path, target_col='Active_Power', seq_len=12, filters=32, kernel_size=3, cnn_epochs=30):
    # file_path为csv文件路径。target_col标签列，要预测的有功功率。seq_len时间步长，用连续12条历史数据当作一段时序样本。
    # filters卷积核数量，最终输出32维特征。kernel_size一次卷积读取连续 3 个时间点的数据。epochesCNN 自编码器一共循环训练 30 轮
    """
    使用CNN进行特征提取的预处理
    """
    # 记录预处理整体开始时间
    preprocessing_start = time.time()

    # 1. 读取数据（保留时间索引）
    df = pd.read_csv(file_path, index_col=0)#读取csv，第一列为索引
    df.index = pd.to_datetime(df.index)#索引转为时间格式

    # 2. 填充缺失值（线性插值+前后向兜底）
    df_interpolated = df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')#插值填充缺失数据.先线性插值，再后面数据填，再前面数据填

    # 3. CNN特征提取
    df_encoded, selected_features, cnn_times = cnn_feature_extraction(
        df_interpolated, target_col, seq_len, filters, kernel_size, cnn_epochs)#调用CNN特征提取函数

    # 4. 数据分布统计
    print(f"\n📈 {target_col}分布统计（CNN特征提取后）：")
    print(f"最小值：{df_interpolated[target_col].min():.6f}")
    non_zero_vals = df_interpolated[df_interpolated[target_col] > 0][target_col]#筛选大于0的功率样本,先筛选功率大于0的行，再选出具体的行和列
    if len(non_zero_vals) > 0:
        print(f"非零最小值：{non_zero_vals.min():.6f}")
    else:
        print(f"非零最小值：无（所有值均为0）")
    print(f"均值：{df_interpolated[target_col].mean():.6f}")
    print(f"接近0的样本数（<0.1）：{sum(df_interpolated[target_col] < 0.1)}")

    # 5. 归一化，min‑max缩放CNN特征到0~1
    scale_params = {}#保存各列归一化参数，推理时复用
    df_scaled = df_encoded.copy()
    for col in df_scaled.columns:#取列名
        x_min = df_encoded[col].min()
        x_max = df_encoded[col].max()
        denominator = max(x_max - x_min, 1e-8)#防止分母为0报错
        df_scaled[col] = (df_encoded[col] - x_min) / denominator#最大最小归一计算
        scale_params[col] = {'min': x_min, 'max': x_max, 'denominator': denominator}

    preprocessing_time = time.time() - preprocessing_start#计算整套预处理耗时

    return df_scaled, selected_features, scale_params, target_col, {#归一化后的数据集，保存CNN特征的字符串,保存归一化数据字典，目标列名字
        'preprocessing_time': preprocessing_time,#整体预处理耗时
        'cnn_training_time': cnn_times['cnn_training_time'],#CNN自编码器训练耗时
        'cnn_total_time': cnn_times['cnn_total_time']#CNN特征提取全套耗时
    }

# ------------------------------------------------------------------------------
# 步骤2：构建LSTM输入样本（针对CNN特征调整）
# ------------------------------------------------------------------------------
def create_lstm_samples_cnn(data, target_col, seq_len=12, test_size=0.2):
    """
    将CNN提取的特征转为LSTM输入格式
    """
    # 提取特征和目标数组
    X = data.drop(columns=[target_col]).values
    y = data[target_col].values
    n_samples = len(X) - seq_len

    # 构建样本
    X_seq, y_seq = [], []
    for i in range(n_samples):
        X_seq.append(X[i:i + seq_len, :])
        y_seq.append(y[i + seq_len])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # 划分训练集/测试集
    split_idx = int(len(X_seq) * (1 - test_size))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    print(f"\n📦 LSTM样本划分完成：")
    print(f"训练集：X={X_train.shape}, y={y_train.shape}")
    print(f"测试集：X={X_test.shape}, y={y_test.shape}")

    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------------------------
# 步骤3：构建并训练LSTM模型（增加时间测量）
# ------------------------------------------------------------------------------
def build_train_lstm(X_train, y_train, seq_len, feature_num, epochs=50, batch_size=32):
    """
    构建LSTM预测模型（增加时间测量）
    """
    # 记录开始时间
    start_time = time.time()

    model = Sequential(name='LSTM_Load_Prediction_CNN')
    model.add(LSTM(units=128, return_sequences=True,
                   input_shape=(seq_len, feature_num), name='LSTM_1'))
    model.add(Dropout(0.3, name='Dropout_1'))

    model.add(LSTM(units=64, return_sequences=False, name='LSTM_2'))
    model.add(Dropout(0.3, name='Dropout_2'))

    model.add(Dense(units=1, name='Output'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # 训练模型
    print(f"\n⏳ 开始LSTM模型训练...")
    training_start = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )
    training_time = time.time() - training_start
    total_time = time.time() - start_time

    # 绘制训练损失曲线
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title(f'LSTM模型训练损失曲线 (训练时间: {training_time:.1f}s)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig('CNN_LSTM_Training_Loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ LSTM模型训练完成")
    print(f"训练时间：{training_time:.2f}秒 ({training_time / 60:.2f}分钟)")
    print(f"总耗时：{total_time:.2f}秒")
    print(f"平均每轮训练时间：{training_time / epochs:.2f}秒")

    return model, history, {
        'training_time': training_time,
        'total_time': total_time
    }


# ------------------------------------------------------------------------------
# 步骤4：预测与评估（增加预测时间测量）
# ------------------------------------------------------------------------------
def predict_evaluate(model, X_test, y_test, scale_params, target_col):
    """
    模型预测+逆归一化+计算评估指标（增加预测时间测量）
    仅保留MAE、RMSE、MAPE、R²四个核心指标
    """
    # 测量预测时间
    print(f"\n⏳ 开始模型预测...")
    predict_start = time.time()

    # 1. 预测与逆归一化
    y_pred = model.predict(X_test, verbose=1)

    predict_time = time.time() - predict_start
    print(f"预测完成，耗时：{predict_time:.2f}秒")
    print(f"平均每个样本预测时间：{predict_time / len(X_test):.4f}秒")

    min_val = scale_params[target_col]['min']
    denominator = scale_params[target_col]['denominator']
    y_test_real = y_test * denominator + min_val
    y_pred_real = y_pred.flatten() * denominator + min_val

    # 2. 计算四个核心评估指标
    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    r2 = r2_score(y_test_real, y_pred_real)

    # 3. 改进的动态阈值计算MAPE
    non_zero_real = y_test_real[y_test_real > 0]

    if len(non_zero_real) > 0:
        # 使用更稳健的阈值计算
        quantile_10 = np.percentile(non_zero_real, 10)
        std_non_zero = non_zero_real.std()
        mean_real = y_test_real.mean()

        # 提高阈值设置，避免过小阈值
        epsilon = max(quantile_10, std_non_zero * 0.15, mean_real * 0.05, 0.5)
    else:
        epsilon = 1.0

    # 应用阈值计算MAPE
    y_test_safe = np.where(y_test_real < epsilon, epsilon, y_test_real)
    mape = (np.mean(np.abs(y_test_real - y_pred_real) / y_test_safe) * 100)

    # 4. 打印简洁的评估结果（仅四个核心指标）
    print(f"\n📊 模型评估结果（{target_col}）：")
    print(f"MAE（平均绝对误差）：{mae:.2f}")
    print(f"RMSE（均方根误差）：{rmse:.2f}")
    print(f"MAPE（平均绝对百分比误差）：{mape:.2f}%")
    print(f"R²（决定系数）：{r2:.4f}")

    # 5. 可视化
    plt.figure(figsize=(12, 4))
    plt.plot(y_test_real[:200], label='真实功率', linewidth=1.5)
    plt.plot(y_pred_real[:200], label='预测功率', linewidth=1.5, linestyle='--')
    plt.title('LSTM光伏功率预测：真实值vs预测值（CNN特征）')
    plt.xlabel('时间步')
    plt.ylabel(f'{target_col}')
    plt.legend()
    plt.savefig('CNN_LSTM_Prediction_Comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6. 误差分布图
    plt.figure(figsize=(10, 4))
    errors = y_pred_real - y_test_real
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', label='零误差线')
    plt.title('LSTM预测误差分布（CNN特征）')
    plt.xlabel('误差值')
    plt.ylabel('频数')
    plt.legend()
    plt.savefig('CNN_Prediction_Error_Distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2,
        'predict_time': predict_time
    }


# ------------------------------------------------------------------------------
# 主流程：增加整体时间测量和性能报告
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # 记录整体开始时间
    total_start_time = time.time()

    # 配置参数
    FILE_PATH = r'PV_YS.csv'
    TARGET_COL = 'Active_Power'
    SEQ_LEN = 12
    FILTERS = 32  # CNN滤波器数量
    KERNEL_SIZE = 3  # 卷积核大小
    CNN_EPOCHS = 30  # CNN训练轮次
    EPOCHS = 50
    BATCH_SIZE = 32

    # 执行完整流程
    print("=" * 60)
    print("开始光伏功率预测流程（CNN特征提取）")
    print("=" * 60)

    # 步骤1：数据加载+CNN特征提取+归一化
    preprocessing_start = time.time()
    df_scaled, selected_features, scale_params, target_col, cnn_times = load_preprocess_cnn(
        file_path=FILE_PATH,
        target_col=TARGET_COL,
        seq_len=SEQ_LEN,
        filters=FILTERS,
        kernel_size=KERNEL_SIZE,
        cnn_epochs=CNN_EPOCHS
    )
    preprocessing_time = cnn_times['preprocessing_time']

    # 步骤2：构建LSTM样本
    sampling_start = time.time()
    X_train, X_test, y_train, y_test = create_lstm_samples_cnn(
        data=df_scaled,
        target_col=target_col,
        seq_len=SEQ_LEN
    )
    sampling_time = time.time() - sampling_start

    # 步骤3：构建并训练LSTM
    training_start = time.time()
    lstm_model, train_history, model_info = build_train_lstm(
        X_train=X_train,
        y_train=y_train,
        seq_len=SEQ_LEN,
        feature_num=len(selected_features),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    training_time = model_info['training_time']

    # 步骤4：预测与评估（仅保留四个核心指标）
    evaluation_start = time.time()
    metrics = predict_evaluate(
        model=lstm_model,
        X_test=X_test,
        y_test=y_test,
        scale_params=scale_params,
        target_col=target_col
    )
    evaluation_time = time.time() - evaluation_start

    # 计算总时间
    total_time = time.time() - total_start_time

    # 生成性能总结报告
    print("\n" + "=" * 60)
    print("🏆 性能总结报告（CNN特征提取版本）")
    print("=" * 60)

    print(f"\n⏱️ 时间统计：")
    print(f"   - 数据预处理：{preprocessing_time:.2f}s")
    print(f"     (其中CNN特征提取：{cnn_times['cnn_total_time']:.2f}s)")
    print(f"     (CNN自动编码器训练：{cnn_times['cnn_training_time']:.2f}s)")
    print(f"   - 样本构建：{sampling_time:.2f}s")
    print(f"   - LSTM模型训练：{training_time:.2f}s ({training_time / 60:.2f}分钟)")
    print(f"   - 预测评估：{evaluation_time:.2f}s")
    print(f"   - 总耗时：{total_time:.2f}s ({total_time / 60:.2f}分钟)")

    print(f"\n🎯 预测性能：")
    print(f"   - MAE：{metrics['MAE']:.2f}")
    print(f"   - RMSE：{metrics['RMSE']:.2f}")
    print(f"   - MAPE：{metrics['MAPE']:.2f}%")
    print(f"   - R²：{metrics['R²']:.4f}")
    print(f"   - 预测速度：{metrics['predict_time'] / len(X_test):.4f}s/样本")

    print(f"\n🔍 CNN特征提取信息：")
    print(f"   - 滤波器数量：{FILTERS}")
    print(f"   - 卷积核大小：{KERNEL_SIZE}")
    print(f"   - 编码维度：{len(selected_features)}")
    print(f"   - CNN训练轮次：{CNN_EPOCHS}")

    print(f"\n📁 生成文件：")
    print(f"   1. CNN_LSTM_Training_Loss.png（LSTM模型训练损失曲线）")
    print(f"   2. CNN_LSTM_Prediction_Comparison.png（真实值vs预测值对比）")
    print(f"   3. CNN_Prediction_Error_Distribution.png（预测误差分布）")
    print("=" * 60)