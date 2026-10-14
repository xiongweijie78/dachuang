import numpy as np
import pandas as pd
import matplotlib
import time  # 新增：用于计时

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import warnings

warnings.filterwarnings('ignore')

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ------------------------------------------------------------------------------
# 自动编码器特征提取
# ------------------------------------------------------------------------------
def autoencoder_feature_extraction(df, target_col='Active_Power', encoding_dim=8, epochs=50):
    """
    使用自动编码器进行特征提取
    """
    # 记录自动编码器开始时间
    ae_start_time = time.time()

    # 1. 准备数据（排除目标列）
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in feature_cols:
        feature_cols.remove(target_col)

    X = df[feature_cols].values

    # 2. 构建自动编码器
    input_dim = X.shape[1]

    # 编码器
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim * 2, activation='relu',
                    activity_regularizer=l1_l2(l1=1e-5, l2=1e-4))(input_layer)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)

    # 解码器
    decoded = Dense(encoding_dim * 2, activation='relu')(encoded)
    decoded = Dropout(0.2)(decoded)
    decoded = Dense(input_dim, activation='linear')(decoded)

    # 构建模型
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    # 编译和训练
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # 训练自动编码器
    print("🔄 训练自动编码器中...")
    ae_training_start = time.time()
    history_ae = autoencoder.fit(X, X,
                                 epochs=epochs,
                                 batch_size=32,
                                 shuffle=True,
                                 validation_split=0.1,
                                 verbose=1)
    ae_training_time = time.time() - ae_training_start

    # 3. 提取特征
    encoded_features = encoder.predict(X, verbose=0)

    # 4. 创建新的特征DataFrame
    encoded_cols = [f'AE_Feature_{i}' for i in range(encoding_dim)]
    df_encoded = pd.DataFrame(encoded_features, columns=encoded_cols, index=df.index)

    # 5. 将编码特征与目标列合并
    df_final = pd.concat([df_encoded, df[[target_col]]], axis=1)

    ae_total_time = time.time() - ae_start_time
    print(f"✅ 自动编码器特征提取完成，耗时：{ae_total_time:.2f}秒")
    print(f"自动编码器训练时间：{ae_training_time:.2f}秒")
    print(f"原始特征数：{len(feature_cols)} → 编码特征数：{encoding_dim}")

    return df_final, encoded_cols, {
        'ae_training_time': ae_training_time,
        'ae_total_time': ae_total_time
    }


# ------------------------------------------------------------------------------
# 步骤1：数据加载+自动编码器特征提取+归一化
# ------------------------------------------------------------------------------
def load_preprocess_ae(file_path, target_col='Active_Power', encoding_dim=8, ae_epochs=30):
    """
    使用自动编码器进行特征提取的预处理
    """
    # 记录预处理开始时间
    preprocessing_start = time.time()

    # 1. 读取数据（保留时间索引）
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)

    # 2. 填充缺失值（线性插值+前后向兜底）
    df_interpolated = df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

    # 3. 自动编码器特征提取
    df_encoded, selected_features, ae_times = autoencoder_feature_extraction(
        df_interpolated, target_col, encoding_dim, ae_epochs)

    # 4. 数据分布统计
    print(f"\n📈 {target_col}分布统计（自动编码器特征提取后）：")
    print(f"最小值：{df_interpolated[target_col].min():.6f}")
    non_zero_vals = df_interpolated[df_interpolated[target_col] > 0][target_col]
    if len(non_zero_vals) > 0:
        print(f"非零最小值：{non_zero_vals.min():.6f}")
    else:
        print(f"非零最小值：无（所有值均为0）")
    print(f"均值：{df_interpolated[target_col].mean():.6f}")
    print(f"接近0的样本数（<0.1）：{sum(df_interpolated[target_col] < 0.1)}")

    # 5. 归一化
    scale_params = {}
    df_scaled = df_encoded.copy()
    for col in df_scaled.columns:
        x_min = df_encoded[col].min()
        x_max = df_encoded[col].max()
        denominator = max(x_max - x_min, 1e-8)
        df_scaled[col] = (df_encoded[col] - x_min) / denominator
        scale_params[col] = {'min': x_min, 'max': x_max, 'denominator': denominator}

    preprocessing_time = time.time() - preprocessing_start

    return df_scaled, selected_features, scale_params, target_col, {
        'preprocessing_time': preprocessing_time,
        'ae_training_time': ae_times['ae_training_time'],
        'ae_total_time': ae_times['ae_total_time']
    }


# ------------------------------------------------------------------------------
# 步骤2：构建LSTM输入样本
# ------------------------------------------------------------------------------
def create_lstm_samples(data, target_col, seq_len=12, test_size=0.2):
    """
    将时序数据转为LSTM输入格式：[样本数, 时间步长, 特征数]
    """
    # 提取特征和目标数组
    X = data.drop(columns=[target_col]).values
    y = data[target_col].values
    n_samples = len(X) - seq_len

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

    model = Sequential(name='LSTM_Load_Prediction')
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
    plt.savefig('Z_LSTM_AE_Training_Loss.png', dpi=300, bbox_inches='tight')
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
    plt.title('LSTM光伏功率预测：真实值vs预测值（自动编码器特征）')
    plt.xlabel('时间步')
    plt.ylabel(f'{target_col}')
    plt.legend()
    plt.savefig('Z_LSTM_AE_Prediction_Comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6. 误差分布图
    plt.figure(figsize=(10, 4))
    errors = y_pred_real - y_test_real
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', label='零误差线')
    plt.title('LSTM预测误差分布（自动编码器特征）')
    plt.xlabel('误差值')
    plt.ylabel('频数')
    plt.legend()
    plt.savefig('Z_AE_Prediction_Error_Distribution.png', dpi=300, bbox_inches='tight')
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
    ENCODING_DIM = 8  # 自动编码器编码维度
    AE_EPOCHS = 30  # 自动编码器训练轮次
    SEQ_LEN = 12
    EPOCHS = 50
    BATCH_SIZE = 32

    # 执行完整流程
    print("=" * 60)
    print("开始光伏功率预测流程（自动编码器特征提取）")
    print("=" * 60)

    # 步骤1：数据加载+自动编码器特征提取+归一化
    preprocessing_start = time.time()
    df_scaled, selected_features, scale_params, target_col, ae_times = load_preprocess_ae(
        file_path=FILE_PATH,
        target_col=TARGET_COL,
        encoding_dim=ENCODING_DIM,
        ae_epochs=AE_EPOCHS
    )
    preprocessing_time = ae_times['preprocessing_time']

    # 步骤2：构建LSTM样本
    sampling_start = time.time()
    X_train, X_test, y_train, y_test = create_lstm_samples(
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
    print("🏆 性能总结报告（自动编码器特征提取版本）")
    print("=" * 60)

    print(f"\n⏱️ 时间统计：")
    print(f"   - 数据预处理：{preprocessing_time:.2f}s")
    print(f"     (其中自动编码器训练：{ae_times['ae_training_time']:.2f}s)")
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

    print(f"\n🔍 特征提取信息：")
    print(f"   - 编码维度：{ENCODING_DIM}")
    print(f"   - 自动编码器训练轮次：{AE_EPOCHS}")

    print(f"\n📁 生成文件：")
    print(f"   1. Z_LSTM_AE_Training_Loss.png（LSTM模型训练损失曲线）")
    print(f"   2. Z_LSTM_AE_Prediction_Comparison.png（真实值vs预测值对比）")
    print(f"   3. Z_AE_Prediction_Error_Distribution.png（预测误差分布）")
    print("=" * 60)