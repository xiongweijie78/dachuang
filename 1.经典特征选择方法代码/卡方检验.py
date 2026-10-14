import numpy as np
import pandas as pd
import matplotlib
import time  # 新增：用于计时

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.feature_selection import SelectKBest, chi2
import warnings

warnings.filterwarnings('ignore')

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ------------------------------------------------------------------------------
# 步骤1：数据加载+卡方检验特征选择+归一化
# ------------------------------------------------------------------------------
def load_preprocess_select_chi2(file_path, target_col='Active_Power', k_features=10, n_bins=5):
    """
    一体化预处理：读取数据→填充缺失值→卡方检验特征选择→归一化
    """
    # 记录卡方检验开始时间
    chi2_start_time = time.time()

    # 1. 读取数据（保留时间索引）
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)

    # 2. 填充缺失值
    df_interpolated = df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

    # 3. 卡方检验特征选择
    # 初始特征：所有数值列（移除目标列）
    feature_cols = df_interpolated.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in feature_cols:
        feature_cols.remove(target_col)

    # 准备特征和目标数据
    X = df_interpolated[feature_cols]
    y = df_interpolated[target_col]

    # 将连续目标变量离散化（分箱）以便使用卡方检验
    y_discrete = pd.cut(y, bins=n_bins, labels=False)

    # 确保所有特征值为非负（卡方检验要求）
    X_non_negative = X - X.min() + 1e-8

    # 使用卡方检验选择特征
    chi2_selector = SelectKBest(score_func=chi2, k=min(k_features, len(feature_cols)))
    X_selected = chi2_selector.fit_transform(X_non_negative, y_discrete)

    # 获取选择的特征列名
    selected_mask = chi2_selector.get_support()
    selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selected_mask[i]]

    chi2_time = time.time() - chi2_start_time
    print(f"✅ 卡方检验特征选择完成，耗时：{chi2_time:.2f}秒")

    # 打印筛选结果
    print(f"📊 卡方检验筛选结果：")
    print(f"初始特征数：{len(feature_cols)} → 筛选后：{len(selected_features)}")
    print(f"筛选后特征：{selected_features}")
    print(f"各特征卡方统计量：")
    for i, feature in enumerate(selected_features):
        print(f"  {feature}: {chi2_selector.scores_[selected_mask][i]:.3f}")

    # 4. 数据分布统计
    print(f"\n📈 {target_col}分布统计（预处理后）：")
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
    df_scaled = df_interpolated[selected_features + [target_col]].copy()
    for col in df_scaled.columns:
        x_min = df_interpolated[col].min()
        x_max = df_interpolated[col].max()
        if abs(x_max - x_min) < 1e-8:
            denominator = 1.0
        else:
            denominator = x_max - x_min
        df_scaled[col] = (df_interpolated[col] - x_min) / denominator
        scale_params[col] = {'min': x_min, 'max': x_max, 'denominator': denominator}

    return df_scaled, selected_features, scale_params, target_col


# ------------------------------------------------------------------------------
# 步骤2：构建LSTM输入样本
# ------------------------------------------------------------------------------
def create_lstm_samples(data, target_col, seq_len=12, test_size=0.2):
    """
    将时序数据转为LSTM输入格式：[样本数, 时间步长, 特征数]
    """
    X = data.drop(columns=[target_col]).values
    y = data[target_col].values
    n_samples = len(X) - seq_len

    X_seq, y_seq = [], []
    for i in range(n_samples):
        X_seq.append(X[i:i + seq_len, :])
        y_seq.append(y[i + seq_len])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

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
    print(f"\n⏳ 开始模型训练...")
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
    plt.savefig('K_LSTM_Training_Loss.png', dpi=300, bbox_inches='tight')
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
    """
    # 测量预测时间
    print(f"\n⏳ 开始模型预测...")
    predict_start = time.time()

    y_pred = model.predict(X_test, verbose=1)

    predict_time = time.time() - predict_start
    print(f"预测完成，耗时：{predict_time:.2f}秒")
    print(f"平均每个样本预测时间：{predict_time / len(X_test):.4f}秒")

    # 1. 预测与逆归一化
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
    plt.title('LSTM光伏功率预测（卡方特征选择）：真实值vs预测值')
    plt.xlabel('时间步')
    plt.ylabel(f'{target_col}')
    plt.legend()
    plt.savefig('K_LSTM_Prediction_Comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6. 误差分布图
    plt.figure(figsize=(10, 4))
    errors = y_pred_real - y_test_real
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', label='零误差线')
    plt.title('LSTM预测误差分布（卡方特征选择）')
    plt.xlabel('误差值')
    plt.ylabel('频数')
    plt.legend()
    plt.savefig('K_Prediction_Error_Distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2,
        'predict_time': predict_time
    }


# ------------------------------------------------------------------------------
# 主流程：使用卡方检验特征选择（增加整体时间测量）
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # 记录整体开始时间
    total_start_time = time.time()

    # 1. 配置参数
    FILE_PATH = r'PV_YS.csv'
    TARGET_COL = 'Active_Power'
    K_FEATURES = 10  # 选择前k个最佳特征
    N_BINS = 5  # 目标变量分箱数
    SEQ_LEN = 12
    EPOCHS = 50
    BATCH_SIZE = 32

    # 2. 执行完整流程
    print("=" * 60)
    print("开始光伏功率预测流程（卡方特征选择）：1.数据预处理→2.样本构建→3.LSTM训练→4.评估")
    print("=" * 60)

    # 步骤1：数据加载+卡方特征选择+归一化
    preprocessing_start = time.time()
    df_scaled, selected_features, scale_params, target_col = load_preprocess_select_chi2(
        file_path=FILE_PATH,
        target_col=TARGET_COL,
        k_features=K_FEATURES,
        n_bins=N_BINS
    )
    preprocessing_time = time.time() - preprocessing_start

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

    # 步骤4：预测与评估
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
    print("🏆 性能总结报告（卡方特征选择版本）")
    print("=" * 60)

    print(f"\n⏱️ 时间统计：")
    print(f"   - 数据预处理：{preprocessing_time:.2f}s")
    print(f"   - 样本构建：{sampling_time:.2f}s")
    print(f"   - 模型训练：{training_time:.2f}s ({training_time / 60:.2f}分钟)")
    print(f"   - 预测评估：{evaluation_time:.2f}s")
    print(f"   - 总耗时：{total_time:.2f}s ({total_time / 60:.2f}分钟)")

    print(f"\n🎯 预测性能：")
    print(f"   - MAE：{metrics['MAE']:.2f}")
    print(f"   - RMSE：{metrics['RMSE']:.2f}")
    print(f"   - MAPE：{metrics['MAPE']:.2f}%")
    print(f"   - R²：{metrics['R²']:.4f}")
    print(f"   - 预测速度：{metrics['predict_time'] / len(X_test):.4f}s/样本")

    print(f"\n📁 生成文件：")
    print(f"   1. K_LSTM_Training_Loss.png（模型训练损失曲线）")
    print(f"   2. K_LSTM_Prediction_Comparison.png（真实值vs预测值对比）")
    print(f"   3. K_Prediction_Error_Distribution.png（预测误差分布）")
    print("=" * 60)