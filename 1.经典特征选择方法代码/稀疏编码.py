import numpy as np
import pandas as pd
import matplotlib
import time  # 新增时间模块

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 新增稀疏编码相关库
from sklearn.decomposition import DictionaryLearning
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
import warnings

warnings.filterwarnings('ignore')

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ------------------------------------------------------------------------------
# 步骤1：数据加载+稀疏编码特征选择+归一化
# ------------------------------------------------------------------------------
def load_preprocess_select(file_path, target_col='Active_Power', n_components=10, alpha=1.0):
    """
    一体化预处理：读取数据→填充缺失值→稀疏编码特征选择→归一化
    """
    # 1. 读取数据（保留时间索引）
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)

    # 2. 填充缺失值（线性插值+前后向兜底）
    df_interpolated = df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

    # 3. 稀疏编码特征选择
    # 提取数值型特征（排除目标列）
    feature_cols = df_interpolated.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in feature_cols:
        feature_cols.remove(target_col)

    X = df_interpolated[feature_cols].values
    y = df_interpolated[target_col].values

    # 标准化特征（稀疏编码对尺度敏感）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 方法1：使用字典学习进行稀疏编码
    dict_learner = DictionaryLearning(
        n_components=n_components,
        alpha=alpha,  # 稀疏性参数
        max_iter=1000,
        random_state=42,
        fit_algorithm='lars'
    )

    # 学习稀疏表示
    sparse_code = dict_learner.fit_transform(X_scaled)

    # 计算特征重要性（基于字典原子的使用频率）
    feature_importance = np.abs(dict_learner.components_).sum(axis=0)

    # 方法2：结合Lasso进行特征选择（可选，增强稳定性）
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X_scaled, y)
    lasso_importance = np.abs(lasso.coef_)

    # 综合两种方法的重要性评分
    combined_importance = 0.7 * feature_importance + 0.3 * lasso_importance

    # 选择重要性排名前50%的特征
    importance_threshold = np.percentile(combined_importance, 50)
    selected_features = [
        feature_cols[i] for i in range(len(feature_cols))
        if combined_importance[i] > importance_threshold
    ]

    # 兜底：无符合条件特征时保留前n_components个最重要的特征
    if not selected_features:
        top_indices = np.argsort(combined_importance)[-n_components:]
        selected_features = [feature_cols[i] for i in top_indices]
        print(f"⚠️ 使用兜底策略，保留前{n_components}个最重要特征")

    # 打印筛选结果
    print(f"📊 稀疏编码特征选择结果：")
    print(f"初始特征数：{len(feature_cols)} → 筛选后：{len(selected_features)}")
    print(f"筛选后特征：{selected_features}")

    # 显示特征重要性排序
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': combined_importance
    }).sort_values('Importance', ascending=False)

    print(f"\n特征重要性排序：")
    print(feature_importance_df.head(10))

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

    # 5. 归一化（含兜底逻辑）
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
# 步骤2：构建LSTM输入样本（时序数据格式转换）
# ------------------------------------------------------------------------------
def create_lstm_samples(data, target_col, seq_len=12, test_size=0.2):
    """
    将时序数据转为LSTM输入格式：[样本数, 时间步长, 特征数]
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
# 步骤3：构建并训练LSTM模型（增加训练时间统计）
# ------------------------------------------------------------------------------
def build_train_lstm(X_train, y_train, seq_len, feature_num, epochs=50, batch_size=32):
    """
    构建LSTM预测模型（增加训练时间统计）
    """
    model = Sequential(name='LSTM_Load_Prediction')
    model.add(LSTM(units=128, return_sequences=True,
                   input_shape=(seq_len, feature_num), name='LSTM_1'))
    model.add(Dropout(0.3, name='Dropout_1'))

    model.add(LSTM(units=64, return_sequences=False, name='LSTM_2'))
    model.add(Dropout(0.3, name='Dropout_2'))

    model.add(Dense(units=1, name='Output'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # 记录训练开始时间
    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

    # 记录训练结束时间
    end_time = time.time()
    training_time = end_time - start_time

    # 将训练时间转换为分钟和秒
    minutes = int(training_time // 60)
    seconds = training_time % 60

    # 绘制训练损失曲线
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('LSTM模型训练损失曲线（稀疏编码特征选择）')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig('S_LSTM_Training_Loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ LSTM模型训练完成，损失曲线已保存")
    print(f"⏱️ 训练时间：{minutes}分{seconds:.2f}秒 ({training_time:.2f}秒)")

    return model, history, training_time


# ------------------------------------------------------------------------------
# 步骤4：预测与评估（改进版：仅保留MAE、RMSE、MAPE、R²）
# ------------------------------------------------------------------------------
def predict_evaluate(model, X_test, y_test, scale_params, target_col):
    """
    模型预测+逆归一化+计算评估指标
    仅保留MAE、RMSE、MAPE、R²四个核心指标
    """
    # 1. 预测与逆归一化
    y_pred = model.predict(X_test, verbose=1)
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
    plt.title('LSTM光伏功率预测：真实值vs预测值（稀疏编码特征选择）')
    plt.xlabel('时间步')
    plt.ylabel(f'{target_col}')
    plt.legend()
    plt.savefig('S_LSTM_Prediction_Comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6. 误差分布图
    plt.figure(figsize=(10, 4))
    errors = y_pred_real - y_test_real
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', label='零误差线')
    plt.title('LSTM预测误差分布（稀疏编码特征选择）')
    plt.xlabel('误差值')
    plt.ylabel('频数')
    plt.legend()
    plt.savefig('S_Prediction_Error_Distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2
    }


# ------------------------------------------------------------------------------
# 主流程（增加整体时间统计）
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # 配置参数
    FILE_PATH = r'PV_YS.csv'
    TARGET_COL = 'Active_Power'
    N_COMPONENTS = 10  # 字典学习组件数
    ALPHA = 1.0  # 稀疏性参数
    SEQ_LEN = 12
    EPOCHS = 50
    BATCH_SIZE = 32

    # 记录整体开始时间
    total_start_time = time.time()

    # 执行完整流程
    print("=" * 50)
    print("开始光伏功率预测流程（稀疏编码特征选择）：1.数据预处理→2.样本构建→3.LSTM训练→4.评估")
    print("=" * 50)

    # 步骤1：数据加载+稀疏编码特征选择+归一化
    preprocessing_start = time.time()
    df_scaled, selected_features, scale_params, target_col = load_preprocess_select(
        file_path=FILE_PATH,
        target_col=TARGET_COL,
        n_components=N_COMPONENTS,
        alpha=ALPHA
    )
    preprocessing_time = time.time() - preprocessing_start
    print(f"⏱️ 数据预处理时间：{preprocessing_time:.2f}秒")

    # 步骤2：构建LSTM样本
    sampling_start = time.time()
    X_train, X_test, y_train, y_test = create_lstm_samples(
        data=df_scaled,
        target_col=target_col,
        seq_len=SEQ_LEN
    )
    sampling_time = time.time() - sampling_start
    print(f"⏱️ 样本构建时间：{sampling_time:.2f}秒")

    # 步骤3：构建并训练LSTM
    training_start = time.time()
    lstm_model, train_history, training_time = build_train_lstm(
        X_train=X_train,
        y_train=y_train,
        seq_len=SEQ_LEN,
        feature_num=len(selected_features),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    actual_training_time = time.time() - training_start

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
    print(f"⏱️ 预测评估时间：{evaluation_time:.2f}秒")

    # 计算总时间
    total_time = time.time() - total_start_time
    total_minutes = int(total_time // 60)
    total_seconds = total_time % 60

    print("\n" + "=" * 50)
    print("基于稀疏编码特征选择的光伏功率预测流程全部完成！")
    print(f"⏱️ 各阶段耗时统计：")
    print(f"  数据预处理：{preprocessing_time:.2f}秒")
    print(f"  样本构建：{sampling_time:.2f}秒")
    print(f"  模型训练：{training_time:.2f}秒")
    print(f"  预测评估：{evaluation_time:.2f}秒")
    print(f"  总耗时：{total_minutes}分{total_seconds:.2f}秒")
    print(f"生成文件：")
    print(f"1. S_LSTM_Training_Loss.png（模型训练损失曲线）")
    print(f"2. S_LSTM_Prediction_Comparison.png（真实值vs预测值对比）")
    print(f"3. S_Prediction_Error_Distribution.png（预测误差分布）")
    print("=" * 50)