import numpy as np
import pandas as pd

import time  # 新增：用于计时
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings('ignore')

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ------------------------------------------------------------------------------
# 步骤1：数据加载+皮尔逊特征选择+归一化（含兜底预处理，去除数据清洗）
# ------------------------------------------------------------------------------
def load_preprocess_select(file_path, target_col='Active_Power', corr_threshold=0.3):
    """
    一体化预处理：读取数据→填充缺失值→皮尔逊特征选择→归一化（含兜底逻辑，无数据清洗）
    """
    # 1. 读取数据（保留时间索引）
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)  # 确保时间索引格式正确

    print(f"📊 原始数据量：{len(df)}行")

    # 2. 填充缺失值（线性插值+前后向兜底）
    df_interpolated = df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

    print(f"✅ 数据预处理完成")
    print(f"处理后数据量：{len(df_interpolated)}行")

    # 3. 皮尔逊特征选择（排除目标列自相关）
    # 初始特征：所有数值列（移除目标列）
    feature_cols = df_interpolated.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in feature_cols:
        feature_cols.remove(target_col)

    # 计算相关系数并筛选
    corr = df_interpolated[feature_cols + [target_col]].corr()[target_col]
    selected_features = [col for col in feature_cols if abs(corr[col]) > corr_threshold]

    # 兜底：无符合条件特征时保留全部
    if not selected_features:
        selected_features = feature_cols
        print(f"⚠️ 无特征超阈值{corr_threshold}，保留所有初始特征")

    # 打印筛选结果
    print(f"📊 皮尔逊筛选结果：")
    print(f"初始特征数：{len(feature_cols)} → 筛选后：{len(selected_features)}")
    print(f"筛选后特征：{selected_features}")
    print(f"各特征与{target_col}相关系数：\n{corr[selected_features].round(3)}")

    # 4. 数据分布统计
    print(f"\n📈 {target_col}分布统计：")
    print(f"最小值：{df_interpolated[target_col].min():.6f}")
    non_zero_vals = df_interpolated[df_interpolated[target_col] > 0][target_col]
    if len(non_zero_vals) > 0:
        print(f"非零最小值：{non_zero_vals.min():.6f}")
    else:
        print(f"非零最小值：无（所有值均为0）")
    print(f"均值：{df_interpolated[target_col].mean():.6f}")
    print(f"接近0的样本数（<0.1）：{sum(df_interpolated[target_col] < 0.1)}")

    # 5. 归一化（含兜底逻辑：避免分母为0或极小值）
    scale_params = {}
    df_scaled = df_interpolated[selected_features + [target_col]].copy()
    for col in df_scaled.columns:
        x_min = df_interpolated[col].min()
        x_max = df_interpolated[col].max()
        # 兜底预处理（避免分母为0）
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
    seq_len：用前12个时间步预测第13个（可调整，如24对应1天）
    """
    # 提取特征和目标数组
    X = data.drop(columns=[target_col]).values
    y = data[target_col].values
    n_samples = len(X) - seq_len

    # 构建样本
    X_seq, y_seq = [], []
    for i in range(n_samples):
        X_seq.append(X[i:i + seq_len, :])  # 前seq_len个时间步的特征
        y_seq.append(y[i + seq_len])  # 第seq_len+1个时间步的目标值

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # 划分训练集/测试集（按时间顺序，避免数据泄露）
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
    构建LSTM预测模型：2层LSTM+Dropout防过拟合+全连接输出
    """
    # 记录开始时间
    start_time = time.time()

    model = Sequential(name='LSTM_Load_Prediction')
    # 第一层LSTM（返回序列供下一层使用）
    model.add(LSTM(units=128, return_sequences=True,
                   input_shape=(seq_len, feature_num), name='LSTM_1'))
    model.add(Dropout(0.3, name='Dropout_1'))  # 防止过拟合

    # 第二层LSTM（不返回序列）
    model.add(LSTM(units=64, return_sequences=False, name='LSTM_2'))
    model.add(Dropout(0.3, name='Dropout_2'))

    # 输出层（预测单个负荷值）
    model.add(Dense(units=1, name='Output'))

    # 编译与训练
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # 训练模型
    print(f"\n⏳ 开始模型训练...")
    training_start = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,  # 训练集中10%作为验证集
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
    plt.savefig('P_LSTM_Training_Loss.png', dpi=300, bbox_inches='tight')
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
    模型预测+逆归一化（恢复真实负荷值）+计算评估指标
    仅保留MAE、RMSE、MAPE、R²四个核心指标
    """
    # 1. 预测（测量预测时间）
    print(f"\n⏳ 开始模型预测...")
    predict_start = time.time()

    y_pred = model.predict(X_test, verbose=1)

    predict_time = time.time() - predict_start
    print(f"预测完成，耗时：{predict_time:.2f}秒")
    print(f"平均每个样本预测时间：{predict_time / len(X_test):.4f}秒")

    # 2. 逆归一化（将归一化值恢复为真实负荷值）
    min_val = scale_params[target_col]['min']
    max_val = scale_params[target_col]['max']
    denominator = scale_params[target_col]['denominator']
    y_test_real = y_test * denominator + min_val
    y_pred_real = y_pred.flatten() * denominator + min_val  # 确保形状一致

    # 3. 计算四个核心评估指标
    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    r2 = r2_score(y_test_real, y_pred_real)

    # 4. 动态阈值计算MAPE（不显示阈值详情）
    non_zero_real = y_test_real[y_test_real > 0]

    if len(non_zero_real) > 0:
        quantile_5 = np.percentile(non_zero_real, 5)
        std_non_zero = non_zero_real.std()
        mean_real = y_test_real.mean()
        epsilon = max(quantile_5, std_non_zero * 0.1, mean_real * 0.02, 0.1)
    else:
        epsilon = 1.0

    # 应用阈值计算MAPE
    y_test_safe = np.where(y_test_real < epsilon, epsilon, y_test_real)
    mape = (np.mean(np.abs(y_test_real - y_pred_real) / y_test_safe) * 100)

    # 5. 打印简洁的评估结果（仅四个核心指标）
    print(f"\n📊 模型评估结果（{target_col}）：")
    print(f"MAE（平均绝对误差）：{mae:.2f}")
    print(f"RMSE（均方根误差）：{rmse:.2f}")
    print(f"MAPE（平均绝对百分比误差）：{mape:.2f}%")
    print(f"R²（决定系数）：{r2:.4f}")

    # 6. 绘制真实值vs预测值对比图
    plt.figure(figsize=(12, 4))
    plt.plot(y_test_real[:200], label='真实功率', linewidth=1.5)  # 只画前200个点，避免图过密
    plt.plot(y_pred_real[:200], label='预测功率', linewidth=1.5, linestyle='--')
    plt.title('LSTM光伏功率预测：真实值vs预测值（前200个时间步）')
    plt.xlabel('时间步')
    plt.ylabel(f'{target_col}')
    plt.legend()
    plt.savefig('P_LSTM_Prediction_Comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 7. 绘制误差分布图
    plt.figure(figsize=(10, 4))
    errors = y_pred_real - y_test_real
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', label='零误差线')
    plt.title('预测误差分布')
    plt.xlabel('误差值')
    plt.ylabel('频数')
    plt.legend()
    plt.savefig('P_Prediction_Error_Distribution.png', dpi=300, bbox_inches='tight')
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

    # 1. 配置参数（根据你的数据调整）
    FILE_PATH = r'PV_YS.csv'  # 你的数据文件路径
    TARGET_COL = 'Active_Power'  # 光伏功率列名（无需修改，与数据匹配）
    CORR_THRESHOLD = 0.3  # 皮尔逊筛选阈值
    SEQ_LEN = 12  # 时间步长（用前12个时间步预测下一个，如每小时1个数据，对应12小时）
    EPOCHS = 50  # 训练轮次（可调整为100，需注意过拟合）
    BATCH_SIZE = 32  # 批次大小（根据内存调整，如16、64）

    # 2. 执行完整流程
    print("=" * 60)
    print("开始光伏功率预测流程：1.数据预处理→2.样本构建→3.LSTM训练→4.评估")
    print("=" * 60)

    # 步骤1：数据加载+特征选择+归一化（无数据清洗）
    preprocessing_start = time.time()
    df_scaled, selected_features, scale_params, target_col = load_preprocess_select(
        file_path=FILE_PATH,
        target_col=TARGET_COL,
        corr_threshold=CORR_THRESHOLD
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
        feature_num=len(selected_features),  # 筛选后的特征数（动态适配）
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
    print("🏆 性能总结报告（皮尔逊特征选择版本，无数据清洗）")
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

    # 在性能总结报告部分，修改特征选择效果显示：
    print(f"\n📊 特征选择效果：")
    # 计算实际的初始特征数（不包括目标列）
    initial_features_count = len(df_scaled.columns) - 1  # 减去目标列
    selected_features_count = len(selected_features)
    reduction_ratio = (1 - selected_features_count / initial_features_count) * 100

    print(f"   - 初始特征数：{initial_features_count}")
    print(f"   - 筛选后特征数：{selected_features_count}")
    print(f"   - 特征减少比例：{reduction_ratio:.1f}%")
    print(f"   - 保留的特征：{selected_features}")

    print(f"\n📁 生成文件：")
    print(f"   1. P_LSTM_Training_Loss_NoCleaning.png（模型训练损失曲线）")
    print(f"   2. P_LSTM_Prediction_Comparison_NoCleaning.png（真实值vs预测值对比）")
    print(f"   3. P_Prediction_Error_Distribution_NoCleaning.png（预测误差分布）")
    print("=" * 60)