import numpy as np
import pandas as pd
import matplotlib
import time  # 用于计时
from tqdm import tqdm

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
from minepy import MINE

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_preprocess_select(file_path, target_col='Active_Power'):
    """
    一体化预处理：读取数据→填充缺失值→MIC特征选择→归一化
    修改：去除数据清洗步骤
    """
    # 记录开始时间
    mic_start_time = time.time()

    # 1. 读取数据（保留时间索引）
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)

    print(f"📊 原始数据量：{len(df)}行")

    # 2. 直接填充缺失值（跳过数据清洗步骤）
    df_interpolated = df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

    print(f"✅ 数据预处理完成")
    print(f"处理后数据量：{len(df_interpolated)}行")

    # 3. MIC特征选择
    numeric_features = df_interpolated.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_features:
        numeric_features.remove(target_col)

    mic_scores = []
    mine = MINE(alpha=0.6, c=15, est="mic_approx")
    target_data = df_interpolated[target_col].values

    print(f"\n🔍 开始计算{len(numeric_features)}个特征的MIC分数...")
    for col in tqdm(numeric_features, desc="MIC计算进度", unit="特征"):
        feature_data = df_interpolated[col].values
        mine.compute_score(feature_data, target_data)
        mic_scores.append(mine.mic())

    mic_time = time.time() - mic_start_time
    print(f"✅ MIC分数计算完成，耗时：{mic_time:.2f}秒")

    mic_df = pd.DataFrame({
        '特征': numeric_features,
        'MIC分数': mic_scores
    }).sort_values('MIC分数', ascending=False)

    print("\n特征与目标负荷的MIC分数（从高到低）：")
    print(mic_df.to_string(index=False))

    # 计算当前筛选阈值（分位数0.35）
    threshold = mic_df['MIC分数'].quantile(0.35)
    strong_mic_features = mic_df[mic_df['MIC分数'] > threshold]['特征'].tolist()

    print(f"\n筛选出的强相关特征（MIC分数前65%，共{len(strong_mic_features)}个）:")
    print(strong_mic_features)

    if not strong_mic_features:
        strong_mic_features = numeric_features
        print(f"⚠️ 无特征超MIC分位数阈值，保留所有初始特征")

    # ---------------------- 绘制MIC分数分布图 ----------------------
    plt.figure(figsize=(10, 6))
    # 绘制直方图（展示所有特征的MIC分数分布）
    n, bins, patches = plt.hist(
        mic_df['MIC分数'],
        bins=10,  # 分10个区间展示
        alpha=0.7,
        edgecolor='black',
        color='#4CAF50'
    )
    # 在图中标注当前筛选阈值
    plt.axvline(
        x=threshold,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'当前筛选阈值 (分位数0.35): {threshold:.4f}'
    )
    # 添加特征名称标签（仅标注分数最高的5个特征，避免拥挤）
    top_n = 5  # 标注前5个高分特征
    for i in range(min(top_n, len(mic_df))):
        plt.text(
            x=mic_df['MIC分数'].iloc[i],
            y=n.max() * 0.9 - i * (n.max() * 0.1),  # 纵向错开排列
            s=f"{mic_df['特征'].iloc[i]}: {mic_df['MIC分数'].iloc[i]:.4f}",
            color='darkblue',
            fontsize=9
        )
    # 图表配置
    plt.title('特征与目标变量的MIC分数分布（无数据清洗）', fontsize=12)
    plt.xlabel('MIC分数（值越高，相关性越强）', fontsize=10)
    plt.ylabel('特征数量', fontsize=10)
    plt.xlim(0, 1.0)  # MIC分数范围是[0,1]
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('MIC_Score_Distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n📊 MIC分数分布图已保存为：MIC_Score_Distribution.png")
    # -------------------------------------------------------------------

    print(f"\n📈 {target_col}分布统计：")
    print(f"最小值：{df_interpolated[target_col].min():.6f}")
    non_zero_vals = df_interpolated[df_interpolated[target_col] > 0][target_col]
    if len(non_zero_vals) > 0:
        print(f"非零最小值：{non_zero_vals.min():.6f}")
    else:
        print(f"非零最小值：无（所有值均为0）")
    print(f"均值：{df_interpolated[target_col].mean():.6f}")
    print(f"接近0的样本数（<0.1）：{sum(df_interpolated[target_col] < 0.1)}")

    # 4. 归一化
    scale_params = {}
    df_scaled = df_interpolated[strong_mic_features + [target_col]].copy()
    for col in df_scaled.columns:
        x_min = df_interpolated[col].min()
        x_max = df_interpolated[col].max()
        if abs(x_max - x_min) < 1e-8:
            denominator = 1.0
        else:
            denominator = x_max - x_min
        df_scaled[col] = (df_interpolated[col] - x_min) / denominator
        scale_params[col] = {'min': x_min, 'max': x_max, 'denominator': denominator}

    return df_scaled, strong_mic_features, scale_params, target_col


def create_lstm_samples(data, target_col, seq_len=12, test_size=0.2):
    """将时序数据转为LSTM输入格式"""
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


def build_train_lstm(X_train, y_train, seq_len, feature_num, epochs=50, batch_size=32):
    """构建2层LSTM模型（增加时间测量）"""
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
    plt.title(f'LSTM模型训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig('MIC_LSTM_Training_Loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ LSTM模型训练完成")
    print(f"训练时间：{training_time:.2f}秒 ({training_time / 60:.2f}分钟)")
    print(f"总耗时：{total_time:.2f}秒")
    print(f"平均每轮训练时间：{training_time / epochs:.2f}秒")

    return model, history, {
        'training_time': training_time,
        'total_time': total_time
    }


def predict_evaluate(model, X_test, y_test, scale_params, target_col):
    """模型预测+评估（增加预测时间测量）"""
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

    # 3. 动态阈值计算MAPE
    non_zero_real = y_test_real[y_test_real > 0]

    if len(non_zero_real) > 0:
        quantile_10 = np.percentile(non_zero_real, 10)
        std_non_zero = non_zero_real.std()
        mean_real = y_test_real.mean()

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
    plt.title('MIC_LSTM光伏功率预测：真实值vs预测值（前200个时间步）')
    plt.xlabel('时间步')
    plt.ylabel(f'{target_col}')
    plt.legend()
    plt.savefig('MIC_LSTM_Prediction_Comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 误差分布图
    plt.figure(figsize=(10, 4))
    errors = y_pred_real - y_test_real
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', label='零误差线')
    plt.title('MIC_LSTM预测误差分布')
    plt.xlabel('误差值')
    plt.ylabel('频数')
    plt.legend()
    plt.savefig('MIC_Prediction_Error_Distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2,
        'predict_time': predict_time
    }


if __name__ == "__main__":
    # 记录整体开始时间
    total_start_time = time.time()

    FILE_PATH = r'PV_YS.csv'
    TARGET_COL = 'Active_Power'
    SEQ_LEN = 12
    EPOCHS = 50
    BATCH_SIZE = 32

    print("=" * 60)
    print("开始光伏功率预测流程")
    print("1.数据预处理→2.MIC特征选择→3.样本构建→4.LSTM训练→5.评估")
    print("=" * 60)

    # 步骤1：数据预处理（不包含数据清洗，仅包含MIC计算）
    preprocessing_start = time.time()
    df_scaled, selected_features, scale_params, target_col = load_preprocess_select(
        file_path=FILE_PATH,
        target_col=TARGET_COL
    )
    preprocessing_time = time.time() - preprocessing_start

    # ========== 新增：保存筛选后的特征 ==========
    print(f"\n💾 保存MIC筛选后的特征数据...")

    # 创建包含筛选特征和目标变量的DataFrame
    features_to_save = df_scaled[selected_features + [target_col]].copy()

    # 添加时间索引（如果原始数据有时间索引）
    try:
        # 重新读取原始数据获取时间索引
        df_original = pd.read_csv(FILE_PATH, index_col=0)
        df_original.index = pd.to_datetime(df_original.index)
        # 确保长度匹配
        if len(features_to_save) == len(df_original):
            features_to_save.index = df_original.index
            print("✅ 已添加时间索引")
    except:
        print("⚠️ 无法添加时间索引，将保存为无索引数据")

    # 保存特征文件
    feature_file_path = 'MIC_Selected_Features.csv'
    features_to_save.to_csv(feature_file_path, index=True if features_to_save.index is not None else False)

    print(f"✅ 筛选特征已保存至: {feature_file_path}")
    print(f"📊 文件维度: {features_to_save.shape}")
    print(f"🔢 包含 {len(selected_features)} 个特征 + 目标变量 '{target_col}'")
    print(f"📋 特征列表: {selected_features}")
    # ==========================================

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
    print("🏆 性能总结报告")
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
    print(f"   1. MIC_Score_Distribution_NoCleaning.png（MIC分数分布图）")
    print(f"   2. MIC_LSTM_Training_Loss_NoCleaning.png（模型训练损失曲线）")
    print(f"   3. MIC_LSTM_Prediction_Comparison_NoCleaning.png（真实值vs预测值对比）")
    print(f"   4. MIC_Prediction_Error_Distribution_NoCleaning.png（预测误差分布）")
    print("=" * 60)