import numpy as np
import pandas as pd
import matplotlib
import time  # 新增：用于计时

matplotlib.use('Agg')#代码生成图片时直接生成本地文件
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score#即mse，mae，r2
from sklearn.linear_model import LassoCV, Lasso#自动找到最合适的alpha，自动去除冗余的特征 做特征选择，去除冗余的特征
from sklearn.preprocessing import StandardScaler#将数据变为均值0，标准差1
from tensorflow.keras.models import Sequential#深度学习框架，高级接口，models模型。
from tensorflow.keras.layers import LSTM, Dense, Dropout#搭建模型的积木。长短期记忆。全连接层。随即关掉一部分神经元。
from tensorflow.keras.optimizers import Adam#优化器工具包，自动更新神经网络权重，减小mae和mse损失。自适应学习率优化器。
import warnings

warnings.filterwarnings('ignore')#忽略不必要的警告

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']#字体变为黑体
plt.rcParams['axes.unicode_minus'] = False#关闭unicode格式负号，使负号正常输出


# ------------------------------------------------------------------------------
# 步骤1：数据加载+L1正则化特征选择+归一化
# ------------------------------------------------------------------------------
def load_preprocess_select(file_path, target_col='Active_Power', alpha=0.01):
    #file_path文件路径。target_col名字。alpha:正则强度，alpha 越大，越多特征系数被压缩到 0。lasso方法参数
    """
    一体化预处理：读取数据→填充缺失值→L1正则化特征选择→归一化
    """
    # 记录L1正则化开始时间
    l1_start_time = time.time()

    # 1. 读取数据（保留时间索引）
    df = pd.read_csv(file_path, index_col=0)#读取数据，第一列为索引
    df.index = pd.to_datetime(df.index)#索引改为时间索引

    # 2. 填充缺失值（线性插值+前后向兜底）
    df_interpolated = df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')#先线性插值，再后边数据再前边数据

    # 3. L1正则化特征选择
    feature_cols = df_interpolated.select_dtypes(include=[np.number]).columns.tolist()#取出数据为数字的列的名字构成python列表
    if target_col in feature_cols:
        feature_cols.remove(target_col)

    # 准备特征和目标数据
    X = df_interpolated[feature_cols].values
    y = df_interpolated[target_col].values

    # 标准化数据（L1正则化对特征尺度敏感）
    scaler_X = StandardScaler()#创造一个标准转换器对象
    X_scaled = scaler_X.fit_transform(X)#计算每一列的均值和标准差，使用计算出的均值和标准差，进行Z_score标准化

    # 使用LassoCV自动选择最佳alpha（正则化强度）
    lasso_cv = LassoCV(alphas=np.logspace(-4, 0, 50), cv=5, random_state=42, max_iter=1000)#生成 50 个候选 alpha 值：从 0.0001 到 1，对数均匀分布。把训练数据分成 5 份，轮流 4 份训练、1 份验证.随机种子。最大迭代次数1000
    lasso_cv.fit(X_scaled, y)#训练模型，上面一行只给了参数，本行训练并保存参数

    # 使用最佳alpha训练Lasso模型
    lasso = Lasso(alpha=lasso_cv.alpha_, max_iter=1000, random_state=42)
    lasso.fit(X_scaled, y)

    # 选择非零系数的特征
    selected_features = [feature_cols[i] for i in range(len(feature_cols)) if abs(lasso.coef_[i]) > 1e-6]#挑出来系数不为0的特征   lasso.coef模型中每个特征的权重

    # 兜底：无符合条件特征时保留全部
    if not selected_features:
        selected_features = feature_cols
        print(f"⚠️ L1正则化未选择任何特征，保留所有初始特征")

    l1_time = time.time() - l1_start_time
    print(f"✅ L1正则化特征选择完成，耗时：{l1_time:.2f}秒")

    # 打印筛选结果
    print(f"📊 L1正则化特征选择结果：")
    print(f"最佳alpha值：{lasso_cv.alpha_:.6f}")
    print(f"初始特征数：{len(feature_cols)} → 筛选后：{len(selected_features)}")
    print(f"筛选后特征：{selected_features}")

    # 打印特征重要性（系数绝对值）
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': lasso.coef_,
        'abs_coefficient': abs(lasso.coef_)
    }).sort_values('abs_coefficient', ascending=False)#三种数据变为dataframe格式后按照系数绝对值的大小排列

    print(f"\n特征重要性排序：")
    print(feature_importance.round(4))#数据保留四位小数

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

    # 5. 归一化（使用MinMax归一化）
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

    return df_scaled, selected_features, scale_params, target_col#返回归一化后的数据集，正则化挑选后的特征，归一化参数，功率名


# ------------------------------------------------------------------------------
# 步骤2：构建LSTM输入样本（时序数据格式转换）
# ------------------------------------------------------------------------------
def create_lstm_samples(data, target_col, seq_len=12, test_size=0.2):
    #数据集，功率，时间步长12即12个数据为一个时序样本，测试集比例为0.2
    """
    将时序数据转为LSTM输入格式：[样本数, 时间步长, 特征数]
    """
    X = data.drop(columns=[target_col]).values
    y = data[target_col].values
    n_samples = len(X) - seq_len#可以生成的时序样本的数量

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
    #两个训练集，时间步长，特征数量，训练次数为50次,一次喂给模型32个样本
    """
    构建LSTM预测模型（增加时间测量）
    """
    # 记录开始时间
    start_time = time.time()

    model = Sequential(name='LSTM_Load_Prediction')
    model.add(LSTM(units=128, return_sequences=True,#设置128个神经元。输出全部12个时间步的结果，保留时间维度，供下一层LSTM接收。
                   input_shape=(seq_len, feature_num), name='LSTM_1'))#定义输入数据结构（），名字为
    model.add(Dropout(0.3, name='Dropout_1'))#去掉0.3的数据防止过拟合
    model.add(LSTM(units=64, return_sequences=False, name='LSTM_2'))#只输出最后一个时间步的结果
    model.add(Dropout(0.3, name='Dropout_2'))
    model.add(Dense(units=1, name='Output'))#输出一个功率预测值

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')#编译模型，，优化器Adam，损失函数均方误差,学习率默认0.001

    # 训练模型
    print(f"\n⏳ 开始模型训练...")
    training_start = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,#拿训练集0.1作为验证集
        verbose=1
    )#history中存history对象，history.history中存了loss和val_loss
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
    plt.savefig('L1_LSTM_Training_Loss.png', dpi=300, bbox_inches='tight')#保存图片到本地文件，分辨率为300，裁掉周围空白
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
    #模型，两个测试集,归一化参数，功率
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
    y_pred_real = y_pred.flatten() * denominator + min_val#模型输出为（样本数，1）

    # 2. 计算四个核心评估指标
    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))#求平方根
    r2 = r2_score(y_test_real, y_pred_real)

    # 3. 改进的动态阈值计算MAPE
    non_zero_real = y_test_real[y_test_real > 0]

    if len(non_zero_real) > 0:
        # 使用更稳健的阈值计算
        quantile_10 = np.percentile(non_zero_real, 10)#求10分位数
        std_non_zero = non_zero_real.std()#求标准差
        mean_real = y_test_real.mean()#求平均值

        # 提高阈值设置，避免过小阈值
        epsilon = max(quantile_10, std_non_zero * 0.15, mean_real * 0.05, 0.5)
    else:
        epsilon = 1.0

    # 应用阈值计算MAPE
    y_test_safe = np.where(y_test_real < epsilon, epsilon, y_test_real)#条件判断，满足条件选择前面，不满足选择后面
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
    plt.title('LSTM光伏功率预测（L1正则化特征选择）：真实值vs预测值')
    plt.xlabel('时间步')
    plt.ylabel(f'{target_col}')
    plt.legend()
    plt.savefig('L1_LSTM_Prediction_Comparison.png', dpi=300, bbox_inches='tight')#清晰度300，去除四周空白部分
    plt.close()

    # 6. 误差分布图
    plt.figure(figsize=(10, 4))
    errors = y_pred_real - y_test_real
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')#误差直方图，50个柱子，透明度为0.7，边框为黑色
    plt.axvline(x=0, color='red', linestyle='--', label='零误差线')#0误差位置放置一条垂直线
    plt.title('LSTM预测误差分布（L1正则化特征选择）')
    plt.xlabel('误差值')
    plt.ylabel('频数')
    plt.legend()
    plt.savefig('L1_Prediction_Error_Distribution.png', dpi=300, bbox_inches='tight')
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
    ALPHA = 0.01  # L1正则化强度参数（LassoCV会自动选择最佳值）
    SEQ_LEN = 12
    EPOCHS = 50
    BATCH_SIZE = 32

    print("=" * 60)
    print("开始光伏功率预测流程（L1正则化特征选择）：1.数据预处理→2.样本构建→3.LSTM训练→4.评估")
    print("=" * 60)

    # 步骤1：数据加载+L1正则化特征选择+归一化
    preprocessing_start = time.time()
    df_scaled, selected_features, scale_params, target_col = load_preprocess_select(
        file_path=FILE_PATH,
        target_col=TARGET_COL,
        alpha=ALPHA
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
    print("🏆 性能总结报告（L1正则化特征选择版本）")
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
    print(f"   1. L1_LSTM_Training_Loss.png（模型训练损失曲线）")
    print(f"   2. L1_LSTM_Prediction_Comparison.png（真实值vs预测值对比）")
    print(f"   3. L1_Prediction_Error_Distribution.png（预测误差分布）")
    print("=" * 60)