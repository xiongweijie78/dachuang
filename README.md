# 面向光伏功率预测的混合特征选择与深度学习建模

本项目是一个**光伏（PV）功率短期预测**的研究性代码仓库，围绕两条主线展开：

1. **特征选择**：在光伏电站实测数据上，系统对比 8 种经典/新兴特征选择方法（过滤法、包装法、嵌入法、特征提取），并统一用 LSTM 验证所选特征子集的预测效果；
2. **预测模型**：在电力负荷/功率数据集上，实现并对比一系列深度学习预测模型（Linear / CNN / LSTM / GRU / CNN-LSTM / Res-LSTM / Transformer / TCN / GCN / KAN / DNM）。

仓库按研究流程组织为三个顶层目录：**特征选择代码 → 预测模型代码 → 文献支撑**。

> ⚠️ 本仓库是论文实验代码合集，**不是工程化软件包**：没有统一的包结构、入口和依赖管理，每个脚本基本可独立运行，但存在代码重复、两套深度学习框架（TensorFlow 与 PyTorch）混用等情况，详见下文「已知问题与注意事项」。

---

## 一、目录结构总览

```
熊大/-
├── 1.经典特征选择方法代码/        # 主线①：特征选择方法对比实验（数据集 PV_YS.csv）
│   ├── PV_YS.csv                  #   光伏电站实测数据（约 4.4 MB，含目标列 Active_Power）
│   ├── 皮尔逊.py                  #   过滤法：Pearson 相关系数
│   ├── 卡方检验.py                #   过滤法：Chi-Square 检验
│   ├── 互信息.py                  #   过滤法：MIC 最大信息系数
│   ├── 特征递归消除.py            #   包装法：RFE（递归特征消除）
│   ├── L1正则化.py                #   嵌入法：Lasso / L1 正则
│   ├── 稀疏编码.py                #   特征提取：稀疏编码（字典学习）
│   ├── 自动编码器特征提取.py      #   特征提取：AutoEncoder
│   └── CNN特征提取.py             #   特征提取：CNN 特征提取
│
├── 2.预测模型代码/                # 主线②：预测模型对比实验（数据集 power.csv）
│   ├── power.csv                  #   电力数据（17567 行，6 列，见「数据集说明」）
│   ├── power_dataset.py           #   ★ 公共数据模块：PyTorch Dataset（滑动窗口 + 归一化）
│   ├── typical_models.py          #   ★ 公共模型库：Linear/CNN/LSTM/GRU/CNN-LSTM/Res-LSTM/Transformer/KAN
│   ├── LSTM_predict.py            #   ★ 训练入口示例：LSTM 训练 + 10 次重复实验 + 指标统计
│   ├── Transformer_model/         #   Transformer 系列（5 个变体迭代 + 汇总脚本）
│   │   ├── my_Transformer.py      #     标准 Transformer 编码器实现
│   │   ├── Transformer_1.py       #     变体 1
│   │   ├── Transformer_2.py       #     变体 2（精简）
│   │   ├── Transformer_3.py       #     变体 3（含完整训练/评估流程，约 300 行）
│   │   ├── Transformer_4.py       #     变体 4（含完整训练/评估流程）
│   │   └── Transformer_5.py       #     变体 5
│   ├── TCN_model/
│   │   └── TCN_fc.py              #   时序卷积网络 TCN（改自 locuslab/TCN，含 Chomp/膨胀因果卷积）
│   ├── GCN_model/
│   │   ├── GCN_1.py               #   GCN + GRU 时空预测（基于 torch_geometric）
│   │   └── GCN_2.py               #   GCN 变体
│   └── DNM_models/
│       ├── DNM_models.py          #   树突神经元模型 DNM（Synapse/Dendritic/Membrane/Soma 四层仿生结构）
│       └── DNM_Dense.py           #   DNM 的全连接封装
│
└── 3.特征选择方法的文献支撑/       # 参考文献 PDF（按方法类别归档，编号对应论文引用 [15]~[24]）
    ├── 基于特征选择和组合模型的短期电力负荷预测_徐宇颂.pdf
    ├── 基于特征选择和GRU-Attention模型组合的短期总线负荷预测.pdf
    ├── 滤波器、包装器和嵌入式特征选择的比较分析...（CCPP）...pdf
    ├── 过滤法/
    │   ├── 皮尔逊相关系数（pcc）[15].pdf
    │   ├── 互信息[16].pdf
    │   └── 卡方检验[17].pdf
    ├── 包装法/
    │   ├── 递归特征消除（RFE)[18].pdf
    │   └── MOOA[24].pdf
    ├── 嵌入法/
    │   ├── L1正则化（Lasso）[19].pdf
    │   └── 基于深度学习的特征选择[20].pdf
    └── 新兴混合方法/
        ├── 集成特征选择[21].pdf
        └── 稀疏编码[23].pdf
```

---

## 二、主线①：特征选择实验（目录 `1.经典特征选择方法代码/`）

### 数据集 `PV_YS.csv`

光伏电站实测多变量时间序列（时间戳为索引），共 12 列：

| 列名 | 含义 |
|---|---|
| `Active_Energy_Delivered_Received` | 有功电能（送出/接收） |
| `Current_Phase_Average_Mean` | 平均相电流 |
| `Wind_Speed` | 风速 |
| `Weather_Temperature_Celsius` | 环境温度 |
| `Air_Pressure` | 气压 |
| `Global_Horizontal_Radiation` | 水平面总辐照度 |
| `Pyranometer_1` | 辐射表读数 |
| `Wind_Direction` | 风向 |
| `Weather_Daily_Rainfall` | 日降雨量 |
| `Temperature_Probe_1 / 2` | 组件温度探头 1 / 2 |
| **`Active_Power`** | **预测目标：有功功率** |

### 各脚本的统一实验流水线

8 个特征选择脚本结构基本一致，均为**端到端独立可运行**，流程为：

```
读取 PV_YS.csv（时间索引）
  → 缺失值处理（线性插值 + 前向/后向填充，刻意不做异常值清洗以保留原始分布）
  → 【特征选择】（各脚本的核心差异所在）
  → Min-Max 归一化（保存缩放参数以便反归一化）
  → 构造 LSTM 监督样本（滑动窗口，默认 seq_len=12，时间顺序 8:2 划分训练/测试）
  → 训练 2 层 LSTM（128 → 64 单元，Dropout 0.3，Adam lr=1e-3，MSE 损失，约 50 epochs）
  → 反归一化后评估：RMSE / MAE / MAPE / R²
  → 输出图表：特征分数分布图、训练损失曲线、预测对比图（PNG，300 dpi）
  → 全程计时（特征选择耗时 / 训练耗时 / 预测耗时）
```

### 各脚本要点

| 脚本 | 方法类别 | 核心实现 | 筛选策略 |
|---|---|---|---|
| `皮尔逊.py` | 过滤法 | `scipy.pearsonr` 计算各特征与目标的相关系数 | 按相关系数绝对值阈值/分位数筛选 |
| `卡方检验.py` | 过滤法 | `sklearn.SelectKBest` + `chi2`（连续特征先分箱离散化，默认 `k_features=10, n_bins=5`） | 选卡方统计量最高的 K 个 |
| `互信息.py` | 过滤法 | `minepy.MINE` 计算 **MIC 最大信息系数**（`alpha=0.6, c=15`） | MIC 分数 > 0.35 分位数（保留前 65%） |
| `特征递归消除.py` | 包装法 | `sklearn.RFE`，以基学习器递归剔除弱特征 | 按排名选优 |
| `L1正则化.py` | 嵌入法 | Lasso 回归的 L1 稀疏约束，系数为 0 的特征被剔除 | 保留非零系数特征 |
| `稀疏编码.py` | 特征提取 | 字典学习/稀疏编码将原始特征映射为稀疏表示 | 以稀疏编码重构特征替代原始特征 |
| `自动编码器特征提取.py` | 特征提取 | AutoEncoder 编码层提取低维特征 | 用隐层表示代替原始特征 |
| `CNN特征提取.py` | 特征提取 | 用 CNN 从时序窗口中提取深度特征 | 用卷积特征代替原始特征 |

> 实验设计意图：在**完全相同的数据、验证模型（LSTM）和评估指标**下，横向对比 8 种方法选出的特征子集对预测精度的影响（以及各自的时间开销）。

---

## 三、主线②：预测模型实验（目录 `2.预测模型代码/`）

### 数据集 `power.csv`

17,567 行 × 6 列的小时级电力数据（2015 年起，含表头）：

```
datetime, Energy, Temperature, Humidity, Precipitation, Wind Speed, Holiday
```

- **预测目标**：`Energy`（用电量/功率）
- 特征：温度、湿度、降水、风速、是否节假日，共 6 列全部参与归一化。

### 公共模块

#### `power_dataset.py` — 数据加载
- `Power_ds(n_hours, n_features, data_type, unsqueeze)`：
  - `series_to_supervised()`：将时序重构为监督学习格式，用**前 `n_hours` 个时刻 × 6 个特征**预测**下一时刻的 Energy**（默认 `n_hours=3, n_features=6`）；
  - 全量数据 `MinMaxScaler(0,1)` 归一化；
  - 按时间顺序 **60% 训练 / 40% 测试**；
  - `unsqueeze=True` 时输出 `[batch, 1, n_hours, n_features]` 供 2D 卷积类模型使用，否则输出 `[batch, n_hours, n_features]`。

#### `typical_models.py` — 基线模型库
| 类 | 结构要点 |
|---|---|
| `Linear_Model` | 3 层全连接（128/64/out），Dropout 0.3 |
| `CNN_model` | 2 层 2D 卷积 + MaxPool + 2 层 FC |
| `LSTM_model` | 单层/多层 LSTM（取最后时间步）→ FC，Dropout 0.4 |
| `GRU_model` | GRU（取最后时间步）→ FC，Dropout 0.5 |
| `CNN_LSTM_model` | 沿特征维卷积 → LSTM → FC |
| `Res_LSTM_model` | 残差卷积连接 + LSTM |
| `Transformer` | `nn.TransformerEncoder` + 因果 mask + Linear 解码 |
| `KANLinear` | **KAN（Kolmogorov-Arnold Network）** 线性层：B 样条基函数 + 可学习系数，含网格自适应更新与正则化 |

### 训练入口 `LSTM_predict.py`
- 用 `Power_ds` + `LSTM_model(feature_num=6, layer_num=1, nodes_num=128)` 训练；
- Adam `lr=1e-3`，`L1Loss`，100 epochs，batch 32，**需要 CUDA**（`device = torch.device('cuda')` 硬编码）；
- **重复实验 10 次**，分别统计 Runtime / MAE / MAPE / RMSE / R² 的均值，并保留 R² 最优一次的预测序列；
- 文件尾部有被注释掉的"将结果写入对比 CSV"代码（残留的个人路径，见「已知问题」）。

### 进阶模型子目录

| 子目录 | 内容 |
|---|---|
| `Transformer_model/` | `my_Transformer.py` 为较规范的编码器实现；`Transformer_1~5.py` 为迭代变体（其中 3、4 是含数据加载、训练、评估、绘图的完整脚本，约 300 行；1、2、5 偏结构实验） |
| `TCN_model/TCN_fc.py` | 标准 TCN：`TemporalBlock`（权重归一化 + 膨胀因果卷积 + Chomp1d 裁剪 + 残差连接），改自 [locuslab/TCN](https://github.com/locuslab/TCN) |
| `GCN_model/` | `GCN_1.py`：两层 `GCNConv`（torch_geometric）逐时间步提空间特征 → GRU 提时序特征 → FC 输出；`GCN_2.py` 为变体 |
| `DNM_models/` | **树突神经元模型（Dendritic Neuron Model）**：`Synapse`（突触层，sigmoid 加权）→ `Dendritic`（树突层，乘法聚合）→ `Membrane`（膜层，加法聚合）→ `Soma`（胞体层，sigmoid 输出）的仿生四层结构；`DNM_Linear1/2` 等封装可直接当回归层用 |

> 注意 `DNM_models.py` 顶部有 `from banama_load.typical_models import KANLinear`——`banama_load` 是作者本机的旧项目目录名，**在本仓库中该导入会失败**，需改为 `from typical_models import KANLinear`（或注释掉，当前文件内未实际使用）。

---

## 四、文献支撑（目录 `3.特征选择方法的文献支撑/`）

纯参考文献 PDF，与目录 1 的代码一一对应，文件名编号即论文中的引用序号：

| 类别 | 文献 | 对应代码 |
|---|---|---|
| 过滤法 | [15] 皮尔逊相关系数、[16] 互信息、[17] 卡方检验 | 皮尔逊.py / 互信息.py / 卡方检验.py |
| 包装法 | [18] 递归特征消除 RFE、[24] 多目标优化 MOOA | 特征递归消除.py |
| 嵌入法 | [19] L1 正则化（Lasso）、[20] 基于深度学习的特征选择 | L1正则化.py |
| 新兴混合方法 | [21] 集成特征选择、[23] 稀疏编码 | 稀疏编码.py 等 |
| 综述/应用 | 徐宇颂《基于特征选择和组合模型的短期电力负荷预测》、GRU-Attention 总线负荷预测、CCPP 滤波/包装/嵌入对比 | — |

---

## 五、环境与运行

### 依赖（无 requirements.txt，按 import 统计）

```bash
# 主线①（特征选择，TensorFlow 系）
pip install numpy pandas matplotlib scikit-learn tensorflow minepy tqdm

# 主线②（预测模型，PyTorch 系）
pip install torch scikit-learn pandas matplotlib
# GCN 模型额外需要：
pip install torch-geometric
```

- Python 3.8+；`LSTM_predict.py` 默认要求 NVIDIA GPU（代码写死 `cuda`）。
- 两套框架（TF/Keras 与 PyTorch）分别用于两个主线，互不影响。

### 运行方式

每个脚本独立运行，**建议先 `cd` 到脚本所在目录**（图表和相对路径数据文件按当前目录解析）：

```bash
# 特征选择实验（任选一个方法）
cd "1.经典特征选择方法代码"
python 互信息.py        # 输出 MIC 分数表 + MIC_Score_Distribution.png + 训练曲线 + 评估指标

# 预测模型实验
cd "2.预测模型代码"
python LSTM_predict.py  # 10 次重复实验，输出平均 RMSE/MAE/MAPE/R²/耗时
```

各模型子目录中的脚本多为**模型结构定义 + `main()` 形状自检**（如 `GCN_1.py`、`TCN_fc.py`、`typical_models.py`），直接运行仅打印输出张量形状，训练需参照 `LSTM_predict.py` 的模式自行接线。

---

## 六、已知问题与注意事项

1. **无统一依赖/入口**：无 `requirements.txt`、`main.py`，脚本各自为政；特征选择 8 个脚本之间大量复制粘贴（预处理、LSTM 验证、评估部分几乎相同），修改公共逻辑需逐文件同步。
2. **框架混用**：目录 1 用 TensorFlow/Keras，目录 2 用 PyTorch，安装环境时两者都需要。
3. **残留个人路径**：`LSTM_predict.py` 末尾注释中有 `C:\Users\ASUS\Desktop\小论文\...` 的写 CSV 代码；`DNM_models.py` 导入了不存在的本地模块 `banama_load`（见上文修复方法）。
4. **GPU 硬编码**：`LSTM_predict.py` 中 `torch.device('cuda')`，无 GPU 环境需改为 `cuda if available else cpu`。
5. **数据划分按时间顺序**（非随机 shuffle 划分），符合时序预测规范；`power.csv` 划了 60/40，特征选择脚本是 80/20，两处不一致是实验设计差异，非 bug。
6. 图表中文依赖 `SimHei` 字体（Windows 自带；Linux/Mac 需自行安装或改字体配置）。
7. 数据文件（`PV_YS.csv`、`power.csv`）必须与对应脚本在同一目录。
