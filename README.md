# 雷达心率估计系统

一个工程化的雷达心率估计系统，支持多种心率估计算法和人体定位方法。

## 📁 项目结构

```
radar_heart_rate_project/
│
├── algorithms/                    # 算法模块
│   ├── heart_rate/               # 心率估计算法
│   │   ├── __init__.py
│   │   ├── base_estimator.py    # 基类
│   │   ├── fft_estimator.py     # FFT方法
│   │   ├── stft_estimator.py    # STFT方法
│   │   ├── wavelet_estimator.py # 小波方法
│   │   ├── dct_estimator.py     # DCT方法
│   │   ├── vmd_estimator.py     # VMD方法
│   │   ├── emd_estimator.py     # EMD方法
│   │   └── eemd_estimator.py    # EEMD方法
│   │
│   └── localization/             # 人体定位算法
│       ├── __init__.py
│       ├── base_localization.py # 基类
│       ├── manual_localization.py     # 手动定位
│       ├── rf_localization.py         # 随机森林定位
│       └── cfar_localization.py       # CFAR定位
│
├── utils/                        # 工具模块
│   ├── data_utils.py            # 数据处理工具
│   └── evaluation.py            # 评估工具
│
├── config.py                     # 配置文件
├── main.py                       # 主程序
├── batch_experiments.py          # 批量实验脚本（新增）
├── README.md                     # 说明文档
└── results/                      # 输出目录（运行时自动创建）
```

## 🚀 快速开始

### 1. 环境要求

```bash
pip install numpy pandas matplotlib seaborn scipy h5py joblib
```

可选依赖（用于特定算法）：
```bash
pip install pywt              # 小波变换
pip install vmdpy             # VMD方法
pip install EMD-signal        # EMD/EEMD方法
```

### 2. 配置文件

编辑 `config.py`：

```python
# 实验文件配置
FILE_CONFIGS = [
    {'path': 'path/to/file.mat', 'distance': 40, 'rb_index': 6, 'rx_index_example': 1},
    # ... 更多文件
]

# 人体定位配置
MANUAL_RB_MAPPING = {40: 6, 50: 7, 60: 8}  # 手动映射

# 随机森林模型路径（使用绝对路径）
RF_MODEL_PATH = r"D:\path\to\model.pkl"
RF_SCALER_PATH = r"D:\path\to\scaler.pkl"
RF_METADATA_PATH = r"D:\path\to\metadata.json"
```

### 3. 运行程序

#### 单次实验

基本用法：
```bash
python main.py --hr_method fft --loc_method manual
```

指定窗口长度：
```bash
python main.py --hr_method fft --loc_method manual --window 60
```

参数说明：
- `--hr_method`: 心率估计方法
  - 可选: `fft`, `stft`, `wavelet`, `dct`, `vmd`, `emd`, `eemd`
- `--loc_method`: 人体定位方法
  - 可选: `manual`, `random_forest` (或 `rf`), `cfar`
- `--window`: 窗口长度（秒），可选，默认使用config.py中的设置

示例：
```bash
# 使用FFT + 手动定位
python main.py --hr_method fft --loc_method manual

# 使用STFT + CFAR定位 + 10秒窗口
python main.py --hr_method stft --loc_method cfar --window 10

# 使用小波 + 随机森林定位 + 90秒窗口
python main.py --hr_method wavelet --loc_method random_forest --window 90
```

#### 批量实验（新功能）

运行多个方法和窗口长度组合的批量实验：

```bash
python batch_experiments.py
```

**实验配置**：
- FFT: 30, 60, 90, 120秒
- STFT: 30, 60, 90, 120秒
- DCT: 30, 60, 90, 120秒
- Wavelet: 30, 60, 90, 120秒
- EMD: 30, 60, 90, 120秒
- EEMD: 30, 60, 90, 120秒
- VMD: 30, 60, 90, 120秒

**输出结果**：
1. **4张对比图**：
   - LoA宽度 vs 窗口长度
   - 平均偏差 vs 窗口长度
   - MAE vs 窗口长度
   - RMSE vs 窗口长度

2. **汇总表格**：
   - Excel文件（每个方法一个sheet）
   - CSV文件（所有数据）
   - Markdown表格（用于文档）

3. **results_cache.json**：所有实验的原始数据

**特性**：
- ✅ 自动运行30个实验配置
- ✅ 进度条显示
- ✅ 中间结果缓存（避免重复运行）
- ✅ 错误处理和日志记录
- ✅ 自动生成对比图和汇总表

## 📊 输出结果

### 单次实验输出

运行完成后，会在 `results/` 目录下生成：

1. **heart_rate_results.csv** - 详细结果表
2. **evaluation_report.json** - 评估报告（包含计算时间）
3. **bland_altman_plot.png** - Bland-Altman一致性图
4. **scatter_with_regression.png** - 散点图+回归线
5. **boxplot_by_distance.png** - 距离分组箱线图
6. **error_distribution.png** - 误差分布直方图

### 批量实验输出

运行 `batch_experiments.py` 后，会在 `batch_results/` 目录下生成：

**📈 图表** (`batch_results/figures/`)：
1. `loa_range_comparison.png` - LoA宽度对比图
2. `mean_difference_comparison.png` - 平均偏差对比图
3. `mae_comparison.png` - MAE对比图
4. `rmse_comparison.png` - RMSE对比图

**📋 表格** (`batch_results/tables/`)：
1. `summary_tables.xlsx` - Excel汇总表（每个方法一个sheet）
2. `summary_all_methods.csv` - CSV格式汇总表
3. `summary_tables.md` - Markdown格式表格

**💾 数据** (`batch_results/`)：
1. `all_results.csv` - 所有实验的完整结果
2. `batch_experiments_YYYYMMDD_HHMMSS.log` - 运行日志

**📦 缓存** (`batch_results/cache/`)：
1. `results_cache.json` - 所有实验的缓存数据
2. 各个实验的详细输出文件夹

## 🔧 算法说明

### 心率估计算法

| 方法 | 说明 | 优势 |
|-----|------|------|
| FFT | 快速傅里叶变换 | 计算快速，适合稳定心率 |
| STFT | 短时傅里叶变换 | 时频分析，适合变化心率 |
| Wavelet | 连续小波变换 | 时频局部化，抗噪声 |
| DCT | 离散余弦变换 | 能量集中，适合缓变信号 |
| VMD | 变分模态分解 | 自适应分解，抗混叠 |
| EMD | 经验模态分解 | 数据驱动，非线性信号 |
| EEMD | 集成经验模态分解 | EMD改进，更鲁棒 |

### 人体定位方法

| 方法 | 说明 | 适用场景 |
|-----|------|----------|
| Manual | 手动映射 | 已知目标距离 |
| Random Forest | 随机森林分类 | 需要预训练模型 |
| CA-CFAR | 恒虚警率检测 | 自动目标检测 |

## ⚙️ 配置参数详解

### 心率估计参数

```python
# 心率频率范围（对应48-120 bpm）
HR_FREQ_RANGE = [0.8, 2.0]

# STFT参数
STFT_NPERSEG = 512      # 窗口长度
STFT_NOVERLAP = 384     # 重叠长度
STFT_WINDOW = 'hann'    # 窗口类型

# VMD参数
VMD_ALPHA = 2000        # 惩罚因子
VMD_K = 7               # 模态数量

# EEMD参数
EEMD_TRIALS = 100       # 集成次数
EEMD_NOISE_STRENGTH = 0.2  # 噪声强度
```

### 人体定位参数

```python
# Manual方法
MANUAL_RB_MAPPING = {40: 6, 50: 7, 60: 8}

# CA-CFAR方法
CFAR_G_R = 1           # Range保护单元
CFAR_L_R = 5           # Range训练单元
CFAR_P_FA = 1e-6       # 虚警概率
```






MSc Dissertation Project - 2025
