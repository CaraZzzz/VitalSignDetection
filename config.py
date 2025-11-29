"""
配置文件
"""

# ============================================================================
# 【实验文件配置】
# ============================================================================
FILE_CONFIGS = [
    # 40cm - RB Index 6
    {'path': r"D:\MSc\Dissertation\Data\250902\test2\RangeFFT\RangeBin6_RX1\test2_NeuLogRadar_aligned_trimmed.mat", 'distance': 40, 'rb_index': 6, 'rx_index_example': 1},
    {'path': r"D:\MSc\Dissertation\Data\250902\test3\RangeFFT\RangeBin6_RX1\test3_NeuLogRadar_aligned_trimmed.mat", 'distance': 40, 'rb_index': 6, 'rx_index_example': 1},
    {'path': r"D:\MSc\Dissertation\Data\250902\test4\RangeFFT\RangeBin6_RX1\test4_NeuLogRadar_aligned_trimmed.mat", 'distance': 40, 'rb_index': 6, 'rx_index_example': 1},
    {'path': r"D:\MSc\Dissertation\Data\250902\test5\RangeFFT\RangeBin6_RX1\test5_NeuLogRadar_aligned_trimmed.mat", 'distance': 40, 'rb_index': 6, 'rx_index_example': 1},
    
    # 50cm - RB Index 7
    {'path': r"D:\MSc\Dissertation\Data\250826\test8\RangeFFT\RangeBin7_RX1\test8_NeuLogRadar_aligned_trimmed.mat", 'distance': 50, 'rb_index': 7, 'rx_index_example': 1},
    {'path': r"D:\MSc\Dissertation\Data\250826\test9\RangeFFT\RangeBin7_RX1\test9_NeuLogRadar_aligned_trimmed.mat", 'distance': 50, 'rb_index': 7, 'rx_index_example': 1},
    {'path': r"D:\MSc\Dissertation\Data\250925\test1\RangeFFT\RangeBin7_RX2\test1_NeuLogRadar_aligned_trimmed.mat", 'distance': 50, 'rb_index': 7, 'rx_index_example': 2},
    {'path': r"D:\MSc\Dissertation\Data\250925\test2\RangeFFT\RangeBin7_RX2\test2_NeuLogRadar_aligned_trimmed.mat", 'distance': 50, 'rb_index': 7, 'rx_index_example': 2},

    # 60cm - RB Index 8
    {'path': r"D:\MSc\Dissertation\Data\250925\test3\RangeFFT\RangeBin8_RX4\test3_NeuLogRadar_aligned_trimmed.mat", 'distance': 60, 'rb_index': 8, 'rx_index_example': 4},
    {'path': r"D:\MSc\Dissertation\Data\250925\test4\RangeFFT\RangeBin8_RX1\test4_NeuLogRadar_aligned_trimmed.mat", 'distance': 60, 'rb_index': 8, 'rx_index_example': 1},
    {'path': r"D:\MSc\Dissertation\Data\251016\test1\RangeFFT\RangeBin8_RX1\test1_NeuLogRadar_aligned_trimmed.mat", 'distance': 60, 'rb_index': 8, 'rx_index_example': 1},
    {'path': r"D:\MSc\Dissertation\Data\251016\test2\RangeFFT\RangeBin8_RX2\test2_NeuLogRadar_aligned_trimmed.mat", 'distance': 60, 'rb_index': 8, 'rx_index_example': 2},
]

# ============================================================================
# 【系统参数配置】
# ============================================================================
WINDOW_DURATION_S = 120  # 窗口长度（秒）
STEP_SIZE_S = 15  # 步长（秒）
SAMPLING_RATE = 20  # 采样率 (Hz)

# ============================================================================
# 【心率估计参数】
# ============================================================================
# 心率频率范围 (Hz) -> 48-120 bpm
HR_FREQ_RANGE = [0.8, 2.0]

# FFT参数
FFT_APPLY_WINDOW = False  # 是否加窗
FFT_DETREND = True  # 是否去趋势

# STFT参数
STFT_NPERSEG = 512  # 窗口长度（样本数）
STFT_NOVERLAP = 384  # 重叠长度（样本数）
STFT_WINDOW = 'hann'  # 窗口类型

# Wavelet参数
WAVELET_TYPE = 'morl'  # 小波类型

# DCT参数
DCT_DETREND = True

# VMD参数
VMD_ALPHA = 2000  # 惩罚因子
VMD_TAU = 0.0  # 噪声容忍度
VMD_K = 4  # 模态数量
VMD_DC = 0
VMD_INIT = 1
VMD_TOL = 1e-7

# EMD参数
EMD_MAX_IMF = -1  # 最大IMF数量

# EEMD参数
EEMD_TRIALS = 100  # 集成次数
EEMD_NOISE_STRENGTH = 0.2  # 噪声强度
EEMD_MAX_IMF = -1

# ============================================================================
# 【人体定位参数】
# ============================================================================
# Manual方法的Range Bin映射（1-based）
# 例如: {40: 6, 50: 7, 60: 8} 表示40cm用RB6，50cm用RB7，60cm用RB8
# 如果为None，则使用FILE_CONFIGS中的rb_index
# MANUAL_RB_MAPPING = None  # {40: 6, 50: 7, 60: 8}
MANUAL_RB_MAPPING = {40: 6, 50: 7, 60: 8}
# MANUAL_RB_MAPPING = {40: 4, 50: 5, 60: 6}
# MANUAL_RB_MAPPING = {40: 5, 50: 6, 60: 7}

# Random Forest模型路径（使用绝对路径）
# 标签为{40: 4, 50: 5, 60: 6}的随机森林
# RF_MODEL_PATH = r"D:\MSc\Dissertation\Code\v5\RandomForest_1120_v4\rf_rangebin_classifier_20251120_210656.pkl"
# RF_SCALER_PATH = r"D:\MSc\Dissertation\Code\v5\RandomForest_1120_v4\rf_rangebin_classifier_scaler_20251120_210656.pkl"
# RF_METADATA_PATH = r"D:\MSc\Dissertation\Code\v5\RandomForest_1120_v4\rf_rangebin_classifier_metadata_20251120_210656.json"
# 标签为{40: 6, 50: 7, 60: 8}的随机森林
RF_MODEL_PATH = r"D:\MSc\Dissertation\Code\v5\RandomForest_1118_v3\rf_rangebin_classifier_20251118_171632.pkl"
RF_SCALER_PATH = r"D:\MSc\Dissertation\Code\v5\RandomForest_1118_v3\rf_rangebin_classifier_scaler_20251118_171632.pkl"
RF_METADATA_PATH = r"D:\MSc\Dissertation\Code\v5\RandomForest_1118_v3\rf_rangebin_classifier_metadata_20251118_171632.json"


# CA-CFAR参数
CFAR_G_R = 1  # Range保护单元半长
CFAR_G_D = 1  # Doppler保护单元半长
CFAR_L_R = 5  # Range训练单元半长
CFAR_L_D = 5  # Doppler训练单元半长
CFAR_P_FA = 1e-4  # 虚警概率
CFAR_N_DOPPLER_FFT = 128  # Doppler FFT点数

# ============================================================================
# 【输出配置】
# ============================================================================
OUTPUT_DIR = './results'