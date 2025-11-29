"""
随机森林 + CFAR Range Bin 分类器 + STFT 心率估计
Leave-One-File-Out 交叉验证 (K=12)
三方对比：Manual, RF, CFAR
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')
import joblib
import json
from datetime import datetime


# 导入自定义模块
from FeatureExtract_1126_v7 import extract_features_from_all_files
from stft_estimator import STFTHeartRateEstimator
from evaluation_stft import STFTDualPathEvaluator
from cfar_detector import CAFARDetector  # 🔥 新增

# ============================================================================
# 【配置区】
# ============================================================================
print("=" * 70)
print("🚀 随机森林 + CFAR Range Bin 预测 + STFT 心率估计系统")
print("   Leave-One-File-Out 交叉验证 (K=12)")
print("   三方对比：Manual, RF, CFAR")
print("=" * 70)

# --- 文件配置 ---
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

# --- 训练配置 ---
TRAIN_WINDOW_DURATION_S = 30
TRAIN_STEP_DURATION_S = 30

# --- 测试配置 ---
TEST_WINDOW_DURATION_S = 120
TEST_STEP_DURATION_S = 15

# --- STFT配置 ---
STFT_NPERSEG = 512
STFT_NOVERLAP = 384
STFT_WINDOW = 'hann'
STFT_FREQ_RANGE = (0.8, 2.0)

# --- CFAR配置 🔥 新增 ---
CFAR_G_R = 1
CFAR_G_D = 1
CFAR_L_R = 5
CFAR_L_D = 5
CFAR_P_FA = 1e-3
CFAR_N_DOPPLER_FFT = 128
CFAR_MIN_RANGE_BIN = 1
CFAR_MAX_RANGE_BIN = 18

# --- 其他配置 ---
TARGET_RX_INDEX = 1
RANDOM_STATE = 42
N_JOBS = -1

# --- 特征名称 ---
FEATURE_NAMES = [
    'amp_mean', 'amp_std', 'amp_p2p', 'amp_skewness', 'amp_kurtosis',
    'phase_diff_std', 'phase_diff_range',
    'amp_energy_resp', 'amp_energy_cardiac', 'amp_life_energy_ratio',
    'amp_freq_peak_pos', 'amp_purity_ratio', 'amp_energy_ratio_C_R', 'amp_snr_db',
    'phase_energy_ratio_C_R', 'phase_snr_db',
    'amp_hr_peak_purity'
]

# --- 输出目录 ---
OUTPUT_DIR = './results_rf_cfar_stft'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"✅ 已创建输出目录: {OUTPUT_DIR}")

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 【辅助函数】
# ============================================================================

def build_samples_from_segments(segments_list, target_rx_index):
    """从片段列表构建二分类样本"""
    samples_data = []
    
    for seg_idx, segment in enumerate(segments_list):
        file_name = segment['original_file']
        segment_id = segment['segment_index']
        heart_count = segment['heart_count']
        range_one_hot = segment['range_one_hot'].squeeze()
        distance = segment['distance']
        
        true_rb_index = np.argmax(range_one_hot) + 1
        rx_idx_0based = target_rx_index - 1
        R = segment['amp_mean'].shape[0]
        
        for r in range(R):
            rb_1based = r + 1
            
            feature_vector = []
            for feat_name in FEATURE_NAMES:
                if feat_name in segment:
                    feat_value = segment[feat_name][r, rx_idx_0based]
                    feature_vector.append(feat_value)
                else:
                    feature_vector.append(0.0)
            
            label = 1 if range_one_hot[r] == 1 else 0
            
            samples_data.append({
                'features': feature_vector,
                'label': label,
                'file_name': file_name,
                'segment_id': segment_id,
                'rb_index': rb_1based,
                'distance': distance,
                'heart_count': heart_count,
                'true_rb_index': true_rb_index,
                'segment_global_id': seg_idx
            })
    
    df = pd.DataFrame(samples_data)
    X = np.array(df['features'].tolist())
    y = df['label'].values
    meta_df = df.drop(columns=['features', 'label'])
    
    return X, y, meta_df


def perform_stft_comparison(segment, pred_rb, true_rb, rx_index, stft_estimator):
    """
    对预测错误的窗口执行STFT双路对比
    
    参数:
        segment: 测试片段字典（包含raw_phase_data）
        pred_rb: 预测的Range Bin (1-based)
        true_rb: 真实的Range Bin (1-based)
        rx_index: RX天线索引 (1-based)
        stft_estimator: STFT估计器实例
    
    返回:
        STFT结果字典
    """
    # 提取原始phase数据
    raw_phase = segment['raw_phase_data']  # (R, nVX, T)
    rx_idx = rx_index - 1
    
    # 路径A: 使用预测的Range Bin
    phase_pred = raw_phase[pred_rb - 1, rx_idx, :]
    hr_pred_result = stft_estimator.estimate(phase_pred, STFT_FREQ_RANGE)
    hr_pred = hr_pred_result['heart_rate_bpm']
    
    # 路径B: 使用真实的Range Bin
    phase_true = raw_phase[true_rb - 1, rx_idx, :]
    hr_true_result = stft_estimator.estimate(phase_true, STFT_FREQ_RANGE)
    hr_true = hr_true_result['heart_rate_bpm']
    
    # NeuLog真实心率
    heart_count = segment['heart_count']
    window_duration = TEST_WINDOW_DURATION_S
    neulog_hr = (heart_count / window_duration) * 60
    
    return {
        'neulog_hr_bpm': neulog_hr,
        'hr_from_pred_rb': hr_pred,
        'hr_from_true_rb': hr_true,
        'mae_pred': abs(hr_pred - neulog_hr) if not np.isnan(hr_pred) else np.nan,
        'mae_true': abs(hr_true - neulog_hr) if not np.isnan(hr_true) else np.nan
    }

# 🔥 新增：CFAR预测函数
def perform_cfar_prediction(segment, cfar_detector, rx_index):
    """
    使用CFAR检测Range Bin
    
    参数:
        segment: 测试片段字典（包含raw_phase_data）
        cfar_detector: CFAR检测器实例
        rx_index: RX天线索引 (1-based，但CFAR用全部RX)
    
    返回:
        detected_rb: 检测到的Range Bin (1-based)
        confidence: 置信度
    """
    # 提取原始phase数据 (R, nVX, T)
    raw_phase = segment['raw_phase_data']
    
    # CFAR需要 (Range x RX x Time) 格式，已经符合
    cfar_result = cfar_detector.detect(raw_phase)
    
    detected_rb = cfar_result['detected_range_bin']
    confidence = cfar_result['confidence']
    
    return detected_rb, confidence


# ============================================================================
# 【主程序：Leave-One-File-Out 交叉验证】
# ============================================================================

print("\n【配置信息】")
print(f"训练窗口: {TRAIN_WINDOW_DURATION_S}s (步长{TRAIN_STEP_DURATION_S}s, 无重叠)")
print(f"测试窗口: {TEST_WINDOW_DURATION_S}s (步长{TEST_STEP_DURATION_S}s, 有重叠)")
print(f"STFT参数: nperseg={STFT_NPERSEG}, noverlap={STFT_NOVERLAP}")
print(f"总文件数: {len(FILE_CONFIGS)} (K={len(FILE_CONFIGS)})")

# 初始化STFT估计器
stft_estimator = STFTHeartRateEstimator(
    fs=20.0,
    detrend=True,
    nperseg=STFT_NPERSEG,
    noverlap=STFT_NOVERLAP,
    window=STFT_WINDOW
)

# 🔥 初始化CFAR检测器
cfar_detector = CAFARDetector(
    G_R=CFAR_G_R,
    G_D=CFAR_G_D,
    L_R=CFAR_L_R,
    L_D=CFAR_L_D,
    P_fa=CFAR_P_FA,
    n_doppler_fft=CFAR_N_DOPPLER_FFT,
    min_range_bin=CFAR_MIN_RANGE_BIN,
    max_range_bin=CFAR_MAX_RANGE_BIN
)

# 存储所有fold的结果
all_fold_summary = []
all_rf_predictions = []
all_cfar_predictions = []  # 🔥 新增
all_feature_importances = []  # 🔥 新增：存储特征重要性

# 开始K-Fold循环
K = len(FILE_CONFIGS)

for fold_idx in range(K):
    print("\n" + "=" * 70)
    print(f"Fold {fold_idx + 1}/{K}: 测试文件 {os.path.basename(FILE_CONFIGS[fold_idx]['path'])} ({FILE_CONFIGS[fold_idx]['distance']}cm)")
    print("=" * 70)
    
    # 1. 划分训练/测试文件
    test_file = FILE_CONFIGS[fold_idx]
    train_files = FILE_CONFIGS[:fold_idx] + FILE_CONFIGS[fold_idx+1:]
    
    # ========================================================================
    # 【RF训练】
    # ========================================================================
    print("\n【RF 训练阶段】")
    train_segments = extract_features_from_all_files(
        train_files,
        window_duration_s=TRAIN_WINDOW_DURATION_S,
        step_duration_s=TRAIN_STEP_DURATION_S,
        keep_raw_phase=False
    )
    
    X_train, y_train, meta_train = build_samples_from_segments(train_segments, TARGET_RX_INDEX)
    
    print(f"RF训练集: {len(train_files)}个文件, {len(np.unique(meta_train['segment_global_id']))}个窗口, {len(y_train)}个样本")
    
    # 标准化 + 训练模型
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )
    
    rf_model.fit(X_train_scaled, y_train)
    print(f"✅ RF模型训练完成")

    # 🔥 新增：保存该fold的特征重要性
    feature_importance = rf_model.feature_importances_
    all_feature_importances.append(feature_importance)
    
    # ========================================================================
    # 【测试阶段：RF + CFAR】
    # ========================================================================
    print("\n【测试阶段：RF + CFAR】")
    test_segments = extract_features_from_all_files(
        [test_file],
        window_duration_s=TEST_WINDOW_DURATION_S,
        step_duration_s=TEST_STEP_DURATION_S,
        keep_raw_phase=True
    )
    
    num_test_windows = len(test_segments)
    print(f"测试集: 1个文件, {num_test_windows}个窗口")
    
    # 对每个测试窗口进行预测
    fold_rf_predictions = []
    fold_cfar_predictions = []
    
    rf_correct_count = 0
    cfar_correct_count = 0
    
    print(f"\n📊 逐窗口预测...")
    
    for window_idx, segment in enumerate(test_segments):
        
        true_rb = segment['rb_index_1based']
        
        # ====================================================================
        # RF 预测
        # ====================================================================
        X_test_window = []
        rx_idx = TARGET_RX_INDEX - 1
        R = segment['amp_mean'].shape[0]
        
        for r in range(R):
            feature_vector = []
            for feat_name in FEATURE_NAMES:
                if feat_name in segment:
                    feat_value = segment[feat_name][r, rx_idx]
                    feature_vector.append(feat_value)
                else:
                    feature_vector.append(0.0)
            X_test_window.append(feature_vector)
        
        X_test_window = np.array(X_test_window)
        X_test_scaled = scaler.transform(X_test_window)
        
        probas = rf_model.predict_proba(X_test_scaled)[:, 1]
        rf_pred_rb = np.argmax(probas) + 1
        rf_is_correct = (rf_pred_rb == true_rb)
        
        if rf_is_correct:
            rf_correct_count += 1
        
        # RF预测记录
        rf_prediction_record = {
            'fold': fold_idx + 1,
            'file_name': segment['original_file'],
            'window_index': segment['segment_index'],
            'distance': segment['distance'],
            'true_rb': true_rb,
            'pred_rb': rf_pred_rb,
            'is_correct': rf_is_correct,
            'pred_proba': probas[rf_pred_rb - 1],
            'heart_count': segment['heart_count']
        }
        
        # 如果RF预测错误，执行STFT
        if not rf_is_correct:
            stft_result = perform_stft_comparison(
                segment, rf_pred_rb, true_rb, TARGET_RX_INDEX, stft_estimator
            )
            rf_prediction_record.update(stft_result)
            
            print(f"  RF ✗ 窗口{segment['segment_index']}: 预测={rf_pred_rb}, 真实={true_rb} | "
                  f"HR: {stft_result['hr_from_pred_rb']:.1f} vs {stft_result['hr_from_true_rb']:.1f} "
                  f"(NeuLog={stft_result['neulog_hr_bpm']:.1f})")
        
        fold_rf_predictions.append(rf_prediction_record)
        
        # ====================================================================
        # CFAR 预测 🔥
        # ====================================================================
        cfar_pred_rb, cfar_confidence = perform_cfar_prediction(
            segment, cfar_detector, TARGET_RX_INDEX
        )
        cfar_is_correct = (cfar_pred_rb == true_rb)
        
        if cfar_is_correct:
            cfar_correct_count += 1
        
        # CFAR预测记录
        cfar_prediction_record = {
            'fold': fold_idx + 1,
            'file_name': segment['original_file'],
            'window_index': segment['segment_index'],
            'distance': segment['distance'],
            'true_rb': true_rb,
            'pred_rb': cfar_pred_rb,
            'is_correct': cfar_is_correct,
            'confidence': cfar_confidence,
            'heart_count': segment['heart_count']
        }
        
        # 如果CFAR预测错误，执行STFT
        if not cfar_is_correct:
            stft_result = perform_stft_comparison(
                segment, cfar_pred_rb, true_rb, TARGET_RX_INDEX, stft_estimator
            )
            cfar_prediction_record.update(stft_result)
            
            print(f"  CFAR ✗ 窗口{segment['segment_index']}: 预测={cfar_pred_rb}, 真实={true_rb} | "
                  f"HR: {stft_result['hr_from_pred_rb']:.1f} vs {stft_result['hr_from_true_rb']:.1f} "
                  f"(NeuLog={stft_result['neulog_hr_bpm']:.1f})")
        
        fold_cfar_predictions.append(cfar_prediction_record)
    
    # ========================================================================
    # Fold结果汇总
    # ========================================================================
    rf_accuracy = rf_correct_count / num_test_windows if num_test_windows > 0 else 0.0
    cfar_accuracy = cfar_correct_count / num_test_windows if num_test_windows > 0 else 0.0
    
    fold_summary = {
        'fold': fold_idx + 1,
        'test_file': os.path.basename(test_file['path']),
        'distance': test_file['distance'],
        'n_test_windows': num_test_windows,
        'rf_correct': rf_correct_count,
        'rf_wrong': num_test_windows - rf_correct_count,
        'rf_accuracy': rf_accuracy,
        'cfar_correct': cfar_correct_count,
        'cfar_wrong': num_test_windows - cfar_correct_count,
        'cfar_accuracy': cfar_accuracy
    }
    
    all_fold_summary.append(fold_summary)
    all_rf_predictions.extend(fold_rf_predictions)
    all_cfar_predictions.extend(fold_cfar_predictions)
    
    print(f"\n✅ Fold {fold_idx + 1} 完成:")
    print(f"   RF:   正确 {rf_correct_count}/{num_test_windows} ({rf_accuracy*100:.1f}%)")
    print(f"   CFAR: 正确 {cfar_correct_count}/{num_test_windows} ({cfar_accuracy*100:.1f}%)")
    
    # 保存模型
    model_path = os.path.join(OUTPUT_DIR, f'rf_model_fold{fold_idx+1}.pkl')
    scaler_path = os.path.join(OUTPUT_DIR, f'scaler_fold{fold_idx+1}.pkl')
    joblib.dump(rf_model, model_path)
    joblib.dump(scaler, scaler_path)

# ============================================================================
# 【汇总所有Fold结果】
# ============================================================================

print("\n" + "=" * 70)
print("✅ 所有Fold完成！")
print("=" * 70)

# 转换为DataFrame
fold_summary_df = pd.DataFrame(all_fold_summary)
rf_predictions_df = pd.DataFrame(all_rf_predictions)
cfar_predictions_df = pd.DataFrame(all_cfar_predictions)

# 保存原始结果
fold_summary_df.to_csv(os.path.join(OUTPUT_DIR, 'fold_summary.csv'), index=False)
rf_predictions_df.to_csv(os.path.join(OUTPUT_DIR, 'rf_predictions.csv'), index=False)
cfar_predictions_df.to_csv(os.path.join(OUTPUT_DIR, 'cfar_predictions.csv'), index=False)

# ============================================================================
# 【统计结果】
# ============================================================================

print("\n【Range Bin 预测准确率汇总】")

# RF准确率
print("\n=== 随机森林 (RF) ===")
print("\n各Fold准确率:")
for _, row in fold_summary_df.iterrows():
    print(f"  Fold {int(row['fold'])} ({int(row['distance'])}cm): "
          f"{int(row['rf_correct'])}/{int(row['n_test_windows'])} = {row['rf_accuracy']*100:.1f}%")

rf_total_windows = fold_summary_df['n_test_windows'].sum()
rf_total_correct = fold_summary_df['rf_correct'].sum()
rf_overall_accuracy = rf_total_correct / rf_total_windows

print(f"\nRF全局统计:")
print(f"  总窗口数: {rf_total_windows}")
print(f"  预测正确: {rf_total_correct} ({rf_overall_accuracy*100:.1f}%)")
print(f"  预测错误: {rf_total_windows - rf_total_correct} ({(1-rf_overall_accuracy)*100:.1f}%)")

print(f"\nRF按距离统计:")
for dist in sorted(fold_summary_df['distance'].unique()):
    dist_data = fold_summary_df[fold_summary_df['distance'] == dist]
    dist_correct = dist_data['rf_correct'].sum()
    dist_total = dist_data['n_test_windows'].sum()
    dist_acc = dist_correct / dist_total
    print(f"  {int(dist)}cm: {int(dist_correct)}/{int(dist_total)} = {dist_acc*100:.1f}%")

# CFAR准确率
print("\n=== CA-CFAR ===")
print("\n各Fold准确率:")
for _, row in fold_summary_df.iterrows():
    print(f"  Fold {int(row['fold'])} ({int(row['distance'])}cm): "
          f"{int(row['cfar_correct'])}/{int(row['n_test_windows'])} = {row['cfar_accuracy']*100:.1f}%")

cfar_total_correct = fold_summary_df['cfar_correct'].sum()
cfar_overall_accuracy = cfar_total_correct / rf_total_windows

print(f"\nCFAR全局统计:")
print(f"  总窗口数: {rf_total_windows}")
print(f"  预测正确: {cfar_total_correct} ({cfar_overall_accuracy*100:.1f}%)")
print(f"  预测错误: {rf_total_windows - cfar_total_correct} ({(1-cfar_overall_accuracy)*100:.1f}%)")

print(f"\nCFAR按距离统计:")
for dist in sorted(fold_summary_df['distance'].unique()):
    dist_data = fold_summary_df[fold_summary_df['distance'] == dist]
    dist_correct = dist_data['cfar_correct'].sum()
    dist_total = dist_data['n_test_windows'].sum()
    dist_acc = dist_correct / dist_total
    print(f"  {int(dist)}cm: {int(dist_correct)}/{int(dist_total)} = {dist_acc*100:.1f}%")

# ============================================================================
# 【STFT心率估计评估】
# ============================================================================

print("\n【STFT 心率估计评估】")

# RF错误预测的STFT评估
rf_wrong_df = rf_predictions_df[rf_predictions_df['is_correct'] == False].copy()
cfar_wrong_df = cfar_predictions_df[cfar_predictions_df['is_correct'] == False].copy()

print(f"\nRF错误窗口数: {len(rf_wrong_df)}")
print(f"CFAR错误窗口数: {len(cfar_wrong_df)}")

# 使用评估模块
if len(rf_wrong_df) > 0:
    print("\n=== RF 错误预测的STFT评估 ===")
    rf_evaluator = STFTDualPathEvaluator(output_dir=os.path.join(OUTPUT_DIR, 'rf_stft_evaluation'))
    rf_eval_results = rf_evaluator.evaluate(rf_predictions_df)

if len(cfar_wrong_df) > 0:
    print("\n=== CFAR 错误预测的STFT评估 ===")
    cfar_evaluator = STFTDualPathEvaluator(output_dir=os.path.join(OUTPUT_DIR, 'cfar_stft_evaluation'))
    cfar_eval_results = cfar_evaluator.evaluate(cfar_predictions_df)



# ============================================================================
# 【三方对比可视化】
# ============================================================================

print("\n【生成三方对比可视化】")

# 0. 🔥 特征重要性可视化
def plot_feature_importance(all_feature_importances, feature_names, save_path=None):
    """
    绘制随机森林特征重要性排序柱状图（跨所有fold的平均）
    
    参数:
        all_feature_importances: 所有fold的特征重要性列表 (K x n_features)
        feature_names: 特征名称列表
        save_path: 保存路径
    """
    # 转换为numpy数组并计算平均值和标准差
    importances_array = np.array(all_feature_importances)  # (K, n_features)
    mean_importances = np.mean(importances_array, axis=0)
    std_importances = np.std(importances_array, axis=0)
    
    # 创建DataFrame并按重要性排序
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_importances,
        'std': std_importances
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(16, 8))
    
    y_pos = np.arange(len(importance_df))
    
    # 使用颜色渐变（从深到浅）
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))

    bars = ax.barh(y_pos, importance_df['importance'],
                xerr=importance_df['std'],
                color=colors,
                edgecolor='black',
                linewidth=1.2,
                error_kw={'elinewidth': 1.5, 'capsize': 3, 'alpha': 0.7})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df['feature'], fontsize=18)
    ax.invert_yaxis()  # 最重要的特征在顶部

    ax.set_xlabel('Mean Feature Importance', fontsize=18, fontweight='bold')
    ax.set_ylabel('Features', fontsize=18, fontweight='bold')
    ax.set_title('Random Forest Feature Importance Ranking\n(Averaged across 12 Folds with Standard Deviation)',
                fontsize=20, fontweight='bold', pad=20)

    # 添加网格
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # 添加数值标签
    for i, (bar, importance, std) in enumerate(zip(bars, importance_df['importance'], importance_df['std'])):
        width = bar.get_width()
        ax.text(width + std + 0.002, bar.get_y() + bar.get_height()/2,
                f'{importance:.4f}',
                # 数值标签设置为 18 号
                ha='left', va='center', fontsize=18, fontweight='bold')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 特征重要性图已保存: {save_path}")
    
    plt.close()
    
    return importance_df


# 调用函数生成特征重要性图
if len(all_feature_importances) > 0:
    print("\n生成特征重要性排序图...")
    feature_importance_path = os.path.join(OUTPUT_DIR, 'feature_importance_ranking.png')
    importance_df = plot_feature_importance(
        all_feature_importances, 
        FEATURE_NAMES, 
        feature_importance_path
    )
    
    # 保存特征重要性数据到CSV
    importance_csv_path = os.path.join(OUTPUT_DIR, 'feature_importance.csv')
    importance_df.to_csv(importance_csv_path, index=False)
    print(f"✅ 特征重要性数据已保存: {importance_csv_path}")
    
    # 打印Top 5特征
    print("\n📊 Top 5 最重要特征:")
    for idx, row in importance_df.head(5).iterrows():
        print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f} (±{row['std']:.4f})")


# 1. 双子图混淆矩阵对比
def plot_dual_confusion_matrix(rf_predictions_df, cfar_predictions_df, save_path=None):
    """
    绘制RF和CFAR的混淆矩阵对比图（双子图）
    
    参数:
        rf_predictions_df: RF预测结果DataFrame
        cfar_predictions_df: CFAR预测结果DataFrame
        save_path: 保存路径
    """
    # 收集所有可能的Range Bin
    all_true_rb = set(rf_predictions_df['true_rb'].unique())
    all_pred_rb_rf = set(rf_predictions_df['pred_rb'].unique())
    all_pred_rb_cfar = set(cfar_predictions_df['pred_rb'].unique())
    
    all_bins = sorted(all_true_rb | all_pred_rb_rf | all_pred_rb_cfar)
    
    # 计算RF混淆矩阵
    rf_cm = confusion_matrix(
        rf_predictions_df['true_rb'], 
        rf_predictions_df['pred_rb'],
        labels=all_bins
    )
    
    # 计算CFAR混淆矩阵
    cfar_cm = confusion_matrix(
        cfar_predictions_df['true_rb'], 
        cfar_predictions_df['pred_rb'],
        labels=all_bins
    )
    
    # 创建双子图
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 子图1: RF混淆矩阵
    # 计算RF准确率（对角线之和 / 总和）
    rf_accuracy = np.trace(rf_cm) / np.sum(rf_cm) * 100
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=all_bins, yticklabels=all_bins,
                cbar_kws={'label': 'Count'}, ax=axes[0],
                linewidths=0.5, linecolor='gray',
                annot_kws={'fontsize': 18})
    axes[0].set_title(f'Random Forest Confusion Matrix\n(Overall Accuracy: {rf_accuracy:.1f}%)', fontsize=20, fontweight='bold', pad=20)
    axes[0].set_xlabel('Predicted Range Bin (1-based)', fontsize=18, fontweight='bold')
    axes[0].set_ylabel('True Range Bin (1-based)', fontsize=18, fontweight='bold')
    # axes[0].text(0.5, 1.05, f'Overall Accuracy: {rf_accuracy:.1f}%', 
    #             ha='center', va='bottom', transform=axes[0].transAxes,
    #             fontsize=14, fontweight='bold', color='darkblue',
    #             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    
    # 子图2: CFAR混淆矩阵
    # 计算CFAR准确率
    cfar_accuracy = np.trace(cfar_cm) / np.sum(cfar_cm) * 100
    sns.heatmap(cfar_cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=all_bins, yticklabels=all_bins,
                cbar_kws={'label': 'Count'}, ax=axes[1],
                linewidths=0.5, linecolor='gray',
                annot_kws={'fontsize': 18})
    axes[1].set_title(f'CA-CFAR Confusion Matrix\n(Overall Accuracy: {cfar_accuracy:.1f}%)', fontsize=20, fontweight='bold', pad=20)
    axes[1].set_xlabel('Predicted Range Bin (1-based)', fontsize=18, fontweight='bold')
    axes[1].set_ylabel('True Range Bin (1-based)', fontsize=18, fontweight='bold')
    # axes[1].text(0.5, 1.05, f'Overall Accuracy: {cfar_accuracy:.1f}%', 
    #             ha='center', va='bottom', transform=axes[1].transAxes,
    #             fontsize=14, fontweight='bold', color='darkgreen',
    #             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))
    
    # 总标题
    fig.suptitle('Range Bin Prediction: Random Forest vs CA-CFAR\nLeave-One-File-Out Cross-Validation (K=12)', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 双子图混淆矩阵已保存: {save_path}")
    
    plt.close()
    
    return rf_cm, cfar_cm, rf_accuracy, cfar_accuracy


print("\n生成双子图混淆矩阵...")
dual_cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_rf_vs_cfar.png')
rf_cm, cfar_cm, rf_cm_acc, cfar_cm_acc = plot_dual_confusion_matrix(
    rf_predictions_df, 
    cfar_predictions_df, 
    dual_cm_path
)


# 2. Range Bin预测准确率对比
fig, ax = plt.subplots(figsize=(12, 6))

methods = ['RF', 'CFAR']
overall_accs = [rf_overall_accuracy * 100, cfar_overall_accuracy * 100]

x = np.arange(len(methods))
bars = ax.bar(x, overall_accs, width=0.6, color=['#FF6B6B', '#4ECDC4'], edgecolor='black', linewidth=1.5)

for bar, acc in zip(bars, overall_accs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{acc:.1f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylim([0, 105])
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=13)
ax.set_ylabel('Overall Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Range Bin Prediction Accuracy Comparison', fontsize=15, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'method_comparison_overall.png'), dpi=300)
plt.close()
print(f"✅ 整体准确率对比图已保存")


# 3. 按距离分组对比
distances = sorted(fold_summary_df['distance'].unique())
rf_dist_accs = []
cfar_dist_accs = []

for dist in distances:
    dist_data = fold_summary_df[fold_summary_df['distance'] == dist]
    
    rf_acc = dist_data['rf_correct'].sum() / dist_data['n_test_windows'].sum() * 100
    cfar_acc = dist_data['cfar_correct'].sum() / dist_data['n_test_windows'].sum() * 100
    
    rf_dist_accs.append(rf_acc)
    cfar_dist_accs.append(cfar_acc)

x = np.arange(len(distances))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, rf_dist_accs, width, label='RF', color='#FF6B6B', edgecolor='black')
bars2 = ax.bar(x + width/2, cfar_dist_accs, width, label='CFAR', color='#4ECDC4', edgecolor='black')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylim([0, 105])
ax.set_xticks(x)
ax.set_xticklabels([f'{int(d)}cm' for d in distances], fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('Distance', fontsize=13, fontweight='bold')
ax.set_title('Range Bin Prediction Accuracy by Distance', fontsize=15, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'method_comparison_by_distance.png'), dpi=300)
plt.close()
print(f"✅ 按距离对比图已保存")

print(f"\n✅ 所有可视化图表已生成！")




## 📊 输出效果
"""
运行后会生成：

results_rf_cfar_stft/
├── feature_importance_ranking.png   # 🔥 新增：特征重要性排序图
├── feature_importance.csv           # 🔥 新增：特征重要性数据
├── confusion_matrix_rf_vs_cfar.png  # 双子图混淆矩阵
├── method_comparison_overall.png    # 整体准确率对比
├── method_comparison_by_distance.png # 按距离对比
└── ...
```

---

## 📈 图表特点

**特征重要性图**：
- **横向柱状图**：易于阅读特征名称
- **颜色渐变**：从重要到不重要用渐变色区分
- **误差线**：显示12个fold之间的标准差
- **数值标签**：精确显示每个特征的重要性得分
- **排序**：从上到下按重要性降序排列

**终端输出示例**：

📊 Top 5 最重要特征:
  1. amp_snr_db: 0.1523 (±0.0089)
  2. amp_energy_cardiac: 0.1247 (±0.0156)
  3. phase_snr_db: 0.1098 (±0.0123)
  4. amp_hr_peak_purity: 0.0876 (±0.0098)
  5. amp_purity_ratio: 0.0754 (±0.0112)
  """