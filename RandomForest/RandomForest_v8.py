## --------------------------------------------------------------##
# v1 对每个单元格进行二分类（是否是人体），暂时不考虑RX，指定RX=1
# 后续可以考虑RX的特征融合
## v2相比v1，强制每个30秒片段RX通道输出有且仅有一个人体所在单元格
## v3相比v2，保存模型以便后续调用
## v4相比v3，修改距离和range bin index的对应关系，40/50/60cm对应range bin index 4/5/6(base 1)
## v5相比v4，增加range bin 级别的confusion matrix绘制
## v6相比v5：K-Fold 交叉验证版本
# 主要改动：使用 StratifiedGroupKFold 进行 K-Fold 交叉验证
## v8改自v6，主要增加按距离输出特征具体的值（用于评估数据质量）和数据质量评估函数Q的得分
## --------------------------------------------------------------##
"""
随机森林 Range Bin 分类器 - K-Fold 交叉验证版
使用 K-Fold 交叉验证，可以在全部数据上进行测试
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold  # 🔥 改为 K-Fold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score, accuracy_score
)
import warnings
warnings.filterwarnings('ignore')
import joblib
import json
from datetime import datetime

# 导入特征提取模块
from FeatureExtract_v6 import extract_features_from_all_files

# ============================================================================
# 【配置区】
# ============================================================================
print("=" * 70)
print("🚀 随机森林Range Bin预测系统 - K-Fold 交叉验证版")
print("=" * 70)

# --- 1. 实验文件配置 ---
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

# --- 2. 模型参数配置 ---
WINDOW_DURATION_S = 30  # 窗口长度（秒）
TARGET_RX_INDEX = 1  # 关注的RX天线索引（1-based）
N_SPLITS = 4  # 🔥 K-Fold 的 K 值（3 或 4）
RANDOM_STATE = 42
N_JOBS = -1

# --- 3. 特征名称 ---
# --- 3. 特征名称 ---
FEATURE_NAMES = [
    'amp_mean', 'amp_std', 'amp_p2p', 'amp_skewness', 'amp_kurtosis',
    'phase_diff_std', 'phase_diff_range',
    'amp_energy_resp', 'amp_energy_cardiac', 'amp_life_energy_ratio',
    'amp_freq_peak_pos', 'amp_purity_ratio', 'amp_energy_ratio_C_R', 'amp_snr_db',
    'phase_energy_ratio_C_R', 'phase_snr_db',
    'amp_hr_peak_purity' # <--- 新增
]

# 🔥 模型保存配置
MODEL_SAVE_DIR = f'./RandomForest_1125_v8'
MODEL_NAME = 'rf_rangebin_classifier'

if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)
    print(f"✅ 已创建模型保存目录: {MODEL_SAVE_DIR}")

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 【Step 0: 调用特征提取模块】
# ============================================================================
print("\n【Step 0】 调用特征提取模块...")
print(f"窗口长度: {WINDOW_DURATION_S} 秒")
print(f"实验文件数: {len(FILE_CONFIGS)}")

ALL_SEGMENTS_WITH_FEATURES = extract_features_from_all_files(
    FILE_CONFIGS, 
    window_duration_s=WINDOW_DURATION_S
)

if len(ALL_SEGMENTS_WITH_FEATURES) == 0:
    raise RuntimeError("❌ 特征提取失败，没有生成任何片段！")

print(f"\n✅ 特征提取完成，共生成 {len(ALL_SEGMENTS_WITH_FEATURES)} 个特征片段。")

# ============================================================================
# 【Step 1: 数据准备 - 构建样本】
# ============================================================================
print("\n【Step 1】 构建二分类样本...")

def build_samples_from_segments(segments_list, target_rx_index):
    """从片段列表构建二分类样本"""
    samples_data = []
    
    for seg_idx, segment in enumerate(segments_list):
        file_name = segment['original_file']
        segment_id = segment['segment_index']
        heart_count = segment['heart_count']
        range_one_hot = segment['range_one_hot'].squeeze()
        
        true_rb_index = np.argmax(range_one_hot) + 1
        
        rb_idx_0based = segment.get('rb_index_1based', true_rb_index) - 1
        if rb_idx_0based == 5:  # 修正：40cm 对应 rb_index=6 (0based=5)
            distance = 40
        elif rb_idx_0based == 6:
            distance = 50
        elif rb_idx_0based == 7:
            distance = 60
        else:
            distance = 50
        
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

X_raw, y_raw, meta_info = build_samples_from_segments(
    ALL_SEGMENTS_WITH_FEATURES, 
    TARGET_RX_INDEX
)

print(f"✅ 样本构建完成！")
print(f"   总样本数: {len(y_raw)}")
print(f"   特征维度: {X_raw.shape[1]}")
print(f"   正样本数: {np.sum(y_raw == 1)} ({np.sum(y_raw == 1)/len(y_raw)*100:.2f}%)")
print(f"   负样本数: {np.sum(y_raw == 0)} ({np.sum(y_raw == 0)/len(y_raw)*100:.2f}%)")

# ============================================================================
# 【Step 1.5: 数据质量评估】🔥 核心：使用 HR Peak Purity 修正 Q 分数
# ============================================================================
print("\n【Step 1.5】 数据质量评估和特征值输出...")

# --- A. 质量评分参数定义 ---
# 权重
W_SNR = 0.4
W_PURITY = 0.3
W_HEART_COUNT = 0.3

# 归一化目标/阈值
SNR_MIN = 10.0
SNR_MAX = 40.0
HR_PURITY_TARGET = 8.0  # 🔥 针对 HR 频带重新设定目标值 (原Purity Target=5.0)
HEART_COUNT_MIN = 1 

# --- B. 数据合并与计算 ---

# 1. 将原始特征数据转换为 DataFrame
X_df = pd.DataFrame(X_raw, columns=FEATURE_NAMES)
# 🔥 关键修复：将标签 y_raw 加入到元数据 DataFrame 中
meta_info['label'] = y_raw 
# 2. 合并特征和元数据
quality_df = pd.concat([X_df, meta_info.reset_index(drop=True)], axis=1)

def calculate_quality_score(row):
    """计算单个片段的质量分数 Q"""
    
    # 1. SNR 归一化 (w1 = 0.4)
    snr_norm = np.clip((row['amp_snr_db'] - SNR_MIN) / (SNR_MAX - SNR_MIN), 0.0, 1.0)
    
    # 2. HR Peak Purity 归一化 (w2 = 0.3)
    purity_norm = np.clip(row['amp_hr_peak_purity'] / HR_PURITY_TARGET, 0.0, 1.0) # <--- 使用新的 HR Purity
    
    # 3. Heart Count (w3 = 0.3)
    heart_norm = 1.0 if row['heart_count'] >= HEART_COUNT_MIN else 0.0
    
    # 最终分数 Q
    Q = W_SNR * snr_norm + W_PURITY * purity_norm + W_HEART_COUNT * heart_norm
    
    return Q

# 计算 Q 分数
quality_df['Quality_Score_Q'] = quality_df.apply(calculate_quality_score, axis=1)

# --- C. 按距离分组输出结果 ---

# 选出要展示的关键特征（包括用于计算 Q 的特征）
KEY_QUALITY_FEATURES = [
    'amp_snr_db', 
    'amp_purity_ratio', # 保留原有的 Purity 用于对比
    'amp_hr_peak_purity', # <--- 新增展示
    'amp_energy_resp', 
    'amp_energy_cardiac', 
    'amp_life_energy_ratio',
    'phase_diff_std',
    'amp_mean', 
    'amp_std', 
    'amp_p2p', 
    'amp_skewness', 
    'amp_kurtosis', 
    'phase_diff_range',
    'amp_freq_peak_pos',
    'amp_energy_ratio_C_R',
    'phase_energy_ratio_C_R', 
    'phase_snr_db'
]
DISPLAY_FEATURES = KEY_QUALITY_FEATURES + ['Quality_Score_Q', 'heart_count', 'label'] # 增加 label 用于方便聚合

# 分组计算平均值和标准差 (计算全样本的 mean/std)
grouped_quality = quality_df.groupby('distance')[DISPLAY_FEATURES].agg(['mean', 'std'])

print("\n" + "=" * 100)
print(f"📊 数据质量评估和关键特征值 (N_segments={quality_df['segment_global_id'].nunique()})")
print(f"   质量评分 Q 公式: Q = {W_SNR}*SNR_norm + {W_PURITY}*HR_Purity_norm (Target={HR_PURITY_TARGET}) + {W_HEART_COUNT}*HeartCount_binary")
print("=" * 100)
# ... (其余输出代码保持不变)

for dist in sorted(quality_df['distance'].unique()):
    # 🔥 正样本 (目标 Range Bin) 筛选
    dist_samples_df = quality_df[
        (quality_df['distance'] == dist) & (quality_df['label'] == 1)
    ].reset_index(drop=True)
    
    # 全样本筛选
    dist_all_samples_df = quality_df[quality_df['distance'] == dist]
    
    n_samples = dist_all_samples_df.shape[0]
    n_segments = dist_all_samples_df['segment_global_id'].nunique()
    
    print(f"\n--- 距离 {dist}cm 汇总 ({n_segments} 个片段, {n_samples} 个样本) ---")
    
    if dist_samples_df.empty:
        print("无正样本（目标 Range Bin 样本）可供展示。")
        continue

    # 只展示目标 Range Bin 的平均特征值 (label=1)
    rb_mean = dist_samples_df[DISPLAY_FEATURES].mean().to_dict()
    
    print(f"{'指标名称':<25} | {'目标RB平均值 (label=1)':<25} | {'全样本平均值 (label=0+1)':<25}")
    print("-" * 100)
    
    # 打印 Q 分数
    q_score_mean_rb = dist_samples_df['Quality_Score_Q'].mean()
    q_score_std_rb = dist_samples_df['Quality_Score_Q'].std()
    q_score_mean_all = dist_all_samples_df['Quality_Score_Q'].mean()
    q_score_std_all = dist_all_samples_df['Quality_Score_Q'].std()
    
    print(f"{'Quality_Score_Q':<25} | {q_score_mean_rb:.4f} ± {q_score_std_rb:.4f} | {q_score_mean_all:.4f} ± {q_score_std_all:.4f}")
    
    # 打印其他关键指标
    for feat in KEY_QUALITY_FEATURES:
        mean_val_rb = rb_mean[feat]
        std_val_rb = dist_samples_df[feat].std()
        
        mean_val_all = dist_all_samples_df[feat].mean()
        std_val_all = dist_all_samples_df[feat].std()
        
        # 确保标准差不是 NaN (对于只有一个样本的情况)
        std_val_rb = std_val_rb if not np.isnan(std_val_rb) else 0.0
        std_val_all = std_val_all if not np.isnan(std_val_all) else 0.0
        
        print(f"{feat:<25} | {mean_val_rb:.6e} ± {std_val_rb:.4e} | {mean_val_all:.6e} ± {std_val_all:.4e}")

print("=" * 100)
print(f"✅ 数据质量评估输出完成。")

# ============================================================================
# 【Step 2: K-Fold 交叉验证】🔥 核心修改
# ============================================================================
print(f"\n【Step 2】 {N_SPLITS}-Fold 交叉验证...")

# 创建分层标签
stratify_labels = meta_info['distance'].astype(str) + '_' + y_raw.astype(str)
groups = meta_info['segment_global_id'].values

# 🔥 使用 StratifiedGroupKFold
skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# 存储所有 fold 的结果
all_fold_results = []
all_test_predictions = []  # 存储所有测试集的预测（用于后续汇总）
fold_models = []  # 存储每个 fold 的模型

print(f"\n开始 {N_SPLITS}-Fold 交叉验证...")

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_raw, stratify_labels, groups=groups)):
    print("\n" + "=" * 70)
    print(f"📊 Fold {fold_idx + 1}/{N_SPLITS}")
    print("=" * 70)
    
    # 划分数据
    X_train_raw = X_raw[train_idx]
    X_test_raw = X_raw[test_idx]
    y_train = y_raw[train_idx]
    y_test = y_raw[test_idx]
    meta_train = meta_info.iloc[train_idx].reset_index(drop=True)
    meta_test = meta_info.iloc[test_idx].reset_index(drop=True)
    
    train_segments = set(meta_train['segment_global_id'].unique())
    test_segments = set(meta_test['segment_global_id'].unique())
    
    print(f"   训练集: {len(y_train)} 样本, {len(train_segments)} 片段")
    print(f"   测试集: {len(y_test)} 样本, {len(test_segments)} 片段")
    
    # 验证无泄露
    assert len(train_segments & test_segments) == 0, "❌ 训练集和测试集有重叠片段！"
    
    # 特征标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # 训练随机森林（使用固定参数，或者你可以加入网格搜索）
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )
    
    rf_model.fit(X_train, y_train)
    
    # 测试集预测
    y_test_pred = rf_model.predict(X_test)
    y_test_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # 评估单元格级性能
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_f1 = f1_score(y_test, y_test_pred)
    
    print(f"   单元格级 AUC: {test_auc:.4f}")
    print(f"   单元格级 F1: {test_f1:.4f}")
    
    # Range Bin 级别评估
    def evaluate_range_bin_level(y_true, y_proba, meta_df):
        """强制每片段选择一个 Range Bin"""
        segment_results = []
        
        for seg_id in meta_df['segment_global_id'].unique():
            seg_mask = meta_df['segment_global_id'] == seg_id
            seg_data = meta_df[seg_mask].iloc[0]
            
            seg_probas = y_proba[seg_mask]
            seg_rb_indices = meta_df[seg_mask]['rb_index'].values
            
            max_idx = np.argmax(seg_probas)
            pred_rb = seg_rb_indices[max_idx]
            true_rb = seg_data['true_rb_index']
            
            segment_results.append({
                'fold': fold_idx + 1,
                'file_name': seg_data['file_name'],
                'segment_id': seg_data['segment_id'],
                'distance': seg_data['distance'],
                'true_rb': true_rb,
                'pred_rb': pred_rb,
                'correct': (pred_rb == true_rb),
                'error': abs(pred_rb - true_rb),
                'max_proba': seg_probas[max_idx],
                'segment_global_id': seg_id
            })
        
        return pd.DataFrame(segment_results)
    
    rb_results = evaluate_range_bin_level(y_test, y_test_proba, meta_test)
    rb_accuracy = rb_results['correct'].mean()
    avg_error = rb_results['error'].mean()
    
    print(f"   Range Bin Top-1 准确率: {rb_accuracy:.2%}")
    print(f"   平均距离误差: {avg_error:.2f} Bins")
    
    # 保存这个 fold 的结果
    fold_result = {
        'fold': fold_idx + 1,
        'n_train': len(y_train),
        'n_test': len(y_test),
        'n_train_segments': len(train_segments),
        'n_test_segments': len(test_segments),
        'cell_auc': test_auc,
        'cell_f1': test_f1,
        'rb_accuracy': rb_accuracy,
        'rb_avg_error': avg_error
    }
    all_fold_results.append(fold_result)
    
    # 保存测试集预测（用于汇总）
    test_predictions = meta_test.copy()
    test_predictions['y_true'] = y_test
    test_predictions['y_pred'] = y_test_pred
    test_predictions['y_proba'] = y_test_proba
    test_predictions['fold'] = fold_idx + 1
    all_test_predictions.append(test_predictions)
    
    # 保存模型和 scaler
    fold_models.append({
        'model': rf_model,
        'scaler': scaler,
        'fold': fold_idx + 1
    })

# ============================================================================
# 【Step 3: 汇总所有 Fold 的结果】🔥 核心优势
# ============================================================================
print("\n" + "=" * 70)
print("📊 汇总所有 Fold 的结果")
print("=" * 70)

# 汇总 fold 性能
fold_summary_df = pd.DataFrame(all_fold_results)
print("\n各 Fold 性能:")
print(fold_summary_df.to_string(index=False))

print(f"\n平均性能 (跨 {N_SPLITS} Folds):")
print(f"   单元格级 AUC: {fold_summary_df['cell_auc'].mean():.4f} ± {fold_summary_df['cell_auc'].std():.4f}")
print(f"   单元格级 F1: {fold_summary_df['cell_f1'].mean():.4f} ± {fold_summary_df['cell_f1'].std():.4f}")
print(f"   Range Bin 准确率: {fold_summary_df['rb_accuracy'].mean():.2%} ± {fold_summary_df['rb_accuracy'].std():.2%}")
print(f"   平均距离误差: {fold_summary_df['rb_avg_error'].mean():.2f} ± {fold_summary_df['rb_avg_error'].std():.2f} Bins")

# 🔥 合并所有测试集预测（这样就有全部数据的预测了！）
all_predictions_df = pd.concat(all_test_predictions, ignore_index=True)

print(f"\n✅ 汇总完成！")
print(f"   总预测样本数: {len(all_predictions_df)}")
print(f"   覆盖的片段数: {all_predictions_df['segment_global_id'].nunique()}")

# 计算全局 Range Bin 准确率
def evaluate_all_range_bins(predictions_df):
    """基于所有预测计算 Range Bin 准确率"""
    segment_results = []
    
    for seg_id in predictions_df['segment_global_id'].unique():
        seg_data = predictions_df[predictions_df['segment_global_id'] == seg_id]
        
        seg_probas = seg_data['y_proba'].values
        seg_rb_indices = seg_data['rb_index'].values
        
        max_idx = np.argmax(seg_probas)
        pred_rb = seg_rb_indices[max_idx]
        true_rb = seg_data['true_rb_index'].iloc[0]
        
        segment_results.append({
            'file_name': seg_data['file_name'].iloc[0],
            'segment_id': seg_data['segment_id'].iloc[0],
            'distance': seg_data['distance'].iloc[0],
            'true_rb': true_rb,
            'pred_rb': pred_rb,
            'correct': (pred_rb == true_rb),
            'error': abs(pred_rb - true_rb),
            'fold': seg_data['fold'].iloc[0]
        })
    
    return pd.DataFrame(segment_results)

global_rb_results = evaluate_all_range_bins(all_predictions_df)

print("\n" + "=" * 70)
print("🎯 全局 Range Bin 性能（基于全部数据）:")
print("=" * 70)
print(f"   Top-1 准确率: {global_rb_results['correct'].mean():.2%}")
print(f"   平均距离误差: {global_rb_results['error'].mean():.2f} Bins")
print(f"   按距离分组准确率:")
for dist in sorted(global_rb_results['distance'].unique()):
    dist_acc = global_rb_results[global_rb_results['distance'] == dist]['correct'].mean()
    n_segments = len(global_rb_results[global_rb_results['distance'] == dist])
    print(f"     {dist}cm: {dist_acc:.2%} ({n_segments} 片段)")

# ============================================================================
# 【Step 4: 保存结果】
# ============================================================================
print("\n【Step 4】 保存结果...")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# 保存 fold 汇总
fold_summary_df.to_csv(
    os.path.join(MODEL_SAVE_DIR, f'fold_summary_{timestamp}.csv'), 
    index=False
)

# 保存全局 Range Bin 结果
global_rb_results.to_csv(
    os.path.join(MODEL_SAVE_DIR, f'global_rb_results_{timestamp}.csv'), 
    index=False
)

# 保存所有预测（用于后续心率估计）
all_predictions_df.to_csv(
    os.path.join(MODEL_SAVE_DIR, f'all_predictions_{timestamp}.csv'), 
    index=False
)

# 保存每个 fold 的模型
for fold_data in fold_models:
    fold_num = fold_data['fold']
    joblib.dump(
        fold_data['model'], 
        os.path.join(MODEL_SAVE_DIR, f'rf_model_fold{fold_num}_{timestamp}.pkl')
    )
    joblib.dump(
        fold_data['scaler'], 
        os.path.join(MODEL_SAVE_DIR, f'scaler_fold{fold_num}_{timestamp}.pkl')
    )

# 保存元信息
metadata = {
    'model_name': MODEL_NAME,
    'timestamp': timestamp,
    'n_splits': N_SPLITS,
    'feature_names': FEATURE_NAMES,
    'window_duration_s': WINDOW_DURATION_S,
    'random_state': RANDOM_STATE,
    'global_rb_accuracy': float(global_rb_results['correct'].mean()),
    'global_rb_avg_error': float(global_rb_results['error'].mean()),
    'fold_performance': fold_summary_df.to_dict('records')
}

with open(os.path.join(MODEL_SAVE_DIR, f'metadata_{timestamp}.json'), 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"✅ 所有结果已保存至: {MODEL_SAVE_DIR}")

# ============================================================================
# 【Step 5: 可视化】
# ============================================================================
print("\n【Step 5】 生成可视化...")

# 1. 全局混淆矩阵（Range Bin 级别）
true_rb_indices = global_rb_results['true_rb'].values
pred_rb_indices = global_rb_results['pred_rb'].values
all_rb_indices = np.arange(1, 22)

cm_rb = confusion_matrix(true_rb_indices, pred_rb_indices, labels=all_rb_indices)

plt.figure(figsize=(12, 10))
sns.heatmap(cm_rb, annot=True, fmt='d', cmap='Blues', 
            xticklabels=all_rb_indices, yticklabels=all_rb_indices)
plt.xlabel('Predicted Range Bin (base-1)', fontsize=14)
plt.ylabel('True Range Bin (base-1)', fontsize=14)
plt.title(f'Confusion Matrix - {N_SPLITS}-Fold CV (All Data)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_SAVE_DIR, 'confusion_matrix_global.png'), 
            dpi=300, bbox_inches='tight')
plt.close()

# 2. Fold 性能对比
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.bar(fold_summary_df['fold'], fold_summary_df['rb_accuracy'] * 100)
ax1.axhline(fold_summary_df['rb_accuracy'].mean() * 100, 
            color='r', linestyle='--', label='Mean')
ax1.set_xlabel('Fold')
ax1.set_ylabel('Range Bin Accuracy (%)')
ax1.set_title('Range Bin Accuracy by Fold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

ax2 = axes[1]
ax2.bar(fold_summary_df['fold'], fold_summary_df['rb_avg_error'])
ax2.axhline(fold_summary_df['rb_avg_error'].mean(), 
            color='r', linestyle='--', label='Mean')
ax2.set_xlabel('Fold')
ax2.set_ylabel('Average Error (Bins)')
ax2.set_title('Average Prediction Error by Fold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_SAVE_DIR, 'fold_performance_comparison.png'), 
            dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ 可视化已保存")

# ============================================================================
# 【总结】
# ============================================================================
print("\n" + "=" * 70)
print("✅ K-Fold 交叉验证完成！")
print("=" * 70)
print(f"\n🎯 关键优势:")
print(f"   1. 使用了全部 {len(ALL_SEGMENTS_WITH_FEATURES)} 个片段进行测试")
print(f"   2. 每个片段都有一次测试机会（在对应的 fold 中）")
print(f"   3. 可以在全部数据上对比手工选择、RF无约束、RF约束")
print(f"\n📊 全局性能:")
print(f"   Range Bin 准确率: {global_rb_results['correct'].mean():.2%}")
print(f"   平均距离误差: {global_rb_results['error'].mean():.2f} Bins")
print(f"\n📁 输出文件:")
print(f"   - fold_summary_*.csv: 各 fold 性能汇总")
print(f"   - global_rb_results_*.csv: 全局 Range Bin 预测结果")
print(f"   - all_predictions_*.csv: 所有预测详情（用于心率估计）")
print(f"   - rf_model_fold*_*.pkl: 各 fold 的模型")
print("=" * 70)

