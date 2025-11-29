##------------------------------------------------------
## v2相比v1，是读取所有数据并提取特征
## v3相比v2，多了示例输出
## v4是v3的封装版
## v5做了时间相关的特征的归一化
## v6加了HR Peak Purity
##------------------------------------------------------
"""
特征提取模块 - 封装版（时间特征归一化）
用于从 .mat 文件中提取雷达特征，供机器学习使用
"""

import numpy as np
import scipy.stats as stats
import scipy.fft as fft
import h5py
import os
from typing import List, Dict, Any

# ========================================================================
# 辅助函数：频谱特征计算 (新增 HR Peak Purity)
# ========================================================================
def calculate_spectrum_features(time_series, fs):
    """ 对时序数据计算频谱特征、SNR 和 HR Peak Purity。"""
    N = len(time_series)
    
    Y = fft.fft(time_series)
    P2 = np.abs(Y / N)
    P1 = P2[:N // 2 + 1]
    P1[1:-1] = 2 * P1[1:-1]
    E_P = P1 ** 2  # 能量谱 (PSD)
    f = fs * np.arange(0, N // 2 + 1) / N
    
    # 定义频带索引 (Hz)
    idx_resp = np.where((f >= 0.15) & (f <= 0.5))[0]
    idx_cardiac = np.where((f >= 0.8) & (f <= 2.0))[0]
    idx_life_signal = np.where((f >= 0.15) & (f <= 2.0))[0] 
    idx_noise = np.where((f >= 3.0) & (f <= 8.0))[0]        
    idx_total = np.where(f > 0)[0] 
    
    # 零能量处理
    if not idx_total.size: 
        return {
            'energy_resp': 0.0, 'energy_cardiac': 0.0, 'energy_total_ratio': 0.0, 
            'freq_peak_pos': 0.0, 'purity_ratio': 0.0, 'energy_ratio_C_R': 0.0, 
            'snr_db': -100.0, 'hr_peak_purity': 0.0 # <--- 新增 hr_peak_purity 零值
        }

    # 能量计算
    E_R = np.sum(E_P[idx_resp])
    E_C = np.sum(E_P[idx_cardiac])
    E_T = np.sum(E_P[idx_total])
    
    # SNR 估计
    signal_energy = np.sum(E_P[idx_life_signal])
    noise_avg_psd = np.mean(E_P[idx_noise]) if idx_noise.size > 0 else E_T / idx_total.size
    
    # 使用 10 * log10 计算 SNR_dB
    snr_db = 10 * np.log10(signal_energy / noise_avg_psd) if noise_avg_psd > 1e-12 else (100.0 if signal_energy > 1e-12 else -100.0)

    # 其他特征
    idx_life = np.where((f >= 0.15) & (f <= 2.0))[0]
    freq_peak_pos = f[idx_life[np.argmax(P1[idx_life])]] if idx_life.size > 0 else 0.0
    
    P1_non_DC = P1[idx_total]
    max_P1 = np.max(P1_non_DC)
    mean_P1 = np.mean(P1_non_DC)
    purity_ratio = max_P1 / mean_P1 if mean_P1 != 0 else 0.0
    
    # 🔥 新增 HR Peak Purity 计算 (心率带内最大峰值P1 / 心率带内平均P1)
    if not idx_cardiac.size or np.sum(P1[idx_cardiac]) == 0:
        hr_peak_purity = 0.0
    else:
        P1_cardiac = P1[idx_cardiac]
        max_P1_cardiac = np.max(P1_cardiac)
        mean_P1_cardiac = np.mean(P1_cardiac)
        # 使用 max/mean 来衡量心率峰值在心率带内的突出程度
        hr_peak_purity = max_P1_cardiac / mean_P1_cardiac if mean_P1_cardiac != 0 else 0.0
    
    energy_ratio_C_R = E_C / E_R if E_R != 0 else 0.0
    energy_total_ratio = (E_R + E_C) / E_T
    
    return {
        'energy_resp': E_R, 
        'energy_cardiac': E_C, 
        'energy_total_ratio': energy_total_ratio, 
        'freq_peak_pos': freq_peak_pos, 
        'purity_ratio': purity_ratio, 
        'energy_ratio_C_R': energy_ratio_C_R, 
        'snr_db': snr_db,
        'hr_peak_purity': hr_peak_purity # <--- 返回新的特征
    }

# ========================================================================
# 🔥 新增：时间特征归一化函数
# ========================================================================
def normalize_time_features(segment_features: Dict[str, np.ndarray], window_duration_s: float) -> Dict[str, np.ndarray]:
    """
    将时间相关的能量特征归一化到单位时间（1秒）
    
    参数:
        segment_features: 包含所有特征的字典
        window_duration_s: 窗口时长（秒）
    
    返回:
        归一化后的特征字典
    """
    # 定义需要归一化的时间相关特征（能量特征）
    time_dependent_features = [
        'amp_energy_resp',      # 呼吸频段能量
        'amp_energy_cardiac',   # 心跳频段能量
    ]
    
    # print(f"   ⏱️ 归一化时间特征（窗口时长: {window_duration_s}s）...")
    
    for feat_name in time_dependent_features:
        if feat_name in segment_features:
            # 除以窗口时长，归一化到 1 秒
            segment_features[feat_name] = segment_features[feat_name] / window_duration_s
    
    return segment_features

# ========================================================================
# 提取 R x nVX 单元格特征 (新增 HR Peak Purity)
# ========================================================================
def extract_segment_features(segment_data: Dict[str, Any], fs: float, window_duration_s: float) -> Dict[str, Any]:
    """ 对一个片段字典进行特征提取（包含归一化）。"""
    
    Amp_data = segment_data['magnitude_fft']
    Phase_data = segment_data['trimmed_unfiltered_phase_fft']
    
    R, nVX, T = Amp_data.shape
    
    # 初始化特征矩阵 - 增加新的特征键
    feature_keys = [
        'amp_mean', 'amp_std', 'amp_p2p', 'amp_skewness', 'amp_kurtosis', 
        'phase_diff_std', 'phase_diff_range', 
        'amp_energy_resp', 'amp_energy_cardiac', 'amp_life_energy_ratio', 
        'amp_freq_peak_pos', 'amp_purity_ratio', 'amp_energy_ratio_C_R', 'amp_snr_db',
        'phase_energy_ratio_C_R', 'phase_snr_db',
        'amp_hr_peak_purity' # <--- 新增特征键
    ]
    
    segment_features = {key: np.zeros((R, nVX)) for key in feature_keys}
    
    # 核心：双层循环提取 R x nVX 单元格特征
    for r in range(R):
        for v in range(nVX):
            
            Amp_ts = Amp_data[r, v, :]
            Phase_ts = Phase_data[r, v, :]
            
            # 1. 振幅统计特征 (略...)
            segment_features['amp_mean'][r, v] = np.mean(Amp_ts)
            segment_features['amp_std'][r, v] = np.std(Amp_ts)
            segment_features['amp_p2p'][r, v] = np.max(Amp_ts) - np.min(Amp_ts)
            segment_features['amp_skewness'][r, v] = stats.skew(Amp_ts)
            segment_features['amp_kurtosis'][r, v] = stats.kurtosis(Amp_ts)
            
            # 2. 相位差统计特征 (略...)
            phase_diff = np.diff(Phase_ts)
            segment_features['phase_diff_std'][r, v] = np.std(phase_diff)
            segment_features['phase_diff_range'][r, v] = np.max(phase_diff) - np.min(phase_diff)
            
            # 3. 振幅频谱特征
            amp_spec_feats = calculate_spectrum_features(Amp_ts, fs)
            segment_features['amp_energy_resp'][r, v] = amp_spec_feats['energy_resp']
            segment_features['amp_energy_cardiac'][r, v] = amp_spec_feats['energy_cardiac']
            segment_features['amp_life_energy_ratio'][r, v] = amp_spec_feats['energy_total_ratio']
            segment_features['amp_freq_peak_pos'][r, v] = amp_spec_feats['freq_peak_pos']
            segment_features['amp_purity_ratio'][r, v] = amp_spec_feats['purity_ratio']
            segment_features['amp_energy_ratio_C_R'][r, v] = amp_spec_feats['energy_ratio_C_R']
            segment_features['amp_snr_db'][r, v] = amp_spec_feats['snr_db']
            segment_features['amp_hr_peak_purity'][r, v] = amp_spec_feats['hr_peak_purity'] # <--- 捕获新的特征
            
            # 4. 相位频谱特征
            phase_spec_feats = calculate_spectrum_features(Phase_ts, fs)
            segment_features['phase_energy_ratio_C_R'][r, v] = phase_spec_feats['energy_ratio_C_R']
            segment_features['phase_snr_db'][r, v] = phase_spec_feats['snr_db']
    
    # 🔥 应用时间特征归一化
    segment_features = normalize_time_features(segment_features, window_duration_s)
    
    # 合并特征和原始标签，并删除时序数据以节省内存
    del segment_data['phase_fft_filtered'] 
    del segment_data['trimmed_unfiltered_phase_fft']
    del segment_data['magnitude_fft']
    del segment_data['peak_mask']
    
    return {**segment_data, **segment_features}

# ========================================================================
# 核心处理函数：读取、切片并提取特征
# ========================================================================
def process_single_file(file_info: Dict[str, Any], window_duration_s: float) -> List[Dict[str, Any]]:
    """ 处理单个 .mat 文件，返回其所有特征片段的列表。"""
    
    file_path = file_info['path'].replace('\\', '/')
    target_distance = file_info['distance']
    target_rb_index = file_info['rb_index'] 
    
    file_segments_list = []
    
    if not os.path.exists(file_path):
        print(f"❌ 文件未找到: {file_path}")
        return []

    try:
        with h5py.File(file_path, 'r') as f:
            
            target_fs = f['target_fs'][()].item()
            
            # 动态计算窗口帧数
            window_size_frames = int(window_duration_s * target_fs)
            step_size_frames = window_size_frames # 无重叠
            
            if int(target_fs) != 20:
                print(f"⚠️ 警告: 文件 {os.path.basename(file_path)} 的采样率是 {target_fs:.1f} Hz。窗口大小为 {window_size_frames} 帧。")

            # 1. 读取并转置数据 (T x nVX x R) -> (R x nVX x T)
            raw_filtered_phase = f['filtered_1_2hz_phase_fft'][:]
            raw_unfiltered_phase = f['trimmed_unfiltered_phase_fft'][:]
            raw_magnitude_fft = f['magnitude_range_fft_trimmed'][:]
            filtered_peak_mask = f['filtered_peak_mask'][:].squeeze().astype(np.int8).reshape(1, -1)
            
            filtered_1_2hz_phase_fft = np.transpose(raw_filtered_phase, (2, 1, 0))
            trimmed_unfiltered_phase_fft = np.transpose(raw_unfiltered_phase, (2, 1, 0))
            magnitude_range_fft_trimmed = np.transpose(raw_magnitude_fft, (2, 1, 0))
            
            R = filtered_1_2hz_phase_fft.shape[0] # Range Bin 总数 (R=21)
            T = filtered_1_2hz_phase_fft.shape[2] # 时间帧总数
            
            # 2. 生成 Range One-Hot 标签 (1, R)
            target_index_0based = int(target_rb_index) - 1
            range_one_hot_label = np.zeros((1, R), dtype=np.float32)
            if 0 <= target_index_0based < R:
                range_one_hot_label[0, target_index_0based] = 1.0
            else:
                 print(f"❌ 错误: Range Bin 索引 {target_rb_index} 超出数据范围 R={R}")
                 return []
            
            # 3. 滑动时间窗切片和特征提取
            num_segments = T // window_size_frames
            
            print(f"✅ 文件: {os.path.basename(file_path)} | 距离: {target_distance}cm | T={T} | 窗口: {window_duration_s}s ({window_size_frames}帧) | 生成 {num_segments} 个片段...")

            for i in range(num_segments):
                start_frame = i * step_size_frames
                end_frame = start_frame + window_size_frames
                
                # 裁剪数据
                segment_data = {
                    'phase_fft_filtered': filtered_1_2hz_phase_fft[:, :, start_frame:end_frame],
                    'trimmed_unfiltered_phase_fft': trimmed_unfiltered_phase_fft[:, :, start_frame:end_frame],
                    'magnitude_fft': magnitude_range_fft_trimmed[:, :, start_frame:end_frame],
                    'peak_mask': filtered_peak_mask[:, start_frame:end_frame],
                    'range_one_hot': range_one_hot_label,
                    'heart_count': int(filtered_peak_mask[:, start_frame:end_frame].sum()),
                    'original_file': os.path.basename(file_path), 
                    'segment_index': i + 1,
                    'rb_index_1based': target_rb_index, 
                    'rx_index_1based_example': file_info.get('rx_index_example', 1) 
                }
                
                # 🔥 特征提取（传入 window_duration_s）
                extracted_segment = extract_segment_features(segment_data, target_fs, window_duration_s)
                file_segments_list.append(extracted_segment)
            
            return file_segments_list
            
    except Exception as e:
        print(f"❌ 处理文件 {os.path.basename(file_path)} 时发生异常: {e}")
        return []

# ========================================================================
# 主函数：批量处理所有文件
# ========================================================================
def extract_features_from_all_files(file_configs: List[Dict], window_duration_s: float = 30) -> List[Dict[str, Any]]:
    """
    批量处理所有实验文件，提取特征
    
    参数:
        file_configs: 文件配置列表，每个元素包含 path, distance, rb_index
        window_duration_s: 窗口长度（秒）
    
    返回:
        ALL_SEGMENTS_WITH_FEATURES: 所有片段的特征列表
    """
    
    ALL_SEGMENTS_WITH_FEATURES = []
    total_processed_files = 0

    print(f"--- 开始批量处理 {len(file_configs)} 组实验数据 (窗口长度: {window_duration_s} 秒) ---")
    print(f"⏱️ 时间相关特征将被归一化到单位时间（1秒）")
    
    for info in file_configs:
        result = process_single_file(info, window_duration_s) 
        if result:
            ALL_SEGMENTS_WITH_FEATURES.extend(result)
            total_processed_files += 1

    # 最终结果总结
    print("\n" + "=" * 50)
    print(f"✅ 批量特征提取完成！")
    print(f"成功处理文件数: {total_processed_files} / {len(file_configs)}")
    total_segments = len(ALL_SEGMENTS_WITH_FEATURES)
    print(f"总共生成的 {window_duration_s} 秒特征片段数: {total_segments}")
    print("=" * 50)

    # QC 检查
    if total_segments > 0:
        first_segment = ALL_SEGMENTS_WITH_FEATURES[0]
        r_idx = first_segment['rb_index_1based'] - 1
        v_idx = first_segment['rx_index_1based_example'] - 1
        
        print("\n--- 首个片段特征 QC 检查 ---")
        print(f"  来源文件: {first_segment['original_file']}")
        print(f"  片段索引: {first_segment['segment_index']}")
        print(f"  心跳计数: {first_segment['heart_count']}")
        print(f"  核心特征形状: {first_segment['amp_mean'].shape}")
        print(f"\n  单元格 R={first_segment['rb_index_1based']}, RX={first_segment['rx_index_1based_example']} 的特征示例:")
        print(f"    -> 振幅均值: {first_segment['amp_mean'][r_idx, v_idx]:.4f}")
        print(f"    -> 归一化呼吸能量 (per sec): {first_segment['amp_energy_resp'][r_idx, v_idx]:.6f}")
        print(f"    -> 归一化心跳能量 (per sec): {first_segment['amp_energy_cardiac'][r_idx, v_idx]:.6f}")
        print(f"    -> Amp SNR (dB): {first_segment['amp_snr_db'][r_idx, v_idx]:.4f}")
        print(f"    -> Phase SNR (dB): {first_segment['phase_snr_db'][r_idx, v_idx]:.4f}")
    
    return ALL_SEGMENTS_WITH_FEATURES


# ========================================================================
# 如果直接运行此文件（用于测试）
# ========================================================================
if __name__ == "__main__":
    
    # 测试用文件配置
    TEST_FILE_CONFIGS = [
        {'path': r"D:\MSc\Dissertation\Data\250902\test2\RangeFFT\RangeBin6_RX1\test2_NeuLogRadar_aligned_trimmed.mat", 
         'distance': 40, 'rb_index': 6, 'rx_index_example': 1},
    ]
    
    # 调用主函数
    test_results = extract_features_from_all_files(TEST_FILE_CONFIGS, window_duration_s=30)
    
    print(f"\n✅ 测试完成！生成了 {len(test_results)} 个特征片段。")