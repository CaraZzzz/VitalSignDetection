"""
雷达心率估计系统 - 主程序

使用方法:
    python main.py --hr_method fft --loc_method manual

参数:
    --hr_method: 心率估计方法
        可选: fft, stft, wavelet, dct, vmd, emd, eemd
    
    --loc_method: 人体定位方法
        可选: manual, random_forest, cfar
"""
"""
相比v1：增加允许调用，便于后续一起跑结果
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 导入配置
import config

# 导入工具函数
from utils.data_utils import SlidingWindowSlicer, NeuLogHeartRateCalculator
from utils.evaluation import HeartRateEvaluator

# 导入心率估计算法
from algorithms.heart_rate import (
    FFTHeartRateEstimator,
    STFTHeartRateEstimator,
    WaveletHeartRateEstimator,
    DCTHeartRateEstimator,
    VMDHeartRateEstimator,
    EMDHeartRateEstimator,
    EEMDHeartRateEstimator,
)

# 导入人体定位算法
from algorithms.localization import (
    ManualLocalization,
    RandomForestLocalization,
    CFARLocalization,
)


def get_heart_rate_estimator(method: str):
    """
    根据方法名称获取心率估计器
    
    参数:
        method: 方法名称
    
    返回:
        心率估计器实例
    """
    method = method.lower()
    
    if method == 'fft':
        return FFTHeartRateEstimator(
            fs=config.SAMPLING_RATE,
            detrend=config.FFT_DETREND,
            apply_window=config.FFT_APPLY_WINDOW
        )
    elif method == 'stft':
        return STFTHeartRateEstimator(
            fs=config.SAMPLING_RATE,
            detrend=config.FFT_DETREND,
            nperseg=config.STFT_NPERSEG,
            noverlap=config.STFT_NOVERLAP,
            window=config.STFT_WINDOW
        )
    elif method == 'wavelet':
        return WaveletHeartRateEstimator(
            fs=config.SAMPLING_RATE,
            detrend=config.FFT_DETREND,
            wavelet=config.WAVELET_TYPE
        )
    elif method == 'dct':
        return DCTHeartRateEstimator(
            fs=config.SAMPLING_RATE,
            detrend=config.DCT_DETREND
        )
    elif method == 'vmd':
        return VMDHeartRateEstimator(
            fs=config.SAMPLING_RATE,
            detrend=config.FFT_DETREND,
            alpha=config.VMD_ALPHA,
            tau=config.VMD_TAU,
            K=config.VMD_K,
            DC=config.VMD_DC,
            init=config.VMD_INIT,
            tol=config.VMD_TOL
        )
    elif method == 'emd':
        return EMDHeartRateEstimator(
            fs=config.SAMPLING_RATE,
            detrend=config.FFT_DETREND,
            max_imf=config.EMD_MAX_IMF
        )
    elif method == 'eemd':
        return EEMDHeartRateEstimator(
            fs=config.SAMPLING_RATE,
            detrend=config.FFT_DETREND,
            trials=config.EEMD_TRIALS,
            noise_strength=config.EEMD_NOISE_STRENGTH,
            max_imf=config.EEMD_MAX_IMF
        )
    else:
        raise ValueError(f"未知的心率估计方法: {method}")


def get_localization_method(method: str):
    """
    根据方法名称获取人体定位方法
    
    参数:
        method: 方法名称
    
    返回:
        人体定位方法实例
    """
    method = method.lower()
    
    if method == 'manual':
        return ManualLocalization(rb_mapping=config.MANUAL_RB_MAPPING)
    elif method == 'random_forest' or method == 'rf':
        return RandomForestLocalization(
            model_path=config.RF_MODEL_PATH,
            scaler_path=config.RF_SCALER_PATH,
            metadata_path=config.RF_METADATA_PATH
        )
    elif method == 'cfar':
        return CFARLocalization(
            G_R=config.CFAR_G_R,
            G_D=config.CFAR_G_D,
            L_R=config.CFAR_L_R,
            L_D=config.CFAR_L_D,
            P_fa=config.CFAR_P_FA,
            n_doppler_fft=config.CFAR_N_DOPPLER_FFT
        )
    else:
        raise ValueError(f"未知的人体定位方法: {method}")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='雷达心率估计系统')
    parser.add_argument('--hr_method', type=str, default='fft',
                       choices=['fft', 'stft', 'wavelet', 'dct', 'vmd', 'emd', 'eemd'],
                       help='心率估计方法')
    parser.add_argument('--loc_method', type=str, default='manual',
                       choices=['manual', 'random_forest', 'rf', 'cfar'],
                       help='人体定位方法')
    parser.add_argument('--window', type=int, default=None,
                       help='窗口长度（秒），覆盖config中的设置')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录，覆盖config中的设置')
    
    args = parser.parse_args()
    
    # 覆盖配置（如果提供了参数）
    if args.window is not None:
        config.WINDOW_DURATION_S = args.window
    if args.output_dir is not None:
        config.OUTPUT_DIR = args.output_dir
    
    print("=" * 70)
    print("🚀 雷达心率估计与评估系统")
    print("=" * 70)
    print(f"\n配置参数:")
    print(f"  心率估计方法: {args.hr_method.upper()}")
    print(f"  人体定位方法: {args.loc_method.upper()}")
    print(f"  窗口长度: {config.WINDOW_DURATION_S} 秒")
    print(f"  步长: {config.STEP_SIZE_S} 秒")
    print(f"  采样率: {config.SAMPLING_RATE} Hz")
    print(f"  心率频率范围: {config.HR_FREQ_RANGE[0]}-{config.HR_FREQ_RANGE[1]} Hz "
          f"({config.HR_FREQ_RANGE[0]*60:.0f}-{config.HR_FREQ_RANGE[1]*60:.0f} bpm)")
    
    # 调用实验运行接口
    evaluation_results, computation_times = run_experiment(
        hr_method=args.hr_method,
        loc_method=args.loc_method,
        window_s=config.WINDOW_DURATION_S,
        output_dir=config.OUTPUT_DIR
    )
    
    # 完成
    print("\n" + "=" * 70)
    print("✅ 心率估计系统运行完成！")
    print("=" * 70)
    print(f"\n生成的输出文件:")
    print(f"  1. heart_rate_results.csv - 详细结果表")
    print(f"  2. evaluation_report.json - 评估报告")
    print(f"  3. bland_altman_plot.png - Bland-Altman图")
    print(f"  4. scatter_with_regression.png - 散点图+回归")
    print(f"  5. boxplot_by_distance.png - 距离分组箱线图")
    print(f"  6. error_distribution.png - 误差分布图")
    print(f"\n所有文件保存在: {config.OUTPUT_DIR}")
    print(f"平均每片段计算时间: {np.mean(computation_times)*1000:.2f} ms")
    print("=" * 70)


def run_experiment(hr_method: str, loc_method: str, window_s: int, output_dir: str):
    """
    可编程调用的实验运行接口
    
    参数:
        hr_method: 心率估计方法
        loc_method: 人体定位方法
        window_s: 窗口长度（秒）
        output_dir: 输出目录
    
    返回:
        evaluation_results: 评估结果字典
        computation_times: 计算时间列表（每个片段的运行时间）
    """
    import time
    
    # 设置配置
    config.WINDOW_DURATION_S = window_s
    config.OUTPUT_DIR = output_dir
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 初始化模块
    slicer = SlidingWindowSlicer(
        window_s=config.WINDOW_DURATION_S,
        step_s=config.STEP_SIZE_S,
        fs=config.SAMPLING_RATE
    )
    
    hr_estimator = get_heart_rate_estimator(hr_method)
    loc_method_obj = get_localization_method(loc_method)
    neulog_calculator = NeuLogHeartRateCalculator(fs=config.SAMPLING_RATE)
    evaluator = HeartRateEvaluator(output_dir=config.OUTPUT_DIR)
    
    # 批量处理文件
    all_results = []
    computation_times = []  # 记录每个片段的运行时间
    
    for file_idx, file_info in enumerate(config.FILE_CONFIGS):
        file_unique_id = f"{file_info['distance']}cm_{os.path.basename(file_info['path'])}"
        
        # 滑动窗口切片
        segments = slicer.slice_file(file_info)
        
        if len(segments) == 0:
            continue
        
        # 逐片段处理
        for seg in segments:
            # 选择Range Bin
            predefined_rb = seg['true_rb_index'] - 1
            pred_rb_0based, selection_info = loc_method_obj.select_range_bin(seg, predefined_rb)
            pred_rb = pred_rb_0based + 1
            
            # 提取选中Range Bin的相位数据
            rx_idx_0based = seg['rx_index'] - 1
            phase_data = seg['filtered_phase'][pred_rb_0based, rx_idx_0based, :]
            
            # 雷达心率估计（计时）
            start_time = time.perf_counter()
            radar_hr_result = hr_estimator.estimate(phase_data, freq_range=config.HR_FREQ_RANGE)
            elapsed_time = time.perf_counter() - start_time
            computation_times.append(elapsed_time)
            
            # NeuLog心率计算
            neulog_hr_result = neulog_calculator.calculate(
                seg['peak_mask'], 
                window_s=config.WINDOW_DURATION_S
            )
            
            # 记录结果
            result = {
                'file_name': seg['file_name'],
                'file_unique_id': seg['file_unique_id'],
                'file_path': seg['file_path'],
                'segment_id': seg['segment_id'],
                'distance': seg['distance'],
                'true_rb_index': seg['true_rb_index'],
                'pred_rb_index': pred_rb,
                'pred_correct': (pred_rb == seg['true_rb_index']),
                'rb_selection_method': selection_info['method'],
                'rb_selection_confidence': selection_info['confidence'],
                'radar_hr_bpm': radar_hr_result['heart_rate_bpm'],
                'radar_peak_freq_hz': radar_hr_result['peak_freq_hz'],
                'radar_peak_magnitude': radar_hr_result['peak_magnitude'],
                'neulog_hr_count': neulog_hr_result['hr_count'],
                'neulog_hr_interval': neulog_hr_result['hr_interval'],
                'neulog_num_peaks': neulog_hr_result['num_peaks'],
                'start_time_s': seg['start_time_s'],
                'end_time_s': seg['end_time_s'],
                'computation_time_s': elapsed_time
            }
            
            all_results.append(result)
    
    # 汇总结果
    results_df = pd.DataFrame(all_results)
    
    # 保存详细结果
    results_path = os.path.join(config.OUTPUT_DIR, 'heart_rate_results.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    
    # 多层级评估
    evaluation_results = evaluator.evaluate(results_df)
    
    # 保存评估报告
    report = {
        'config': {
            'hr_method': hr_method,
            'loc_method': loc_method,
            'window_duration_s': window_s,
            'step_size_s': config.STEP_SIZE_S,
            'sampling_rate': config.SAMPLING_RATE,
            'freq_range_hz': config.HR_FREQ_RANGE,
        },
        'metrics': evaluation_results,
        'computation_time': {
            'mean_per_segment_s': np.mean(computation_times),
            'std_per_segment_s': np.std(computation_times),
            'min_per_segment_s': np.min(computation_times),
            'max_per_segment_s': np.max(computation_times),
            'total_segments': len(computation_times)
        }
    }
    
    report_path = os.path.join(config.OUTPUT_DIR, 'evaluation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    return evaluation_results, computation_times


if __name__ == "__main__":
    main()