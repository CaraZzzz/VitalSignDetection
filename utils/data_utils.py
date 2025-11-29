"""
数据处理工具函数
"""
import os
import h5py
import numpy as np
from typing import List, Dict, Any


class SlidingWindowSlicer:
    """
    对单个.mat文件进行滑动窗口切片
    """
    
    def __init__(self, window_s: float = 30, step_s: float = 15, fs: float = 20):
        """
        初始化滑动窗口切片器
        
        参数:
            window_s: 窗口长度（秒）
            step_s: 步长（秒）
            fs: 采样率 (Hz)
        """
        self.window_s = window_s
        self.step_s = step_s
        self.fs = fs
        self.window_frames = int(window_s * fs)
        self.step_frames = int(step_s * fs)
    
    def slice_file(self, file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        对单个文件进行滑动窗口切片
        
        参数:
            file_info: 文件配置字典 (包含path, distance, rb_index, rx_index_example)
        
        返回:
            segments: 切片后的片段列表
        """
        file_path = file_info['path'].replace('\\', '/')
        distance = file_info['distance']
        true_rb_index = file_info['rb_index']
        rx_index = file_info['rx_index_example']
        
        # 创建文件唯一标识：距离_文件名
        file_name = os.path.basename(file_path)
        file_unique_id = f"{distance}cm_{file_name}"
        
        if not os.path.exists(file_path):
            print(f"❌ 文件未找到: {file_path}")
            return []
        
        segments = []
        
        try:
            with h5py.File(file_path, 'r') as f:
                # 读取关键数据
                target_fs = f['target_fs'][()].item()
                if int(target_fs) != self.fs:
                    print(f"⚠️ 警告: 文件采样率 {target_fs} Hz 与配置不符 ({self.fs} Hz)")
                
                # 读取数据并转置 (T x nVX x R) -> (R x nVX x T)
                raw_filtered_phase = f['filtered_1_2hz_phase_fft'][:]
                raw_magnitude = f['magnitude_range_fft_trimmed'][:]
                raw_unfiltered_phase = f['trimmed_unfiltered_phase_fft'][:]
                filtered_peak_mask = f['filtered_peak_mask'][:].squeeze().astype(np.int8)
                
                # 转置
                filtered_phase = np.transpose(raw_filtered_phase, (2, 1, 0))
                magnitude = np.transpose(raw_magnitude, (2, 1, 0))
                unfiltered_phase = np.transpose(raw_unfiltered_phase, (2, 1, 0))
                
                R, nVX, T = filtered_phase.shape
                
                # 滑动窗口切片
                num_segments = (T - self.window_frames) // self.step_frames + 1
                
                print(f"  文件: {file_unique_id} | "
                      f"总帧数: {T} | 生成片段数: {num_segments}")
                
                for i in range(num_segments):
                    start_frame = i * self.step_frames
                    end_frame = start_frame + self.window_frames
                    
                    # 提取片段
                    segment = {
                        'filtered_phase': filtered_phase[:, :, start_frame:end_frame],
                        'magnitude': magnitude[:, :, start_frame:end_frame],
                        'unfiltered_phase': unfiltered_phase[:, :, start_frame:end_frame],
                        'peak_mask': filtered_peak_mask[start_frame:end_frame],
                        'file_name': file_name,
                        'file_unique_id': file_unique_id,
                        'file_path': file_path,
                        'distance': distance,
                        'true_rb_index': true_rb_index,
                        'rx_index': rx_index,
                        'segment_id': i + 1,
                        'start_time_s': start_frame / self.fs,
                        'end_time_s': end_frame / self.fs
                    }
                    
                    segments.append(segment)
                
                return segments
                
        except Exception as e:
            print(f"❌ 处理文件时发生错误: {e}")
            return []


class NeuLogHeartRateCalculator:
    """
    从NeuLog峰值掩码计算心率
    """
    
    def __init__(self, fs: float = 20):
        """
        初始化NeuLog心率计算器
        
        参数:
            fs: 采样率 (Hz)
        """
        self.fs = fs
    
    def calculate(self, peak_mask: np.ndarray, window_s: float = 30) -> Dict[str, float]:
        """
        计算两种心率
        
        参数:
            peak_mask: 1D二值数组 (长度 = window_s * fs)
            window_s: 窗口长度（秒）
        
        返回:
            {'hr_count': float, 'hr_interval': float, 'num_peaks': int}
        """
        # 方法1: 计数法
        num_peaks = int(np.sum(peak_mask))
        hr_count = (num_peaks / window_s) * 60
        
        # 方法2: 间隔法
        peak_indices = np.where(peak_mask == 1)[0]
        
        if len(peak_indices) < 2:
            hr_interval = np.nan
        else:
            intervals_frames = np.diff(peak_indices)
            avg_interval_s = np.mean(intervals_frames) / self.fs
            hr_interval = 60 / avg_interval_s
        
        return {
            'hr_count': hr_count,
            'hr_interval': hr_interval,
            'num_peaks': num_peaks
        }