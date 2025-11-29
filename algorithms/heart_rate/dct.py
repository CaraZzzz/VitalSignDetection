"""
DCT方法心率估计
"""
import numpy as np
from scipy.fft import dct
from typing import Dict, Tuple
from .base_estimation import BaseHeartRateEstimator


class DCTHeartRateEstimator(BaseHeartRateEstimator):
    """
    基于离散余弦变换（DCT-II）的心率估计
    
    优势：
    - 对缓变信号更鲁棒
    - 能量集中在低频系数
    - 计算效率高（接近FFT）
    """
    
    def __init__(self, fs: float = 20, detrend: bool = True):
        """
        初始化DCT心率估计器
        
        参数:
            fs: 采样率 (Hz)
            detrend: 是否去趋势
        """
        super().__init__(fs=fs, detrend=detrend)
    
    def estimate(self, phase_data: np.ndarray, freq_range: Tuple[float, float] = (0.8, 2.0)) -> Dict[str, float]:
        """
        使用DCT估计心率
        
        参数:
            phase_data: 相位时间序列
            freq_range: 心率频率范围 (Hz)
            
        返回:
            {'heart_rate_bpm': float, 'peak_freq_hz': float, 'peak_magnitude': float}
        """
        try:
            # 预处理
            phase_data = self.preprocess(phase_data)
            
            # 执行DCT-II变换
            N = len(phase_data)
            dct_result = dct(phase_data, type=2, norm='ortho')
            
            # DCT频率轴计算
            # DCT的第k个系数对应频率：f_k = k * fs / (2*N)
            freq_indices = np.arange(N)
            freq_axis = freq_indices * self.fs / (2 * N)
            
            # 计算功率谱（DCT系数的平方）
            power_spectrum = dct_result ** 2
            
            # 在心跳频率范围内搜索峰值
            freq_min, freq_max = freq_range
            valid_idx = (freq_axis >= freq_min) & (freq_axis <= freq_max)
            valid_freqs = freq_axis[valid_idx]
            valid_power = power_spectrum[valid_idx]
            
            if len(valid_power) == 0:
                return {
                    'heart_rate_bpm': np.nan,
                    'peak_freq_hz': np.nan,
                    'peak_magnitude': np.nan
                }
            
            # 找最大功率对应的频率
            peak_idx = np.argmax(valid_power)
            peak_freq_hz = valid_freqs[peak_idx]
            peak_magnitude = valid_power[peak_idx]
            heart_rate_bpm = peak_freq_hz * 60
            
            return {
                'heart_rate_bpm': heart_rate_bpm,
                'peak_freq_hz': peak_freq_hz,
                'peak_magnitude': peak_magnitude
            }
            
        except Exception as e:
            print(f"⚠️ DCT方法执行失败: {e}")
            return {
                'heart_rate_bpm': np.nan,
                'peak_freq_hz': np.nan,
                'peak_magnitude': np.nan
            }
    
    def get_method_name(self) -> str:
        """获取方法名称"""
        return "DCT"