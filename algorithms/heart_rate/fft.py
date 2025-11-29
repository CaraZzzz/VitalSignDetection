"""
FFT方法心率估计
"""
import numpy as np
from scipy.fft import fft, fftfreq
from typing import Dict, Tuple
from .base_estimation import BaseHeartRateEstimator


class FFTHeartRateEstimator(BaseHeartRateEstimator):
    """
    基于快速傅里叶变换（FFT）的心率估计
    
    原理：
    将时域相位信号转换到频域，在心跳频率范围内找到最大峰值
    """
    
    def __init__(self, fs: float = 20, detrend: bool = True, apply_window: bool = False):
        """
        初始化FFT心率估计器
        
        参数:
            fs: 采样率 (Hz)
            detrend: 是否去趋势
            apply_window: 是否应用汉明窗
        """
        super().__init__(fs=fs, detrend=detrend)
        self.apply_window = apply_window
    
    def estimate(self, phase_data: np.ndarray, freq_range: Tuple[float, float] = (0.8, 2.0)) -> Dict[str, float]:
        """
        使用FFT估计心率
        
        参数:
            phase_data: 相位时间序列
            freq_range: 心率频率范围 (Hz)
            
        返回:
            {'heart_rate_bpm': float, 'peak_freq_hz': float, 'peak_magnitude': float}
        """
        # 预处理
        phase_data = self.preprocess(phase_data)
        
        N = len(phase_data)
        
        # 可选：加窗
        if self.apply_window:
            from scipy import signal as sp_signal
            window = sp_signal.hamming(N)
            phase_data = phase_data * window
        
        # FFT
        fft_result = fft(phase_data)
        freq_axis = fftfreq(N, 1/self.fs)
        
        # 只取正频率 + 心跳范围
        valid_idx = (freq_axis >= freq_range[0]) & (freq_axis <= freq_range[1])
        valid_freqs = freq_axis[valid_idx]
        valid_magnitude = np.abs(fft_result[valid_idx])
        
        if len(valid_magnitude) == 0:
            return {
                'heart_rate_bpm': np.nan,
                'peak_freq_hz': np.nan,
                'peak_magnitude': np.nan
            }
        
        # 找峰值
        peak_idx = np.argmax(valid_magnitude)
        peak_freq_hz = valid_freqs[peak_idx]
        peak_magnitude = valid_magnitude[peak_idx]
        heart_rate_bpm = peak_freq_hz * 60
        
        return {
            'heart_rate_bpm': heart_rate_bpm,
            'peak_freq_hz': peak_freq_hz,
            'peak_magnitude': peak_magnitude
        }
    
    def get_method_name(self) -> str:
        """获取方法名称"""
        return "FFT"