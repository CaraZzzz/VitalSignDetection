"""
心率估计算法基类
"""
import numpy as np
from scipy import signal
from typing import Dict, Tuple


class BaseHeartRateEstimator:
    """
    心率估计算法基类
    所有具体的心率估计方法都应继承此类
    """
    
    def __init__(self, fs: float = 20, detrend: bool = True):
        """
        初始化心率估计器
        
        参数:
            fs: 采样率 (Hz)
            detrend: 是否进行去趋势处理
        """
        self.fs = fs
        self.detrend = detrend
    
    def preprocess(self, phase_data: np.ndarray) -> np.ndarray:
        """
        数据预处理（去趋势）
        
        参数:
            phase_data: 相位时间序列
            
        返回:
            预处理后的数据
        """
        if self.detrend:
            return signal.detrend(phase_data)
        return phase_data
    
    def estimate(self, phase_data: np.ndarray, freq_range: Tuple[float, float] = (0.8, 2.0)) -> Dict[str, float]:
        """
        估计心率（需要在子类中实现）
        
        参数:
            phase_data: 相位时间序列
            freq_range: 心率频率范围 (Hz)
            
        返回:
            {'heart_rate_bpm': float, 'peak_freq_hz': float, 'peak_magnitude': float}
        """
        raise NotImplementedError("子类必须实现estimate方法")
    
    def get_method_name(self) -> str:
        """
        获取方法名称（需要在子类中实现）
        """
        raise NotImplementedError("子类必须实现get_method_name方法")