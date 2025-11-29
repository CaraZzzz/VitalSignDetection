"""
STFT方法心率估计
"""
import numpy as np
from scipy import signal as sp_signal
from typing import Dict, Tuple


class STFTHeartRateEstimator:
    """
    基于短时傅里叶变换（STFT）的心率估计
    """
    
    def __init__(self, fs: float = 20, detrend: bool = True,
                 nperseg: int = 512, noverlap: int = 384, window: str = 'hann'):
        """
        初始化STFT心率估计器
        
        参数:
            fs: 采样率 (Hz)
            detrend: 是否去趋势
            nperseg: STFT窗口长度（样本数）
            noverlap: STFT重叠长度（样本数）
            window: 窗口类型 ('hann', 'hamming', 'blackman')
        """
        self.fs = fs
        self.detrend = detrend
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.window = window
    
    def preprocess(self, phase_data: np.ndarray) -> np.ndarray:
        """预处理：去趋势"""
        if self.detrend:
            return sp_signal.detrend(phase_data, type='linear')
        return phase_data
    
    def estimate(self, phase_data: np.ndarray, freq_range: Tuple[float, float] = (0.8, 2.0)) -> Dict[str, float]:
        """
        使用STFT估计心率
        
        参数:
            phase_data: 相位时间序列
            freq_range: 心率频率范围 (Hz)
            
        返回:
            {'heart_rate_bpm': float, 'peak_freq_hz': float, 'peak_magnitude': float}
        """
        try:
            # 预处理
            phase_data = self.preprocess(phase_data)
            
            # 确保窗口长度不超过信号长度
            nperseg = min(self.nperseg, len(phase_data))
            noverlap = min(self.noverlap, nperseg - 1)
            
            # 执行STFT
            f, t, Zxx = sp_signal.stft(
                phase_data,
                fs=self.fs,
                window=self.window,
                nperseg=nperseg,
                noverlap=noverlap,
                detrend=False,
                return_onesided=True,
                boundary='zeros',
                padded=True
            )
            
            # 计算功率谱密度（时间平均）
            power_spectrum = np.mean(np.abs(Zxx) ** 2, axis=1)
            
            # 在心跳频率范围内搜索峰值
            freq_min, freq_max = freq_range
            valid_idx = (f >= freq_min) & (f <= freq_max)
            valid_freqs = f[valid_idx]
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
            
            # 峰值细化（抛物线插值）
            if 0 < peak_idx < len(valid_power) - 1:
                y1, y2, y3 = valid_power[peak_idx-1:peak_idx+2]
                delta = 0.5 * (y3 - y1) / (2*y2 - y1 - y3)
                if abs(delta) < 1.0:
                    freq_resolution = valid_freqs[1] - valid_freqs[0] if len(valid_freqs) > 1 else 0.01
                    peak_freq_hz = peak_freq_hz + delta * freq_resolution
            
            heart_rate_bpm = peak_freq_hz * 60
            
            return {
                'heart_rate_bpm': heart_rate_bpm,
                'peak_freq_hz': peak_freq_hz,
                'peak_magnitude': peak_magnitude
            }
            
        except Exception as e:
            print(f"⚠️ STFT方法执行失败: {e}")
            return {
                'heart_rate_bpm': np.nan,
                'peak_freq_hz': np.nan,
                'peak_magnitude': np.nan
            }
    
    def get_method_name(self) -> str:
        """获取方法名称"""
        return "STFT"