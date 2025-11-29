"""
Wavelet方法心率估计
"""
import numpy as np
from typing import Dict, Tuple
from .base_estimation import BaseHeartRateEstimator


class WaveletHeartRateEstimator(BaseHeartRateEstimator):
    """
    基于连续小波变换（CWT）的心率估计
    
    优势：
    - 时频局部化，适合非平稳信号
    - 对噪声和运动伪影更鲁棒
    - 不需要加窗
    """
    
    def __init__(self, fs: float = 20, detrend: bool = True, wavelet: str = 'morl'):
        """
        初始化Wavelet心率估计器
        
        参数:
            fs: 采样率 (Hz)
            detrend: 是否去趋势
            wavelet: 小波类型 ('morl' for Morlet)
        """
        super().__init__(fs=fs, detrend=detrend)
        self.wavelet = wavelet
    
    def _morlet_wavelet(self, t, scale, w=6.0):
        """
        Morlet小波
        
        参数:
            t: 时间数组
            scale: 尺度参数
            w: 小波参数（默认6.0）
        """
        normalized_t = t / scale
        wavelet = np.exp(1j * w * normalized_t) * np.exp(-0.5 * normalized_t**2)
        wavelet *= 1.0 / np.sqrt(scale)
        return wavelet
    
    def estimate(self, phase_data: np.ndarray, freq_range: Tuple[float, float] = (0.8, 2.0)) -> Dict[str, float]:
        """
        使用小波变换估计心率
        
        参数:
            phase_data: 相位时间序列
            freq_range: 心率频率范围 (Hz)
            
        返回:
            {'heart_rate_bpm': float, 'peak_freq_hz': float, 'peak_magnitude': float}
        """
        try:
            # 尝试使用pywt
            try:
                import pywt
                use_pywt = True
            except ImportError:
                use_pywt = False
            
            # 预处理
            phase_data = self.preprocess(phase_data)
            
            freq_min, freq_max = freq_range
            
            if use_pywt:
                # 使用PyWavelets
                center_freq = pywt.central_frequency(self.wavelet)
                scale_max = center_freq * self.fs / freq_min
                scale_min = center_freq * self.fs / freq_max
                num_scales = 100
                # scales = np.logspace(np.log10(scale_min), np.log10(scale_max), num=num_scales)
                scales = np.linspace(scale_min, scale_max, num=num_scales)
                
                coefficients, frequencies = pywt.cwt(
                    phase_data, scales, self.wavelet, sampling_period=1/self.fs
                )
                power = np.mean(np.abs(coefficients) ** 2, axis=1)
            else:
                # 手动实现Morlet小波CWT
                w = 6.0
                center_freq = w / (2 * np.pi)
                scale_max = center_freq * self.fs / freq_min
                scale_min = center_freq * self.fs / freq_max
                num_scales = 100
                scales = np.logspace(np.log10(scale_min), np.log10(scale_max), num=num_scales)
                
                N = len(phase_data)
                coefficients = np.zeros((num_scales, N), dtype=complex)
                t = np.arange(N)
                
                for i, scale in enumerate(scales):
                    wavelet = self._morlet_wavelet(t - N/2, scale, w)
                    signal_fft = np.fft.fft(phase_data)
                    wavelet_fft = np.fft.fft(np.conj(wavelet[::-1]))
                    coefficients[i, :] = np.fft.ifft(signal_fft * wavelet_fft)
                
                frequencies = center_freq * self.fs / scales
                power = np.mean(np.abs(coefficients) ** 2, axis=1)
            
            # 在指定频率范围内找峰值
            valid_idx = (frequencies >= freq_min) & (frequencies <= freq_max)
            valid_freqs = frequencies[valid_idx]
            valid_power = power[valid_idx]
            
            if len(valid_power) == 0:
                return {
                    'heart_rate_bpm': np.nan,
                    'peak_freq_hz': np.nan,
                    'peak_magnitude': np.nan
                }
            
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
            print(f"⚠️ 小波方法执行失败: {e}")
            return {
                'heart_rate_bpm': np.nan,
                'peak_freq_hz': np.nan,
                'peak_magnitude': np.nan
            }
    
    def get_method_name(self) -> str:
        """获取方法名称"""
        return "Wavelet"