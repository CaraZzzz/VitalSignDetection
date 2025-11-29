"""
EMD方法心率估计
"""
import numpy as np
from typing import Dict, Tuple
# from PyEMD import EMD
from .base_estimation import BaseHeartRateEstimator


class EMDHeartRateEstimator(BaseHeartRateEstimator):
    """
    基于经验模态分解（EMD）的心率估计
    
    优势：
    - 完全数据驱动，无需预设基函数
    - 自适应分解非线性、非平稳信号
    
    劣势：
    - 对噪声敏感
    - 存在模态混叠问题
    """
    
    def __init__(self, fs: float = 20, detrend: bool = True, max_imf: int = -1):
        """
        初始化EMD心率估计器
        
        参数:
            fs: 采样率 (Hz)
            detrend: 是否去趋势
            max_imf: 最大IMF数量（-1=自动）
        """
        super().__init__(fs=fs, detrend=detrend)
        self.max_imf = max_imf
    
    def estimate(self, phase_data: np.ndarray, freq_range: Tuple[float, float] = (0.8, 2.0)) -> Dict[str, float]:
        """
        使用EMD估计心率
        
        参数:
            phase_data: 相位时间序列
            freq_range: 心率频率范围 (Hz)
            
        返回:
            {'heart_rate_bpm': float, 'peak_freq_hz': float, 'peak_magnitude': float}
        """
        try:
            # 尝试导入PyEMD
            try:
                from PyEMD import EMD
            except ImportError:
                print("⚠️ PyEMD未安装")
                return {
                    'heart_rate_bpm': np.nan,
                    'peak_freq_hz': np.nan,
                    'peak_magnitude': np.nan
                }
            
            # 预处理
            phase_data = self.preprocess(phase_data)
            
            # 执行EMD分解
            emd = EMD()
            if self.max_imf > 0:
                IMFs = emd.emd(phase_data, max_imf=self.max_imf)
            else:
                IMFs = emd.emd(phase_data)
            
            # 选择心率频段内能量最大的IMF
            freq_min, freq_max = freq_range
            best_imf_idx = -1
            max_energy = 0
            
            for i in range(IMFs.shape[0]):
                imf = IMFs[i, :]
                
                # 计算该IMF的FFT
                fft_result = np.fft.fft(imf)
                freqs = np.fft.fftfreq(len(imf), 1/self.fs)
                power = np.abs(fft_result) ** 2
                
                # 计算心率频段内的能量
                valid_idx = (freqs >= freq_min) & (freqs <= freq_max) & (freqs >= 0)
                energy = np.sum(power[valid_idx])
                
                if energy > max_energy:
                    max_energy = energy
                    best_imf_idx = i
            
            if best_imf_idx == -1 or max_energy == 0:
                return {
                    'heart_rate_bpm': np.nan,
                    'peak_freq_hz': np.nan,
                    'peak_magnitude': np.nan
                }
            
            # 对选中的IMF进行FFT分析
            best_imf = IMFs[best_imf_idx, :]
            fft_result = np.fft.fft(best_imf)
            freqs = np.fft.fftfreq(len(best_imf), 1/self.fs)
            power = np.abs(fft_result) ** 2
            
            # 在心率频段内找峰值
            valid_idx = (freqs >= freq_min) & (freqs <= freq_max) & (freqs >= 0)
            valid_freqs = freqs[valid_idx]
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
            
            # 峰值细化
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
            print(f"⚠️ EMD方法执行失败: {e}")
            return {
                'heart_rate_bpm': np.nan,
                'peak_freq_hz': np.nan,
                'peak_magnitude': np.nan
            }
    
    def get_method_name(self) -> str:
        """获取方法名称"""
        return "EMD"