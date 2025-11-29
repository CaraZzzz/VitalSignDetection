"""
VMD方法心率估计
"""
import numpy as np
from typing import Dict, Tuple
from .base_estimation import BaseHeartRateEstimator
# from vmdpy import VMD
# from PyEMD import VMD


class VMDHeartRateEstimator(BaseHeartRateEstimator):
    """
    基于变分模态分解（VMD）的心率估计
    
    优势：
    - 自适应信号分解，无模态混叠
    - 比EMD更鲁棒，对噪声不敏感
    - 可以精确控制模态数量和带宽
    """
    
    def __init__(self, fs: float = 20, detrend: bool = True,
                 alpha: int = 2000, tau: float = 0.0, K: int = 5,
                 DC: int = 0, init: int = 1, tol: float = 1e-7):
        """
        初始化VMD心率估计器
        
        参数:
            fs: 采样率 (Hz)
            detrend: 是否去趋势
            alpha: 惩罚因子，控制带宽
            tau: 噪声容忍度
            K: 模态数量
            DC: 是否包含DC分量
            init: 初始化方式
            tol: 收敛容差
        """
        super().__init__(fs=fs, detrend=detrend)
        self.alpha = alpha
        self.tau = tau
        self.K = K
        self.DC = DC
        self.init = init
        self.tol = tol
    
    def estimate(self, phase_data: np.ndarray, freq_range: Tuple[float, float] = (0.8, 2.0)) -> Dict[str, float]:
        """
        使用VMD估计心率
        
        参数:
            phase_data: 相位时间序列
            freq_range: 心率频率范围 (Hz)
            
        返回:
            {'heart_rate_bpm': float, 'peak_freq_hz': float, 'peak_magnitude': float}
        """
        try:
            # 尝试导入vmdpy或PyEMD
            try:
                from vmdpy import VMD
            except ImportError:
                try:
                    from PyEMD import VMD
                except ImportError:
                    print("⚠️ vmdpy和PyEMD都未安装")
                    return {
                        'heart_rate_bpm': np.nan,
                        'peak_freq_hz': np.nan,
                        'peak_magnitude': np.nan
                    }
            
            # 预处理
            phase_data = self.preprocess(phase_data)
            
            # 执行VMD分解
            try:
                # vmdpy版本
                u, u_hat, omega = VMD(phase_data, self.alpha, self.tau, self.K, 
                                     self.DC, self.init, self.tol)
            except:
                # PyEMD版本
                vmd = VMD()
                u = vmd(phase_data, K=self.K)
            
            # 选择心率频段内能量最大的IMF
            freq_min, freq_max = freq_range
            best_imf_idx = -1
            max_energy = 0
            
            for i in range(u.shape[0]):
                imf = u[i, :]
                
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
            
            if best_imf_idx == -1:
                return {
                    'heart_rate_bpm': np.nan,
                    'peak_freq_hz': np.nan,
                    'peak_magnitude': np.nan
                }
            
            # 对选中的IMF进行FFT分析
            best_imf = u[best_imf_idx, :]
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
            print(f"⚠️ VMD方法执行失败: {e}")
            return {
                'heart_rate_bpm': np.nan,
                'peak_freq_hz': np.nan,
                'peak_magnitude': np.nan
            }
    
    def get_method_name(self) -> str:
        """获取方法名称"""
        return "VMD"