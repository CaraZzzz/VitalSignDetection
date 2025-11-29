"""
CA-CFAR方法人体定位
"""
import numpy as np
from scipy.signal import windows
from typing import Dict, Any, Tuple
from .base_localization import BaseLocalizationMethod


class CFARLocalization(BaseLocalizationMethod):
    """
    基于CA-CFAR（Cell Averaging - Constant False Alarm Rate）的人体定位方法
    
    使用Range-Doppler图进行目标检测
    """
    
    def __init__(self, G_R: int = 1, G_D: int = 1, L_R: int = 5, L_D: int = 5,
                 P_fa: float = 1e-6, n_doppler_fft: int = 128):
        """
        初始化CA-CFAR定位方法
        
        参数:
            G_R: Range保护单元半长
            G_D: Doppler保护单元半长
            L_R: Range训练单元半长
            L_D: Doppler训练单元半长
            P_fa: 虚警概率
            n_doppler_fft: Doppler FFT点数
        """
        super().__init__()
        self.G_R = G_R
        self.G_D = G_D
        self.L_R = L_R
        self.L_D = L_D
        self.P_fa = P_fa
        self.n_doppler_fft = n_doppler_fft
    
    def _compute_range_doppler_map(self, phase_data: np.ndarray, fs: float) -> np.ndarray:
        """
        计算Range-Doppler图
        
        参数:
            phase_data: 相位数据 (R, nVX, T)
            fs: 采样率
        
        返回:
            rd_map: Range-Doppler功率图 (R, Doppler)
        """
        N_range, N_rx, N_time = phase_data.shape
        rd_map = np.zeros((N_range, self.n_doppler_fft))
        
        for r in range(N_range):
            # 对所有RX求和（相干积累）
            range_time_data = np.sum(phase_data[r, :, :], axis=0)
            
            # 加Hann窗
            window = windows.hann(N_time)
            windowed_data = range_time_data * window
            
            # Doppler FFT
            doppler_spectrum = np.fft.fft(windowed_data, n=self.n_doppler_fft)
            doppler_spectrum = np.fft.fftshift(doppler_spectrum)
            
            # 功率（幅度平方）
            rd_map[r, :] = np.abs(doppler_spectrum) ** 2
        
        return rd_map
    
    def _ca_cfar_2d(self, range_doppler_map: np.ndarray) -> np.ndarray:
        """
        2D CA-CFAR检测器
        
        参数:
            range_doppler_map: Range-Doppler图 (Range x Doppler)
        
        返回:
            detection_map: 检测结果 (Range x Doppler)，True表示检测到目标
        """
        N_range, N_doppler = range_doppler_map.shape
        detection_map = np.zeros_like(range_doppler_map, dtype=bool)
        
        # 计算阈值因子
        N_train = 4 * self.L_R * self.L_D
        alpha = N_train * (self.P_fa**(-1/N_train) - 1)
        
        # 遍历每个待测单元
        for r in range(N_range):
            for d in range(N_doppler):
                # 确定训练窗口范围
                r_start = max(0, r - self.G_R - self.L_R)
                r_end = min(N_range, r + self.G_R + self.L_R + 1)
                d_start = max(0, d - self.G_D - self.L_D)
                d_end = min(N_doppler, d + self.G_D + self.L_D + 1)
                
                # 提取训练区域
                train_region = range_doppler_map[r_start:r_end, d_start:d_end].copy()
                
                # 移除保护单元和待测单元
                guard_r_start = max(0, r - self.G_R - r_start)
                guard_r_end = min(r_end - r_start, r + self.G_R + 1 - r_start)
                guard_d_start = max(0, d - self.G_D - d_start)
                guard_d_end = min(d_end - d_start, d + self.G_D + 1 - d_start)
                
                # 创建掩码（排除保护单元）
                mask = np.ones_like(train_region, dtype=bool)
                mask[guard_r_start:guard_r_end, guard_d_start:guard_d_end] = False
                
                # 计算训练单元平均功率
                train_cells = train_region[mask]
                if len(train_cells) > 0:
                    noise_level = np.mean(train_cells)
                    threshold = alpha * noise_level
                else:
                    threshold = 0
                
                # 检测判决
                if range_doppler_map[r, d] > threshold:
                    detection_map[r, d] = True
        
        return detection_map
    
    def select_range_bin(self, segment_data: Dict[str, Any], 
                        predefined_rb_index: int = None) -> Tuple[int, Dict[str, Any]]:
        """
        使用CA-CFAR检测选择Range Bin
        
        参数:
            segment_data: 片段数据字典
            predefined_rb_index: 预定义的Range Bin索引（在回退时使用）
        
        返回:
            (selected_rb_index, selection_info)
        """
        try:
            # 计算Range-Doppler图
            unfiltered_phase = segment_data['unfiltered_phase']
            rd_map = self._compute_range_doppler_map(unfiltered_phase, fs=20)
            
            # 执行CA-CFAR检测
            detection_map = self._ca_cfar_2d(rd_map)
            
            # 分析检测结果
            range_detections = np.sum(detection_map, axis=1)
            
            if np.max(range_detections) > 0:
                # 有检测到目标，选择检测次数最多的Range Bin
                selected_idx = np.argmax(range_detections)
                confidence = range_detections[selected_idx] / detection_map.shape[1]
                fallback = False
            else:
                # 没有检测到，回退到功率最大的Range Bin
                magnitude = segment_data['magnitude']
                range_power = np.mean(magnitude ** 2, axis=(1, 2))
                selected_idx = np.argmax(range_power)
                confidence = 0.0
                fallback = True
                print("⚠️ CA-CFAR未检测到目标，回退到功率方法")
            
            selection_info = {
                'method': 'ca_cfar',
                'confidence': float(confidence),
                'detections_per_bin': range_detections.tolist(),
                'fallback': fallback
            }
            
            return selected_idx, selection_info
            # return min(20,selected_idx+2), selection_info
            
        except Exception as e:
            print(f"⚠️ CA-CFAR检测失败: {e}")
            # 回退到功率方法
            magnitude = segment_data['magnitude']
            range_power = np.mean(magnitude ** 2, axis=(1, 2))
            selected_idx = np.argmax(range_power)
            
            selection_info = {
                'method': 'ca_cfar',
                'confidence': 0.0,
                'fallback': True,
                'error': str(e)
            }
            
            return selected_idx, selection_info
    
    def get_method_name(self) -> str:
        """获取方法名称"""
        return "CA-CFAR"