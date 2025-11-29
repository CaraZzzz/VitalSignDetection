"""
CA-CFAR 检测器封装
"""
import numpy as np
from scipy.signal import windows
from typing import Tuple, Dict


class CAFARDetector:
    """
    2D CA-CFAR 人体位置检测器
    """
    
    def __init__(self, G_R: int = 1, G_D: int = 1, L_R: int = 5, L_D: int = 5,
                 P_fa: float = 1e-3, n_doppler_fft: int = 128,
                 min_range_bin: int = 1, max_range_bin: int = 18):
        """
        初始化CFAR检测器
        
        参数:
            G_R, G_D: Range和Doppler保护单元半长
            L_R, L_D: Range和Doppler训练单元半长
            P_fa: 虚警概率
            n_doppler_fft: Doppler FFT点数
            min_range_bin: 最小有效Range Bin (1-based)
            max_range_bin: 最大有效Range Bin (1-based)
        """
        self.G_R = G_R
        self.G_D = G_D
        self.L_R = L_R
        self.L_D = L_D
        self.P_fa = P_fa
        self.n_doppler_fft = n_doppler_fft
        self.min_range_bin = min_range_bin
        self.max_range_bin = max_range_bin
        
        # 计算阈值因子
        N_train = 4 * L_R * L_D
        self.alpha = N_train * (P_fa**(-1/N_train) - 1)
    
    def compute_range_doppler_map(self, phase_data: np.ndarray, window_type: str = 'hann') -> np.ndarray:
        """
        计算Range-Doppler图
        
        参数:
            phase_data: 相位数据 (Range x RX x Time)
            window_type: 窗函数类型
        
        返回:
            rd_map: Range-Doppler功率图 (Range x Doppler)
        """
        N_range, N_rx, N_time = phase_data.shape
        
        rd_map_complex = np.zeros((N_range, self.n_doppler_fft), dtype=complex)
        
        for r in range(N_range):
            # 对所有RX求和（coherent integration）
            range_time_data = np.sum(phase_data[r, :, :], axis=0)
            
            # 加窗
            if window_type == 'hann':
                window = windows.hann(N_time)
            elif window_type == 'hamming':
                window = windows.hamming(N_time)
            else:
                window = np.ones(N_time)
            
            windowed_data = range_time_data * window
            
            # Doppler FFT
            doppler_spectrum = np.fft.fft(windowed_data, n=self.n_doppler_fft)
            doppler_spectrum = np.fft.fftshift(doppler_spectrum)
            
            rd_map_complex[r, :] = doppler_spectrum
        
        # 计算功率
        rd_map = np.abs(rd_map_complex) ** 2
        
        return rd_map
    
    def ca_cfar_2d(self, range_doppler_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        2D CA-CFAR检测
        
        参数:
            range_doppler_map: Range-Doppler图 (Range x Doppler)
        
        返回:
            detection_map: 检测结果 (Range x Doppler)
            threshold_map: 阈值图
        """
        N_range, N_doppler = range_doppler_map.shape
        detection_map = np.zeros_like(range_doppler_map, dtype=bool)
        threshold_map = np.zeros_like(range_doppler_map)
        
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
                
                # 创建掩码
                mask = np.ones_like(train_region, dtype=bool)
                mask[guard_r_start:guard_r_end, guard_d_start:guard_d_end] = False
                
                # 计算训练单元平均功率
                train_cells = train_region[mask]
                if len(train_cells) > 0:
                    noise_level = np.mean(train_cells)
                    threshold = self.alpha * noise_level
                else:
                    threshold = 0
                
                threshold_map[r, d] = threshold
                
                # 检测判决
                if range_doppler_map[r, d] > threshold:
                    detection_map[r, d] = True
        
        return detection_map, threshold_map
    
    def detect(self, phase_data: np.ndarray) -> Dict:
        """
        检测人体位置
        
        参数:
            phase_data: 相位数据 (Range x RX x Time)
        
        返回:
            结果字典，包含detected_range_bin, rd_map, detection_map, confidence
        """
        # 计算Range-Doppler图
        rd_map = self.compute_range_doppler_map(phase_data)
        
        # 执行2D CA-CFAR
        detection_map, threshold_map = self.ca_cfar_2d(rd_map)
        
        # 统计每个Range Bin的检测次数
        range_detections = np.sum(detection_map, axis=1)
        
        # 创建有效范围掩码
        valid_range_mask = np.zeros_like(range_detections, dtype=bool)
        valid_range_mask[self.min_range_bin:self.max_range_bin+1] = True
        
        # 屏蔽无效范围
        range_detections_masked = range_detections.copy()
        range_detections_masked[~valid_range_mask] = 0
        
        if np.max(range_detections_masked) > 0:
            # 在有效范围内找到检测次数最多的Bin
            detected_range_bin_0based = np.argmax(range_detections_masked)
            detected_range_bin = detected_range_bin_0based + 1  # 转换为1-based
            confidence = range_detections[detected_range_bin_0based] / detection_map.shape[1]
        else:
            # 如果没有检测到，使用有效范围内功率最大的Range Bin
            range_powers = np.sum(rd_map, axis=1)
            range_powers_masked = range_powers.copy()
            range_powers_masked[~valid_range_mask] = 0
            
            detected_range_bin_0based = np.argmax(range_powers_masked)
            detected_range_bin = detected_range_bin_0based + 1
            confidence = 0.0
        
        return {
            'detected_range_bin': detected_range_bin,
            'rd_map': rd_map,
            'detection_map': detection_map,
            'threshold_map': threshold_map,
            'confidence': confidence
        }