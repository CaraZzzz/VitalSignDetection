"""
Random Forest方法人体定位 - 改进版（限制选择范围）
使用完整版特征提取，不限制range bin 选择范围
"""


import numpy as np
import joblib
import json
import os
from typing import Dict, Any, Tuple
from scipy import stats
import scipy.fft as fft
from .base_localization import BaseLocalizationMethod


class RandomForestLocalization(BaseLocalizationMethod):
    """
    基于随机森林模型的人体定位方法（完整特征版）
    
    使用与训练时相同的16个特征进行预测
    """
    
    def __init__(self, model_path: str, scaler_path: str, metadata_path: str):
        """
        初始化Random Forest定位方法
        
        参数:
            model_path: 模型文件路径（.pkl）- 使用绝对路径
            scaler_path: 标准化器文件路径（.pkl）- 使用绝对路径
            metadata_path: 元数据文件路径（.json）- 使用绝对路径
        """
        super().__init__()
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"标准化器文件未找到: {scaler_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"元数据文件未找到: {metadata_path}")
        
        # 加载模型
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # 获取采样率（从元数据或使用默认值）
        self.fs = self.metadata.get('sampling_rate', 20.0)  # 默认20Hz
        
        print(f"✅ 成功加载随机森林模型: {os.path.basename(model_path)}")
        print(f"   测试集AUC: {self.metadata.get('test_auc', 'N/A')}")
        print(f"   测试集F1: {self.metadata.get('test_f1', 'N/A')}")
        print(f"   特征数量: {len(self.metadata['feature_names'])}")
        print(f"   采样率: {self.fs} Hz")
    
    def _calculate_spectrum_features(self, time_series: np.ndarray) -> Dict[str, float]:
        """
        计算频谱特征（与训练时完全一致）
        
        参数:
            time_series: 时间序列数据
        
        返回:
            特征字典
        """
        N = len(time_series)
        
        # FFT
        Y = fft.fft(time_series)
        P2 = np.abs(Y / N)
        P1 = P2[:N // 2 + 1]
        P1[1:-1] = 2 * P1[1:-1]
        E_P = P1 ** 2  # 能量谱 (PSD)
        f = self.fs * np.arange(0, N // 2 + 1) / N
        
        # 定义频带索引 (Hz)
        idx_resp = np.where((f >= 0.15) & (f <= 0.5))[0]
        idx_cardiac = np.where((f >= 0.8) & (f <= 2.0))[0]
        idx_life_signal = np.where((f >= 0.15) & (f <= 2.0))[0] 
        idx_noise = np.where((f >= 3.0) & (f <= 8.0))[0]        
        idx_total = np.where(f > 0)[0] 
        
        # 零能量处理
        if not idx_total.size: 
            return {
                'energy_resp': 0.0, 
                'energy_cardiac': 0.0, 
                'energy_total_ratio': 0.0, 
                'freq_peak_pos': 0.0, 
                'purity_ratio': 0.0, 
                'energy_ratio_C_R': 0.0, 
                'snr_db': -100.0
            }
        
        # 能量计算
        E_R = np.sum(E_P[idx_resp])
        E_C = np.sum(E_P[idx_cardiac])
        E_T = np.sum(E_P[idx_total])
        
        # SNR 估计
        signal_energy = np.sum(E_P[idx_life_signal])
        noise_avg_psd = np.mean(E_P[idx_noise]) if idx_noise.size > 0 else E_T / idx_total.size
        
        # 使用 10 * log10 计算 SNR_dB
        if noise_avg_psd > 1e-12:
            snr_db = 10 * np.log10(signal_energy / noise_avg_psd)
        else:
            snr_db = 100.0 if signal_energy > 1e-12 else -100.0
        
        # 其他特征
        idx_life = np.where((f >= 0.15) & (f <= 2.0))[0]
        freq_peak_pos = f[idx_life[np.argmax(P1[idx_life])]] if idx_life.size > 0 else 0.0
        
        P1_non_DC = P1[idx_total]
        max_P1 = np.max(P1_non_DC)
        mean_P1 = np.mean(P1_non_DC)
        purity_ratio = max_P1 / mean_P1 if mean_P1 != 0 else 0.0
        
        energy_ratio_C_R = E_C / E_R if E_R != 0 else 0.0
        energy_total_ratio = (E_R + E_C) / E_T if E_T != 0 else 0.0
        
        return {
            'energy_resp': E_R, 
            'energy_cardiac': E_C, 
            'energy_total_ratio': energy_total_ratio, 
            'freq_peak_pos': freq_peak_pos, 
            'purity_ratio': purity_ratio, 
            'energy_ratio_C_R': energy_ratio_C_R, 
            'snr_db': snr_db
        }
    
    def _extract_features(self, segment_data: Dict[str, Any]) -> np.ndarray:
        """
        提取完整特征（与训练时完全一致）
        
        参数:
            segment_data: 片段数据字典
        
        返回:
            features: (R x nVX x n_features) 特征数组
        """
        magnitude = segment_data['magnitude']
        unfiltered_phase = segment_data['unfiltered_phase']
        
        R, nVX, T = magnitude.shape
        feature_names = self.metadata['feature_names']
        n_features = len(feature_names)
        
        # 验证特征数量
        expected_features = 16  # 训练时使用的特征数量
        if n_features != expected_features:
            print(f"⚠️ 警告: 元数据中的特征数量 ({n_features}) "
                  f"与预期 ({expected_features}) 不符")
        
        features = np.zeros((R, nVX, n_features))
        
        # 遍历每个 Range Bin 和 RX
        for r in range(R):
            for v in range(nVX):
                amp_ts = magnitude[r, v, :]
                phase_ts = unfiltered_phase[r, v, :]
                
                # 1. 振幅统计特征
                amp_mean = np.mean(amp_ts)
                amp_std = np.std(amp_ts)
                amp_p2p = np.max(amp_ts) - np.min(amp_ts)
                amp_skewness = stats.skew(amp_ts)
                amp_kurtosis = stats.kurtosis(amp_ts)
                
                # 2. 相位差统计特征
                phase_diff = np.diff(phase_ts)
                phase_diff_std = np.std(phase_diff)
                phase_diff_range = np.max(phase_diff) - np.min(phase_diff) if len(phase_diff) > 0 else 0.0
                
                # 3. 振幅频谱特征（包含 SNR）
                amp_spec_feats = self._calculate_spectrum_features(amp_ts)
                
                # 4. 相位频谱特征（包含 SNR）
                phase_spec_feats = self._calculate_spectrum_features(phase_ts)
                
                # 5. 按照训练时的特征顺序组装特征向量
                feature_vector = [
                    amp_mean,                                    # 0: amp_mean
                    amp_std,                                     # 1: amp_std
                    amp_p2p,                                     # 2: amp_p2p
                    amp_skewness,                                # 3: amp_skewness
                    amp_kurtosis,                                # 4: amp_kurtosis
                    phase_diff_std,                              # 5: phase_diff_std
                    phase_diff_range,                            # 6: phase_diff_range
                    amp_spec_feats['energy_resp'],               # 7: amp_energy_resp
                    amp_spec_feats['energy_cardiac'],            # 8: amp_energy_cardiac
                    amp_spec_feats['energy_total_ratio'],        # 9: amp_life_energy_ratio
                    amp_spec_feats['freq_peak_pos'],             # 10: amp_freq_peak_pos
                    amp_spec_feats['purity_ratio'],              # 11: amp_purity_ratio
                    amp_spec_feats['energy_ratio_C_R'],          # 12: amp_energy_ratio_C_R
                    amp_spec_feats['snr_db'],                    # 13: amp_snr_db
                    phase_spec_feats['energy_ratio_C_R'],        # 14: phase_energy_ratio_C_R
                    phase_spec_feats['snr_db']                   # 15: phase_snr_db
                ]
                
                features[r, v, :] = feature_vector
        
        return features
    
    def select_range_bin(self, segment_data: Dict[str, Any], 
                        predefined_rb_index: int = None) -> Tuple[int, Dict[str, Any]]:
        """
        使用随机森林模型选择Range Bin
        
        参数:
            segment_data: 片段数据字典
            predefined_rb_index: 预定义的Range Bin索引（未使用，为了保持接口一致）
        
        返回:
            (selected_rb_index, selection_info)
        """
        # 提取完整特征
        features = self._extract_features(segment_data)
        
        # 获取RX索引
        rx_index = segment_data['rx_index']  # 1-based
        rx_idx_0based = rx_index - 1
        
        R = features.shape[0]
        
        # 提取该RX的特征并标准化
        features_flat = features[:, rx_idx_0based, :]  # (R x n_features)
        
        # 验证特征维度
        if features_flat.shape[1] != len(self.metadata['feature_names']):
            raise ValueError(
                f"特征维度不匹配: 提取了 {features_flat.shape[1]} 个特征, "
                f"但模型需要 {len(self.metadata['feature_names'])} 个特征"
            )
        
        features_scaled = self.scaler.transform(features_flat)
        
        # 预测概率
        probabilities = self.model.predict_proba(features_scaled)[:, 1]  # 人体类别的概率
        
        # 选择概率最高的Range Bin
        predicted_rb_0based = np.argmax(probabilities)
        predicted_confidence = probabilities[predicted_rb_0based]
        
        selection_info = {
            'method': 'random_forest_full_features',
            'confidence': float(predicted_confidence),
            'probabilities': probabilities.tolist(),
            'n_features_used': features_flat.shape[1]
        }
        
        return predicted_rb_0based, selection_info
    
    def get_method_name(self) -> str:
        """获取方法名称"""
        return "RandomForest_Full"


# ============================================================================
# 使用示例
# ============================================================================
"""
from RandomForestLocalization_Full import RandomForestLocalization

# 初始化
rf_localizer = RandomForestLocalization(
    model_path='path/to/rf_rangebin_classifier_timestamp.pkl',
    scaler_path='path/to/rf_rangebin_classifier_scaler_timestamp.pkl',
    metadata_path='path/to/rf_rangebin_classifier_metadata_timestamp.json'
)

# 使用
segment_data = {
    'magnitude': ...,           # (R, nVX, T) 振幅数据
    'unfiltered_phase': ...,    # (R, nVX, T) 未滤波相位数据
    'rx_index': 1               # 1-based RX索引
}

selected_bin, info = rf_localizer.select_range_bin(segment_data)
print(f"选择的 Range Bin: {selected_bin + 1} (base-1)")
print(f"置信度: {info['confidence']:.4f}")
print(f"使用的特征数: {info['n_features_used']}")
"""