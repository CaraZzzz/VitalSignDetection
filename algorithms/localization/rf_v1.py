"""
Random Forest方法人体定位
使用简化版特征提取，无range bin 范围限制
"""

import numpy as np
import joblib
import json
import os
from typing import Dict, Any, Tuple
from scipy import stats
from .base_localization import BaseLocalizationMethod


class RandomForestLocalization(BaseLocalizationMethod):
    """
    基于随机森林模型的人体定位方法
    
    使用预训练的随机森林模型预测人体所在的Range Bin
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
        
        print(f"✅ 成功加载随机森林模型: {os.path.basename(model_path)}")
        print(f"   测试集AUC: {self.metadata.get('test_auc', 'N/A')}")
        print(f"   测试集F1: {self.metadata.get('test_f1', 'N/A')}")
    
    def _extract_features(self, segment_data: Dict[str, Any]) -> np.ndarray:
        """
        提取特征
        
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
        
        features = np.zeros((R, nVX, n_features))
        
        for r in range(R):
            for v in range(nVX):
                amp_ts = magnitude[r, v, :]
                phase_ts = unfiltered_phase[r, v, :]
                
                feature_vector = []
                
                for feat_name in feature_names:
                    if 'amp_mean' in feat_name:
                        feature_vector.append(np.mean(amp_ts))
                    elif 'amp_std' in feat_name:
                        feature_vector.append(np.std(amp_ts))
                    elif 'amp_p2p' in feat_name:
                        feature_vector.append(np.max(amp_ts) - np.min(amp_ts))
                    elif 'amp_skewness' in feat_name:
                        feature_vector.append(stats.skew(amp_ts))
                    elif 'amp_kurtosis' in feat_name:
                        feature_vector.append(stats.kurtosis(amp_ts))
                    elif 'phase_diff_std' in feat_name:
                        phase_diff = np.diff(phase_ts)
                        feature_vector.append(np.std(phase_diff))
                    elif 'phase_diff_range' in feat_name:
                        phase_diff = np.diff(phase_ts)
                        feature_vector.append(np.max(phase_diff) - np.min(phase_diff))
                    elif 'amp_snr_db' in feat_name or 'phase_snr_db' in feat_name:
                        feature_vector.append(10.0)
                    else:
                        feature_vector.append(0.0)
                
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
        # 提取特征
        features = self._extract_features(segment_data)
        
        # 获取RX索引
        rx_index = segment_data['rx_index']  # 1-based
        rx_idx_0based = rx_index - 1
        
        R = features.shape[0]
        
        # 提取该RX的特征并标准化
        features_flat = features[:, rx_idx_0based, :]  # (R x n_features)
        features_scaled = self.scaler.transform(features_flat)
        
        # 预测概率
        probabilities = self.model.predict_proba(features_scaled)[:, 1]  # 人体类别的概率
        
        # 选择概率最高的Range Bin
        predicted_rb_0based = np.argmax(probabilities)
        predicted_confidence = probabilities[predicted_rb_0based]
        
        selection_info = {
            'method': 'random_forest',
            'confidence': float(predicted_confidence),
            'probabilities': probabilities.tolist()
        }
        
        return predicted_rb_0based, selection_info
    
    def get_method_name(self) -> str:
        """获取方法名称"""
        return "RandomForest"