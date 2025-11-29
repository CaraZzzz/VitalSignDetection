"""
心率估计算法包
"""
from .base_estimation import BaseHeartRateEstimator
from .fft import FFTHeartRateEstimator
from .stft import STFTHeartRateEstimator
from .wavelet import WaveletHeartRateEstimator
from .dct import DCTHeartRateEstimator
from .vmd import VMDHeartRateEstimator
from .emd import EMDHeartRateEstimator
from .eemd import EEMDHeartRateEstimator

__all__ = [
    'BaseHeartRateEstimator',
    'FFTHeartRateEstimator',
    'STFTHeartRateEstimator',
    'WaveletHeartRateEstimator',
    'DCTHeartRateEstimator',
    'VMDHeartRateEstimator',
    'EMDHeartRateEstimator',
    'EEMDHeartRateEstimator',
]