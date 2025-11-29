"""
人体定位算法基类
"""
import numpy as np
from typing import Dict, Any, Tuple


class BaseLocalizationMethod:
    """
    人体定位算法基类
    所有具体的定位方法都应继承此类
    """
    
    def __init__(self):
        """初始化定位方法"""
        pass
    
    def select_range_bin(self, segment_data: Dict[str, Any], 
                        predefined_rb_index: int = None) -> Tuple[int, Dict[str, Any]]:
        """
        选择最佳Range Bin（需要在子类中实现）
        
        参数:
            segment_data: 片段数据字典，包含:
                - filtered_phase: (R, nVX, T)
                - magnitude: (R, nVX, T)
                - unfiltered_phase: (R, nVX, T)
                - distance: 距离
                - true_rb_index: 真实RB索引（1-based）
                - rx_index: RX索引（1-based）
            predefined_rb_index: 预定义的Range Bin索引（0-based）
        
        返回:
            (selected_rb_index, selection_info)
            - selected_rb_index: 选中的Range Bin索引（0-based）
            - selection_info: 选择信息字典
        """
        raise NotImplementedError("子类必须实现select_range_bin方法")
    
    def get_method_name(self) -> str:
        """
        获取方法名称（需要在子类中实现）
        """
        raise NotImplementedError("子类必须实现get_method_name方法")