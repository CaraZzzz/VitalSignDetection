"""
Manual方法人体定位
"""
import numpy as np
from typing import Dict, Any, Tuple, Optional
from .base_localization import BaseLocalizationMethod


class ManualLocalization(BaseLocalizationMethod):
    """
    手动人体定位方法
    
    使用预定义的距离到Range Bin的映射表
    """
    
    def __init__(self, rb_mapping: Optional[Dict[int, int]] = None):
        """
        初始化Manual定位方法
        
        参数:
            rb_mapping: 距离到Range Bin的映射表（1-based）
                       例如: {40: 6, 50: 7, 60: 8}
                       表示40cm用RB6，50cm用RB7，60cm用RB8
                       如果为None，则使用predefined_rb_index
        """
        super().__init__()
        self.rb_mapping = rb_mapping
    
    def select_range_bin(self, segment_data: Dict[str, Any], 
                        predefined_rb_index: int = None) -> Tuple[int, Dict[str, Any]]:
        """
        使用手动映射选择Range Bin
        
        参数:
            segment_data: 片段数据字典
            predefined_rb_index: 预定义的Range Bin索引（0-based）
        
        返回:
            (selected_rb_index, selection_info)
        """
        if self.rb_mapping is not None:
            # 使用映射表
            distance = segment_data.get('distance', None)
            if distance in self.rb_mapping:
                # 映射表中的值是1-based，需要转换为0-based
                selected_rb_1based = self.rb_mapping[distance]
                selected_rb_0based = selected_rb_1based - 1
                
                selection_info = {
                    'method': 'manual',
                    'source': 'mapping',
                    'mapping_used': f"{distance}cm -> RB{selected_rb_1based}",
                    'confidence': 1.0
                }
                
                return selected_rb_0based, selection_info
            else:
                print(f"⚠️ 距离 {distance}cm 不在映射表中")
                if predefined_rb_index is None:
                    raise ValueError(f"距离 {distance}cm 不在映射表中，且未提供predefined_rb_index")
        
        # 使用预定义索引
        if predefined_rb_index is None:
            raise ValueError("Manual方法需要提供rb_mapping或predefined_rb_index")
        
        selection_info = {
            'method': 'manual',
            'source': 'predefined',
            'confidence': 1.0
        }
        
        return predefined_rb_index, selection_info
    
    def get_method_name(self) -> str:
        """获取方法名称"""
        return "Manual"