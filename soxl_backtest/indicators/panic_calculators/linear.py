import numpy as np
import pandas as pd
from .base import BasePanicCalculator

class LinearPanicCalculator(BasePanicCalculator):
    """基于线性映射的恐慌指数计算器"""
    
    def calculate_base_score(self, vix):
        """线性映射VIX值到基础恐慌指数
        
        使用分段线性映射：
        1. VIX <= vix_low: 最大乐观值 (fear_max)
        2. vix_low < VIX < vix_high: 线性映射
        3. VIX >= vix_high: 最大恐慌值 (fear_min)
        """
        vix_low = self.config['VIX_LOW']
        vix_high = self.config['VIX_HIGH']
        fear_min = self.config['FEAR_MIN']
        fear_max = self.config['FEAR_MAX']
        
        # 将VIX值限制在指定范围内
        vix_clipped = max(min(vix, vix_high), vix_low)
        
        # 线性映射
        slope = (fear_min - fear_max) / (vix_high - vix_low)
        intercept = fear_max - slope * vix_low
        
        return slope * vix_clipped + intercept
    
    def calculate_final_score(self, base_score, rsi6):
        """根据RSI6调整基础恐慌指数，计算最终得分
        
        使用与SigmoidPanicCalculator相同的RSI调整逻辑，
        以便于比较两种方法的差异主要来自基础分数的计算方式
        """
        if pd.isna(rsi6):
            return base_score
            
        rsi6_value = float(rsi6)
        
        if rsi6_value <= 30:
            # RSI超卖区域，恐慌指数下调
            adjustment = 20 + (30 - rsi6_value) * 2.5
            new_score = base_score - adjustment
            return max(-110, new_score)
        elif rsi6_value >= 70:
            # RSI超买区域，恐慌指数上调
            adjustment = 20 + (rsi6_value - 70)
            new_score = base_score + adjustment
            return min(110, new_score)
        else:
            return base_score 