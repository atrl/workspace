import numpy as np
import pandas as pd
from .base import BasePanicCalculator

class SigmoidPanicCalculator(BasePanicCalculator):
    """基于S型映射的恐慌指数计算器"""
    
    def calculate_base_score(self, vix):
        """S型映射VIX值到基础恐慌指数"""
        vix_low = self.config['VIX_LOW']
        vix_high = self.config['VIX_HIGH']
        fear_min = self.config['FEAR_MIN']
        fear_max = self.config['FEAR_MAX']
        sigmoid_steepness = self.config['SIGMOID_STEEPNESS']
        
        vix_clipped = max(min(vix, vix_high), vix_low)
        vix_mid = (vix_high + vix_low) / 2
        x = -(vix_clipped - vix_mid) * sigmoid_steepness
        sigmoid_value = 1 / (1 + np.exp(-x))
        
        return sigmoid_value * (fear_max - fear_min) + fear_min
    
    def calculate_final_score(self, base_score, rsi6):
        """根据RSI6调整基础恐慌指数，计算最终得分"""
        if pd.isna(rsi6):
            return base_score
            
        rsi6_value = float(rsi6)
        
        if rsi6_value <= 30:
            # RSI超卖区域，恐慌指数下调
            adjustment = 20 + (30 - rsi6_value) * 2.5
            new_score = base_score - adjustment
            return max(-150, new_score)
        elif rsi6_value >= 70:
            # RSI超买区域，恐慌指数上调
            adjustment = 20 + (rsi6_value - 70)
            new_score = base_score + adjustment
            return min(150, new_score)
        else:
            return base_score 