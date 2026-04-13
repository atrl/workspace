import numpy as np
import pandas as pd
from .base import BasePanicCalculator

class EnhancedPanicCalculator(BasePanicCalculator):
    """增强型恐慌指数计算器
    
    综合考虑VIX、交易量和RSI6三个因素：
    1. VIX作为基础恐慌值的计算依据
    2. 交易量作为情绪放大器
    3. RSI6作为短期超买超卖指标的调节因子
    """
    
    def calculate_volume_multiplier(self, volume, avg_volume):
        """计算交易量放大系数
        
        Args:
            volume: 当前交易量
            avg_volume: 过去N天的平均交易量
            
        Returns:
            float: 交易量放大系数，范围在[0.8, 1.5]之间
        """
        if pd.isna(volume) or pd.isna(avg_volume) or avg_volume == 0:
            return 1.0
            
        # 计算相对交易量
        relative_volume = volume / avg_volume
        
        # 使用对数函数平滑处理，避免极端值
        log_relative_volume = np.log1p(relative_volume)
        
        # 将放大系数限制在[0.8, 1.5]范围内
        multiplier = 1.0 + 0.5 * np.tanh(log_relative_volume - 1)
        return max(0.8, min(1.5, multiplier))
    
    def calculate_base_score(self, vix):
        """计算基础恐慌指数
        
        使用S型函数映射VIX值，以获得更平滑的变化曲线
        """
        vix_low = self.config['VIX_LOW']
        vix_high = self.config['VIX_HIGH']
        fear_min = self.config['FEAR_MIN']
        fear_max = self.config['FEAR_MAX']
        sigmoid_steepness = self.config['SIGMOID_STEEPNESS']
        
        # 将VIX值限制在指定范围内
        vix_clipped = max(min(vix, vix_high), vix_low)
        
        # 计算中点和相对位置
        vix_mid = (vix_high + vix_low) / 2
        x = -(vix_clipped - vix_mid) * sigmoid_steepness
        
        # 使用sigmoid函数进行映射
        sigmoid_value = 1 / (1 + np.exp(-x))
        
        return sigmoid_value * (fear_max - fear_min) + fear_min
    
    def calculate_final_score(self, base_score, rsi6):
        """计算最终恐慌指数
        
        根据RSI6值对基础恐慌指数进行调整，并考虑交易量的影响
        """
        if pd.isna(rsi6):
            return base_score
            
        rsi6_value = float(rsi6)
        
        # RSI6调整系数
        if rsi6_value <= 30:
            # 超卖区域，降低恐慌程度
            adjustment = 25 + (30 - rsi6_value) * 3
            new_score = base_score - adjustment
        elif rsi6_value >= 70:
            # 超买区域，提高恐慌程度
            adjustment = 25 + (rsi6_value - 70) * 1.5
            new_score = base_score + adjustment
        else:
            # 中性区域，保持原值
            return base_score
            
        # 确保最终得分在有效范围内
        return max(-110, min(110, new_score))
    
    def process_data(self, df):
        """处理数据，计算考虑交易量的恐慌指数"""
        # 计算20日平均交易量
        if 'volume' in df.columns:
            df['avg_volume'] = df['volume'].rolling(window=20).mean()
            df['volume_multiplier'] = df.apply(
                lambda row: self.calculate_volume_multiplier(row['volume'], row['avg_volume']),
                axis=1
            )
        else:
            df['volume_multiplier'] = 1.0
        
        # 计算基础恐慌指数
        df['base_panic_score'] = df['vix'].apply(self.calculate_base_score)
        
        # 应用交易量放大系数
        df['volume_adjusted_score'] = df['base_panic_score'] * df['volume_multiplier']
        
        # 计算最终恐慌指数
        df['panic_score'] = df.apply(
            lambda row: self.calculate_final_score(row['volume_adjusted_score'], row['RSI6']),
            axis=1
        )
        
        return df