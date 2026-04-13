import pandas as pd
import numpy as np
from .panic_calculators import SigmoidPanicCalculator

class IndicatorCalculator:
    """技术指标计算器"""
    
    def __init__(self, config):
        """初始化指标计算器"""
        self.config = config
        self.panic_calculator = SigmoidPanicCalculator(config)
        
    def calculate_rsi(self, prices, period=6):
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).ewm(span=period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=period, adjust=False).mean()
        rs = gain / (loss + 1e-5)
        return 100 - (100 / (1 + rs))
        
    def calculate_moving_averages(self, df):
        """计算移动平均线"""
        df['MA5'] = df['price'].rolling(window=5).mean()
        df['MA20'] = df['price'].rolling(window=180).mean()
        return df
        
    def calculate_historical_volatility(self, prices, window=20):
        """计算历史波动率"""
        returns = np.log(prices / prices.shift(1))
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # 年化
        return volatility
        
    def calculate_volatility_coefficient(self, df):
        """计算波动率系数"""
        volatility_median = self.config['VOLATILITY_MEDIAN']
        df['VOL_COEF'] = (df['HV_20'] / volatility_median).clip(0.5, 2.0)
        return df
        
    def process_data(self, df):
        """处理所有技术指标"""
        # 计算RSI
        df['RSI6'] = self.calculate_rsi(df['price'], 6)
        
        # 计算移动平均线
        df = self.calculate_moving_averages(df)
        
        # 计算历史波动率
        df['HV_20'] = self.calculate_historical_volatility(df['price'], window=20)
        df['HV_30'] = self.calculate_historical_volatility(df['price'], window=30)
        
        # 将波动率的NaN值填充为0
        df['HV_20'] = df['HV_20'].fillna(0)
        df['HV_30'] = df['HV_30'].fillna(0)
        
        # 计算波动率系数
        df = self.calculate_volatility_coefficient(df)
        
        # 计算恐慌指数
        df = self.panic_calculator.process_data(df)
        
        return df