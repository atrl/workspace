from abc import ABC, abstractmethod

class BasePanicCalculator(ABC):
    """恐慌指数计算器基类"""
    
    def __init__(self, config):
        """初始化恐慌指数计算器"""
        self.config = config
        
    @abstractmethod
    def calculate_base_score(self, vix):
        """计算基础恐慌指数"""
        pass
        
    @abstractmethod
    def calculate_final_score(self, base_score, rsi6):
        """计算最终恐慌指数"""
        pass
        
    def process_data(self, df):
        """处理数据，计算恐慌指数"""
        # 计算基础恐慌指数
        df['base_panic_score'] = df['vix'].apply(self.calculate_base_score)
        
        # 计算最终恐慌指数
        df['panic_score'] = df.apply(
            lambda row: self.calculate_final_score(row['base_panic_score'], row['RSI6']), 
            axis=1
        )
        
        return df 