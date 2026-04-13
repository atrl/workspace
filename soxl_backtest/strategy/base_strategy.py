from abc import ABC

class BaseStrategy(ABC):
    """交易策略基类"""
    
    def __init__(self, config, position_manager):
        self.config = config
        self.position_manager = position_manager
        self.cumulative_buy_cooldown = 0
        self.last_buy_was_cumulative = False
        
    def get_panic_score(self, row):
        """获取恐慌指数得分"""
        return row['panic_score']
        
    def calculate_max_cumulative_return(self, df, current_idx, last_buy_date=None):
        """计算最大累积跌幅
        Args:
            df: 数据DataFrame
            current_idx: 当前索引
            last_buy_date: 上次买入日期，用于限制计算窗口
        """
        max_cumulative_return = 0
        current_price = df.iloc[current_idx]['price']
        self.actual_cumulative_days = 0
        
        if current_idx >= 2:
            # 计算最大窗口大小
            max_window = self.config['CUMULATIVE_DAYS']
            if last_buy_date is not None:
                days_since_last_buy = (df.iloc[current_idx]['date'] - last_buy_date).days
                max_window = min(max_window, days_since_last_buy)
            
            for days in range(2, max_window + 1):
                if current_idx >= days:
                    price_days_ago = df.iloc[current_idx - days]['price']
                    cumulative_return = (current_price - price_days_ago) / price_days_ago
                    if cumulative_return < max_cumulative_return:
                        max_cumulative_return = cumulative_return
                        self.actual_cumulative_days = days
        
        return max_cumulative_return