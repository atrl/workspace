from datetime import datetime

class BacktestAnalyzer:
    """回测结果分析类"""
    
    def __init__(self, config, position_manager, trades, df, capital_history):
        """初始化分析器"""
        self.config = config
        self.position_manager = position_manager
        self.trades = trades
        self.df = df
        self.capital_history = capital_history
        
    def calculate_annual_stats(self):
        """计算年度统计数据"""
        annual_stats = {}
        trades_by_year = {}
        
        for trade in self.trades:
            year = datetime.strptime(trade['date'], '%Y-%m-%d').year
            if year not in trades_by_year:
                trades_by_year[year] = []
            trades_by_year[year].append(trade)
        
        for year in trades_by_year.keys():
            year_data = self.df[self.df['date'].dt.year == year]
            year_trades = trades_by_year[year]
            
            if len(year_data) > 0:
                first_day = year_data.iloc[0]
                last_day = year_data.iloc[-1]
                
                year_start_idx = self.df.index.get_loc(first_day.name)
                year_end_idx = self.df.index.get_loc(last_day.name)
                
                if year_end_idx >= 20:
                    if year_start_idx < 20:
                        year_start_capital = self.config['INITIAL_CAPITAL']
                    else:
                        year_start_capital = self.capital_history[year_start_idx - 20]
                    
                    year_end_capital = self.capital_history[year_end_idx - 20]
                    year_return = (year_end_capital - year_start_capital) / year_start_capital * 100
                    
                    max_drawdown = 0
                    peak = year_start_capital
                    
                    for capital in self.capital_history[max(0, year_start_idx - 20):year_end_idx - 19]:
                        if capital > peak:
                            peak = capital
                        drawdown = (peak - capital) / peak * 100
                        max_drawdown = max(max_drawdown, drawdown)
                    
                    annual_stats[year] = {
                        'return': year_return,
                        'trades': len(year_trades),
                        'max_drawdown': max_drawdown
                    }
        
        return annual_stats
        
    def calculate_total_return(self):
        """计算总收益"""
        initial_capital = self.config['INITIAL_CAPITAL']
        final_capital = self.capital_history[-1]
        return (final_capital - initial_capital) / initial_capital * 100
        
    def calculate_max_drawdown(self):
        """计算最大回撤"""
        max_drawdown = 0
        peak = self.capital_history[0]
        
        for capital in self.capital_history:
            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
            
        return max_drawdown 