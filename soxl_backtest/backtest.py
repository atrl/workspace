from datetime import datetime
from .config.backtest_config import BacktestConfig
from .data.data_fetcher import LongbridgeDataFetcher, YahooDataFetcher
from .indicators.indicator_calculator import IndicatorCalculator
from .position.position_manager import PositionManager
from .strategy.buy_strategy import BuyStrategy
from .strategy.sell_strategy import SellStrategy
from .analysis.backtest_analyzer import BacktestAnalyzer

class SOXLBacktest:
    """SOXL回测主类"""
    
    def __init__(self, params):
        """初始化回测"""
        # 初始化配置
        self.config = BacktestConfig(params)
        
        # 初始化数据获取器
        self.data_fetcher = LongbridgeDataFetcher() if self.config['USE_LONGBRIDGE'] else YahooDataFetcher()
        
        # 获取数据
        self.df = self.data_fetcher.fetch_data(self.config)
        
        # 过滤到实际开始日期
        self.df = self.df[self.df['date'] >= self.config['START_DATE']]
        
        # 初始化指标计算器
        self.indicator_calculator = IndicatorCalculator(self.config)
        
        # 计算所有技术指标（包括恐慌指数）
        self.df = self.indicator_calculator.process_data(self.df)
        
        # 初始化仓位管理器
        self.position_manager = PositionManager(self.config)
        
        # 初始化交易策略
        self.buy_strategy = BuyStrategy(self.config, self.position_manager)
        self.sell_strategy = SellStrategy(self.config, self.position_manager)
        
        # 初始化历史记录
        self.position_history = []
        self.capital_history = []
        self.price_history = []
        self.panic_score_history = []
        self.hv_20_history = []
        self.hv_30_history = []
        self.vol_coef_history = []
        self.trades = []
        
        # 最后一个交易日状态
        self.last_day_status = {
            'date': None,
            'price': None,
            'position': None,
            'panic_score': None,
            'no_trade_reason': None
        }
        
    def execute_buy_overvalued(self, current_data, total_value, buy_step):
        """执行高估时的买入操作"""
        # 高估时买入步长为正常步长的40%
        overvalued_buy_step = buy_step * 0.4
        
        # 计算目标仓位和需要买入的金额
        target_position = min(self.position_manager.position + overvalued_buy_step, self.config['MAX_POSITION'])
        buy_amount = total_value * (target_position - self.position_manager.position)
        
        # 计算实际可以买入的股数
        new_shares = int(buy_amount / current_data['price'])
        if new_shares > 0:
            cost = new_shares * current_data['price']
            self.position_manager.add_position(current_data['price'], new_shares)
            self.position_manager.capital -= cost
            
            # # 如果是通过累积跌幅条件买入，设置冷却期
            # if self.buy_strategy.last_buy_was_cumulative:
            #     self.buy_strategy.cumulative_buy_cooldown = self.config['CUMULATIVE_DAYS']
            
            # 更新仓位
            new_total_value = self.position_manager.update_position(current_data['price'])
            
            return {
                'date': current_data['date'].strftime('%Y-%m-%d'),
                'type': 'buy',
                'price': current_data['price'],
                'position': self.position_manager.position,
                'score': current_data['panic_score'],
                'total_value': new_total_value,
                'shares': self.position_manager.shares,
                'avg_price': self.position_manager.avg_price,
                'buy_type': 'cumulative' if self.buy_strategy.last_buy_was_cumulative else 'daily',
                'market_condition': 'overvalued',
                'buy_step': overvalued_buy_step,
                'trigger_conditions': self.buy_strategy.get_last_buy_conditions()
            }
        return None

    def execute_buy_undervalued(self, current_data, total_value, buy_step):
        """执行低估时的买入操作"""
        # 计算目标仓位和需要买入的金额
        target_position = min(self.position_manager.position + buy_step, self.config['MAX_POSITION'])
        buy_amount = total_value * (target_position - self.position_manager.position)
        
        # 计算实际可以买入的股数
        new_shares = int(buy_amount / current_data['price'])
        if new_shares > 0:
            cost = new_shares * current_data['price']
            self.position_manager.add_position(current_data['price'], new_shares)
            self.position_manager.capital -= cost
            
            # # 如果是通过累积跌幅条件买入，设置冷却期
            # if self.buy_strategy.last_buy_was_cumulative:
            #     self.buy_strategy.cumulative_buy_cooldown = self.config['CUMULATIVE_DAYS']
            
            # 更新仓位
            new_total_value = self.position_manager.update_position(current_data['price'])
            
            return {
                'date': current_data['date'].strftime('%Y-%m-%d'),
                'type': 'buy',
                'price': current_data['price'],
                'position': self.position_manager.position,
                'score': current_data['panic_score'],
                'total_value': new_total_value,
                'shares': self.position_manager.shares,
                'avg_price': self.position_manager.avg_price,
                'buy_type': 'cumulative' if self.buy_strategy.last_buy_was_cumulative else 'daily',
                'market_condition': 'undervalued',
                'buy_step': buy_step,
                'trigger_conditions': self.buy_strategy.get_last_buy_conditions()
            }
        return None

    def execute_sell(self, current_data, sell_type):
        """执行卖出操作"""
        total_value = self.position_manager.capital + self.position_manager.shares * current_data['price']
        
        if sell_type == 'take_profit':
            # 大幅止盈，卖出95%的仓位
            sell_amount = self.position_manager.position * self.config['TAKE_PROFIT_POSITION']
            sell_shares = int((total_value * sell_amount) / current_data['price'])
            sell_type_str = 'take_profit'
        elif sell_type == 'stop_loss':
            # 止损，卖出配置的止损仓位
            sell_amount = self.position_manager.position * self.config['STOP_LOSS_POSITION']
            sell_shares = int((total_value * sell_amount) / current_data['price'])
            sell_type_str = 'stop_loss'
        else:
            # 获取最后一次买入的持仓信息
            last_buy = self.position_manager.get_last_position()
            if last_buy:
                last_buy_profit = (current_data['price'] - last_buy['price']) / last_buy['price']
            else:
                last_buy_profit = 0
            
            # 如果是针对最近一次买入的获利了结，则卖出最后一次买入的股数，但确保不会低于最小仓位
            if (current_data['panic_score'] >= self.config['SELL_SCORE_THRESHOLD'] and
                last_buy_profit >= self.config['LAST_BUY_PROFIT_TARGET']):
                # 计算卖出后的仓位
                remaining_shares = self.position_manager.shares - last_buy['shares']
                remaining_value = remaining_shares * current_data['price']
                new_total_value = self.position_manager.capital + remaining_value + last_buy['shares'] * current_data['price']
                new_position = remaining_value / new_total_value
                
                if new_position >= self.config['MIN_POSITION']:
                    # 如果卖出全部最后一次买入的股数后仍然高于最小仓位，则全部卖出
                    sell_shares = last_buy['shares']
                else:
                    # 否则只卖出部分股数，确保保持最小仓位
                    target_value = new_total_value * self.config['MIN_POSITION']
                    keep_shares = int(target_value / current_data['price'])
                    sell_shares = self.position_manager.shares - keep_shares
            else:
                # 原有的卖出逻辑
                sell_amount = min(self.config['SELL_AMOUNT'], 
                                self.position_manager.position - self.config['MIN_POSITION'])
                sell_shares = int((total_value * sell_amount) / current_data['price'])
            sell_type_str = 'sell'
        
        if sell_shares > 0:
            # 按FIFO原则卖出股票
            self.position_manager.reduce_position(sell_shares)
            self.position_manager.capital += sell_shares * current_data['price']
            
            # 更新仓位
            new_total_value = self.position_manager.update_position(current_data['price'])
            
            return {
                'date': current_data['date'].strftime('%Y-%m-%d'),
                'type': sell_type_str,
                'price': current_data['price'],
                'position': self.position_manager.position,
                'score': current_data['panic_score'],
                'gain': ((current_data['price'] - self.position_manager.avg_price) / self.position_manager.avg_price) * 100 if self.position_manager.avg_price > 0 else 0,
                'total_value': new_total_value,
                'shares': self.position_manager.shares,
                'avg_price': self.position_manager.avg_price,
                'trigger_conditions': self.sell_strategy.get_last_sell_conditions()
            }
        return None

    def run(self):
        """运行回测"""
        print(f"开始运行回测...")
        days_since_update = 0
        max_capital = self.config['INITIAL_CAPITAL']  # 记录历史最高资金
        current_drawdown = 0  # 当前回撤
        max_drawdown = 0  # 最大回撤
        
        for i in range(0, len(self.df)):
            current_row = self.df.iloc[i]
            prev_row = self.df.iloc[i-1] if i > 0 else None
            
            # 更新累积跌幅买入的冷却计数器
            if self.buy_strategy.cumulative_buy_cooldown > 0:
                self.buy_strategy.cumulative_buy_cooldown -= 1
            
            # 根据配置的频率更新实际仓位
            days_since_update += 1
            if days_since_update >= 20:
                if self.position_manager.shares > 0:
                    total_value = self.position_manager.update_position(current_row['price'])
                    
                    # 检查是否完成建仓
                    if not self.position_manager.initial_position_built and self.position_manager.position >= self.config['MIN_POSITION']:
                        self.position_manager.initial_position_built = True
                        print(f"首次建仓完成 - 日期: {current_row['date'].strftime('%Y-%m-%d')}, 仓位: {self.position_manager.position:.2%}")
                        
                days_since_update = 0
            
            # 买入条件
            if self.buy_strategy.should_buy(current_row, self.df, i):
                total_value = self.position_manager.capital + self.position_manager.shares * current_row['price']
                
                # 如果是首次建仓（仓位小于最小仓位），使用初始建仓步长
                if not self.position_manager.initial_position_built:
                    buy_step = self.config['BUY_STEP_INITIAL']
                else:
                    buy_step = self.config['BUY_STEP_NORMAL']  # 正常加仓时的买入步长
                
                # 执行买入操作
                is_overvalued = current_row['price'] > current_row['MA20']
                trade_info = (self.execute_buy_overvalued(current_row, total_value, buy_step) if is_overvalued 
                            else self.execute_buy_undervalued(current_row, total_value, buy_step))
                
                if trade_info:
                    self.trades.append(trade_info)
                    if is_overvalued:
                        print(f"执行高估时买入 - 日期: {trade_info['date']}, 买入股数: {trade_info['shares']}, 价格: {trade_info['price']:.2f}, 买入步长: {trade_info['buy_step']:.1%}")
                    else:
                        print(f"执行低估时买入 - 日期: {trade_info['date']}, 买入股数: {trade_info['shares']}, 价格: {trade_info['price']:.2f}, 买入步长: {trade_info['buy_step']:.1%}")
            
            # 卖出条件
            elif (sell_type := self.sell_strategy.should_sell(current_row)):
                trade_info = self.execute_sell(current_row, sell_type)
                if trade_info:
                    self.trades.append(trade_info)
                    if sell_type == 'take_profit':
                        print(f"触发止盈 - 日期: {trade_info['date']}, 累积收益: {trade_info['gain']:.2f}%")
                    elif sell_type == 'stop_loss':
                        print(f"触发止损 - 日期: {trade_info['date']}, 累积亏损: {trade_info['gain']:.2f}%")
                    else:
                        print(f"执行卖出 - 日期: {trade_info['date']}, 卖出股数: {trade_info['shares']}, 价格: {trade_info['price']:.2f}")
            
            # 记录每日数据
            self.position_history.append(self.position_manager.position)
            self.price_history.append(current_row['price'])
            self.panic_score_history.append(current_row['panic_score'])
            self.hv_20_history.append(current_row['HV_20'])
            self.hv_30_history.append(current_row['HV_30'])
            self.vol_coef_history.append(current_row['VOL_COEF'])
            total_value = self.position_manager.capital + self.position_manager.shares * current_row['price']
            self.capital_history.append(total_value)
            
            # 更新回撤相关数据
            if total_value > max_capital:
                max_capital = total_value
            current_drawdown = (max_capital - total_value) / max_capital * 100
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown
                print(f"新的最大回撤 - 日期: {current_row['date'].strftime('%Y-%m-%d')}, 回撤: {max_drawdown:.2f}%")
            
            # 记录最后一个交易日的状态
            self.last_day_status = {
                'date': current_row['date'].strftime('%Y-%m-%d'),
                'price': current_row['price'],
                'position': self.position_manager.position,
                'panic_score': current_row['panic_score'],
                'no_trade_reason': None
            }
            
            # 如果没有执行买入或卖出操作，记录原因
            if not self.buy_strategy.should_buy(current_row, self.df, i) and not self.sell_strategy.should_sell(current_row):
                buy_reason = self.buy_strategy.get_no_buy_reason()
                sell_reason = self.sell_strategy.get_no_sell_reason()
                
                if buy_reason and sell_reason:
                    self.last_day_status['no_trade_reason'] = f"未买入: {buy_reason}; 未卖出: {sell_reason}"
                elif buy_reason:
                    self.last_day_status['no_trade_reason'] = f"未买入: {buy_reason}"
                elif sell_reason:
                    self.last_day_status['no_trade_reason'] = f"未卖出: {sell_reason}"
                
    def calculate_annual_stats(self):
        """计算年度统计"""
        analyzer = BacktestAnalyzer(self.config, self.position_manager, self.trades, self.df, self.capital_history)
        return analyzer.calculate_annual_stats()
        
   