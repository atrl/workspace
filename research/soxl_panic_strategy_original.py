#!/usr/bin/env python3
"""
SOXL Panic Strategy - Original Version Comparison

使用原始策略参数运行回测，与Qlib版本对比

Author: clawQuant
Date: 2026-04-02
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import sys
sys.path.insert(0, '/Users/rziiiii/.openclaw/media/inbound')

# 导入原始策略组件
from soxl_backtest.config.backtest_config import BacktestConfig
from soxl_backtest.position.position_manager import PositionManager
from soxl_backtest.strategy.buy_strategy import BuyStrategy
from soxl_backtest.strategy.sell_strategy import SellStrategy
from soxl_backtest.indicators.indicator_calculator import IndicatorCalculator


class SOXLOriginalBacktest:
    """原始策略回测"""
    
    def __init__(self, params=None):
        """初始化回测"""
        # 初始化配置
        self.config = BacktestConfig(params)
        
        # 初始化仓位管理器
        self.position_manager = PositionManager(self.config)
        
        # 初始化交易策略
        self.buy_strategy = BuyStrategy(self.config, self.position_manager)
        self.sell_strategy = SellStrategy(self.config, self.position_manager)
        
        # 初始化历史记录
        self.position_history = []
        self.capital_history = []
        self.trades = []
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备数据，使用VIX替代方案"""
        df = df.copy()
        
        # 计算RSI6
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))
        avg_gain = gain.rolling(window=6).mean()
        avg_loss = loss.rolling(window=6).mean()
        rs = avg_gain / avg_loss
        df['RSI6'] = 100 - (100 / (1 + rs))
        
        # 计算历史波动率作为VIX替代
        returns = df['close'].pct_change()
        df['HV_20'] = returns.rolling(window=20).std() * np.sqrt(252)
        df['HV_30'] = returns.rolling(window=30).std() * np.sqrt(252)
        
        # 将HV映射到VIX范围 (SOXL HV通常在0.3-0.8，映射到VIX 10-30)
        # vix = hv * 40 (近似)
        df['vix'] = df['HV_20'] * 40
        
        # 填充NaN
        df['RSI6'] = df['RSI6'].fillna(50)
        df['HV_20'] = df['HV_20'].fillna(self.config['VOLATILITY_MEDIAN'])
        df['HV_30'] = df['HV_30'].fillna(self.config['VOLATILITY_MEDIAN'])
        df['vix'] = df['vix'].fillna(20)
        
        # 计算波动率系数
        if self.config['VOLATILITY_MEDIAN'] != 0:
            df['VOL_COEF'] = (df['HV_20'] / self.config['VOLATILITY_MEDIAN']).clip(0.5, 2.0)
        else:
            df['VOL_COEF'] = 1.0
        
        # 计算移动平均
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        
        # 计算恐慌指数 (使用Sigmoid)
        df = self.calculate_panic_score(df)
        
        return df
    
    def calculate_panic_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算恐慌指数"""
        vix = df['vix']
        rsi6 = df['RSI6']
        
        # Sigmoid映射
        vix_low = self.config['VIX_LOW']
        vix_high = self.config['VIX_HIGH']
        fear_min = self.config['FEAR_MIN']
        fear_max = self.config['FEAR_MAX']
        sigmoid_steepness = self.config['SIGMOID_STEEPNESS']
        
        vix_clipped = vix.clip(vix_low, vix_high)
        vix_mid = (vix_high + vix_low) / 2
        x = -(vix_clipped - vix_mid) * sigmoid_steepness
        sigmoid_value = 1 / (1 + np.exp(-x))
        
        base_score = sigmoid_value * (fear_max - fear_min) + fear_min
        
        # RSI6调整
        df['panic_score'] = base_score.copy()
        
        # 超卖调整
        oversold = rsi6 <= 30
        df.loc[oversold, 'panic_score'] = base_score[oversold] - (25 + (30 - rsi6[oversold]) * 3)
        
        # 超买调整
        overbought = rsi6 >= 70
        df.loc[overbought, 'panic_score'] = base_score[overbought] + (25 + (rsi6[overbought] - 70) * 1.5)
        
        # 限制范围
        df['panic_score'] = df['panic_score'].clip(-110, 110)
        
        return df
    
    def execute_buy(self, current_data: pd.Series, buy_step: float) -> Dict:
        """执行买入"""
        total_value = self.position_manager.capital + self.position_manager.shares * current_data['close']
        
        # 计算目标仓位
        target_position = min(self.position_manager.position + buy_step, self.config['MAX_POSITION'])
        buy_amount = total_value * (target_position - self.position_manager.position)
        
        # 计算买入股数
        new_shares = int(buy_amount / current_data['close'])
        if new_shares <= 0:
            return None
        
        # 执行买入
        cost = new_shares * current_data['close']
        self.position_manager.add_position(current_data['close'], new_shares)
        self.position_manager.capital -= cost
        
        # 更新仓位
        new_total_value = self.position_manager.update_position(current_data['close'])
        
        # 标记初始建仓
        if not self.position_manager.initial_position_built and self.position_manager.position >= self.config['MIN_POSITION']:
            self.position_manager.initial_position_built = True
        
        return {
            'type': 'buy',
            'date': current_data['date'],
            'price': current_data['close'],
            'shares': new_shares,
            'position': self.position_manager.position,
            'avg_price': self.position_manager.avg_price,
            'total_value': new_total_value
        }
    
    def execute_sell(self, current_data: pd.Series, sell_type: str) -> Dict:
        """执行卖出"""
        total_value = self.position_manager.capital + self.position_manager.shares * current_data['close']
        
        # 确定卖出仓位
        if sell_type == 'take_profit':
            sell_amount = self.position_manager.position * self.config['TAKE_PROFIT_POSITION']
        elif sell_type == 'stop_loss':
            sell_amount = self.position_manager.position * self.config['STOP_LOSS_POSITION']
        else:
            sell_amount = min(self.config['SELL_AMOUNT'], 
                             self.position_manager.position - self.config['MIN_POSITION'])
        
        sell_shares = int((total_value * sell_amount) / current_data['close'])
        if sell_shares <= 0:
            return None
        
        # 执行卖出
        self.position_manager.reduce_position(sell_shares)
        self.position_manager.capital += sell_shares * current_data['close']
        
        # 更新仓位
        new_total_value = self.position_manager.update_position(current_data['close'])
        
        profit = (current_data['close'] - self.position_manager.avg_price) / self.position_manager.avg_price if self.position_manager.avg_price > 0 else 0
        
        return {
            'type': sell_type,
            'date': current_data['date'],
            'price': current_data['close'],
            'shares': sell_shares,
            'position': self.position_manager.position,
            'profit': profit,
            'total_value': new_total_value
        }
    
    def run_backtest(self, df: pd.DataFrame, verbose: bool = True) -> Dict:
        """运行回测"""
        if verbose:
            print("开始运行原始策略回测...")
        
        # 准备数据
        df = self.prepare_data(df)
        
        max_capital = self.config['INITIAL_CAPITAL']
        max_drawdown = 0
        
        for i in range(len(df)):
            current_data = df.iloc[i]
            
            # 更新冷却计数器
            if self.buy_strategy.cumulative_buy_cooldown > 0:
                self.buy_strategy.cumulative_buy_cooldown -= 1
            
            # 检查买入
            if self.buy_strategy.should_buy(current_data, df, i):
                # 确定买入步长
                if not self.position_manager.initial_position_built:
                    buy_step = self.config['BUY_STEP_INITIAL']
                else:
                    buy_step = self.config['BUY_STEP_NORMAL']
                
                trade = self.execute_buy(current_data, buy_step)
                if trade:
                    self.trades.append(trade)
                    if verbose:
                        print(f"买入 - {trade['date'].strftime('%Y-%m-%d')}, 价格: {trade['price']:.2f}, 仓位: {trade['position']:.1%}")
            
            # 检查卖出
            elif sell_type := self.sell_strategy.should_sell(current_data):
                if isinstance(sell_type, str):
                    trade = self.execute_sell(current_data, sell_type)
                    if trade:
                        self.trades.append(trade)
                        if verbose:
                            print(f"卖出({sell_type}) - {trade['date'].strftime('%Y-%m-%d')}, 价格: {trade['price']:.2f}")
            
            # 记录历史
            total_value = self.position_manager.capital + self.position_manager.shares * current_data['close']
            self.capital_history.append(total_value)
            self.position_history.append(self.position_manager.position)
            
            # 更新最大回撤
            if total_value > max_capital:
                max_capital = total_value
            current_drawdown = (max_capital - total_value) / max_capital
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown
        
        # 计算统计
        final_value = self.capital_history[-1]
        total_return = (final_value - self.config['INITIAL_CAPITAL']) / self.config['INITIAL_CAPITAL']
        
        # 计算Sharpe
        returns = pd.Series(self.capital_history).pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # 年化收益
        days = len(df)
        years = days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        return {
            'initial_capital': self.config['INITIAL_CAPITAL'],
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'final_position': self.position_manager.position
        }


def load_data_from_csv(csv_path: str) -> pd.DataFrame:
    """从CSV加载数据"""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


if __name__ == '__main__':
    # 加载数据
    csv_path = '/Users/rziiiii/.openclaw/workspace/data/symbols/SOXL.csv'
    df = load_data_from_csv(csv_path)
    
    # 过滤日期
    df = df[df['date'] >= '2020-01-01']
    
    print(f"加载数据: {len(df)} 条记录")
    print(f"日期范围: {df['date'].min()} 到 {df['date'].max()}")
    print()
    
    # 创建策略
    backtest = SOXLOriginalBacktest()
    
    # 运行回测
    results = backtest.run_backtest(df)
    
    # 打印结果
    print("\n" + "="*60)
    print("原始策略回测结果")
    print("="*60)
    print(f"初始资金: ${results['initial_capital']:,.0f}")
    print(f"最终资金: ${results['final_value']:,.0f}")
    print(f"总收益率: {results['total_return']:.2%}")
    print(f"年化收益率: {results['annual_return']:.2%}")
    print(f"Sharpe比率: {results['sharpe']:.2f}")
    print(f"最大回撤: {results['max_drawdown']:.2%}")
    print(f"总交易次数: {results['total_trades']}")
    print(f"最终仓位: {results['final_position']:.2%}")
    print("="*60)
