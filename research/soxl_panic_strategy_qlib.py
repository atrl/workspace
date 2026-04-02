#!/usr/bin/env python3
"""
SOXL Panic Strategy - Qlib Native Implementation

基于恐慌指数的SOXL交易策略，转换为Qlib框架

策略核心：
1. 恐慌指数 = sigmoid((RSI - median) / volatility) * 100
2. 买入：恐慌指数低 + 跌幅达到阈值
3. 卖出：恐慌指数高 + 收益达到阈值
4. 风控：止损、止盈、波动率系数调整

Author: clawQuant
Date: 2026-04-02
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


class SOXLPanicStrategyQlib:
    """
    SOXL恐慌指数策略 - Qlib版本
    """
    
    def __init__(
        self,
        max_position: float = 0.9,
        min_position: float = 0.2,
        initial_capital: float = 100000,
        
        # 恐慌指数参数
        vix_low: float = 10,
        vix_high: float = 30,
        fear_min: float = -100,
        fear_max: float = 100,
        sigmoid_steepness: float = 0.5,
        
        # 买入参数
        buy_score_threshold: float = 35,
        max_panic_threshold: float = 20,
        daily_return_threshold: float = -0.02,
        cumulative_return_threshold: float = -0.05,
        cumulative_days: int = 5,
        buy_step_initial: float = 0.1,
        buy_step_normal: float = 0.05,
        
        # 卖出参数
        sell_score_threshold: float = 65,
        stop_loss_threshold: float = -0.08,
        take_profit_threshold: float = 0.15,
        take_profit_position: float = 0.95,
        sell_profit_target: float = 0.08,
        last_buy_profit_target: float = 0.05,
        sell_amount: float = 0.1,
        
        # 波动率参数
        volatility_median: float = 1.0,
    ):
        """初始化策略"""
        # 仓位参数
        self.max_position = max_position
        self.min_position = min_position
        self.initial_capital = initial_capital
        
        # 恐慌指数参数
        self.vix_low = vix_low
        self.vix_high = vix_high
        self.fear_min = fear_min
        self.fear_max = fear_max
        self.sigmoid_steepness = sigmoid_steepness
        
        # 买入参数
        self.buy_score_threshold = buy_score_threshold
        self.max_panic_threshold = max_panic_threshold
        self.daily_return_threshold = daily_return_threshold
        self.cumulative_return_threshold = cumulative_return_threshold
        self.cumulative_days = cumulative_days
        self.buy_step_initial = buy_step_initial
        self.buy_step_normal = buy_step_normal
        
        # 卖出参数
        self.sell_score_threshold = sell_score_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.take_profit_threshold = take_profit_threshold
        self.take_profit_position = take_profit_position
        self.sell_profit_target = sell_profit_target
        self.last_buy_profit_target = last_buy_profit_target
        self.sell_amount = sell_amount
        
        # 波动率参数
        self.volatility_median = volatility_median
        
        # 状态变量
        self.position = 0.0
        self.shares = 0
        self.avg_price = 0
        self.capital = initial_capital
        self.initial_position_built = False
        self.last_buy_date = None
        self.cumulative_buy_cooldown = 0
        self.position_fifo = []  # FIFO仓位管理
        
        # 历史记录
        self.position_history = []
        self.capital_history = []
        self.trades = []
        
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_sigmoid_panic_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算Sigmoid恐慌指数"""
        rsi = df['rsi_14']
        hv_20 = df['hv_20']
        
        # 确保没有除零错误
        hv_20_safe = hv_20.replace(0, np.nan)
        
        # Sigmoid变换
        x = (rsi - self.vix_low) / (self.vix_high - self.vix_low)
        x = np.clip(x, -10, 10)  # 防止溢出
        
        score = self.fear_max / (1 + np.exp(-self.sigmoid_steepness * x))
        
        # 归一化到0-100
        score = (score - self.fear_min) / (self.fear_max - self.fear_min) * 100
        score = np.clip(score, 0, 100)
        
        return score
    
    def calculate_volatility_coefficient(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算波动率系数"""
        hv_20 = df['hv_20']
        hv_median = self.volatility_median
        
        # 波动率系数 = 当前波动率 / 中位数
        vol_coef = hv_20 / hv_median if hv_median > 0 else 1.0
        vol_coef = vol_coef.fillna(1.0)
        
        return vol_coef
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备数据，计算所有指标"""
        df = df.copy()
        
        # 计算RSI
        df['rsi_14'] = self.calculate_rsi(df['close'])
        
        # 计算历史波动率
        returns = df['close'].pct_change()
        df['hv_20'] = returns.rolling(window=20).std() * np.sqrt(252)
        df['hv_30'] = returns.rolling(window=30).std() * np.sqrt(252)
        
        # 填充NaN
        df['hv_20'] = df['hv_20'].fillna(1.0)
        df['hv_30'] = df['hv_30'].fillna(1.0)
        
        # 计算波动率系数
        df['vol_coef'] = self.calculate_volatility_coefficient(df)
        
        # 计算恐慌指数
        df['panic_score'] = self.calculate_sigmoid_panic_score(df)
        
        # 计算移动平均
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma60'] = df['close'].rolling(window=60).mean()
        
        return df
    
    def calculate_cumulative_return(self, df: pd.DataFrame, current_idx: int):
        """计算最大累积跌幅"""
        if current_idx < 2:
            return 0, 0
        
        max_cumulative_return = 0
        current_price = df.iloc[current_idx]['close']
        
        # 计算窗口大小
        max_window = self.cumulative_days
        if self.last_buy_date is not None:
            days_since_last_buy = (df.iloc[current_idx]['date'] - self.last_buy_date).days
            max_window = min(max_window, days_since_last_buy)
        
        actual_cumulative_days = 0
        for days in range(2, max_window + 1):
            if current_idx >= days:
                price_days_ago = df.iloc[current_idx - days]['close']
                cumulative_return = (current_price - price_days_ago) / price_days_ago
                if cumulative_return < max_cumulative_return:
                    max_cumulative_return = cumulative_return
                    actual_cumulative_days = days
        
        return max_cumulative_return, actual_cumulative_days
    
    def should_buy(self, current_data: pd.Series, df: pd.DataFrame, current_idx: int) -> bool:
        """判断是否应该买入"""
        # 检查最大仓位
        if self.position >= self.max_position:
            return False
        
        # 获取恐慌指数
        panic_score = current_data['panic_score']
        
        # 检查恐慌指数阈值
        if panic_score <= self.max_panic_threshold:
            return False
        
        # 检查冷却期
        if self.cumulative_buy_cooldown > 0:
            return False
        
        # 计算可用仓位
        available_position = self.max_position - self.position
        if available_position < 0.01:
            return False
        
        # 检查恐慌指数
        if panic_score > self.buy_score_threshold + 10:
            return False
        
        # 计算日收益率
        if current_idx > 0:
            daily_return = (current_data['close'] - df.iloc[current_idx - 1]['close']) / df.iloc[current_idx - 1]['close']
        else:
            daily_return = 0
        
        # 计算累积跌幅
        max_cumulative_return, actual_days = self.calculate_cumulative_return(df, current_idx)
        
        # 获取波动率系数
        vol_coef = current_data['vol_coef']
        
        # 调整阈值
        if not self.initial_position_built:
            daily_threshold = self.daily_return_threshold * 0.5 * vol_coef
            cumulative_threshold = self.cumulative_return_threshold * 0.5 * vol_coef
        else:
            is_overvalued = current_data['close'] > current_data['ma20']
            if is_overvalued:
                daily_threshold = self.daily_return_threshold * vol_coef
                cumulative_threshold = self.cumulative_return_threshold * vol_coef
            else:
                daily_threshold = self.daily_return_threshold * vol_coef
                cumulative_threshold = self.cumulative_return_threshold * vol_coef
        
        # 检查条件
        daily_condition = daily_return <= daily_threshold
        cumulative_condition = max_cumulative_return <= cumulative_threshold
        
        return daily_condition or cumulative_condition
    
    def should_sell(self, current_data: pd.Series) -> str:
        """判断是否应该卖出"""
        if self.position <= 0:
            return None
        
        panic_score = current_data['panic_score']
        profit = (current_data['close'] - self.avg_price) / self.avg_price if self.avg_price > 0 else 0
        
        # 计算可卖出仓位
        max_sell_position = self.position - self.min_position
        if max_sell_position < 0.01:
            return None
        
        # 止损
        if profit <= self.stop_loss_threshold:
            return 'stop_loss'
        
        # 获取最后买入的收益
        last_buy_profit = 0
        if self.position_fifo:
            last_buy = self.position_fifo[-1]
            last_buy_profit = (current_data['close'] - last_buy['price']) / last_buy['price']
        
        # 大幅止盈
        take_profit_condition = (
            panic_score >= self.sell_score_threshold and
            profit * self.position >= self.take_profit_threshold and
            self.position >= self.min_position
        )
        if take_profit_condition:
            return 'take_profit'
        
        # 正常卖出
        vol_coef = current_data['vol_coef']
        normal_sell_1 = (
            profit * self.position >= self.sell_profit_target and
            panic_score >= self.sell_score_threshold and
            profit > 0
        )
        normal_sell_2 = (
            panic_score >= self.sell_score_threshold and
            last_buy_profit >= self.last_buy_profit_target * vol_coef
        )
        
        if normal_sell_1 or normal_sell_2:
            return 'normal_sell'
        
        return None
    
    def execute_buy(self, current_data: pd.Series, df: pd.DataFrame, current_idx: int):
        """执行买入操作"""
        total_value = self.capital + self.shares * current_data['close']
        
        # 确定买入步长
        if not self.initial_position_built:
            buy_step = self.buy_step_initial
        else:
            is_overvalued = current_data['close'] > current_data['ma20']
            buy_step = self.buy_step_normal * 0.4 if is_overvalued else self.buy_step_normal
        
        # 计算目标仓位
        target_position = min(self.position + buy_step, self.max_position)
        buy_amount = total_value * (target_position - self.position)
        
        # 计算买入股数
        new_shares = int(buy_amount / current_data['close'])
        if new_shares <= 0:
            return None
        
        # 更新仓位
        cost = new_shares * current_data['close']
        self.capital -= cost
        
        # 更新平均价格
        total_cost = self.avg_price * self.shares + cost
        self.shares += new_shares
        self.avg_price = total_cost / self.shares
        
        # 更新仓位比例
        self.position = (self.shares * current_data['close']) / (self.capital + self.shares * current_data['close'])
        
        # 添加到FIFO
        self.position_fifo.append({
            'price': current_data['close'],
            'shares': new_shares,
            'date': current_data['date']
        })
        
        # 更新状态
        self.last_buy_date = current_data['date']
        
        # 标记初始建仓完成
        if not self.initial_position_built and self.position >= self.min_position:
            self.initial_position_built = True
        
        return {
            'type': 'buy',
            'price': current_data['close'],
            'shares': new_shares,
            'position': self.position,
            'avg_price': self.avg_price
        }
    
    def execute_sell(self, current_data: pd.Series, sell_type: str):
        """执行卖出操作"""
        total_value = self.capital + self.shares * current_data['close']
        
        # 确定卖出比例
        if sell_type == 'take_profit':
            sell_ratio = self.take_profit_position
        elif sell_type == 'stop_loss':
            sell_ratio = 0.5  # 止损卖出50%
        else:
            sell_ratio = self.sell_amount
        
        # 计算卖出股数
        sell_value = total_value * min(sell_ratio, self.position - self.min_position)
        sell_shares = int(sell_value / current_data['close'])
        
        if sell_shares <= 0:
            return None
        
        # 更新资金
        self.capital += sell_shares * current_data['close']
        
        # 更新FIFO
        remaining_shares = sell_shares
        while remaining_shares > 0 and self.position_fifo:
            first_position = self.position_fifo[0]
            if first_position['shares'] <= remaining_shares:
                remaining_shares -= first_position['shares']
                self.position_fifo.pop(0)
            else:
                first_position['shares'] -= remaining_shares
                remaining_shares = 0
        
        # 更新股数
        self.shares -= sell_shares
        
        # 更新仓位
        if self.shares > 0:
            self.position = (self.shares * current_data['close']) / (self.capital + self.shares * current_data['close'])
        else:
            self.position = 0
            self.avg_price = 0
        
        return {
            'type': sell_type,
            'price': current_data['close'],
            'shares': sell_shares,
            'position': self.position,
            'profit': (current_data['close'] - self.avg_price) / self.avg_price if self.avg_price > 0 else 0
        }
    
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """运行回测"""
        print("开始运行回测...")
        
        # 准备数据
        df = self.prepare_data(df)
        
        max_capital = self.initial_capital
        max_drawdown = 0
        
        for i in range(len(df)):
            current_data = df.iloc[i]
            
            # 更新冷却计数器
            if self.cumulative_buy_cooldown > 0:
                self.cumulative_buy_cooldown -= 1
            
            # 检查买入
            if self.should_buy(current_data, df, i):
                trade = self.execute_buy(current_data, df, i)
                if trade:
                    self.trades.append(trade)
                    print(f"买入 - 日期: {current_data['date']}, 价格: {trade['price']:.2f}, 仓位: {trade['position']:.2%}")
            
            # 检查卖出
            elif sell_type := self.should_sell(current_data):
                trade = self.execute_sell(current_data, sell_type)
                if trade:
                    self.trades.append(trade)
                    print(f"卖出 - 日期: {current_data['date']}, 类型: {sell_type}, 价格: {trade['price']:.2f}")
            
            # 记录历史
            total_value = self.capital + self.shares * current_data['close']
            self.capital_history.append(total_value)
            self.position_history.append(self.position)
            
            # 更新最大回撤
            if total_value > max_capital:
                max_capital = total_value
            current_drawdown = (max_capital - total_value) / max_capital
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown
        
        # 计算统计
        final_value = self.capital_history[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # 计算Sharpe
        returns = pd.Series(self.capital_history).pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        return {
            'final_value': final_value,
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'final_position': self.position,
            'trades': self.trades
        }


def load_data_from_csv(csv_path: str) -> pd.DataFrame:
    """从CSV加载数据"""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


if __name__ == '__main__':
    # 加载数据
    csv_path = 'data/symbols/SOXL.csv'
    df = load_data_from_csv(csv_path)
    
    # 过滤日期
    df = df[df['date'] >= '2020-01-01']
    
    print(f"加载数据: {len(df)} 条记录")
    
    # 创建策略
    strategy = SOXLPanicStrategyQlib()
    
    # 运行回测
    results = strategy.run_backtest(df)
    
    # 打印结果
    print("\n" + "="*60)
    print("回测结果")
    print("="*60)
    print(f"初始资金: ${strategy.initial_capital:,.0f}")
    print(f"最终资金: {results['final_value']:,.0f}")
    print(f"总收益率: {results['total_return']:.2%}")
    print(f"Sharpe比率: {results['sharpe']:.2f}")
    print(f"最大回撤: {results['max_drawdown']:.2%}")
    print(f"总交易次数: {results['total_trades']}")
    print(f"最终仓位: {results['final_position']:.2%}")
    print("="*60)
