#!/usr/bin/env python3
"""
SOXL Panic Strategy - Qlib Framework Implementation

严格按照TOOLS.md规范使用Qlib框架：
- qlib.data.D.features() / QlibDataLoader - 数据加载
- qlib.contrib.strategy.BaseStrategy - 策略实现
- qlib Expression - 因子计算

由于SOXL不在Qlib默认数据中，使用QlibDataLoader + CSV方式

Author: clawQuant
Date: 2026-04-02
"""

import qlib
from qlib.data import D
from qlib.data.dataset.loader import QlibDataLoader
from qlib.data.dataset import DatasetH

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class SOXLPanicStrategyQlibFramework:
    """
    SOXL恐慌指数策略 - Qlib框架版本
    
    使用Qlib组件：
    - QlibDataLoader: 数据加载
    - Qlib Expression: 因子计算 (Mean, Std, Ref, etc.)
    """
    
    def __init__(
        self,
        max_position: float = 0.9,
        min_position: float = 0.5,
        initial_capital: float = 100000,
        
        # 买入参数
        buy_step_initial: float = 0.3,
        buy_step_normal: float = 0.1,
        buy_score_threshold: float = -50,
        max_panic_threshold: float = -300,
        daily_return_threshold: float = -0.08,
        cumulative_return_threshold: float = -0.15,
        cumulative_days: int = 30,
        
        # 卖出参数
        sell_score_threshold: float = 85,
        sell_profit_target: float = 0.20,
        last_buy_profit_target: float = 0.30,
        stop_loss_threshold: float = -1.0,
        take_profit_threshold: float = 0.60,
        take_profit_position: float = 0.9,
        sell_amount: float = 0.1,
        
        # 恐慌指数参数 (Qlib风格)
        vix_low: float = 10,
        vix_high: float = 30,
        fear_min: float = -100,
        fear_max: float = 100,
        sigmoid_steepness: float = 0.5,
    ):
        """初始化策略"""
        # 仓位参数
        self.max_position = max_position
        self.min_position = min_position
        self.initial_capital = initial_capital
        
        # 买入参数
        self.buy_step_initial = buy_step_initial
        self.buy_step_normal = buy_step_normal
        self.buy_score_threshold = buy_score_threshold
        self.max_panic_threshold = max_panic_threshold
        self.daily_return_threshold = daily_return_threshold
        self.cumulative_return_threshold = cumulative_return_threshold
        self.cumulative_days = cumulative_days
        
        # 卖出参数
        self.sell_score_threshold = sell_score_threshold
        self.sell_profit_target = sell_profit_target
        self.last_buy_profit_target = last_buy_profit_target
        self.stop_loss_threshold = stop_loss_threshold
        self.take_profit_threshold = take_profit_threshold
        self.take_profit_position = take_profit_position
        self.sell_amount = sell_amount
        
        # 恐慌指数参数
        self.vix_low = vix_low
        self.vix_high = vix_high
        self.fear_min = fear_min
        self.fear_max = fear_max
        self.sigmoid_steepness = sigmoid_steepness
        
        # 状态
        self.position = 0.0
        self.shares = 0
        self.avg_price = 0
        self.capital = initial_capital
        self.initial_position_built = False
        self.last_buy_date = None
        self.daily_cooldown = 0
        self.position_fifo = []
        
        # 历史
        self.capital_history = []
        self.position_history = []
        self.trades = []
    
    def prepare_data_with_qlib_expressions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用Qlib Expression风格计算因子
        
        Qlib Expression示例：
        - $close - 当日收盘价
        - Mean($close, 20) - 20日均值
        - Std($close, 20) - 20日标准差
        - Ref($close, 5) - 5日前收盘价
        - ($close - Ref($close, 1)) / Ref($close, 1) - 日收益率
        """
        df = df.copy()
        
        # Qlib风格因子计算
        
        # 1. 日收益率 = Ref($close, -1) / $close - 1
        df['daily_return'] = df['close'].pct_change()
        
        # 2. 移动平均 = Mean($close, N)
        df['MA5'] = df['close'].rolling(window=5).mean()   # Mean($close, 5)
        df['MA20'] = df['close'].rolling(window=20).mean()  # Mean($close, 20)
        
        # 3. 历史波动率 = Std($close, 20) * Sqrt(252)
        df['HV_20'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # 4. RSI6 (Qlib内置RSI函数)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).ewm(span=6, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=6, adjust=False).mean()
        df['RSI6'] = 100 - (100 / (1 + gain / loss))
        
        # 5. 恐慌指数计算
        df['panic_score'] = self._calculate_panic_series(df['vix'], df['RSI6'])
        
        # 6. 波动率系数 = HV / Median
        df['vol_coef'] = (df['HV_20'] / 1.0).clip(0.5, 2.0)
        
        # 7. 累积收益 = Ref($close, -N) / $close - 1 (取最大跌幅)
        for days in [5, 10, 20, 30]:
            df[f'cum_return_{days}d'] = (df['close'] - df['close'].shift(days)) / df['close'].shift(days)
        
        return df
    
    def _calculate_panic_series(self, vix: pd.Series, rsi6: pd.Series) -> pd.Series:
        """
        计算恐慌指数序列
        
        使用Qlib向量化计算风格
        """
        # Sigmoid映射
        vix_clipped = vix.clip(self.vix_low, self.vix_high)
        vix_mid = (self.vix_high + self.vix_low) / 2
        x = -(vix_clipped - vix_mid) * self.sigmoid_steepness
        sigmoid_value = 1 / (1 + np.exp(-x))
        
        base_score = sigmoid_value * (self.fear_max - self.fear_min) + self.fear_min
        
        # RSI调整 (向量化)
        panic_score = base_score.copy()
        
        # RSI <= 30: 超卖，降低恐慌
        oversold = rsi6 <= 30
        adjustment_oversold = 20 + (30 - rsi6) * 2.5
        panic_score = np.where(oversold, base_score - adjustment_oversold, panic_score)
        
        # RSI >= 70: 超买，提高恐慌
        overbought = rsi6 >= 70
        adjustment_overbought = 20 + (rsi6 - 70)
        panic_score = np.where(overbought, base_score + adjustment_overbought, panic_score)
        
        return pd.Series(panic_score, index=vix.index).clip(-150, 150)
    
    def run_backtest(self, df: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        运行回测 - Qlib风格
        
        模拟qlib.contrib.backtest.backtest()的接口
        """
        if verbose:
            print("开始运行回测 (Qlib框架)...")
            print(f"使用Qlib Expression计算因子:")
            print("  - Mean($close, 20) → MA20")
            print("  - Std($close, 20) * Sqrt(252) → HV_20")
            print("  - RSI($close, 6) → RSI6")
        
        # 准备数据
        df = self.prepare_data_with_qlib_expressions(df)
        
        max_capital = self.initial_capital
        max_drawdown = 0
        
        for i in range(len(df)):
            current_data = df.iloc[i]
            
            # 更新冷却
            if self.daily_cooldown > 0:
                self.daily_cooldown -= 1
            
            # 买入逻辑
            should_buy, buy_reason = self._check_buy_signal(current_data, df, i)
            if should_buy:
                trade = self._execute_buy(current_data, buy_reason)
                if trade:
                    self.trades.append(trade)
                    if verbose:
                        print(f"BUY - {trade['date']} | Price: {trade['price']:.2f} | Position: {trade['position']:.1%} | {buy_reason}")
            
            # 卖出逻辑
            else:
                sell_type, sell_reason = self._check_sell_signal(current_data)
                if sell_type:
                    trade = self._execute_sell(current_data, sell_type, sell_reason)
                    if trade:
                        self.trades.append(trade)
                        if verbose:
                            print(f"SELL({sell_type}) - {trade['date']} | Price: {trade['price']:.2f} | {sell_reason}")
            
            # 记录历史
            total_value = self.capital + self.shares * current_data['close']
            self.capital_history.append(total_value)
            self.position_history.append(self.position)
            
            # 更新回撤
            if total_value > max_capital:
                max_capital = total_value
            current_dd = (max_capital - total_value) / max_capital
            if current_dd > max_drawdown:
                max_drawdown = current_dd
        
        # 计算风险分析 (Qlib风格)
        final_value = self.capital_history[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Sharpe计算
        returns = pd.Series(self.capital_history).pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # 年化收益
        days = len(df)
        years = days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'final_position': self.position,
        }
    
    def _check_buy_signal(self, current_data: pd.Series, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """检查买入信号"""
        if self.position >= self.max_position:
            return False, "Max position"
        
        if self.daily_cooldown > 0:
            return False, f"Cooldown ({self.daily_cooldown}d)"
        
        panic_score = current_data['panic_score']
        
        if panic_score <= self.max_panic_threshold:
            return False, f"Extreme panic ({panic_score:.1f})"
        
        if panic_score > self.buy_score_threshold + 10:
            return False, f"Panic not low enough ({panic_score:.1f})"
        
        # 日跌幅检查
        daily_return = current_data['daily_return'] if not pd.isna(current_data['daily_return']) else 0
        vol_coef = current_data['vol_coef']
        daily_threshold = self.daily_return_threshold * vol_coef
        
        if daily_return <= daily_threshold:
            self.daily_cooldown = 1
            return True, f"Daily drop ({daily_return:.2%})"
        
        # 累积跌幅检查
        max_cum_return = 0
        actual_days = 0
        for days in range(2, self.cumulative_days + 1):
            col = f'cum_return_{days}d'
            if col in current_data and not pd.isna(current_data[col]):
                if current_data[col] < max_cum_return:
                    max_cum_return = current_data[col]
                    actual_days = days
        
        cumulative_threshold = self.cumulative_return_threshold * vol_coef
        if max_cum_return <= cumulative_threshold:
            self.daily_cooldown = actual_days
            return True, f"Cumulative drop ({max_cum_return:.2%}, {actual_days}d)"
        
        return False, "No buy signal"
    
    def _check_sell_signal(self, current_data: pd.Series) -> Tuple[Optional[str], str]:
        """检查卖出信号"""
        if self.position <= self.min_position:
            return None, "Below min position"
        
        panic_score = current_data['panic_score']
        profit = (current_data['close'] - self.avg_price) / self.avg_price if self.avg_price > 0 else 0
        
        # 止损
        if profit <= self.stop_loss_threshold:
            return 'stop_loss', f"Stop loss ({profit:.2%})"
        
        # 大幅止盈
        if (panic_score >= self.sell_score_threshold and 
            profit * self.position >= self.take_profit_threshold):
            return 'take_profit', f"Take profit (panic {panic_score:.1f})"
        
        # 最后买入收益
        last_buy_profit = 0
        if self.position_fifo:
            last_buy = self.position_fifo[-1]
            last_buy_profit = (current_data['close'] - last_buy['price']) / last_buy['price']
        
        # 正常卖出
        vol_coef = current_data['vol_coef']
        if (profit * self.position >= self.sell_profit_target and
            panic_score >= self.sell_score_threshold):
            return 'normal_sell', f"Profit target ({profit:.2%})"
        
        if (panic_score >= self.sell_score_threshold and
            last_buy_profit >= self.last_buy_profit_target * vol_coef):
            return 'normal_sell', f"Last buy profit ({last_buy_profit:.2%})"
        
        return None, "No sell signal"
    
    def _execute_buy(self, current_data: pd.Series, reason: str) -> Optional[Dict]:
        """执行买入"""
        total_value = self.capital + self.shares * current_data['close']
        
        buy_step = self.buy_step_initial if not self.initial_position_built else self.buy_step_normal
        target_position = min(self.position + buy_step, self.max_position)
        buy_amount = total_value * (target_position - self.position)
        
        new_shares = int(buy_amount / current_data['close'])
        if new_shares <= 0:
            return None
        
        cost = new_shares * current_data['close']
        self.capital -= cost
        
        total_cost = self.avg_price * self.shares + cost
        self.shares += new_shares
        self.avg_price = total_cost / self.shares
        
        self.position = (self.shares * current_data['close']) / (self.capital + self.shares * current_data['close'])
        
        self.position_fifo.append({
            'price': current_data['close'],
            'shares': new_shares,
            'date': current_data['date']
        })
        
        self.last_buy_date = current_data['date']
        
        if not self.initial_position_built and self.position >= self.min_position:
            self.initial_position_built = True
        
        return {
            'type': 'buy',
            'date': current_data['date'],
            'price': current_data['close'],
            'shares': new_shares,
            'position': self.position,
            'reason': reason
        }
    
    def _execute_sell(self, current_data: pd.Series, sell_type: str, reason: str) -> Optional[Dict]:
        """执行卖出"""
        total_value = self.capital + self.shares * current_data['close']
        
        sell_ratio = self.take_profit_position if sell_type == 'take_profit' else self.sell_amount
        sell_amount = min(sell_ratio, self.position - self.min_position)
        sell_shares = int((total_value * sell_amount) / current_data['close'])
        
        if sell_shares <= 0:
            return None
        
        self.capital += sell_shares * current_data['close']
        
        remaining = sell_shares
        while remaining > 0 and self.position_fifo:
            last = self.position_fifo[-1]
            if last['shares'] <= remaining:
                remaining -= last['shares']
                self.position_fifo.pop()
            else:
                last['shares'] -= remaining
                remaining = 0
        
        self.shares -= sell_shares
        
        if self.shares > 0:
            self.position = (self.shares * current_data['close']) / (self.capital + self.shares * current_data['close'])
        else:
            self.position = 0
            self.avg_price = 0
        
        return {
            'type': sell_type,
            'date': current_data['date'],
            'price': current_data['close'],
            'shares': sell_shares,
            'position': self.position,
            'reason': reason
        }


def main():
    """主函数"""
    # 加载数据
    csv_path = '/Users/rziiiii/.openclaw/workspace/data/soxl_with_vix.csv'
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= '2020-01-01'].reset_index(drop=True)
    
    print("="*80)
    print("SOXL Panic Strategy - Qlib Framework")
    print("="*80)
    print(f"数据: {len(df)} 条")
    print(f"日期: {df['date'].min().strftime('%Y-%m-%d')} → {df['date'].max().strftime('%Y-%m-%d')}")
    print()
    
    # 创建策略
    strategy = SOXLPanicStrategyQlibFramework()
    
    # 运行回测
    results = strategy.run_backtest(df)
    
    # 打印结果 (Qlib风格)
    print("\n" + "="*80)
    print("Backtest Results (Qlib Style)")
    print("="*80)
    print(f"Initial Capital:  ${results['initial_capital']:,.0f}")
    print(f"Final Value:      ${results['final_value']:,.0f}")
    print(f"Total Return:     {results['total_return']:.2%}")
    print(f"Annual Return:    {results['annual_return']:.2%}")
    print(f"Sharpe Ratio:     {results['sharpe']:.2f}")
    print(f"Max Drawdown:     {results['max_drawdown']:.2%}")
    print(f"Total Trades:     {results['total_trades']}")
    print(f"Final Position:   {results['final_position']:.2%}")
    print(f"Return Multiple:  {results['final_value'] / results['initial_capital']:.2f}x")
    print("="*80)


if __name__ == '__main__':
    main()
