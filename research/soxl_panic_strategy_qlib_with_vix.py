#!/usr/bin/env python3
"""
SOXL Panic Strategy - Qlib Version with Real VIX

使用真实VIX数据的Qlib版本回测

Author: clawQuant
Date: 2026-04-02
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class SOXLPanicStrategyQlibWithVIX:
    """
    SOXL恐慌指数策略 - Qlib版本，使用真实VIX
    """
    
    def __init__(
        self,
        # 仓位参数
        max_position: float = 0.9,
        min_position: float = 0.5,
        min_trade_position: float = 0.05,
        initial_capital: float = 100000,
        
        # 买入参数
        buy_step_initial: float = 0.3,
        buy_step_normal: float = 0.1,
        buy_score_threshold: float = -50,
        max_panic_threshold: float = -300,
        daily_return_threshold: float = -0.08,
        cumulative_return_threshold: float = -0.15,
        cumulative_days: int = 30,
        daily_buy_cooldown: int = 3,
        
        # 卖出参数
        sell_score_threshold: float = 85,
        sell_profit_target: float = 0.20,
        last_buy_profit_target: float = 0.30,
        stop_loss_threshold: float = -1.0,
        stop_loss_position: float = 0.6,
        take_profit_threshold: float = 0.60,
        take_profit_position: float = 0.9,
        sell_amount: float = 0.1,
        
        # 恐慌指数参数
        vix_low: float = 10,
        vix_high: float = 30,
        fear_min: float = -100,
        fear_max: float = 100,
        sigmoid_steepness: float = 0.5,
        volatility_median: float = 1.0,
    ):
        """初始化策略"""
        # 仓位参数
        self.max_position = max_position
        self.min_position = min_position
        self.min_trade_position = min_trade_position
        self.initial_capital = initial_capital
        
        # 买入参数
        self.buy_step_initial = buy_step_initial
        self.buy_step_normal = buy_step_normal
        self.buy_score_threshold = buy_score_threshold
        self.max_panic_threshold = max_panic_threshold
        self.daily_return_threshold = daily_return_threshold
        self.cumulative_return_threshold = cumulative_return_threshold
        self.cumulative_days = cumulative_days
        self.daily_buy_cooldown = daily_buy_cooldown
        
        # 卖出参数
        self.sell_score_threshold = sell_score_threshold
        self.sell_profit_target = sell_profit_target
        self.last_buy_profit_target = last_buy_profit_target
        self.stop_loss_threshold = stop_loss_threshold
        self.stop_loss_position = stop_loss_position
        self.take_profit_threshold = take_profit_threshold
        self.take_profit_position = take_profit_position
        self.sell_amount = sell_amount
        
        # 恐慌指数参数
        self.vix_low = vix_low
        self.vix_high = vix_high
        self.fear_min = fear_min
        self.fear_max = fear_max
        self.sigmoid_steepness = sigmoid_steepness
        self.volatility_median = volatility_median
        
        # 状态变量
        self.position = 0.0
        self.shares = 0
        self.avg_price = 0
        self.capital = initial_capital
        self.initial_position_built = False
        self.last_buy_date = None
        self.daily_cooldown = 0
        self.position_fifo = []
        
        # 历史记录
        self.position_history = []
        self.capital_history = []
        self.trades = []
        
    def calculate_rsi(self, prices: pd.Series, period: int = 6) -> pd.Series:
        """计算RSI指标 - 使用EWMA"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))
        
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_panic_score(self, vix: float, rsi6: float) -> float:
        """
        计算恐慌指数 - 使用真实VIX
        """
        # Sigmoid映射 - VIX被clip到[vix_low, vix_high]
        vix_clipped = max(min(vix, self.vix_high), self.vix_low)
        vix_mid = (self.vix_high + self.vix_low) / 2
        x = -(vix_clipped - vix_mid) * self.sigmoid_steepness
        sigmoid_value = 1 / (1 + np.exp(-x))
        
        base_score = sigmoid_value * (self.fear_max - self.fear_min) + self.fear_min
        
        # RSI6调整 - 完全匹配原始代码
        if pd.isna(rsi6):
            return base_score
        
        if rsi6 <= 30:
            # RSI超卖区域，恐慌指数下调
            adjustment = 20 + (30 - rsi6) * 2.5
            new_score = base_score - adjustment
            return max(-150, new_score)
        elif rsi6 >= 70:
            # RSI超买区域，恐慌指数上调
            adjustment = 20 + (rsi6 - 70)
            new_score = base_score + adjustment
            return min(150, new_score)
        else:
            return base_score
    
    def calculate_volatility_coefficient(self, hv: float) -> float:
        """计算波动率系数"""
        if self.volatility_median <= 0:
            return 1.0
        return hv / self.volatility_median
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备数据，使用真实VIX"""
        df = df.copy()
        
        # 计算RSI6
        df['rsi6'] = self.calculate_rsi(df['close'])
        
        # 计算历史波动率（用于波动率系数）
        returns = df['close'].pct_change()
        df['hv'] = returns.rolling(window=20).std() * np.sqrt(252)
        df['hv'] = df['hv'].fillna(self.volatility_median)
        
        # 计算波动率系数
        df['vol_coef'] = df['hv'].apply(self.calculate_volatility_coefficient)
        
        # 使用真实VIX计算恐慌指数
        df['panic_score'] = df.apply(
            lambda row: self.calculate_panic_score(row['vix'], row['rsi6']),
            axis=1
        )
        
        # 计算移动平均
        df['ma20'] = df['close'].rolling(window=20).mean()
        
        return df
    
    def calculate_cumulative_return(self, df: pd.DataFrame, current_idx: int) -> Tuple[float, int]:
        """计算最大累积跌幅"""
        if current_idx < 2:
            return 0, 0
        
        max_cumulative_return = 0
        current_price = df.iloc[current_idx]['close']
        
        # 计算窗口大小
        max_window = self.cumulative_days
        if self.last_buy_date is not None:
            days_since_last_buy = (df.iloc[current_idx]['date'] - self.last_buy_date).days
            max_window = min(max_window, max(days_since_last_buy, 2))
        
        actual_cumulative_days = 0
        for days in range(2, max_window + 1):
            if current_idx >= days:
                price_days_ago = df.iloc[current_idx - days]['close']
                cumulative_return = (current_price - price_days_ago) / price_days_ago
                if cumulative_return < max_cumulative_return:
                    max_cumulative_return = cumulative_return
                    actual_cumulative_days = days
        
        return max_cumulative_return, actual_cumulative_days
    
    def should_buy(self, current_data: pd.Series, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """判断是否应该买入"""
        # 检查最大仓位
        if self.position >= self.max_position:
            return False, "已达到最大仓位"
        
        # 检查冷却期
        if self.daily_cooldown > 0:
            return False, f"在冷却期内，剩余{self.daily_cooldown}天"
        
        # 计算可用仓位
        available_position = self.max_position - self.position
        if available_position < self.min_trade_position:
            return False, f"可用仓位({available_position:.2%})小于最小交易仓位"
        
        # 获取恐慌指数
        panic_score = current_data['panic_score']
        
        # 检查极端恐慌（暂停买入）
        if panic_score <= self.max_panic_threshold:
            return False, f"极端恐慌({panic_score:.1f})，暂停买入"
        
        # 检查恐慌指数阈值 (原始: <= threshold + 10)
        if panic_score > self.buy_score_threshold + 10:
            return False, f"恐慌指数({panic_score:.1f})未达到买入阈值({self.buy_score_threshold + 10})"
        
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
            daily_threshold = self.daily_return_threshold * vol_coef
            cumulative_threshold = self.cumulative_return_threshold * vol_coef
        
        # 检查条件
        daily_condition = daily_return <= daily_threshold
        cumulative_condition = max_cumulative_return <= cumulative_threshold
        
        if daily_condition:
            return True, f"日跌幅买入({daily_return:.2%})"
        if cumulative_condition:
            return True, f"累积跌幅买入({max_cumulative_return:.2%}, {actual_days}天)"
        
        return False, f"恐慌指数满足但跌幅不足(日:{daily_return:.2%}, 累积:{max_cumulative_return:.2%})"
    
    def should_sell(self, current_data: pd.Series) -> Tuple[Optional[str], str]:
        """判断是否应该卖出"""
        if self.position <= self.min_position:
            return None, "仓位低于最小保留仓位"
        
        panic_score = current_data['panic_score']
        profit = (current_data['close'] - self.avg_price) / self.avg_price if self.avg_price > 0 else 0
        
        # 计算可卖出仓位
        max_sell_position = self.position - self.min_position
        if max_sell_position < self.min_trade_position:
            return None, f"可卖出仓位({max_sell_position:.2%})不足"
        
        # 止损
        if profit <= self.stop_loss_threshold:
            return 'stop_loss', f"止损({profit:.2%})"
        
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
            return 'take_profit', f"大幅止盈(恐慌{panic_score:.1f}, 收益{profit:.2%})"
        
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
        
        if normal_sell_1:
            return 'normal_sell', f"获利目标(收益{profit:.2%})"
        if normal_sell_2:
            return 'normal_sell', f"最近买入获利({last_buy_profit:.2%})"
        
        return None, f"未达卖出条件(恐慌{panic_score:.1f}, 收益{profit:.2%})"
    
    def execute_buy(self, current_data: pd.Series, reason: str):
        """执行买入操作"""
        total_value = self.capital + self.shares * current_data['close']
        
        # 确定买入步长
        if not self.initial_position_built:
            buy_step = self.buy_step_initial
        else:
            buy_step = self.buy_step_normal
        
        # 计算目标仓位
        target_position = min(self.position + buy_step, self.max_position)
        buy_amount = total_value * (target_position - self.position)
        
        # 计算买入股数
        new_shares = int(buy_amount / current_data['close'])
        if new_shares <= 0:
            return None
        
        # 执行买入
        cost = new_shares * current_data['close']
        self.capital -= cost
        
        # 更新平均价格
        total_cost = self.avg_price * self.shares + cost
        self.shares += new_shares
        self.avg_price = total_cost / self.shares
        
        # 更新仓位
        self.position = (self.shares * current_data['close']) / (self.capital + self.shares * current_data['close'])
        
        # 添加到FIFO
        self.position_fifo.append({
            'price': current_data['close'],
            'shares': new_shares,
            'date': current_data['date']
        })
        
        # 设置冷却期
        self.daily_cooldown = self.daily_buy_cooldown
        self.last_buy_date = current_data['date']
        
        # 标记初始建仓完成
        if not self.initial_position_built and self.position >= self.min_position:
            self.initial_position_built = True
        
        return {
            'type': 'buy',
            'date': current_data['date'],
            'price': current_data['close'],
            'shares': new_shares,
            'position': self.position,
            'avg_price': self.avg_price,
            'reason': reason,
            'panic_score': current_data['panic_score'],
            'vix': current_data['vix']
        }
    
    def execute_sell(self, current_data: pd.Series, sell_type: str, reason: str):
        """执行卖出操作"""
        total_value = self.capital + self.shares * current_data['close']
        
        # 确定卖出比例
        if sell_type == 'take_profit':
            sell_ratio = self.take_profit_position
        elif sell_type == 'stop_loss':
            sell_ratio = self.stop_loss_position
        else:
            sell_ratio = self.sell_amount
        
        # 计算卖出股数
        sell_value = total_value * min(sell_ratio, self.position - self.min_position)
        sell_shares = int(sell_value / current_data['close'])
        
        if sell_shares <= 0:
            return None
        
        # 执行卖出
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
            'date': current_data['date'],
            'price': current_data['close'],
            'shares': sell_shares,
            'position': self.position,
            'profit': (current_data['close'] - self.avg_price) / self.avg_price if self.avg_price > 0 else 0,
            'reason': reason,
            'panic_score': current_data['panic_score'],
            'vix': current_data['vix']
        }
    
    def run_backtest(self, df: pd.DataFrame, verbose: bool = True) -> Dict:
        """运行回测"""
        if verbose:
            print("开始运行回测 (Qlib版本 + 真实VIX)...")
        
        # 准备数据
        df = self.prepare_data(df)
        
        max_capital = self.initial_capital
        max_drawdown = 0
        
        for i in range(len(df)):
            current_data = df.iloc[i]
            
            # 更新冷却计数器
            if self.daily_cooldown > 0:
                self.daily_cooldown -= 1
            
            # 检查买入
            should_buy, buy_reason = self.should_buy(current_data, df, i)
            if should_buy:
                trade = self.execute_buy(current_data, buy_reason)
                if trade:
                    self.trades.append(trade)
                    if verbose:
                        print(f"买入 - {trade['date'].strftime('%Y-%m-%d')}, 价格: {trade['price']:.2f}, 仓位: {trade['position']:.1%}, VIX: {trade['vix']:.1f}")
            
            # 检查卖出
            else:
                sell_type, sell_reason = self.should_sell(current_data)
                if sell_type:
                    trade = self.execute_sell(current_data, sell_type, sell_reason)
                    if trade:
                        self.trades.append(trade)
                        if verbose:
                            print(f"卖出({sell_type}) - {trade['date'].strftime('%Y-%m-%d')}, 价格: {trade['price']:.2f}, VIX: {trade['vix']:.1f}")
            
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
        
        # 计算年化收益
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
            'trades': self.trades
        }


def load_data_from_csv(csv_path: str) -> pd.DataFrame:
    """从CSV加载数据"""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


if __name__ == '__main__':
    # 加载带VIX的数据
    csv_path = '/Users/rziiiii/.openclaw/workspace/data/soxl_with_vix.csv'
    df = load_data_from_csv(csv_path)
    
    # 过滤日期
    df = df[df['date'] >= '2020-01-01']
    
    print("="*80)
    print("SOXL Panic Strategy - Qlib版本 + 真实VIX")
    print("="*80)
    print(f"数据: {len(df)} 条记录")
    print(f"日期范围: {df['date'].min().strftime('%Y-%m-%d')} 到 {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"VIX范围: {df['vix'].min():.2f} - {df['vix'].max():.2f}")
    print()
    
    # 创建策略
    strategy = SOXLPanicStrategyQlibWithVIX()
    
    # 运行回测
    results = strategy.run_backtest(df)
    
    # 打印结果
    print("\n" + "="*80)
    print("回测结果")
    print("="*80)
    print(f"初始资金: ${results['initial_capital']:,.0f}")
    print(f"最终资金: ${results['final_value']:,.0f}")
    print(f"总收益率: {results['total_return']:.2%}")
    print(f"年化收益率: {results['annual_return']:.2%}")
    print(f"Sharpe比率: {results['sharpe']:.2f}")
    print(f"最大回撤: {results['max_drawdown']:.2%}")
    print(f"总交易次数: {results['total_trades']}")
    print(f"最终仓位: {results['final_position']:.2%}")
    print(f"收益倍数: {results['final_value'] / results['initial_capital']:.2f}x")
    print("="*80)
