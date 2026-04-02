#!/usr/bin/env python3
"""
SOXL Panic Strategy - Original Logic Comparison

使用原始策略的核心逻辑进行对比回测

Author: clawQuant
Date: 2026-04-02
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class OriginalPanicStrategy:
    """
    原始恐慌指数策略
    严格按照原始代码逻辑实现
    """
    
    def __init__(self):
        """初始化策略 - 使用原始默认参数"""
        # 原始配置参数
        self.config = {
            'INITIAL_CAPITAL': 100000,
            
            # VIX相关参数
            'VIX_LOW': 10,
            'VIX_HIGH': 30,
            'FEAR_MIN': -100,
            'FEAR_MAX': 100,
            'SIGMOID_STEEPNESS': 0.5,
            
            # 波动率参数
            'VOLATILITY_MEDIAN': 1.0,
            
            # 仓位控制
            'MAX_POSITION': 0.9,
            'MIN_POSITION': 0.5,
            'MIN_TRADE_POSITION': 0.05,
            'BUY_STEP_INITIAL': 0.3,
            'BUY_STEP_NORMAL': 0.1,
            'SELL_AMOUNT': 0.1,
            
            # 买入条件
            'BUY_SCORE_THRESHOLD': -50,
            'MAX_PANIC_THRESHOLD': -300,
            'DAILY_RETURN_THRESHOLD': -0.08,
            'CUMULATIVE_RETURN_THRESHOLD': -0.15,
            'CUMULATIVE_DAYS': 30,
            
            # 卖出条件
            'SELL_SCORE_THRESHOLD': 85,
            'SELL_PROFIT_TARGET': 0.20,
            'LAST_BUY_PROFIT_TARGET': 0.30,
            'STOP_LOSS_THRESHOLD': -1.0,
            'STOP_LOSS_POSITION': 0.6,
            'TAKE_PROFIT_THRESHOLD': 0.60,
            'TAKE_PROFIT_POSITION': 0.9,
        }
        
        # 仓位状态
        self.position = 0.0
        self.shares = 0
        self.avg_price = 0
        self.capital = self.config['INITIAL_CAPITAL']
        self.initial_position_built = False
        self.position_fifo = []
        
        # 买入状态
        self.last_buy_was_cumulative = False
        self.cumulative_buy_cooldown = 0
        self.last_buy_date = None
        self.actual_cumulative_days = 0
        
        # 历史记录
        self.capital_history = []
        self.position_history = []
        self.trades = []
    
    def calculate_rsi6(self, prices: pd.Series) -> pd.Series:
        """计算RSI6"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta.where(delta < 0, 0))
        avg_gain = gain.rolling(window=6).mean()
        avg_loss = loss.rolling(window=6).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def calculate_hv(self, prices: pd.Series, window: int) -> pd.Series:
        """计算历史波动率"""
        returns = np.log(prices / prices.shift(1))
        return returns.rolling(window=window).std() * np.sqrt(252)
    
    def calculate_panic_score(self, vix: float, rsi6: float) -> float:
        """
        计算恐慌指数 - 使用Sigmoid
        """
        # Sigmoid映射
        vix_clipped = max(min(vix, self.config['VIX_HIGH']), self.config['VIX_LOW'])
        vix_mid = (self.config['VIX_HIGH'] + self.config['VIX_LOW']) / 2
        x = -(vix_clipped - vix_mid) * self.config['SIGMOID_STEEPNESS']
        sigmoid_value = 1 / (1 + np.exp(-x))
        
        base_score = sigmoid_value * (self.config['FEAR_MAX'] - self.config['FEAR_MIN']) + self.config['FEAR_MIN']
        
        # RSI6调整
        if pd.isna(rsi6):
            return base_score
        
        if rsi6 <= 30:
            adjustment = 25 + (30 - rsi6) * 3
            new_score = base_score - adjustment
            return max(-150, new_score)
        elif rsi6 >= 70:
            adjustment = 25 + (rsi6 - 70) * 1.5
            new_score = base_score + adjustment
            return min(150, new_score)
        else:
            return base_score
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备数据"""
        df = df.copy()
        
        # 计算RSI6
        df['RSI6'] = self.calculate_rsi6(df['close'])
        
        # 计算历史波动率
        df['HV_20'] = self.calculate_hv(df['close'], 20)
        df['HV_30'] = self.calculate_hv(df['close'], 30)
        
        # 填充NaN
        df['RSI6'] = df['RSI6'].fillna(50)
        df['HV_20'] = df['HV_20'].fillna(self.config['VOLATILITY_MEDIAN'])
        df['HV_30'] = df['HV_30'].fillna(self.config['VOLATILITY_MEDIAN'])
        
        # 计算波动率系数
        if self.config['VOLATILITY_MEDIAN'] != 0:
            df['VOL_COEF'] = (df['HV_20'] / self.config['VOLATILITY_MEDIAN']).clip(0.5, 2.0)
        else:
            df['VOL_COEF'] = 1.0
        
        # 计算VIX替代 (HV * 40)
        df['vix'] = df['HV_20'] * 40
        
        # 计算恐慌指数
        df['panic_score'] = df.apply(
            lambda row: self.calculate_panic_score(row['vix'], row['RSI6']),
            axis=1
        )
        
        # 计算MA
        df['MA20'] = df['close'].rolling(window=20).mean()
        
        return df
    
    def calculate_max_cumulative_return(self, df: pd.DataFrame, current_idx: int) -> float:
        """计算最大累积跌幅"""
        if current_idx < 2:
            return 0
        
        max_cumulative_return = 0
        current_price = df.iloc[current_idx]['close']
        
        max_window = self.config['CUMULATIVE_DAYS']
        if self.last_buy_date is not None:
            days_since_last_buy = (df.iloc[current_idx]['date'] - self.last_buy_date).days
            max_window = min(max_window, max(days_since_last_buy, 2))
        
        self.actual_cumulative_days = 0
        for days in range(2, max_window + 1):
            if current_idx >= days:
                price_days_ago = df.iloc[current_idx - days]['close']
                cumulative_return = (current_price - price_days_ago) / price_days_ago
                if cumulative_return < max_cumulative_return:
                    max_cumulative_return = cumulative_return
                    self.actual_cumulative_days = days
        
        return max_cumulative_return
    
    def should_buy(self, current_data: pd.Series, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """
        判断是否应该买入 - 原始逻辑
        """
        # 检查仓位上限
        if self.position >= self.config['MAX_POSITION']:
            return False, "已达最大仓位"
        
        # 获取恐慌指数
        panic_score = current_data['panic_score']
        
        # 检查极端恐慌
        if panic_score <= self.config['MAX_PANIC_THRESHOLD']:
            return False, f"极端恐慌({panic_score:.1f})"
        
        # 检查冷却期
        if self.cumulative_buy_cooldown > 0:
            return False, f"冷却期({self.cumulative_buy_cooldown}天)"
        
        # 检查可用仓位
        available_position = self.config['MAX_POSITION'] - self.position
        if available_position < self.config['MIN_TRADE_POSITION']:
            return False, "可用仓位不足"
        
        # 检查恐慌指数阈值 (原始: <= threshold + 10)
        if panic_score > self.config['BUY_SCORE_THRESHOLD'] + 10:
            return False, f"恐慌指数({panic_score:.1f})未达阈值"
        
        # 计算日收益率
        if current_idx > 0:
            daily_return = (current_data['close'] - df.iloc[current_idx - 1]['close']) / df.iloc[current_idx - 1]['close']
        else:
            daily_return = 0
        
        # 计算累积跌幅
        max_cumulative_return = self.calculate_max_cumulative_return(df, current_idx)
        
        # 获取波动率系数
        vol_coef = current_data['VOL_COEF']
        
        # 确定阈值
        if not self.initial_position_built:
            # 初始建仓 - 原始代码使用 LOW_POSITION_THRESHOLD_MULTIPLIER (默认1)
            daily_threshold = self.config['DAILY_RETURN_THRESHOLD'] * vol_coef
            cumulative_threshold = self.config['CUMULATIVE_RETURN_THRESHOLD'] * vol_coef
        else:
            # 正常买入
            is_overvalued = current_data['close'] > current_data['MA20']
            daily_threshold = self.config['DAILY_RETURN_THRESHOLD'] * vol_coef
            cumulative_threshold = self.config['CUMULATIVE_RETURN_THRESHOLD'] * vol_coef
        
        # 检查条件
        daily_condition = daily_return <= daily_threshold
        cumulative_condition = max_cumulative_return <= cumulative_threshold
        
        if daily_condition:
            self.last_buy_was_cumulative = False
            self.cumulative_buy_cooldown = 1
            return True, f"日跌幅({daily_return:.2%})"
        
        if cumulative_condition:
            self.last_buy_was_cumulative = True
            self.cumulative_buy_cooldown = self.actual_cumulative_days
            return True, f"累积跌幅({max_cumulative_return:.2%}, {self.actual_cumulative_days}天)"
        
        return False, "不满足买入条件"
    
    def should_sell(self, current_data: pd.Series) -> Tuple[Optional[str], str]:
        """
        判断是否应该卖出 - 原始逻辑
        """
        if self.position <= self.config['MIN_POSITION']:
            return None, "低于最小仓位"
        
        panic_score = current_data['panic_score']
        profit = (current_data['close'] - self.avg_price) / self.avg_price if self.avg_price > 0 else 0
        
        # 检查可卖出仓位
        max_sell_position = self.position - self.config['MIN_POSITION']
        if max_sell_position < self.config['MIN_TRADE_POSITION']:
            return None, "可卖出仓位不足"
        
        # 止损
        if profit <= self.config['STOP_LOSS_THRESHOLD']:
            return 'stop_loss', f"止损({profit:.2%})"
        
        # 最后买入收益
        last_buy_profit = 0
        if self.position_fifo:
            last_buy = self.position_fifo[-1]
            last_buy_profit = (current_data['close'] - last_buy['price']) / last_buy['price']
        
        # 大幅止盈
        take_profit_condition = (
            panic_score >= self.config['SELL_SCORE_THRESHOLD'] and
            profit * self.position >= self.config['TAKE_PROFIT_THRESHOLD'] and
            self.position >= self.config['MIN_POSITION']
        )
        if take_profit_condition:
            return 'take_profit', f"大幅止盈(恐慌{panic_score:.1f})"
        
        # 正常卖出 - 条件1
        normal_sell_1 = (
            profit * self.position >= self.config['SELL_PROFIT_TARGET'] and
            panic_score >= self.config['SELL_SCORE_THRESHOLD'] and
            profit > 0
        )
        
        # 正常卖出 - 条件2
        vol_coef = current_data['VOL_COEF']
        if self.config['VOLATILITY_MEDIAN'] != 0:
            normal_sell_2 = (
                panic_score >= self.config['SELL_SCORE_THRESHOLD'] and
                last_buy_profit >= self.config['LAST_BUY_PROFIT_TARGET'] * vol_coef
            )
        else:
            normal_sell_2 = (
                panic_score >= self.config['SELL_SCORE_THRESHOLD'] and
                last_buy_profit >= self.config['LAST_BUY_PROFIT_TARGET']
            )
        
        if normal_sell_1 or normal_sell_2:
            return 'normal_sell', f"正常卖出(恐慌{panic_score:.1f})"
        
        return None, "不满足卖出条件"
    
    def execute_buy(self, current_data: pd.Series, reason: str) -> Optional[Dict]:
        """执行买入"""
        total_value = self.capital + self.shares * current_data['close']
        
        # 确定买入步长
        if not self.initial_position_built:
            buy_step = self.config['BUY_STEP_INITIAL']
        else:
            buy_step = self.config['BUY_STEP_NORMAL']
        
        # 计算目标仓位
        target_position = min(self.position + buy_step, self.config['MAX_POSITION'])
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
        
        # 更新最后买入日期
        self.last_buy_date = current_data['date']
        
        # 标记初始建仓
        if not self.initial_position_built and self.position >= self.config['MIN_POSITION']:
            self.initial_position_built = True
        
        return {
            'type': 'buy',
            'date': current_data['date'],
            'price': current_data['close'],
            'shares': new_shares,
            'position': self.position,
            'reason': reason
        }
    
    def execute_sell(self, current_data: pd.Series, sell_type: str, reason: str) -> Optional[Dict]:
        """执行卖出"""
        total_value = self.capital + self.shares * current_data['close']
        
        # 确定卖出仓位
        if sell_type == 'take_profit':
            sell_amount = self.position * self.config['TAKE_PROFIT_POSITION']
        elif sell_type == 'stop_loss':
            sell_amount = self.position * self.config['STOP_LOSS_POSITION']
        else:
            sell_amount = min(self.config['SELL_AMOUNT'], self.position - self.config['MIN_POSITION'])
        
        sell_shares = int((total_value * sell_amount) / current_data['close'])
        if sell_shares <= 0:
            return None
        
        # 执行卖出
        self.capital += sell_shares * current_data['close']
        
        # 更新FIFO (从最后买入开始卖)
        remaining_shares = sell_shares
        while remaining_shares > 0 and self.position_fifo:
            last_pos = self.position_fifo[-1]
            if last_pos['shares'] <= remaining_shares:
                remaining_shares -= last_pos['shares']
                self.position_fifo.pop()
            else:
                last_pos['shares'] -= remaining_shares
                remaining_shares = 0
        
        # 更新股数
        self.shares -= sell_shares
        
        # 更新仓位
        if self.shares > 0:
            self.position = (self.shares * current_data['close']) / (self.capital + self.shares * current_data['close'])
        else:
            self.position = 0
            self.avg_price = 0
        
        profit = (current_data['close'] - self.avg_price) / self.avg_price if self.avg_price > 0 else 0
        
        return {
            'type': sell_type,
            'date': current_data['date'],
            'price': current_data['close'],
            'shares': sell_shares,
            'position': self.position,
            'profit': profit,
            'reason': reason
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
            if self.cumulative_buy_cooldown > 0:
                self.cumulative_buy_cooldown -= 1
            
            # 检查买入
            should_buy, buy_reason = self.should_buy(current_data, df, i)
            if should_buy:
                trade = self.execute_buy(current_data, buy_reason)
                if trade:
                    self.trades.append(trade)
                    if verbose:
                        print(f"买入 - {trade['date'].strftime('%Y-%m-%d')}, 价格: {trade['price']:.2f}, 仓位: {trade['position']:.1%}")
            
            # 检查卖出
            else:
                sell_type, sell_reason = self.should_sell(current_data)
                if sell_type:
                    trade = self.execute_sell(current_data, sell_type, sell_reason)
                    if trade:
                        self.trades.append(trade)
                        if verbose:
                            print(f"卖出({sell_type}) - {trade['date'].strftime('%Y-%m-%d')}, 价格: {trade['price']:.2f}")
            
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
            'final_position': self.position
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
    
    print("="*60)
    print("原始策略回测")
    print("="*60)
    print(f"数据: {len(df)} 条记录")
    print(f"日期范围: {df['date'].min().strftime('%Y-%m-%d')} 到 {df['date'].max().strftime('%Y-%m-%d')}")
    print()
    
    # 创建策略
    strategy = OriginalPanicStrategy()
    
    # 运行回测
    results = strategy.run_backtest(df)
    
    # 打印结果
    print("\n" + "="*60)
    print("回测结果")
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
