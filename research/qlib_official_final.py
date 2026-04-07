#!/usr/bin/env python3
"""
qlib 官方回测 + DQN（最终优化版）

目标：从 163% 提升到接近 792%

优化点：
1. 使用更精确的仓位计算
2. 减少交易频率（降低成本）
3. 优化订单生成逻辑
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import qlib
from qlib.data import D
from qlib.strategy.base import BaseStrategy
from qlib.backtest.decision import TradeDecisionWO, Order
from qlib.backtest.executor import SimulatorExecutor
from qlib.backtest import backtest
from stable_baselines3 import DQN
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

print("="*70)
print("qlib 官方回测 + DQN（最终优化版）")
print("="*70)

# 1. 初始化
qlib.init(provider_uri="/Users/rziiiii/.openclaw/workspace/qlib_soxl_binary")

# 2. 加载模型
model = DQN.load("models/qlib_rl_v2/dqn_model.zip")
print("✅ 加载 DQN 模型")

# 3. 加载完整数据
df = D.features(
    instruments=['SOXL'],
    fields=['$close', '$volume', '$vix'],
    start_time='2010-03-11',
    end_time='2026-04-02'
)

df = df.reset_index()
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime').sort_index()

if 'instrument' in df.columns:
    df = df.drop('instrument', axis=1)

df.columns = ['close', 'volume', 'vix']

# 计算特征（与训练时完全一致）
df['return'] = df['close'].pct_change()
df['ma5'] = df['close'].rolling(5).mean()
df['ma20'] = df['close'].rolling(20).mean()

delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))

df['vix_zscore'] = (df['vix'] - df['vix'].rolling(20).mean()) / (df['vix'].rolling(20).std() + 1e-8)

df['panic_score'] = (
    df['vix_zscore'] * 40 +
    df['vix'].rolling(20).rank(pct=True) * 60 +
    np.where(df['return'] < 0, np.minimum(df['return'] / -0.05, 1) * (-50), 0)
)

df = df.dropna()

print(f"✅ 数据准备完成: {len(df)} 条")

# 4. 定义策略（最终优化版）
class FinalRLStrategy(BaseStrategy):
    """最终优化版 RL 策略"""
    
    def __init__(self, model, df, **kwargs):
        super().__init__(**kwargs)
        
        self.model = model
        self.df = df
        self.target_position = 0.0
        self.last_trade_step = -100  # 上次交易步骤
        
        # 统计
        self.trade_count = 0
        self.position_history = []
        
    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        total_steps = self.trade_calendar.get_trade_len()
        
        if trade_step >= total_steps - 2:
            return TradeDecisionWO([], self)
        
        trade_date = self.trade_calendar.get_step_time(trade_step)[0]
        next_date = self.trade_calendar.get_step_time(trade_step + 1)[0]
        
        # 查找数据
        if trade_date not in self.df.index:
            return TradeDecisionWO([], self)
        
        idx = self.df.index.get_loc(trade_date)
        
        if idx < 60:
            return TradeDecisionWO([], self)
        
        # 获取数据
        row = self.df.iloc[idx]
        lb = self.df.iloc[idx-60:idx]
        
        # 17维观察（与训练时完全一致）
        obs = np.nan_to_num(np.array([
            row['return'],
            row['ma5'] / row['ma20'] - 1,
            lb['return'].mean(),
            lb['return'].std() * np.sqrt(252),
            row['vix'],
            row['vix_zscore'],
            lb['vix'].mean(),
            row['rsi'] / 100,
            row['panic_score'] / 100,
            np.log(row['volume'] / (lb['volume'].mean() + 1e-8)),
            self.target_position,
            0,
            idx / len(self.df),
            lb['return'].skew() if len(lb) > 3 else 0,
            lb['return'].kurtosis() if len(lb) > 4 else 0,
            0,
            0,
        ], dtype=np.float32)).reshape(1, -1)
        
        # 预测
        action, _ = self.model.predict(obs, deterministic=True)
        new_position = int(action.item()) * 0.1
        
        # 仓位变化太小则不交易
        if abs(new_position - self.target_position) < 0.1:  # 提高阈值
            self.position_history.append(self.target_position)
            return TradeDecisionWO([], self)
        
        # 交易间隔限制（避免过度交易）
        if trade_step - self.last_trade_step < 5:  # 至少间隔5天
            self.position_history.append(self.target_position)
            return TradeDecisionWO([], self)
        
        # 计算目标市值（使用固定初始值）
        account_value = 100000
        target_value = account_value * new_position
        target_shares = int(target_value / row['close'])
        
        if target_shares > 0:
            order = Order(
                stock_id='SOXL',
                amount=target_shares,
                direction=Order.BUY,
                start_time=trade_date,
                end_time=next_date
            )
            
            # 更新状态
            self.target_position = new_position
            self.last_trade_step = trade_step
            self.trade_count += 1
            self.position_history.append(new_position)
            
            # 调试（只打印前10次）
            if self.trade_count <= 10:
                print(f"  交易 #{self.trade_count}: 仓位 {new_position:.0%}, 股数 {target_shares}")
            
            return TradeDecisionWO([order], self)
        
        self.position_history.append(self.target_position)
        return TradeDecisionWO([], self)

# 5. 回测
print("\n运行官方回测...")

strategy = FinalRLStrategy(model=model, df=df)
executor = SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)

portfolio_metrics, indicator_metrics = backtest(
    start_time="2020-01-02",
    end_time="2026-03-31",
    strategy=strategy,
    executor=executor,
    benchmark="SOXL",
    account=100000,
    exchange_kwargs={
        "freq": "day",
        "deal_price": "close",
        "open_cost": 0.001,
        "close_cost": 0.001,
        "min_cost": 5,
    }
)

print("\n✅ 回测完成")

# 6. 提取结果
if isinstance(portfolio_metrics, dict) and '1day' in portfolio_metrics:
    day_data = portfolio_metrics['1day']
    if isinstance(day_data, tuple):
        result_df = day_data[0]
        
        if isinstance(result_df, pd.DataFrame):
            final_value = result_df['account'].iloc[-1]
            total_return = (final_value / 100000 - 1) * 100
            
            daily_returns = np.diff(result_df['account'].values) / result_df['account'].values[:-1]
            sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
            max_dd = np.max((np.maximum.accumulate(result_df['account'].values) - result_df['account'].values) / np.maximum.accumulate(result_df['account'].values))
            
            print(f"\n" + "="*70)
            print("最终结果（官方回测）")
            print("="*70)
            print(f"最终资产: ${final_value:,.2f}")
            print(f"总收益: {total_return:.2f}%")
            print(f"Sharpe: {sharpe:.2f}")
            print(f"MaxDD: {max_dd:.2%}")
            print(f"交易次数: {strategy.trade_count}")
            
            # 仓位分布
            if strategy.position_history:
                pos_array = np.array(strategy.position_history)
                print(f"\n仓位分布:")
                for i in range(10):
                    pos = i * 0.1
                    count = (pos_array == pos).sum()
                    pct = count / len(pos_array)
                    if count > 0:
                        print(f"  {pos:.0%}: {count:4d} ({pct:.1%})")
            
            print(f"\n对比:")
            print(f"  目标: 792%")
            print(f"  官方回测（最终优化版）: {total_return:.2f}%")
            
            # 绘图
            dates = result_df.index
            account_values = result_df['account'].values
            bench_values = result_df['bench'].values
            buy_hold = 100000 * (1 + bench_values)
            
            fig, ax = plt.subplots(figsize=(14, 7))
            
            ax.plot(dates, account_values / 1000, 
                    label='Qlib Official Backtest + DQN (Final)', 
                    linewidth=2.5, color='#2E86AB')
            ax.plot(dates, buy_hold / 1000, 
                    label='Buy & Hold (SOXL)', 
                    linewidth=2, color='#DC3545', alpha=0.6, linestyle='--')
            ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
            ax.fill_between(dates, 100, account_values / 1000, alpha=0.2, color='#2E86AB')
            
            ax.set_ylabel('Portfolio Value ($K)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=14, fontweight='bold')
            ax.set_title(f'Qlib Official Backtest + DQN (Final Optimized) - {total_return:.1f}% Return', 
                          fontsize=16, fontweight='bold', pad=20)
            
            ax.legend(loc='upper left', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}K'))
            
            # 文本框
            textstr = f'Official Qlib + DQN\n\nStart: $100K\nEnd: ${final_value/1000:.1f}K\nReturn: {total_return:.1f}%\nSharpe: {sharpe:.2f}\nTrades: {strategy.trade_count}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
            ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right', bbox=props,
                     fontfamily='monospace')
            
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            
            plt.tight_layout()
            plt.savefig('qlib_official_final.png', dpi=300, bbox_inches='tight', facecolor='white')
            print(f"\n✅ 图表已保存: qlib_official_final.png")
            
            # 保存结果
            with open('qlib_official_result.txt', 'w') as f:
                f.write(f"Qlib Official Backtest + DQN (Final)\n")
                f.write(f"="*70 + "\n")
                f.write(f"Final Value: ${final_value:,.2f}\n")
                f.write(f"Total Return: {total_return:.2f}%\n")
                f.write(f"Sharpe: {sharpe:.2f}\n")
                f.write(f"MaxDD: {max_dd:.2%}\n")
                f.write(f"Trade Count: {strategy.trade_count}\n")
            
            print(f"✅ 结果已保存: qlib_official_result.txt")

print("\n" + "="*70)
print("完成")
print("="*70)
