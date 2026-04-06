#!/usr/bin/env python3
"""
测试已训练的10档仓位模型
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['KMP_INIT_AT_FORK'] = 'FALSE'
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
from stable_baselines3 import DQN

print("测试10档仓位模型...")

# 加载模型
model = DQN.load("models/qlib_rl_10level/dqn_model.zip")

# 加载数据
df = pd.read_csv("data/soxl_soxx_vix_full.csv", parse_dates=['date'])
df = df.set_index('date')

# 计算特征（简化版）
close = df['soxl_close']
vix = df['vix']
df['return'] = close.pct_change()
df['ma5'] = close.rolling(5).mean()
df['ma20'] = close.rolling(20).mean()
delta = close.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
df['vix_zscore'] = (vix - vix.rolling(20).mean()) / (vix.rolling(20).std() + 1e-8)
df['panic_score'] = (
    df['vix_zscore'] * 40 +
    vix.rolling(20).rank(pct=True) * 60 +
    np.where(df['return'] < 0, np.minimum(df['return'] / -0.05, 1) * (-50), 0)
)
df = df.dropna()

test_df = df[df.index >= '2020-01-01']

# 简单回测
POSITION_MAP = {i: i * 0.1 for i in range(10)}

cash = 100000
position = 0.0
values = [100000]
positions = [0.0]

for i in range(60, len(test_df)-1):
    row = test_df.iloc[i]
    
    # 构造观察（简化）
    lookback = test_df.iloc[i-60:i]
    obs = np.array([
        row['return'],
        row['ma5'] / row['ma20'] - 1,
        lookback['return'].mean(),
        lookback['return'].std() * np.sqrt(252),
        row['vix'],
        row['vix_zscore'],
        lookback['vix'].mean(),
        row['rsi'] / 100,
        lookback['rsi'].mean() / 100,
        row['panic_score'] / 100,
        lookback['panic_score'].mean() / 100,
        np.log(row['soxl_volume'] / lookback['soxl_volume'].mean()),
        lookback['return'].std() * np.sqrt(252),
        position,
        (cash - 100000) / 100000,
        i / len(test_df),
        row['soxl_close'] / lookback['soxl_close'].iloc[-10:].mean() - 1,
        lookback['return'].skew() if len(lookback) > 3 else 0,
        lookback['return'].kurtosis() if len(lookback) > 4 else 0,
        0,  # ATR
        0,  # bb_width
        0,  # momentum_10
        0,  # vix_percentile
        0,  # vix_regime
        0,  # market_stress
        0,  # macd
        0,  # volume_trend
        0,  # unrealized_pnl
        0,  # days_held
        0,  # day_of_week
    ], dtype=np.float32)
    
    obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0).reshape(1, -1)
    
    # 预测动作
    action = model.predict(obs, deterministic=True)[0]
    target_position = POSITION_MAP[int(action)]
    
    # 交易
    current_price = row['soxl_close']
    next_price = test_df.iloc[i+1]['soxl_close']
    
    commission = abs(target_position - position) * 0.001
    position = target_position
    
    price_return = (next_price - current_price) / current_price
    portfolio_return = position * price_return - commission
    
    cash *= (1 + portfolio_return)
    values.append(cash)
    positions.append(position)

# 统计
values = np.array(values)
total_return = (values[-1] - 100000) / 100000
daily_returns = np.diff(values) / values[:-1]
sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
max_dd = np.max((np.maximum.accumulate(values) - values) / np.maximum.accumulate(values))

# 仓位分布
positions = np.array(positions)
print(f"\n仓位分布:")
for i in range(10):
    count = (positions == i * 0.1).sum()
    pct = (positions == i * 0.1).mean()
    print(f"  {i} ({i*10:2d}%): {count:4d} ({pct:.1%})")

print(f"\n总收益: {total_return:.2%}")
print(f"Sharpe: {sharpe:.2f}")
print(f"MaxDD: {max_dd:.2%}")

print(f"\n对比:")
print(f"  目标: 792%")
print(f"  旧版（2档）: 806%")
print(f"  新版（10档）: {total_return:.2%}")

if total_return > 8.06:
    print(f"\n✅ 改进！ {total_return:.2%} > 806%")
