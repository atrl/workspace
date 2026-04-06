#!/usr/bin/env python3
"""
测试已训练的 Qlib RL 模型
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
from stable_baselines3 import DQN
import qlib
from qlib.data import D

print("测试真正的 Qlib RL 模型...")

# 初始化 Qlib
qlib.init(provider_uri="/Users/rziiiii/.openclaw/workspace/qlib_soxl_binary")

# 加载数据
df = D.features(
    instruments=['SOXL'],
    fields=['$close', '$open', '$high', '$low', '$volume', '$vix',
            'Mean($close, 5)', 'Mean($close, 20)', 'Std($close, 20)'],
    start_time='2010-03-11',
    end_time='2026-04-02'
)

df = df.reset_index()
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')

# 特征
df['return'] = df['$close'].pct_change()
df['ma5'] = df['Mean($close, 5)']
df['ma20'] = df['Mean($close, 20)']
df['volatility'] = df['Std($close, 20)'] * np.sqrt(252)
df['vix_zscore'] = (df['$vix'] - df['$vix'].rolling(20).mean()) / (df['$vix'].rolling(20).std() + 1e-8)
df['panic_score'] = df['vix_zscore'] * 40 + df['$vix'].rolling(20).rank(pct=True) * 60

df = df.dropna()
test_df = df[df.index >= '2020-01-01']

print(f"测试数据: {len(test_df)} 条")

# 加载模型
try:
    model = DQN.load("models/qlib_rl_native/dqn_model.zip")
    print("✅ 模型加载成功")
except:
    print("❌ 模型不存在，使用随机策略")
    model = None

# 回测
POSITION_MAP = {i: i * 0.1 for i in range(10)}
cash = 100000
position = 0.0
values = [100000]

for i in range(60, len(test_df)-1):
    row = test_df.iloc[i]
    lb = test_df.iloc[i-60:i]
    
    obs = np.nan_to_num(np.array([
        row['return'],
        row['ma5'] / row['ma20'] - 1,
        lb['return'].mean(),
        lb['return'].std() * np.sqrt(252),
        row['$vix'],
        row['vix_zscore'],
        lb['$vix'].mean(),
        row['volatility'],
        row['panic_score'] / 100,
        np.log(row['$volume'] / lb['$volume'].mean()),
        position,
        (cash - 100000) / 100000,
        i / len(test_df),
        lb['return'].skew() if len(lb) > 3 else 0,
        lb['return'].kurtosis() if len(lb) > 4 else 0,
        0,
        0,
    ], dtype=np.float32)).reshape(1, -1)
    
    if model:
        action, _ = model.predict(obs, deterministic=True)
        target_pos = POSITION_MAP[int(action.item())]
    else:
        target_pos = 0.7  # 默认高仓位
    
    current_price = row['$close']
    next_price = test_df.iloc[i+1]['$close']
    
    commission = abs(target_pos - position) * 0.001
    position = target_pos
    
    ret = (next_price - current_price) / current_price
    port_ret = position * ret - commission
    
    cash *= (1 + port_ret)
    values.append(cash)

values = np.array(values)
total_return = (values[-1] - 100000) / 100000
daily_returns = np.diff(values) / values[:-1]
sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
max_dd = np.max((np.maximum.accumulate(values) - values) / np.maximum.accumulate(values))

print(f"\n总收益: {total_return:.2%}")
print(f"Sharpe: {sharpe:.2f}")
print(f"MaxDD: {max_dd:.2%}")

print(f"\n对比:")
print(f"  目标: 792%")
print(f"  真正Qlib版: {total_return:.2%}")

if total_return > 7.92:
    print(f"\n✅ 超过目标！")
