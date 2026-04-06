#!/usr/bin/env python3
"""
简化版：生成回测图表
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['KMP_INIT_AT_FORK'] = 'FALSE'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
import sys

print("生成回测图表...")

# 加载数据
df = pd.read_csv("data/soxl_soxx_vix_full.csv", parse_dates=['date'])
df = df.set_index('date')

# 计算特征
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

# 测试数据
test_df = df[df.index >= '2020-01-01']

# 加载模型
sys.path.insert(0, '/Users/rziiiii/.openclaw/workspace/research')
from train_qlib_rl_imitation import ExpertEnv

model = DQN.load("models/qlib_rl_imitation/dqn_model.zip")

# 回测
test_env = ExpertEnv(test_df)
obs, _ = test_env.reset()

test_values = [100000]
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = test_env.step(action)
    test_values.append(info.get('cash', test_values[-1]))

test_values = np.array(test_values)

# 指标
total_return = (test_values[-1] - 100000) / 100000
daily_returns = np.diff(test_values) / test_values[:-1]
sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
max_dd = np.max((np.maximum.accumulate(test_values) - test_values) / np.maximum.accumulate(test_values))

# 买入持有
buy_hold = 100000 * (test_df['soxl_close'] / test_df['soxl_close'].iloc[0])

print(f"RL收益: {total_return:.2%}, Sharpe: {sharpe:.2f}")

# 绘图
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# 图1: 资金曲线
axes[0].plot(test_df.index, test_values[:-1], label='RL Strategy', linewidth=2)
axes[0].plot(test_df.index, buy_hold, label='Buy & Hold', linewidth=2, alpha=0.7)
axes[0].axhline(y=100000, color='gray', linestyle='--', alpha=0.5)
axes[0].fill_between(test_df.index, 100000, test_values[:-1], alpha=0.3)
axes[0].set_title(f'Qlib RL Strategy Backtest (2020-2026)\n'
                  f'Return: {total_return:.1%} | Sharpe: {sharpe:.2f} | MaxDD: {max_dd:.2%}', 
                  fontsize=14, fontweight='bold')
axes[0].set_ylabel('Portfolio Value ($)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 图2: SOXL价格
axes[1].plot(test_df.index, test_df['soxl_close'], linewidth=1.5)
axes[1].set_title('SOXL Price (2020-2026)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Price ($)')
axes[1].set_xlabel('Date')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('research/qlib_rl_backtest_chart.png', dpi=150)
print("✅ 保存: research/qlib_rl_backtest_chart.png")
