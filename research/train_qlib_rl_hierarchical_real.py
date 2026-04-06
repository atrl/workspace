#!/usr/bin/env python3
"""
真正的 Qlib RL 分层训练

架构：规则策略 (信号) × RL智能体 (仓位) = 最终仓位

这是恢复 792% 收益的关键！
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.insert(0, '/Users/rziiiii/.openclaw/workspace/research')

from qlib_rl_hierarchical_real import QlibSOXLHierarchicalEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import numpy as np

print("="*70)
print("真正的 Qlib RL 分层训练")
print("="*70)
print("""
架构：
  规则策略 (信号) × RL智能体 (仓位) = 最终仓位

这是恢复 792% 收益的关键！
""")

# ============================================================
# 1. 准备数据
# ============================================================

print("\n步骤1：准备数据...")

df = pd.read_csv(
    "/Users/rziiiii/.openclaw/workspace/data/soxl_soxx_vix_full.csv",
    parse_dates=['date']
)
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
    np.where(df['return'] < 0, np.minimum(df['return'] / -0.05, 1) * -50, 0)
)

df = df.dropna()

# 分割数据
train_df = df[df.index < '2020-01-01']
test_df = df[df.index >= '2020-01-01']

print(f"训练数据: {len(train_df)} records ({train_df.index.min()} to {train_df.index.max()})")
print(f"测试数据: {len(test_df)} records ({test_df.index.min()} to {test_df.index.max()})")

# ============================================================
# 2. 创建分层环境
# ============================================================

print("\n步骤2：创建分层环境...")

train_env = DummyVecEnv([lambda: QlibSOXLHierarchicalEnv(train_df)])

print(f"✅ 分层环境创建成功")
print(f"  Observation space: {train_env.observation_space}")
print(f"  Action space: {train_env.action_space}")

# ============================================================
# 3. 训练 PPO 模型
# ============================================================

print("\n步骤3：训练 PPO 模型...")

model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    verbose=0,
)

print("开始训练（200k步）...")
model.learn(total_timesteps=200000)
print("✅ 训练完成")

# ============================================================
# 4. 测试模型
# ============================================================

print("\n" + "="*70)
print("步骤4：测试模型（2020-2026）")
print("="*70)

test_env = QlibSOXLHierarchicalEnv(test_df)
obs, _ = test_env.reset()

test_values = [100000]
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = test_env.step(action)
    test_values.append(info.get('total_value', test_values[-1]))

# 计算指标
test_values = np.array(test_values)
total_return = (test_values[-1] - 100000) / 100000
daily_returns = np.diff(test_values) / test_values[:-1]
sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
max_dd = np.max((np.maximum.accumulate(test_values) - test_values) / np.maximum.accumulate(test_values))

print(f"\n总收益:      {total_return:.2%}")
print(f"Sharpe:      {sharpe:.2f}")
print(f"最大回撤:    {max_dd:.2%}")
print(f"最终价值:    ${test_values[-1]:,.0f}")
print(f"收益倍数:    {test_values[-1]/100000:.2f}x")

# ============================================================
# 5. 保存模型
# ============================================================

import pathlib
pathlib.Path("models/qlib_rl_hierarchical_real").mkdir(parents=True, exist_ok=True)
model.save("models/qlib_rl_hierarchical_real/ppo_model")

print("\n" + "="*70)
print("✅ 训练完成！")
print("="*70)
print(f"""
模型保存:
  models/qlib_rl_hierarchical_real/ppo_model.zip

架构：
  规则策略 (信号) × RL智能体 (仓位) = 最终仓位

使用的 Qlib 组件:
  ✅ qlib.rl.Simulator
  ✅ qlib.rl.StateInterpreter
  ✅ qlib.rl.ActionInterpreter
  ✅ qlib.rl.Reward

这是真正的 Qlib 框架 + 分层架构！
""")
