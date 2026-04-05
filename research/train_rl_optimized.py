#!/usr/bin/env python3
"""
优化版RL训练 - 快速测试（200k步）

关键优化：
1. 奖励函数调整（降低回撤惩罚）
2. 限制做空（最多30%）
3. 持仓奖励
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.insert(0, '/Users/rziiiii/.openclaw/workspace/research')

from rl_hierarchical_env_optimized import SOXLHierarchicalEnvOptimized
from rl_hierarchical_env import SOXLHierarchicalTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

print("="*70)
print("优化版RL训练（200k步）")
print("="*70)
print("""
优化点：
  1. 回撤惩罚降低：50 → 15（仅>15%时惩罚）
  2. 波动惩罚降低：20 → 5
  3. 收益奖励提高：100 → 150
  4. 限制做空：最多30%
  5. 新增持仓奖励
    """)

# 1. 创建优化版环境
env = DummyVecEnv([lambda: SOXLHierarchicalEnvOptimized()])

print(f"✅ 环境创建成功")
print(f"   观察空间: {env.observation_space.shape}")
print(f"   动作空间: {env.action_space}")
print()

# 2. 创建模型
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    verbose=0,
)

# 3. 训练
print("开始训练（200k步）...")
model.learn(total_timesteps=200000)
print("✅ 训练完成")
print()

# 4. 回测RL策略
print("="*70)
print("回测优化版RL策略")
print("="*70)

test_env = SOXLHierarchicalEnvOptimized()
obs, _ = test_env.reset()

total_values = [100000]
rl_positions = []
rule_signals = []

done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = test_env.step(action)
    
    total_values.append(info['total_value'])
    rl_positions.append(info['rl_position'])
    rule_signals.append(info['rule_signal'])

pv = np.array(total_values)
total_return = (pv[-1] - 100000) / 100000
daily_returns = np.diff(pv) / pv[:-1]
sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
max_dd = np.max((np.maximum.accumulate(pv) - pv) / np.maximum.accumulate(pv))

rl_pos = np.array(rl_positions)

print(f"总收益:      {total_return:.2%}")
print(f"Sharpe:      {sharpe:.2f}")
print(f"最大回撤:    {max_dd:.2%}")
print(f"最终价值:    ${pv[-1]:,.0f}")
print(f"收益倍数:    {pv[-1]/100000:.2f}x")
print()
print(f"平均仓位:    {np.mean(rl_pos):.2f}")
print(f"做空比例:    {np.sum(rl_pos < 0) / len(rl_pos):.2%}")
print(f"仓位范围:    [{np.min(rl_pos):.2f}, {np.max(rl_pos):.2f}]")

# 5. 对比原版
print()
print("="*70)
print("对比：原版 vs 优化版")
print("="*70)

# 原版回测
env_orig = SOXLHierarchicalTradingEnv()
obs_orig, _ = env_orig.reset()

pv_orig = [100000]
done_orig = False
while not done_orig:
    action_orig = env_orig.action_space.sample()
    obs_orig, reward_orig, done_orig, truncated_orig, info_orig = env_orig.step(action_orig)
    pv_orig.append(info_orig['total_value'])

pv_orig = np.array(pv_orig)
return_orig = (pv_orig[-1] - 100000) / 100000
sharpe_orig = np.mean(np.diff(pv_orig) / pv_orig[:-1]) / (np.std(np.diff(pv_orig) / pv_orig[:-1]) + 1e-8) * np.sqrt(252)
dd_orig = np.max((np.maximum.accumulate(pv_orig) - pv_orig) / np.maximum.accumulate(pv_orig))

print(f"{'指标':<15} {'原版（随机）':<15} {'优化版（训练）':<15}")
print("-"*70)
print(f"{'总收益':<15} {return_orig:>13.2%} {total_return:>13.2%}")
print(f"{'Sharpe':<15} {sharpe_orig:>13.2f} {sharpe:>13.2f}")
print(f"{'最大回撤':<15} {dd_orig:>13.2%} {max_dd:>13.2%}")
print(f"{'收益倍数':<15} {pv_orig[-1]/100000:>13.2f}x {pv[-1]/100000:>13.2f}x")

print()
print("="*70)
print("✅ 优化完成！")
print("="*70)

# 6. 保存模型
model.save("models/hierarchical_rl_optimized/ppo_model")
print(f"\n✅ 模型已保存: models/hierarchical_rl_optimized/ppo_model.zip")
