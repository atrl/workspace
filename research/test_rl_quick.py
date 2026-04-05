#!/usr/bin/env python3
"""
分层RL训练脚本 - 快速测试版（10000步）
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'  # 限制OMP线程数

import sys
sys.path.insert(0, '/Users/rziiiii/.openclaw/workspace/research')

from rl_hierarchical_env import SOXLHierarchicalTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

print("="*70)
print("分层RL训练 - 快速测试（10000步）")
print("="*70)

# 1. 创建环境
env = DummyVecEnv([lambda: SOXLHierarchicalTradingEnv()])

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
print("开始训练...")
model.learn(total_timesteps=10000)
print("✅ 训练完成")
print()

# 4. 快速回测
print("="*70)
print("快速回测（前100步）")
print("="*70)

test_env = SOXLHierarchicalTradingEnv()
obs, _ = test_env.reset()

total_values = [100000]
rl_positions = []
rule_signals = []

for i in range(100):
    # RL预测
    action, _ = model.predict(obs, deterministic=True)
    
    # 执行
    obs, reward, done, truncated, info = test_env.step(action)
    
    total_values.append(info['total_value'])
    rl_positions.append(info['rl_position'])
    rule_signals.append(info['rule_signal'])
    
    if done:
        break

# 5. 结果
pv = np.array(total_values)
total_return = (pv[-1] - 100000) / 100000

print(f"总收益:      {total_return:.2%}")
print(f"最终价值:    ${pv[-1]:,.0f}")
print(f"收益倍数:    {pv[-1]/100000:.2f}x")
print()

# RL行为分析
rl_pos = np.array(rl_positions)
print(f"RL平均仓位:      {np.mean(rl_pos):.2f}")
print(f"RL仓位范围:      [{np.min(rl_pos):.2f}, {np.max(rl_pos):.2f}]")
print(f"RL做空比例:      {np.sum(rl_pos < 0) / len(rl_pos):.2%}")

print()
print("="*70)
print("测试完成！可以运行完整训练：python3 train_hierarchical_rl.py")
print("="*70)
