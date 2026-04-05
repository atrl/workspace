#!/usr/bin/env python3
"""
Qlib + RL 分层策略训练脚本

关键改进：
1. ✅ 使用 Qlib 框架
2. ✅ 训练/测试时间分开
3. ✅ 独立测试集验证
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.insert(0, '/Users/rziiiii/.openclaw/workspace/research')

from qlib_rl_hierarchical_env import SOXLQlibHierarchicalEnv, SOXLQlibProvider
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import pandas as pd

print("="*70)
print("Qlib + RL 分层策略训练")
print("="*70)
print("""
使用组件：
  ✅ qlib.data.BaseProvider
  ✅ Qlib Expression (Mean, RSI)
  ✅ 训练/测试时间分开
  ✅ Qlib 字段命名 ($close, $volume, $vix)
    """)

# ============================================================
# 1. 创建训练/测试环境
# ============================================================

print("创建环境...")
print()

# 训练环境（70%数据）
train_env = DummyVecEnv([lambda: SOXLQlibHierarchicalEnv(mode='train')])

# 测试环境（30%数据）
test_env = SOXLQlibHierarchicalEnv(mode='test')

print()
print("="*70)
print("训练配置")
print("="*70)
print(f"训练数据: {len(train_env.envs[0].df)} records")
print(f"测试数据: {len(test_env.df)} records")
print(f"训练比例: 70%")
print(f"测试比例: 30%")
print()

# ============================================================
# 2. 创建 PPO 模型
# ============================================================

model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=0,
)

# ============================================================
# 3. 训练
# ============================================================

print("开始训练（200k步）...")
model.learn(total_timesteps=200000)
print("✅ 训练完成")
print()

# ============================================================
# 4. 测试（使用未见过的数据）
# ============================================================

print("="*70)
print("测试（使用未见过的30%数据）")
print("="*70)

obs, _ = test_env.reset()
test_values = [100000]
test_positions = []
test_returns = []

done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = test_env.step(action)
    
    test_values.append(info['total_value'])
    test_positions.append(info['rl_position'])
    test_returns.append(info['return'])

# 计算测试指标
test_values = np.array(test_values)
test_return = (test_values[-1] - 100000) / 100000
test_daily_returns = np.diff(test_values) / test_values[:-1]
test_sharpe = np.mean(test_daily_returns) / (np.std(test_daily_returns) + 1e-8) * np.sqrt(252)
test_max_dd = np.max((np.maximum.accumulate(test_values) - test_values) / np.maximum.accumulate(test_values))

print(f"总收益:      {test_return:.2%}")
print(f"Sharpe:      {test_sharpe:.2f}")
print(f"最大回撤:    {test_max_dd:.2%}")
print(f"最终价值:    ${test_values[-1]:,.0f}")
print(f"收益倍数:    {test_values[-1]/100000:.2f}x")
print()

test_pos = np.array(test_positions)
print(f"平均仓位:    {np.mean(test_pos):.2f}")
print(f"做空比例:    {np.sum(test_pos < 0) / len(test_pos):.2%}")
print(f"仓位范围:    [{np.min(test_pos):.2f}, {np.max(test_pos):.2f}]")

# ============================================================
# 5. 对比：训练集 vs 测试集
# ============================================================

print()
print("="*70)
print("对比：训练集 vs 测试集")
print("="*70)

# 训练集回测
train_env_single = SOXLQlibHierarchicalEnv(mode='train')
obs_train, _ = train_env_single.reset()
train_values = [100000]

done_train = False
while not done_train:
    action_train, _ = model.predict(obs_train, deterministic=True)
    obs_train, reward_train, done_train, truncated_train, info_train = train_env_single.step(action_train)
    train_values.append(info_train['total_value'])

train_values = np.array(train_values)
train_return = (train_values[-1] - 100000) / 100000
train_daily_returns = np.diff(train_values) / train_values[:-1]
train_sharpe = np.mean(train_daily_returns) / (np.std(train_daily_returns) + 1e-8) * np.sqrt(252)
train_max_dd = np.max((np.maximum.accumulate(train_values) - train_values) / np.maximum.accumulate(train_values))

print(f"{'指标':<15} {'训练集':>15} {'测试集':>15}")
print("-"*70)
print(f"{'总收益':<15} {train_return:>13.2%} {test_return:>13.2%}")
print(f"{'Sharpe':<15} {train_sharpe:>13.2f} {test_sharpe:>13.2f}")
print(f"{'最大回撤':<15} {train_max_dd:>13.2%} {test_max_dd:>13.2%}")
print(f"{'收益倍数':<15} {train_values[-1]/100000:>13.2f}x {test_values[-1]/100000:>13.2f}x")
print("="*70)

# ============================================================
# 6. 检查过拟合
# ============================================================

print()
print("="*70)
print("过拟合检查")
print("="*70)

sharpe_ratio = test_sharpe / train_sharpe if train_sharpe > 0 else 0
return_ratio = test_return / train_return if train_return > 0 else 0

print(f"Sharpe比 (测试/训练):   {sharpe_ratio:.2f}")
print(f"收益比 (测试/训练):     {return_ratio:.2f}")
print()

if sharpe_ratio < 0.8:
    print("⚠️ 警告：可能存在过拟合（测试Sharpe < 训练Sharpe * 0.8）")
elif sharpe_ratio > 1.2:
    print("✅ 测试表现优于训练（可能数据集特性不同）")
else:
    print("✅ 泛化能力良好")

# ============================================================
# 7. 保存模型
# ============================================================

import pathlib
pathlib.Path("models/qlib_rl_hierarchical").mkdir(parents=True, exist_ok=True)
model.save("models/qlib_rl_hierarchical/ppo_model")

print()
print("="*70)
print("✅ 训练完成！")
print("="*70)
print(f"模型已保存: models/qlib_rl_hierarchical/ppo_model.zip")
print()
print("使用方式：")
print("  1. 加载模型: model = PPO.load('models/qlib_rl_hierarchical/ppo_model.zip')")
print("  2. 创建测试环境: env = SOXLQlibHierarchicalEnv(mode='test')")
print("  3. 运行回测: obs = env.reset(); action = model.predict(obs)")
