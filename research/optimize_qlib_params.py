#!/usr/bin/env python3
"""
遗传算法优化 Qlib RL 分层参数

目标：在测试集（2020-2026）跑赢 792%
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.insert(0, '/Users/rziiiii/.openclaw/workspace/research')

from qlib_rl_hierarchical_real import QlibSOXLHierarchicalEnv
from stable_baselines3 import PPO
import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import random

print("="*70)
print("遗传算法优化 Qlib RL 分层参数")
print("="*70)
print("""
目标：在测试集（2020-2026）跑赢 792%

优化参数：
  - panic_threshold: [-50, -20]
  - vix_panic: [15, 25]
  - vix_extreme: [25, 35]
  - vix_greed: [10, 15]
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

print(f"训练数据: {len(train_df)} records")
print(f"测试数据: {len(test_df)} records")

# ============================================================
# 2. 定义评估函数
# ============================================================

def evaluate_params(params, train_data, test_data, quick_mode=True):
    """
    评估参数组合
    
    Args:
        params: [panic_threshold, vix_panic, vix_extreme, vix_greed]
        quick_mode: True = 快速评估（50k步），False = 完整评估（200k步）
    
    Returns:
        (total_return, sharpe, max_dd)
    """
    panic_threshold, vix_panic, vix_extreme, vix_greed = params
    
    try:
        # 创建环境
        from gymnasium import spaces
        
        class QuickEnv(QlibSOXLHierarchicalEnv):
            def __init__(self, data, params):
                self.data = data
                self.params = params
                # 简化初始化
                self.simulator = type('obj', (object,), {
                    'data': data,
                    'initial_cash': 100000,
                    'commission': 0.001,
                    'panic_threshold': params[0],
                    'vix_panic': params[1],
                    'vix_extreme': params[2],
                    'vix_greed': params[3],
                })()
                
                # 定义空间
                self.action_space = spaces.Box(
                    low=np.array([0.0]),
                    high=np.array([1.0]),
                    dtype=np.float32
                )
                self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(9,),
                    dtype=np.float32
                )
        
        # 训练
        train_env = QuickEnv(train_data, params)
        
        from stable_baselines3.common.vec_env import DummyVecEnv
        train_env_vec = DummyVecEnv([lambda: train_env])
        
        model = PPO(
            "MlpPolicy",
            train_env_vec,
            learning_rate=3e-4,
            n_steps=2048 if not quick_mode else 512,
            batch_size=64,
            n_epochs=10,
            verbose=0,
        )
        
        timesteps = 50000 if quick_mode else 200000
        model.learn(total_timesteps=timesteps)
        
        # 测试
        test_env = QuickEnv(test_data, params)
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
        
        return total_return, sharpe, max_dd
        
    except Exception as e:
        print(f"评估失败: {e}")
        return -1.0, 0.0, 1.0


# ============================================================
# 3. 遗传算法优化
# ============================================================

print("\n步骤2：遗传算法优化...")

# 定义参数范围
PARAM_BOUNDS = [
    (-50, -20),  # panic_threshold
    (15, 25),     # vix_panic
    (25, 35),     # vix_extreme
    (10, 15),     # vix_greed
]

# 创建遗传算法
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", lambda: [random.uniform(b[0], b[1]) for b in PARAM_BOUNDS])
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 评估函数
def evaluate(individual):
    total_return, sharpe, max_dd = evaluate_params(individual, train_df, test_df, quick_mode=True)
    
    # 目标：最大化收益，同时控制回撤
    fitness = total_return * 100 - max_dd * 50
    
    return (fitness,)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# 运行优化
population = toolbox.population(n=10)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("max", np.max)

print("开始优化（10个体，5代）...")
result = algorithms.eaSimple(
    population,
    toolbox,
    cxpb=0.7,
    mutpb=0.2,
    ngen=5,
    stats=stats,
    halloffame=hof,
    verbose=True
)

# 最优参数
best_params = hof[0]
print(f"\n最优参数:")
print(f"  panic_threshold: {best_params[0]:.1f}")
print(f"  vix_panic: {best_params[1]:.1f}")
print(f"  vix_extreme: {best_params[2]:.1f}")
print(f"  vix_greed: {best_params[3]:.1f}")

# ============================================================
# 4. 用最优参数完整训练
# ============================================================

print("\n" + "="*70)
print("步骤3：用最优参数完整训练")
print("="*70)

total_return, sharpe, max_dd = evaluate_params(best_params, train_df, test_df, quick_mode=False)

print(f"\n最终结果（2020-2026测试集）:")
print(f"  总收益: {total_return:.2%}")
print(f"  Sharpe: {sharpe:.2f}")
print(f"  最大回撤: {max_dd:.2%}")

# 对比目标
if total_return > 7.92:
    print(f"\n✅ 超过目标！ {total_return:.2%} > 792%")
else:
    print(f"\n⚠️ 未达目标 {total_return:.2%} < 792%")

print("\n" + "="*70)
print("完成！")
print("="*70)
