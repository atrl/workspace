#!/usr/bin/env python3
"""
训练增强版 Qlib RL 模型（添加 SOXX 特征）

对比：
1. 基线版本（11维特征）: SOXL + VIX
2. 增强版本（16维特征）: SOXL + VIX + SOXX
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.insert(0, '/Users/rziiiii/.openclaw/workspace/research')

from qlib_rl_env_enhanced import SOXLQlibHierarchicalEnvEnhanced
from train_2019_test_2020 import SOXLQlibHierarchicalEnvSplit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import pandas as pd
import time

print("="*70)
print("对比：基线 vs 增强版（添加SOXX特征）")
print("="*70)
print("""
基线版本（11维）：
  - SOXL: close, volume, RSI, 趋势
  - VIX:  zscore  normalized
  - 恐慌得分
  
增强版本（16维）：
  - 基线全部特征
  - SOXX收益率
  - SOXX RSI
  - 杠杆比率（SOXL/SOXX）
  - SOXX趋势
  - 基准偏离度

训练配置：
  - 训练集：2010-2019
  - 测试集：2020-2026
  - 200k步训练
    """)

# ============================================================
# 1. 基线版本训练
# ============================================================

print("\n" + "="*70)
print("训练基线版本（11维特征）")
print("="*70)

start_time_baseline = time.time()

train_env_baseline = DummyVecEnv([lambda: SOXLQlibHierarchicalEnvSplit(mode='train')])

model_baseline = PPO("MlpPolicy", train_env_baseline, learning_rate=3e-4, verbose=0)
model_baseline.learn(total_timesteps=200000)

baseline_train_time = time.time() - start_time_baseline

print(f"✅ 基线版本训练完成 ({baseline_train_time:.1f}秒)")

# ============================================================
# 2. 增强版本训练
# ============================================================

print("\n" + "="*70)
print("训练增强版本（16维特征）")
print("="*70)

start_time_enhanced = time.time()

train_env_enhanced = DummyVecEnv([lambda: SOXLQlibHierarchicalEnvEnhanced(mode='train')])

model_enhanced = PPO("MlpPolicy", train_env_enhanced, learning_rate=3e-4, verbose=0)
model_enhanced.learn(total_timesteps=200000)

enhanced_train_time = time.time() - start_time_enhanced

print(f"✅ 增强版本训练完成 ({enhanced_train_time:.1f}秒)")

# ============================================================
# 3. 测试两个版本
# ============================================================

print("\n" + "="*70)
print("测试对比（2020-2026）")
print("="*70)

def backtest_model(model, env_class, name):
    """回测模型"""
    test_env = env_class(mode='test')
    obs, _ = test_env.reset()
    
    values = [100000]
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        values.append(info['total_value'])
    
    values = np.array(values)
    total_return = (values[-1] - 100000) / 100000
    daily_returns = np.diff(values) / values[:-1]
    sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
    max_dd = np.max((np.maximum.accumulate(values) - values) / np.maximum.accumulate(values))
    
    return {
        'name': name,
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'final_value': values[-1],
        'times': len(values)
    }

# 测试基线版本
print("\n测试基线版本...")
baseline_results = backtest_model(model_baseline, SOXLQlibHierarchicalEnvSplit, "基线版本")

# 测试增强版本
print("测试增强版本...")
enhanced_results = backtest_model(model_enhanced, SOXLQlibHierarchicalEnvEnhanced, "增强版本")

# ============================================================
# 4. 结果对比
# ============================================================

print("\n" + "="*70)
print("对比结果")
print("="*70)

print(f"\n{'指标':<20} {'基线版本 (11维)':>20} {'增强版本 (16维)':>20}")
print("-"*70)
print(f"{'总收益':<20} {baseline_results['total_return']:>18.2%} {enhanced_results['total_return']:>18.2%}")
print(f"{'Sharpe':<20} {baseline_results['sharpe']:>18.2f} {enhanced_results['sharpe']:>18.2f}")
print(f"{'最大回撤':<20} {baseline_results['max_dd']:>18.2%} {enhanced_results['max_dd']:>18.2%}")
print(f"{'最终价值':<20} ${baseline_results['final_value']:>16,.0f} ${enhanced_results['final_value']:>16,.0f}")
print(f"{'收益倍数':<20} {baseline_results['final_value']/100000:>18.2f}x {enhanced_results['final_value']/100000:>18.2f}x")
print("="*70)

# ============================================================
# 5. 改进分析
# ============================================================

print("\n" + "="*70)
print("改进分析")
print("="*70)

sharpe_improve = enhanced_results['sharpe'] - baseline_results['sharpe']
return_improve = enhanced_results['total_return'] - baseline_results['total_return']
dd_improve = enhanced_results['max_dd'] - baseline_results['max_dd']

print(f"\nSharpe改进:   {sharpe_improve:+.2f} ({sharpe_improve/baseline_results['sharpe']*100:+.1f}%)")
print(f"收益改进:     {return_improve:+.2%} ({return_improve/baseline_results['total_return']*100:+.1f}%)")
print(f"回撤改进:     {dd_improve:+.2%}")

if sharpe_improve > 0.1:
    print(f"\n✅ SOXX特征显著提升性能！")
elif sharpe_improve > 0:
    print(f"\n✅ SOXX特征略有提升")
elif sharpe_improve > -0.1:
    print(f"\n⚠️ SOXX特征略有下降")
else:
    print(f"\n❌ SOXX特征无帮助")

# ============================================================
# 6. 保存模型
# ============================================================

import pathlib
pathlib.Path("models/qlib_rl_enhanced").mkdir(parents=True, exist_ok=True)
model_enhanced.save("models/qlib_rl_enhanced/ppo_model")

print("\n" + "="*70)
print("✅ 完成！")
print("="*70)
print(f"\n基线模型: {baseline_train_time:.1f}秒训练")
print(f"增强模型: {enhanced_train_time:.1f}秒训练")
print(f"\n增强模型已保存: models/qlib_rl_enhanced/ppo_model.zip")

# ============================================================
# 7. 特征重要性分析
# ============================================================

print("\n" + "="*70)
print("SOXX特征分析")
print("="*70)

print("""
新增的SOXX特征（5维）：
  1. SOXX收益率 - 半导体基准收益
  2. SOXX RSI    - 基准超买超卖
  3. 杠杆比率    - SOXL/SOXX 比率（监测杠杆效应）
  4. SOXX趋势    - 基准市场方向
  5. 基准偏离度   - SOXL相对SOXX的偏离

这些特征帮助RL模型：
  - 理解市场整体趋势（SOXX）
  - 识别过度杠杆（杠杆比率）
  - 发现价格异常（基准偏离度）
""")
