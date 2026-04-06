#!/usr/bin/env python3
"""
诊断：为什么 Qlib RL 分层性能这么差？

对比：
  纯规则：990%
  RL分层：179%
  
可能原因：
  1. 奖励函数设计问题
  2. RL学到错误策略
  3. 规则信号被RL"破坏"
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("诊断：为什么 RL 性能差？")
print("="*70)

# ============================================================
# 1. 准备数据
# ============================================================

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

# 测试数据
test_df = df[df.index >= '2020-01-01']

print(f"测试数据: {len(test_df)} records")

# ============================================================
# 2. 规则信号分析
# ============================================================

print("\n" + "="*70)
print("规则信号分析")
print("="*70)

panic_threshold = -25
vix_panic = 22
vix_extreme = 26
vix_greed = 15

signals = []
positions_pure_rule = []

cash = 100000
position = 0.0

for i in range(60, len(test_df)):
    row = test_df.iloc[i]
    
    # 规则信号
    if row['panic_score'] <= panic_threshold:
        if row['vix'] > vix_extreme:
            signal = 1.0
        elif row['vix'] > vix_panic:
            signal = 0.9
        else:
            signal = 0.7
    elif row['rsi'] < 65 and row['ma5'] > row['ma20']:
        signal = 0.6
    elif row['vix'] < vix_greed:
        if row['rsi'] > 65:
            signal = 0.2
        else:
            signal = 0.4
    else:
        signal = 0.5
    
    signals.append(signal)
    
    # 纯规则仓位
    target_position = signal * 0.8
    
    if i < len(test_df) - 1:
        price_return = (test_df.iloc[i+1]['soxl_close'] - row['soxl_close']) / row['soxl_close']
        portfolio_return = target_position * price_return - abs(target_position - position) * 0.001
        cash *= (1 + portfolio_return)
        position = target_position
        positions_pure_rule.append(target_position)

signals = np.array(signals)
positions_pure_rule = np.array(positions_pure_rule)

print(f"信号分布:")
print(f"  Min: {signals.min():.2f}")
print(f"  Max: {signals.max():.2f}")
print(f"  Mean: {signals.mean():.2f}")
print(f"  Std: {signals.std():.2f}")

print(f"\n规则信号出现频率:")
print(f"  1.0 (极端恐慌): {(signals == 1.0).sum()} ({(signals == 1.0).mean():.1%})")
print(f"  0.9 (恐慌):     {(signals == 0.9).sum()} ({(signals == 0.9).mean():.1%})")
print(f"  0.7 (恐慌):     {(signals == 0.7).sum()} ({(signals == 0.7).mean():.1%})")
print(f"  0.6 (趋势):     {(signals == 0.6).sum()} ({(signals == 0.6).mean():.1%})")
print(f"  0.5 (中性):     {(signals == 0.5).sum()} ({(signals == 0.5).mean():.1%})")
print(f"  0.4 (贪婪):     {(signals == 0.4).sum()} ({(signals == 0.4).mean():.1%})")
print(f"  0.2 (超买):     {(signals == 0.2).sum()} ({(signals == 0.2).mean():.1%})")

# ============================================================
# 3. 加载训练好的RL模型，看RL学到了什么
# ============================================================

print("\n" + "="*70)
print("RL 学到了什么？")
print("="*70)

try:
    import sys
    sys.path.insert(0, '/Users/rziiiii/.openclaw/workspace/research')
    from qlib_rl_hierarchical_real import QlibSOXLHierarchicalEnv
    from stable_baselines3 import PPO
    
    # 加载模型
    model = PPO.load("models/qlib_rl_optimized/ppo_model.zip")
    
    # 测试
    env = QlibSOXLHierarchicalEnv(
        test_df,
        panic_threshold=-25,
        vix_panic=22,
        vix_extreme=26,
        vix_greed=15,
    )
    
    obs, _ = env.reset()
    
    rl_positions = []
    rule_signals_recorded = []
    
    done = False
    while not done:
        # RL动作
        action, _ = model.predict(obs, deterministic=True)
        
        # 记录规则信号
        rule_signals_recorded.append(env.simulator._get_rule_signal())
        
        # RL仓位
        rl_pos = float(np.clip(action[0], 0, 1))
        rl_positions.append(rl_pos)
        
        obs, reward, done, truncated, info = env.step(action)
    
    rl_positions = np.array(rl_positions)
    rule_signals_recorded = np.array(rule_signals_recorded)
    
    print(f"\nRL仓位分布:")
    print(f"  Min: {rl_positions.min():.2f}")
    print(f"  Max: {rl_positions.max():.2f}")
    print(f"  Mean: {rl_positions.mean():.2f}")
    print(f"  Std: {rl_positions.std():.2f}")
    
    print(f"\n最终仓位 = 规则信号 × RL仓位")
    final_positions = rule_signals_recorded * rl_positions
    print(f"  Min: {final_positions.min():.2f}")
    print(f"  Max: {final_positions.max():.2f}")
    print(f"  Mean: {final_positions.mean():.2f}")
    
    print(f"\n对比:")
    print(f"  纯规则平均仓位: {positions_pure_rule.mean():.2f}")
    print(f"  RL分层平均仓位: {final_positions.mean():.2f}")
    
    if final_positions.mean() < positions_pure_rule.mean() * 0.5:
        print(f"\n⚠️ 问题：RL仓位太低！")
        print(f"   RL学到'保守'策略，导致收益低")
    
except Exception as e:
    print(f"无法加载RL模型: {e}")

# ============================================================
# 4. 奖励函数分析
# ============================================================

print("\n" + "="*70)
print("奖励函数分析")
print("="*70)

print("""
当前奖励函数:
  reward = portfolio_return × 150
  reward += mean_ret / std_ret × 15  (风险调整)
  reward -= (dd - 0.15) × 30         (回撤惩罚)

问题：
  1. ❌ 回撤惩罚太强（-30），RL学会保守
  2. ❌ 风险调整（sharpe）可能误导
  3. ❌ 收益奖励（×150）可能不够

建议：
  1. ✅ 降低回撤惩罚：30 → 10
  2. ✅ 增加收益奖励：150 → 200
  3. ✅ 移除风险调整，让RL专注收益
""")

print("\n" + "="*70)
print("诊断完成")
print("="*70)
