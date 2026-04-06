#!/usr/bin/env python3
"""
RL学习规则参数

思路：
  - 规则策略已证明有效（990%）
  - RL不学仓位，学参数
  - 动作空间：[panic_threshold, vix_panic, vix_extreme, vix_greed]

这是超参数优化，用RL来做。
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

print("="*70)
print("RL 学习规则参数")
print("="*70)
print("""
架构：
  RL智能体 → 参数 [panic_threshold, vix_panic, vix_extreme, vix_greed]
           ↓
  规则策略 → 信号 (0-1)
           ↓
  固定仓位乘数 0.8 → 最终仓位

优势：
  ✅ 规则策略已证明有效（990%）
  ✅ RL只学参数，空间更小
  ✅ 更可控、更可解释
""")

# ============================================================
# 1. 准备数据
# ============================================================

print("\n准备数据...")

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
    np.where(df['return'] < 0, np.minimum(df['return'] / -0.05, 1) * (-50), 0)
)

df = df.dropna()

# 分割数据
train_df = df[df.index < '2020-01-01']
test_df = df[df.index >= '2020-01-01']

print(f"训练数据: {len(train_df)} records")
print(f"测试数据: {len(test_df)} records")

# ============================================================
# 2. 规则策略（固定）
# ============================================================

def evaluate_rule_params(data, panic_threshold, vix_panic, vix_extreme, vix_greed, verbose=False):
    """
    用给定参数回测规则策略
    
    返回：total_return, sharpe, max_dd
    """
    cash = 100000
    position = 0.0
    values = [cash]
    
    for i in range(60, len(data)):
        row = data.iloc[i]
        
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
        
        # 目标仓位（固定乘数0.8）
        target_position = signal * 0.8
        
        if i < len(data) - 1:
            price_return = (data.iloc[i+1]['soxl_close'] - row['soxl_close']) / row['soxl_close']
            portfolio_return = target_position * price_return - abs(target_position - position) * 0.001
            cash *= (1 + portfolio_return)
            position = target_position
            values.append(cash)
    
    # 计算指标
    total_return = (cash - 100000) / 100000
    daily_returns = np.diff(values) / values[:-1]
    sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
    max_dd = np.max((np.maximum.accumulate(values) - values) / np.maximum.accumulate(values))
    
    if verbose:
        print(f"  Return: {total_return:.2%}, Sharpe: {sharpe:.2f}, MaxDD: {max_dd:.2%}")
    
    return total_return, sharpe, max_dd

# ============================================================
# 3. RL 环境：学习参数
# ============================================================

class RuleParamEnv(gym.Env):
    """
    RL环境：学习规则参数
    
    状态：市场统计（VIX分布、收益分布等）
    动作：[panic_threshold, vix_panic, vix_extreme, vix_greed]
    奖励：策略收益
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, data, window=252):
        super().__init__()
        
        self.data = data
        self.window = window
        self.current_idx = window
        
        # 动作空间：4个参数
        # panic_threshold: [-50, -20]
        # vix_panic: [15, 25]
        # vix_extreme: [25, 35]
        # vix_greed: [10, 15]
        self.action_space = spaces.Box(
            low=np.array([-50, 15, 25, 10]),
            high=np.array([-20, 25, 35, 15]),
            dtype=np.float32
        )
        
        # 观察空间：市场统计
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32
        )
        
        self.best_return = -np.inf
        self.best_params = None
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self.current_idx = self.window
        self.best_return = -np.inf
        self.best_params = None
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """获取市场统计"""
        window_data = self.data.iloc[self.current_idx-self.window:self.current_idx]
        
        obs = np.array([
            window_data['vix'].mean(),
            window_data['vix'].std(),
            window_data['return'].mean(),
            window_data['return'].std(),
            window_data['panic_score'].mean(),
            window_data['panic_score'].std(),
            window_data['rsi'].mean(),
            window_data['soxl_close'].pct_change().std() * np.sqrt(252),
            self.current_idx / len(self.data),
            len(self.data) / 5000,
        ], dtype=np.float32)
        
        return np.nan_to_num(obs)
    
    def step(self, action):
        """
        执行动作
        
        Args:
            action: [panic_threshold, vix_panic, vix_extreme, vix_greed]
        """
        # 解析参数
        panic_threshold = float(action[0])
        vix_panic = float(action[1])
        vix_extreme = float(action[2])
        vix_greed = float(action[3])
        
        # 在当前窗口回测
        window_data = self.data.iloc[self.current_idx:self.current_idx+63]  # 3个月
        
        if len(window_data) < 63:
            return self._get_obs(), 0, True, False, {}
        
        total_return, sharpe, max_dd = evaluate_rule_params(
            window_data,
            panic_threshold, vix_panic, vix_extreme, vix_greed
        )
        
        # 奖励 = 收益 - 回撤惩罚
        reward = total_return * 100 - max_dd * 50
        
        # 更新最佳
        if total_return > self.best_return:
            self.best_return = total_return
            self.best_params = action.copy()
        
        # 前进
        self.current_idx += 63
        
        done = self.current_idx >= len(self.data) - 63
        
        return self._get_obs(), reward, done, False, {
            'return': total_return,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'params': action.copy(),
        }


# ============================================================
# 4. 训练
# ============================================================

print("\n" + "="*70)
print("训练 RL 学习参数")
print("="*70)

train_env = DummyVecEnv([lambda: RuleParamEnv(train_df)])

# 使用 SAC（连续动作空间更擅长）
model = SAC(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=256,
    verbose=0,
)

print("训练（100k步，快速测试）...")
model.learn(total_timesteps=100000)
print("✅ 训练完成")

# ============================================================
# 5. 获取学习到的参数
# ============================================================

print("\n" + "="*70)
print("RL 学习到的参数")
print("="*70)

# 用学到的参数在训练集测试
obs, _ = train_env.envs[0].reset()
best_params = None
best_return = -np.inf

for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = train_env.step([action])
    
    if info[0].get('return', 0) > best_return:
        best_return = info[0]['return']
        best_params = info[0]['params']
    
    if done[0]:
        break

if best_params is not None:
    print(f"\n最优参数（RL学习）:")
    print(f"  panic_threshold: {best_params[0]:.1f}")
    print(f"  vix_panic:       {best_params[1]:.1f}")
    print(f"  vix_extreme:     {best_params[2]:.1f}")
    print(f"  vix_greed:       {best_params[3]:.1f}")
    
    # ============================================================
    # 6. 测试
    # ============================================================
    
    print("\n" + "="*70)
    print("测试（2020-2026）")
    print("="*70)
    
    total_return, sharpe, max_dd = evaluate_rule_params(
        test_df,
        best_params[0], best_params[1], best_params[2], best_params[3],
        verbose=True
    )
    
    print(f"\n总收益:      {total_return:.2%}")
    print(f"Sharpe:      {sharpe:.2f}")
    print(f"最大回撤:    {max_dd:.2%}")
    
    # ============================================================
    # 7. 对比
    # ============================================================
    
    print("\n" + "="*70)
    print("对比")
    print("="*70)
    print(f"目标（Pandas）: 792%, Sharpe 0.88")
    print(f"网格搜索（规则）: 990%, Sharpe 1.02")
    print(f"RL学习参数:      {total_return:.2%}, Sharpe {sharpe:.2f}")
    
    if total_return > 7.92:
        print(f"\n✅ 超过目标！ {total_return:.2%} > 792%")
        if total_return > 9.90:
            print(f"✅ 超过网格搜索！ {total_return:.2%} > 990%")
    else:
        print(f"\n⚠️ 未达目标 {total_return:.2%} < 792%")
    
    # ============================================================
    # 8. 保存
    # ============================================================
    
    import pathlib
    pathlib.Path("models/qlib_rl_params").mkdir(parents=True, exist_ok=True)
    model.save("models/qlib_rl_params/sac_model")
    
    import json
    with open("models/qlib_rl_params/best_params.json", 'w') as f:
        json.dump({
            'panic_threshold': float(best_params[0]),
            'vix_panic': float(best_params[1]),
            'vix_extreme': float(best_params[2]),
            'vix_greed': float(best_params[3]),
            'return': float(total_return),
            'sharpe': float(sharpe),
            'max_dd': float(max_dd),
        }, f, indent=2)
    
    print("\n模型保存: models/qlib_rl_params/")
    print("参数保存: models/qlib_rl_params/best_params.json")

print("\n" + "="*70)
print("✅ 完成！")
print("="*70)
print("""
架构：
  RL智能体 → 参数
           ↓
  规则策略 → 信号
           ↓
  固定仓位 0.8 → 最终仓位

这是真正的 RL + Qlib！
""")
