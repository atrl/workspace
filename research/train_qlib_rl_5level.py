#!/usr/bin/env python3
"""
改进版：支持5档仓位

动作空间：
  0 = 空仓（0%）
  1 = 轻仓（20%）
  2 = 中仓（50%）
  3 = 重仓（80%）
  4 = 满仓（95%）
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['KMP_INIT_AT_FORK'] = 'FALSE'

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

print("="*70)
print("改进版：5档仓位")
print("="*70)
print("""
动作空间（5档）：
  0 = 空仓（0%）
  1 = 轻仓（20%）
  2 = 中仓（50%）
  3 = 重仓（80%）
  4 = 满仓（95%）
""")

# ============================================================
# 准备数据
# ============================================================

df = pd.read_csv("data/soxl_soxx_vix_full.csv", parse_dates=['date'])
df = df.set_index('date')

# 计算特征（省略...）
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

train_df = df[df.index < '2020-01-01']
test_df = df[df.index >= '2020-01-01']

print(f"训练: {len(train_df)}, 测试: {len(test_df)}")

# ============================================================
# 专家策略（贝叶斯优化）
# ============================================================

EXPERT_PARAMS = {
    'panic_threshold': -25,
    'vix_panic': 22,
    'vix_extreme': 26,
    'vix_greed': 15,
}

def expert_policy(row):
    """专家策略：返回5档仓位"""
    if row['panic_score'] <= EXPERT_PARAMS['panic_threshold']:
        if row['vix'] > EXPERT_PARAMS['vix_extreme']:
            return 4  # 满仓（95%）
        elif row['vix'] > EXPERT_PARAMS['vix_panic']:
            return 3  # 重仓（80%）
        else:
            return 2  # 中仓（50%）
    
    if row['rsi'] < 65 and row['ma5'] > row['ma20']:
        return 3  # 重仓（80%）
    
    if row['vix'] < EXPERT_PARAMS['vix_greed']:
        if row['rsi'] > 65:
            return 0  # 空仓（0%）
        else:
            return 1  # 轻仓（20%）
    
    return 2  # 中仓（50%）

# ============================================================
# 改进版环境
# ============================================================

class ImprovedEnv(gym.Env):
    """支持5档仓位"""
    
    # 仓位映射
    POSITION_MAP = {
        0: 0.0,   # 空仓
        1: 0.2,   # 轻仓
        2: 0.5,   # 中仓
        3: 0.8,   # 重仓
        4: 0.95,  # 满仓
    }
    
    def __init__(self, data, lookback=60):
        super().__init__()
        
        self.data = data
        self.lookback = lookback
        
        # 5个动作
        self.action_space = spaces.Discrete(5)
        
        # 17维观察
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
        )
        
        self.current_idx = lookback
        self.position = 0.0
        self.cash = 100000
        self.values = [100000]
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_idx = self.lookback
        self.position = 0.0
        self.cash = 100000
        self.values = [100000]
        return self._get_obs(), {}
    
    def _get_obs(self):
        if self.current_idx >= len(self.data):
            return np.zeros(17, dtype=np.float32)
        
        row = self.data.iloc[self.current_idx]
        lookback = self.data.iloc[self.current_idx-self.lookback:self.current_idx]
        
        obs = np.array([
            row['return'],
            row['ma5'] / row['ma20'] - 1,
            lookback['return'].mean(),
            lookback['return'].std() * np.sqrt(252),
            row['vix'],
            row['vix_zscore'],
            lookback['vix'].mean(),
            lookback['vix'].std(),
            row['rsi'] / 100,
            lookback['rsi'].mean() / 100,
            row['panic_score'] / 100,
            lookback['panic_score'].mean() / 100,
            np.log(row['soxl_volume'] / lookback['soxl_volume'].mean()),
            lookback['soxl_close'].pct_change().std() * np.sqrt(252),
            self.position,
            (self.cash - 100000) / 100000,
            self.current_idx / len(self.data),
        ], dtype=np.float32)
        
        return np.nan_to_num(obs)
    
    def step(self, action):
        if self.current_idx >= len(self.data) - 1:
            return self._get_obs(), 0, True, False, {}
        
        # 映射到实际仓位
        target_position = self.POSITION_MAP[action]
        
        row = self.data.iloc[self.current_idx]
        current_price = row['soxl_close']
        
        # 交易成本
        commission = abs(target_position - self.position) * 0.001
        self.position = target_position
        
        # 计算收益
        next_price = self.data.iloc[self.current_idx + 1]['soxl_close']
        price_return = (next_price - current_price) / current_price
        portfolio_return = self.position * price_return - commission
        
        self.cash *= (1 + portfolio_return)
        self.values.append(self.cash)
        
        # 奖励
        row = self.data.iloc[self.current_idx]
        expert_action = expert_policy(row)
        
        if action == expert_action:
            reward = 1.0
        else:
            reward = -0.5
        
        reward += portfolio_return * 10
        
        self.current_idx += 1
        done = self.current_idx >= len(self.data) - 1
        
        return self._get_obs(), reward, done, False, {'cash': self.cash, 'position': self.position}

# ============================================================
# 训练
# ============================================================

print("\n训练（200k步）...")
train_env = DummyVecEnv([lambda: ImprovedEnv(train_df)])
model = DQN("MlpPolicy", train_env, learning_rate=1e-3, buffer_size=50000,
            learning_starts=1000, batch_size=64, verbose=0)
model.learn(total_timesteps=200000)
print("✅ 训练完成")

# ============================================================
# 测试
# ============================================================

print("\n测试（2020-2026）...")
test_env = ImprovedEnv(test_df)
obs, _ = test_env.reset()

test_values = [100000]
actions_taken = []
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    actions_taken.append(action)
    obs, reward, done, truncated, info = test_env.step(action)
    test_values.append(info['cash'])

# 统计
test_values = np.array(test_values)
total_return = (test_values[-1] - 100000) / 100000
daily_returns = np.diff(test_values) / test_values[:-1]
sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
max_dd = np.max((np.maximum.accumulate(test_values) - test_values) / np.maximum.accumulate(test_values))

actions_taken = np.array(actions_taken)
print(f"\n仓位分布:")
for i, pos in ImprovedEnv.POSITION_MAP.items():
    count = (actions_taken == i).sum()
    pct = (actions_taken == i).mean()
    print(f"  {i} ({pos:.0%}): {count} ({pct:.1%})")

print(f"\n总收益: {total_return:.2%}")
print(f"Sharpe: {sharpe:.2f}")
print(f"MaxDD: {max_dd:.2%}")

# 保存
import pathlib
pathlib.Path("models/qlib_rl_5level").mkdir(parents=True, exist_ok=True)
model.save("models/qlib_rl_5level/dqn_model")

print(f"\n对比:")
print(f"  目标: 792%")
print(f"  旧版（2档）: 806%")
print(f"  新版（5档）: {total_return:.2%}")

if total_return > 8.06:
    print(f"✅ 改进！ {total_return:.2%} > 806%")
