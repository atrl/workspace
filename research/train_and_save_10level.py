#!/usr/bin/env python3
"""
完整训练并保存10档模型
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['KMP_INIT_AT_FORK'] = 'FALSE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import pathlib

print("="*70)
print("10档仓位完整训练")
print("="*70)

# 准备数据
df = pd.read_csv("data/soxl_soxx_vix_full.csv", parse_dates=['date'])
df = df.set_index('date')

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

print(f"训练数据: {len(train_df)}")

# 简化环境
class SimpleEnv(gym.Env):
    POSITION_MAP = {i: i * 0.1 for i in range(10)}
    
    def __init__(self, data, lookback=60):
        super().__init__()
        self.data = data
        self.lookback = lookback
        
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(-np.inf, np.inf, (17,), np.float32)
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.idx = self.lookback
        self.position = 0.0
        self.cash = 100000
        return self._obs(), {}
    
    def _obs(self):
        if self.idx >= len(self.data):
            return np.zeros(17, dtype=np.float32)
        
        row = self.data.iloc[self.idx]
        lb = self.data.iloc[self.idx-self.lookback:self.idx]
        
        return np.nan_to_num(np.array([
            row['return'],
            row['ma5'] / row['ma20'] - 1,
            lb['return'].mean(),
            lb['return'].std() * np.sqrt(252),
            row['vix'],
            row['vix_zscore'],
            lb['vix'].mean(),
            row['rsi'] / 100,
            row['panic_score'] / 100,
            np.log(row['soxl_volume'] / lb['soxl_volume'].mean()),
            self.position,
            (self.cash - 100000) / 100000,
            self.idx / len(self.data),
            lb['return'].skew() if len(lb) > 3 else 0,
            lb['return'].kurtosis() if len(lb) > 4 else 0,
            row['momentum_10'] if 'momentum_10' in row else 0,
            0,  # padding
        ], dtype=np.float32))
    
    def step(self, action):
        if self.idx >= len(self.data) - 1:
            return self._obs(), 0, True, False, {}
        
        action = int(action)
        target_pos = self.POSITION_MAP[action]
        
        row = self.data.iloc[self.idx]
        next_price = self.data.iloc[self.idx + 1]['soxl_close']
        current_price = row['soxl_close']
        
        commission = abs(target_pos - self.position) * 0.001
        self.position = target_pos
        
        ret = (next_price - current_price) / current_price
        port_ret = self.position * ret - commission
        
        self.cash *= (1 + port_ret)
        
        # 简单奖励
        reward = port_ret * 100
        if target_pos > 0.7 and row['panic_score'] < -25:
            reward += 1.0  # 恐慌时高仓位奖励
        elif target_pos < 0.2 and row['vix'] > 30:
            reward += 1.0  # 高VIX时低仓位奖励
        
        self.idx += 1
        done = self.idx >= len(self.data) - 1
        
        return self._obs(), reward, done, False, {'cash': self.cash}

# 训练
print("\n训练（200k步）...")
env = DummyVecEnv([lambda: SimpleEnv(train_df)])
model = DQN("MlpPolicy", env, learning_rate=1e-3, buffer_size=50000,
            learning_starts=1000, batch_size=64, verbose=0)
model.learn(total_timesteps=200000)

# 保存
pathlib.Path("models/qlib_rl_10level").mkdir(parents=True, exist_ok=True)
model.save("models/qlib_rl_10level/dqn_model")

print("✅ 训练完成")
print(f"保存: models/qlib_rl_10level/dqn_model.zip")
