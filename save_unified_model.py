#!/usr/bin/env python3
"""
保存统一数据源训练的模型
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path

print("保存统一数据源训练的模型...")

# 加载数据
csv_path = '/Users/rziiiii/.openclaw/workspace/data/soxl_soxx_vix_full.csv'
df = pd.read_csv(csv_path, parse_dates=['date'])
df = df.set_index('date')

data = df[['soxl_close', 'soxl_volume', 'vix']].copy()
data.columns = ['close', 'volume', 'vix']

# 特征工程
close = data['close']
vix = data['vix']

data['return'] = close.pct_change()
data['ma5'] = close.rolling(5).mean()
data['ma20'] = close.rolling(20).mean()

delta = close.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
data['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))

data['vix_zscore'] = (vix - vix.rolling(20).mean()) / (vix.rolling(20).std() + 1e-8)

data['panic_score'] = (
    data['vix_zscore'] * 40 +
    vix.rolling(20).rank(pct=True) * 60 +
    np.where(data['return'] < 0, np.minimum(data['return'] / -0.05, 1) * (-50), 0)
)

data = data.dropna()
train_data = data[data.index < '2020-01-01']

# 创建环境
class TradingEnv(gym.Env):
    POSITION_MAP = {i: i * 0.1 for i in range(10)}
    
    def __init__(self, data, lookback=60):
        super().__init__()
        self.data = data
        self.lookback = lookback
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(-np.inf, np.inf, (17,), np.float32)
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
        lb = self.data.iloc[self.current_idx-self.lookback:self.current_idx]
        
        obs = np.nan_to_num(np.array([
            row['return'],
            row['ma5'] / row['ma20'] - 1,
            lb['return'].mean(),
            lb['return'].std() * np.sqrt(252),
            row['vix'],
            row['vix_zscore'],
            lb['vix'].mean(),
            row['rsi'] / 100,
            row['panic_score'] / 100,
            np.log(row['volume'] / (lb['volume'].mean() + 1e-8)),
            self.position,
            (self.cash - 100000) / 100000,
            self.current_idx / len(self.data),
            lb['return'].skew() if len(lb) > 3 else 0,
            lb['return'].kurtosis() if len(lb) > 4 else 0,
            0,
            0,
        ], dtype=np.float32))
        
        return obs
    
    def step(self, action):
        if self.current_idx >= len(self.data) - 1:
            return self._get_obs(), 0, True, False, {}
        
        action = int(action.item() if hasattr(action, 'item') else action)
        target_pos = self.POSITION_MAP[action]
        
        row = self.data.iloc[self.current_idx]
        current_price = row['close']
        
        commission = abs(target_pos - self.position) * 0.001
        self.position = target_pos
        
        next_price = self.data.iloc[self.current_idx + 1]['close']
        price_return = (next_price - current_price) / current_price
        portfolio_return = self.position * price_return - commission
        
        self.cash *= (1 + portfolio_return)
        self.values.append(self.cash)
        
        expert_action = self._expert_policy(row)
        
        if action == expert_action:
            imitation_reward = 1.0
        else:
            imitation_reward = -0.5
        
        return_reward = portfolio_return * 10
        reward = imitation_reward + return_reward
        
        self.current_idx += 1
        done = self.current_idx >= len(self.data) - 1
        
        return self._get_obs(), reward, done, False, {'cash': self.cash}
    
    def _expert_policy(self, row):
        if row['panic_score'] <= -25:
            if row['vix'] > 28:
                return 9
            elif row['vix'] > 22:
                return 8
            else:
                return 6
        
        if row['rsi'] < 65 and row['ma5'] > row['ma20']:
            return 6
        
        if row['vix'] < 15:
            if row['rsi'] > 70:
                return 0
            elif row['rsi'] > 65:
                return 1
            else:
                return 3
        
        return 5

# 训练
print("训练 DQN（300k步）...")
train_env = DummyVecEnv([lambda: TradingEnv(train_data)])

model = DQN(
    "MlpPolicy",
    train_env,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    verbose=0,
)

model.learn(total_timesteps=300000)

# 保存
Path("models/unified_drl").mkdir(parents=True, exist_ok=True)
model.save("models/unified_drl/dqn_model")

print(f"✅ 模型已保存: models/unified_drl/dqn_model.zip")
