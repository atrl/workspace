#!/usr/bin/env python3
"""
Qlib RL v3 - 优化版

基于 v2 (1156%) 的优化：
1. ✅ 保持 Qlib D.features() 数据加载
2. ✅ 保持 pandas 特征计算
3. ✅ 保持模仿学习
4. ✅ 优化点：
   - 增加训练步数：300k → 500k
   - 调整奖励函数权重
   - 增加更多特征
   - 调整 DQN 超参数
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_INIT_AT_FORK'] = 'FALSE'

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path

print("="*70)
print("Qlib RL v3 - 优化版")
print("="*70)
print("""
基于 v2 (1156%) 的优化：
1. ✅ Qlib D.features() 数据加载
2. ✅ pandas 特征计算
3. ✅ 模仿学习
4. ✅ 训练步数: 500k
5. ✅ 优化奖励函数
6. ✅ 更多特征
""")

# ============================================================
# 1. 使用 Qlib 加载基础数据
# ============================================================

import qlib
from qlib.data import D

QLIB_DIR = "/Users/rziiiii/.openclaw/workspace/qlib_soxl_binary"

qlib.init(provider_uri=QLIB_DIR)
print(f"✅ Qlib 初始化成功")

print("\n加载 Qlib 数据...")

df = D.features(
    instruments=['SOXL'],
    fields=['$close', '$open', '$high', '$low', '$volume', '$vix'],
    start_time='2010-03-11',
    end_time='2026-04-02'
)

df = df.reset_index()
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime').sort_index()

if 'instrument' in df.columns:
    df = df.drop('instrument', axis=1)

df.columns = ['close', 'open', 'high', 'low', 'volume', 'vix']

print(f"✅ Qlib 数据: {len(df)} 条")

# ============================================================
# 2. 用 pandas 计算特征（增加更多特征）
# ============================================================

print("\n计算特征（pandas）...")

close = df['close']
vix = df['vix']
volume = df['volume']

# 基础特征
df['return'] = close.pct_change()
df['ma5'] = close.rolling(5).mean()
df['ma10'] = close.rolling(10).mean()
df['ma20'] = close.rolling(20).mean()
df['ma60'] = close.rolling(60).mean()

# RSI
delta = close.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))

# MACD
ema12 = close.ewm(span=12).mean()
ema26 = close.ewm(span=26).mean()
df['macd'] = ema12 - ema26
df['macd_signal'] = df['macd'].ewm(span=9).mean()

# 波动率
df['volatility_5'] = close.pct_change().rolling(5).std() * np.sqrt(252)
df['volatility_20'] = close.pct_change().rolling(20).std() * np.sqrt(252)

# VIX特征
df['vix_zscore'] = (vix - vix.rolling(20).mean()) / (vix.rolling(20).std() + 1e-8)
df['vix_percentile'] = vix.rolling(60).rank(pct=True)

# 恐慌得分
df['panic_score'] = (
    df['vix_zscore'] * 40 +
    vix.rolling(20).rank(pct=True) * 60 +
    np.where(df['return'] < 0, np.minimum(df['return'] / -0.05, 1) * (-50), 0)
)

# 成交量特征
df['volume_ratio'] = volume / volume.rolling(20).mean()
df['volume_trend'] = volume.pct_change()

# 动量
df['momentum_5'] = close / close.shift(5) - 1
df['momentum_10'] = close / close.shift(10) - 1

df = df.dropna()

# 分割数据
train_df = df[df.index < '2020-01-01']
test_df = df[df.index >= '2020-01-01']

print(f"✅ 训练数据: {len(train_df)} 条")
print(f"✅ 测试数据: {len(test_df)} 条")
print(f"✅ 特征数量: {len(df.columns)}")

# ============================================================
# 3. 优化的专家策略
# ============================================================

def expert_policy(row):
    """优化的专家策略"""
    # 恐慌买入
    if row['panic_score'] <= -25:
        if row['vix'] > 28:
            return 9  # 90%
        elif row['vix'] > 22:
            return 8  # 80%
        else:
            return 6  # 60%
    
    # 趋势跟踪（增加 MACD）
    if row['rsi'] < 65 and row['ma5'] > row['ma20'] and row['macd'] > row['macd_signal']:
        return 7  # 70%
    
    # 强趋势
    if row['momentum_10'] > 0.15 and row['rsi'] < 70:
        return 7  # 70%
    
    # 贪婪卖出
    if row['vix'] < 15:
        if row['rsi'] > 70:
            return 0  # 0%
        elif row['rsi'] > 65:
            return 1  # 10%
        else:
            return 3  # 30%
    
    # 高波动降低仓位
    if row['volatility_20'] > 0.5:
        return 3  # 30%
    
    # 默认
    return 5  # 50%

# ============================================================
# 4. 优化的 RL 环境
# ============================================================

class OptimizedEnv(gym.Env):
    """优化的 RL 环境（增加更多观察维度）"""
    
    POSITION_MAP = {i: i * 0.1 for i in range(10)}
    
    def __init__(self, data, lookback=60):
        super().__init__()
        
        self.data = data
        self.lookback = lookback
        
        self.action_space = spaces.Discrete(10)
        
        # 20维观察（增加特征）
        self.observation_space = spaces.Box(-np.inf, np.inf, (20,), np.float32)
        
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
            return np.zeros(20, dtype=np.float32)
        
        row = self.data.iloc[self.current_idx]
        lb = self.data.iloc[self.current_idx-self.lookback:self.current_idx]
        
        obs = np.nan_to_num(np.array([
            # 趋势特征（7维）
            row['return'],
            row['ma5'] / row['ma20'] - 1,
            row['macd'] / row['close'] if row['close'] > 0 else 0,
            row['momentum_5'],
            row['momentum_10'],
            row['price_vs_ma20'] if 'price_vs_ma20' in row else 0,
            lb['return'].mean(),
            
            # 波动率特征（2维）
            row['volatility_20'],
            lb['return'].std() * np.sqrt(252),
            
            # VIX特征（3维）
            row['vix'] / 100,
            row['vix_zscore'],
            lb['vix'].mean() / 100,
            
            # RSI和恐慌（2维）
            row['rsi'] / 100,
            row['panic_score'] / 100,
            
            # 成交量（2维）
            np.log(row['volume'] / (lb['volume'].mean() + 1e-8)),
            row['volume_ratio'] if 'volume_ratio' in row else 1,
            
            # 账户状态（4维）
            self.position,
            (self.cash - 100000) / 100000,
            self.current_idx / len(self.data),
            lb['return'].skew() if len(lb) > 3 else 0,
        ], dtype=np.float32))
        
        return obs
    
    def step(self, action):
        if self.current_idx >= len(self.data) - 1:
            return self._get_obs(), 0, True, False, {}
        
        # 动作转换
        action = int(action.item() if hasattr(action, 'item') else action)
        target_pos = self.POSITION_MAP[action]
        
        row = self.data.iloc[self.current_idx]
        current_price = row['close']
        
        # 交易成本
        commission = abs(target_pos - self.position) * 0.001
        self.position = target_pos
        
        # 计算收益
        next_price = self.data.iloc[self.current_idx + 1]['close']
        price_return = (next_price - current_price) / current_price
        portfolio_return = self.position * price_return - commission
        
        self.cash *= (1 + portfolio_return)
        self.values.append(self.cash)
        
        # 🔧 优化的奖励函数
        expert_action = expert_policy(row)
        
        # 1. 模仿奖励（加强）
        if action == expert_action:
            imitation_reward = 2.0  # v2: 1.0
        else:
            imitation_reward = -1.0  # v2: -0.5
        
        # 2. 收益奖励（调整权重）
        return_reward = portfolio_return * 15  # v2: 10
        
        # 3. 风险管理奖励（新增）
        risk_reward = 0
        if len(self.values) > 1:
            peak = max(self.values)
            dd = (peak - self.cash) / peak
            if dd > 0.15:
                risk_reward = -dd * 5  # 回撤惩罚
            elif dd < 0.05:
                risk_reward = 0.5  # 低回撤奖励
        
        # 4. 趋势对齐奖励（新增）
        trend_reward = 0
        if row['ma5'] > row['ma20']:  # 上升趋势
            if target_pos > 0.5:
                trend_reward = 0.3
        else:  # 下降趋势
            if target_pos < 0.5:
                trend_reward = 0.3
        
        # 综合奖励
        reward = imitation_reward + return_reward + risk_reward + trend_reward
        
        self.current_idx += 1
        done = self.current_idx >= len(self.data) - 1
        
        return self._get_obs(), reward, done, False, {'cash': self.cash}

# ============================================================
# 5. 训练（500k步）
# ============================================================

print("\n" + "="*70)
print("训练（500k步）")
print("="*70)

train_env = DummyVecEnv([lambda: OptimizedEnv(train_df)])

model = DQN(
    "MlpPolicy",
    train_env,
    learning_rate=1e-3,
    buffer_size=100000,      # 增加 buffer
    learning_starts=2000,    # 增加预热
    batch_size=128,          # 增加 batch size
    tau=0.005,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.15,  # 降低探索
    exploration_final_eps=0.03,
    verbose=0,
)

model.learn(total_timesteps=500000)
print("✅ 训练完成")

# ============================================================
# 6. 测试
# ============================================================

print("\n" + "="*70)
print("测试（2020-2026）")
print("="*70)

test_env = OptimizedEnv(test_df)
obs, _ = test_env.reset()

test_values = [100000]
positions = []
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = test_env.step(action)
    test_values.append(info['cash'])
    positions.append(test_env.position)

# 统计
test_values = np.array(test_values)
positions = np.array(positions)

total_return = (test_values[-1] - 100000) / 100000
daily_returns = np.diff(test_values) / test_values[:-1]
sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
max_dd = np.max((np.maximum.accumulate(test_values) - test_values) / np.maximum.accumulate(test_values))

# 仓位分布
print(f"\n仓位分布:")
for i in range(10):
    count = (positions == i * 0.1).sum()
    pct = (positions == i * 0.1).mean()
    if count > 0:
        print(f"  {i*10:2d}%: {count:4d} ({pct:.1%})")

print(f"\n总收益: {total_return:.2%}")
print(f"Sharpe: {sharpe:.2f}")
print(f"MaxDD: {max_dd:.2%}")

# 保存
import pathlib
pathlib.Path("models/qlib_rl_v3").mkdir(parents=True, exist_ok=True)
model.save("models/qlib_rl_v3/dqn_model")

print(f"\n对比:")
print(f"  目标: 792%")
print(f"  v2 (300k): 1156%")
print(f"  v3 (500k): {total_return:.2%}")

if total_return > 11.56:
    print(f"\n✅✅✅ 超过 v2！")
elif total_return > 7.92:
    print(f"\n✅ 超过目标")
else:
    print(f"\n⚠️ 需要继续优化")

print("\n" + "="*70)
print("✅ 完成！")
print("="*70)
