#!/usr/bin/env python3
"""
增强版：10档仓位 + 完整状态空间

动作空间：10档
  0=0%, 1=10%, 2=20%, ..., 9=90%

状态空间：30维
  - 价格特征（5维）
  - 收益统计（4维）
  - 波动率（3维）
  - VIX（4维）
  - 动量（3维）
  - 情绪（3维）
  - 成交量（2维）
  - 账户（4维）
  - 时间（2维）
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['KMP_INIT_AT_FORK'] = 'FALSE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

print("="*70)
print("增强版：10档仓位 + 30维状态空间")
print("="*70)
print("""
动作空间（10档）：
  0=0%, 1=10%, 2=20%, 3=30%, 4=40%, 
  5=50%, 6=60%, 7=70%, 8=80%, 9=90%

状态空间（30维）：
  ✅ 价格特征（5维）
  ✅ 收益统计（4维）
  ✅ 波动率（3维）
  ✅ VIX（4维）
  ✅ 动量（3维）
  ✅ 情绪（3维）
  ✅ 成交量（2维）
  ✅ 账户（4维）
  ✅ 时间（2维）
""")

# ============================================================
# 准备数据
# ============================================================

print("准备数据...")

df = pd.read_csv("data/soxl_soxx_vix_full.csv", parse_dates=['date'])
df = df.set_index('date')

# 计算特征
close = df['soxl_close']
high = df['soxl_high']
low = df['soxl_low']
volume = df['soxl_volume']
vix = df['vix']

# 基础特征
df['return'] = close.pct_change()
df['ma5'] = close.rolling(5).mean()
df['ma20'] = close.rolling(20).mean()
df['ma60'] = close.rolling(60).mean()

# RSI
delta = close.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))

# ATR
df['tr'] = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
df['atr'] = df['tr'].rolling(14).mean()

# 布林带
df['bb_mid'] = close.rolling(20).mean()
df['bb_std'] = close.rolling(20).std()
df['bb_width'] = 2 * df['bb_std'] / df['bb_mid']

# MACD
exp1 = close.ewm(span=12).mean()
exp2 = close.ewm(span=26).mean()
df['macd'] = exp1 - exp2

# 动量
df['momentum_10'] = close / close.shift(10) - 1

# VIX特征
df['vix_zscore'] = (vix - vix.rolling(20).mean()) / (vix.rolling(20).std() + 1e-8)
df['vix_percentile'] = vix.rolling(252).rank(pct=True)
df['vix_regime'] = pd.cut(vix, bins=[0, 15, 25, 100], labels=[0, 1, 2]).astype(float)

# 恐慌得分
df['panic_score'] = (
    df['vix_zscore'] * 40 +
    vix.rolling(20).rank(pct=True) * 60 +
    np.where(df['return'] < 0, np.minimum(df['return'] / -0.05, 1) * (-50), 0)
)

# 市场压力
df['market_stress'] = (
    (vix > 25).astype(float) * 0.5 +
    (df['return'] < -0.05).astype(float) * 0.3 +
    (df['bb_width'] > df['bb_width'].rolling(60).quantile(0.8)).astype(float) * 0.2
)

# 成交量趋势
df['volume_trend'] = volume.rolling(5).mean() / volume.rolling(20).mean() - 1

# 星期几
df['day_of_week'] = df.index.dayofweek

df = df.dropna()

train_df = df[df.index < '2020-01-01']
test_df = df[df.index >= '2020-01-01']

print(f"训练: {len(train_df)}, 测试: {len(test_df)}")

# ============================================================
# 专家策略
# ============================================================

def expert_policy(row):
    """
    专家策略：返回10档仓位
    
    基于贝叶斯优化的参数，扩展到10档
    """
    # 恐慌买入
    if row['panic_score'] <= -25:
        if row['vix'] > 28:
            return 9  # 90%满仓
        elif row['vix'] > 22:
            return 8  # 80%重仓
        else:
            return 6  # 60%中高
    
    # 趋势跟踪
    if row['rsi'] < 65 and row['ma5'] > row['ma20']:
        if row['momentum_10'] > 0.1:
            return 7  # 70%
        else:
            return 6  # 60%
    
    # 贪婪卖出
    if row['vix'] < 15:
        if row['rsi'] > 70:
            return 0  # 0%空仓
        elif row['rsi'] > 65:
            return 1  # 10%轻仓
        else:
            return 3  # 30%
    
    # 波动率调整（简化）
    if row['vix'] > 30:
        return 2  # 20%低仓位
    elif row['vix'] > 25:
        return 4  # 40%
    else:
        return 5  # 50%中性

# ============================================================
# 增强版环境
# ============================================================

class EnhancedEnv(gym.Env):
    """10档仓位 + 30维状态空间"""
    
    # 10档仓位映射
    POSITION_MAP = {i: i * 0.1 for i in range(10)}
    
    def __init__(self, data, lookback=60):
        super().__init__()
        
        self.data = data
        self.lookback = lookback
        
        # 10个动作
        self.action_space = spaces.Discrete(10)
        
        # 30维观察空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32
        )
        
        self.current_idx = lookback
        self.position = 0.0
        self.cash = 100000
        self.values = [100000]
        self.num_days_held = 0
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_idx = self.lookback
        self.position = 0.0
        self.cash = 100000
        self.values = [100000]
        self.num_days_held = 0
        return self._get_obs(), {}
    
    def _get_obs(self):
        """30维状态空间"""
        if self.current_idx >= len(self.data):
            return np.zeros(30, dtype=np.float32)
        
        row = self.data.iloc[self.current_idx]
        lookback = self.data.iloc[self.current_idx-self.lookback:self.current_idx]
        
        # 1. 价格特征（5维）
        price_features = [
            row['return'],
            row['soxl_close'] / row['ma5'] - 1,
            row['soxl_close'] / row['ma20'] - 1,
            row['soxl_close'] / row['ma60'] - 1 if row['ma60'] > 0 else 0,
            row['soxl_high'] / row['soxl_low'] - 1,
        ]
        
        # 2. 收益率统计（4维）
        return_features = [
            lookback['return'].mean(),
            lookback['return'].std(),
            lookback['return'].skew() if len(lookback) > 3 else 0,
            lookback['return'].kurtosis() if len(lookback) > 4 else 0,
        ]
        
        # 3. 波动率特征（3维）
        vol_features = [
            lookback['return'].std() * np.sqrt(252),
            row['atr'] / row['soxl_close'] if row['atr'] > 0 else 0,
            row['bb_width'],
        ]
        
        # 4. VIX特征（4维）
        vix_features = [
            row['vix'] / 100,
            row['vix_zscore'],
            lookback['vix'].mean() / 100,
            row['vix_percentile'],
        ]
        
        # 5. 动量特征（3维）
        momentum_features = [
            row['rsi'] / 100,
            row['macd'] / row['soxl_close'] if row['macd'] != 0 else 0,
            row['momentum_10'],
        ]
        
        # 6. 情绪特征（3维）
        sentiment_features = [
            row['panic_score'] / 100,
            row['vix_regime'] / 2,
            row['market_stress'],
        ]
        
        # 7. 成交量特征（2维）
        volume_features = [
            np.log(row['soxl_volume'] / lookback['soxl_volume'].mean()),
            row['volume_trend'],
        ]
        
        # 8. 账户状态（4维）
        account_features = [
            self.position,
            (self.cash - 100000) / 100000,
            self.position * (row['soxl_close'] - lookback['soxl_close'].iloc[0]) / lookback['soxl_close'].iloc[0] if self.position > 0 else 0,
            self.num_days_held / 20,
        ]
        
        # 9. 时间特征（2维）
        time_features = [
            self.current_idx / len(self.data),
            row['day_of_week'] / 4,
        ]
        
        obs = np.concatenate([
            price_features,       # 5
            return_features,      # 4
            vol_features,         # 3
            vix_features,         # 4
            momentum_features,    # 3
            sentiment_features,   # 3
            volume_features,      # 2
            account_features,     # 4
            time_features,        # 2
        ])
        
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
    
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
        
        # 更新持仓天数
        if self.position > 0:
            self.num_days_held += 1
        else:
            self.num_days_held = 0
        
        # 奖励：模仿 + 收益
        expert_action = expert_policy(row)
        
        if action == expert_action:
            reward = 1.0
        else:
            reward = -0.5
        
        reward += portfolio_return * 10
        
        self.current_idx += 1
        done = self.current_idx >= len(self.data) - 1
        
        return self._get_obs(), reward, done, False, {
            'cash': self.cash,
            'position': self.position,
        }

# ============================================================
# 训练
# ============================================================

print("\n训练（300k步）...")
train_env = DummyVecEnv([lambda: EnhancedEnv(train_df)])
model = DQN("MlpPolicy", train_env, learning_rate=1e-3, buffer_size=100000,
            learning_starts=1000, batch_size=64, verbose=0)
model.learn(total_timesteps=300000)
print("✅ 训练完成")

# ============================================================
# 测试
# ============================================================

print("\n测试（2020-2026）...")
test_env = EnhancedEnv(test_df)
obs, _ = test_env.reset()

test_values = [100000]
actions_taken = []
positions = [0.0]
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)  # 转换为int
    actions_taken.append(action)
    obs, reward, done, truncated, info = test_env.step(action)
    test_values.append(info['cash'])
    positions.append(info['position'])

# 统计
test_values = np.array(test_values)
total_return = (test_values[-1] - 100000) / 100000
daily_returns = np.diff(test_values) / test_values[:-1]
sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
max_dd = np.max((np.maximum.accumulate(test_values) - test_values) / np.maximum.accumulate(test_values))

actions_taken = np.array(actions_taken)
print(f"\n仓位分布:")
for i in range(10):
    count = (actions_taken == i).sum()
    pct = (actions_taken == i).mean()
    pos = i * 10
    print(f"  {i} ({pos:2d}%): {count:4d} ({pct:.1%})")

print(f"\n总收益: {total_return:.2%}")
print(f"Sharpe: {sharpe:.2f}")
print(f"MaxDD: {max_dd:.2%}")

# 保存
import pathlib
pathlib.Path("models/qlib_rl_10level").mkdir(parents=True, exist_ok=True)
model.save("models/qlib_rl_10level/dqn_model")

print(f"\n对比:")
print(f"  目标: 792%")
print(f"  旧版（2档）: 806%")
print(f"  新版（10档）: {total_return:.2%}")

if total_return > 8.06:
    print(f"\n✅ 改进！ {total_return:.2%} > 806%")
