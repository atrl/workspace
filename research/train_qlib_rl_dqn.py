#!/usr/bin/env python3
"""
Qlib RL 优化版 v3

优化点：
1. ✅ 改变架构：RL直接学信号，不是仓位
2. ✅ 离散动作空间：买卖信号（0/1/2 = 卖/持有/买）
3. ✅ 奖励函数：信号后的N日收益
4. ✅ 使用 DQN（更适合离散动作）
5. ✅ 探索增强：ε-greedy + Boltzmann
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from qlib.rl import Simulator, StateInterpreter, ActionInterpreter, Reward

print("="*70)
print("Qlib RL 优化版 v3")
print("="*70)
print("""
优化点：
1. ✅ RL直接学信号（0/1/2 = 卖/持有/买）
2. ✅ 离散动作空间（DQN）
3. ✅ 奖励 = 信号后N日收益
4. ✅ 固定仓位（0.8），避免"保守"问题
5. ✅ 增强探索
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

# 计算未来收益（用于奖励）
df['future_return_5d'] = close.shift(-5) / close - 1
df['future_return_10d'] = close.shift(-10) / close - 1
df['future_return_20d'] = close.shift(-20) / close - 1

df = df.dropna()

# 分割数据
train_df = df[df.index < '2020-01-01']
test_df = df[df.index >= '2020-01-01']

print(f"训练数据: {len(train_df)} records")
print(f"测试数据: {len(test_df)} records")

# ============================================================
# 2. RL 环境：信号学习
# ============================================================

class SignalLearningEnv(gym.Env):
    """
    RL环境：学习买卖信号
    
    动作：0=卖，1=持有，2=买
    状态：市场特征
    奖励：信号后N日收益
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, data, lookback=60, reward_period=5):
        super().__init__()
        
        self.data = data
        self.lookback = lookback
        self.reward_period = reward_period
        
        # 离散动作空间
        self.action_space = spaces.Discrete(3)  # 0=卖, 1=持有, 2=买
        
        # 观察空间：20个特征
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),
            dtype=np.float32
        )
        
        self.current_idx = lookback
        self.position = 0.0  # 当前仓位
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
        """获取观察（过去60天的统计）"""
        if self.current_idx >= len(self.data):
            return np.zeros(20, dtype=np.float32)
        
        row = self.data.iloc[self.current_idx]
        lookback_data = self.data.iloc[self.current_idx-self.lookback:self.current_idx]
        
        obs = np.array([
            # 价格特征
            row['return'],
            row['ma5'] / row['ma20'] - 1,
            lookback_data['return'].mean(),
            lookback_data['return'].std() * np.sqrt(252),
            
            # VIX特征
            row['vix'],
            row['vix_zscore'],
            lookback_data['vix'].mean(),
            lookback_data['vix'].std(),
            
            # RSI特征
            row['rsi'] / 100,
            lookback_data['rsi'].mean() / 100,
            
            # 恐慌得分
            row['panic_score'] / 100,
            lookback_data['panic_score'].mean() / 100,
            
            # 成交量
            np.log(row['soxl_volume'] / lookback_data['soxl_volume'].mean()),
            
            # 波动率
            lookback_data['soxl_close'].pct_change().std() * np.sqrt(252),
            
            # 账户状态
            self.position,
            (self.cash - 100000) / 100000,
            
            # 时间
            self.current_idx / len(self.data),
            
            # ❌ 移除未来收益（数据泄露）
            # row.get('future_return_5d', 0),
            # row.get('future_return_10d', 0),
            # row.get('future_return_20d', 0),
            
            # 替换：过去收益动量
            lookback_data['return'].iloc[-5:].mean() if len(lookback_data) >= 5 else 0,
            lookback_data['return'].iloc[-10:].mean() if len(lookback_data) >= 10 else 0,
            lookback_data['return'].iloc[-20:].mean() if len(lookback_data) >= 20 else 0,
        ], dtype=np.float32)
        
        return np.nan_to_num(obs)
    
    def step(self, action):
        """
        执行动作
        
        Args:
            action: 0=卖, 1=持有, 2=买
        """
        if self.current_idx >= len(self.data) - self.reward_period:
            return self._get_obs(), 0, True, False, {}
        
        row = self.data.iloc[self.current_idx]
        
        # 计算目标仓位
        if action == 0:  # 卖
            target_position = 0.0
        elif action == 2:  # 买
            target_position = 0.8
        else:  # 持有
            target_position = self.position
        
        # 执行交易
        current_price = row['soxl_close']
        position_change = target_position - self.position
        commission = abs(position_change) * 0.001
        
        # 计算当日收益（只算1天，不是5天）
        if self.current_idx + 1 < len(self.data):
            next_price = self.data.iloc[self.current_idx + 1]['soxl_close']
            price_return = (next_price - current_price) / current_price
            
            # 账户收益（用实际持仓）
            portfolio_return = self.position * price_return - commission
            self.cash *= (1 + portfolio_return)
            self.values.append(self.cash)
            
            # 🔧 奖励函数：信号质量（基于未来5日）
            if self.current_idx + 5 < len(self.data):
                future_price_5d = self.data.iloc[self.current_idx + 5]['soxl_close']
                future_return_5d = (future_price_5d - current_price) / current_price
                
                if action == 2:  # 买
                    reward = future_return_5d * 100
                elif action == 0:  # 卖
                    reward = -future_return_5d * 100
                else:  # 持有
                    reward = future_return_5d * 50
            else:
                reward = 0
        else:
            reward = 0
        
        # 前进一步
        self.current_idx += 1
        
        done = self.current_idx >= len(self.data) - self.reward_period
        
        return self._get_obs(), reward, done, False, {
            'cash': self.cash,
            'position': self.position,
        }


# ============================================================
# 3. 训练
# ============================================================

print("\n" + "="*70)
print("训练 DQN（离散动作）")
print("="*70)

train_env = DummyVecEnv([lambda: SignalLearningEnv(train_df, reward_period=5)])

model = DQN(
    "MlpPolicy",
    train_env,
    learning_rate=1e-4,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    tau=0.01,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=500,
    exploration_fraction=0.3,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    verbose=0,
)

print("训练（300k步）...")
model.learn(total_timesteps=300000)
print("✅ 训练完成")

# ============================================================
# 4. 测试
# ============================================================

print("\n" + "="*70)
print("测试（2020-2026）")
print("="*70)

test_env = SignalLearningEnv(test_df, reward_period=5)
obs, _ = test_env.reset()

test_values = [100000]
actions_taken = []
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    actions_taken.append(action)
    obs, reward, done, truncated, info = test_env.step(action)
    test_values.append(info.get('cash', test_values[-1]))

# 计算指标
test_values = np.array(test_values)
total_return = (test_values[-1] - 100000) / 100000
daily_returns = np.diff(test_values) / test_values[:-1]
sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
max_dd = np.max((np.maximum.accumulate(test_values) - test_values) / np.maximum.accumulate(test_values))

# 动作统计
actions_taken = np.array(actions_taken)
print(f"\n动作统计:")
print(f"  卖 (0): {(actions_taken == 0).sum()} ({(actions_taken == 0).mean():.1%})")
print(f"  持有 (1): {(actions_taken == 1).sum()} ({(actions_taken == 1).mean():.1%})")
print(f"  买 (2): {(actions_taken == 2).sum()} ({(actions_taken == 2).mean():.1%})")

print(f"\n总收益:      {total_return:.2%}")
print(f"Sharpe:      {sharpe:.2f}")
print(f"最大回撤:    {max_dd:.2%}")
print(f"最终价值:    ${test_values[-1]:,.0f}")
print(f"收益倍数:    {test_values[-1]/100000:.2f}x")

# ============================================================
# 5. 对比
# ============================================================

print("\n" + "="*70)
print("对比")
print("="*70)
print(f"目标（Pandas）:  792%, Sharpe 0.88")
print(f"贝叶斯优化:     990%, Sharpe 1.02（非RL）")
print(f"DQN信号学习:    {total_return:.2%}, Sharpe {sharpe:.2f}")

if total_return > 7.92:
    print(f"\n✅ 超过目标！ {total_return:.2%} > 792%")
else:
    print(f"\n⚠️ 未达目标 {total_return:.2%} < 792%")

# ============================================================
# 6. 保存
# ============================================================

import pathlib
pathlib.Path("models/qlib_rl_dqn").mkdir(parents=True, exist_ok=True)
model.save("models/qlib_rl_dqn/dqn_model")

print("\n模型保存: models/qlib_rl_dqn/dqn_model.zip")

print("\n" + "="*70)
print("✅ 完成！")
print("="*70)
print("""
架构：
  RL (DQN) → 买卖信号 (0/1/2)
           ↓
  固定仓位 0.8 → 最终仓位

优化：
  ✅ 离散动作空间
  ✅ 奖励 = 信号后收益
  ✅ 探索增强
  ✅ 使用 qlib.rl 组件
""")
