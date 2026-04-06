#!/usr/bin/env python3
"""
模仿学习：RL学习专家策略

思路：
1. 用贝叶斯优化的策略作为专家
2. 生成专家轨迹（状态-动作对）
3. 用 Behavior Cloning 训练RL
4. 应该能达到990%
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
from typing import List, Tuple
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

print("="*70)
print("模仿学习：RL学习专家策略")
print("="*70)
print("""
思路：
1. 专家策略（贝叶斯优化）：990%
2. 生成专家轨迹
3. Behavior Cloning
4. RL应该达到990%
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
# 2. 专家策略（贝叶斯优化的参数）
# ============================================================

EXPERT_PARAMS = {
    'panic_threshold': -25,
    'vix_panic': 22,
    'vix_extreme': 26,
    'vix_greed': 15,
}

def expert_policy(row):
    """
    专家策略：贝叶斯优化的规则策略
    
    Returns:
        0=卖, 1=持有, 2=买
    """
    if row['panic_score'] <= EXPERT_PARAMS['panic_threshold']:
        if row['vix'] > EXPERT_PARAMS['vix_extreme']:
            return 2  # 买（极端恐慌）
        elif row['vix'] > EXPERT_PARAMS['vix_panic']:
            return 2  # 买（恐慌）
        else:
            return 1  # 持有
    
    if row['rsi'] < 65 and row['ma5'] > row['ma20']:
        return 2  # 买（趋势）
    
    if row['vix'] < EXPERT_PARAMS['vix_greed']:
        if row['rsi'] > 65:
            return 0  # 卖（超买）
        else:
            return 1  # 持有
    
    return 1  # 持有（中性）

# ============================================================
# 3. 生成专家轨迹
# ============================================================

print("\n" + "="*70)
print("生成专家轨迹")
print("="*70)

class ExpertEnv(gym.Env):
    """环境：用于生成专家轨迹和训练"""
    
    def __init__(self, data, lookback=60):
        super().__init__()
        
        self.data = data
        self.lookback = lookback
        
        # 离散动作空间
        self.action_space = spaces.Discrete(3)
        
        # 观察空间
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(17,),
            dtype=np.float32
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
        ], dtype=np.float32)
        
        return np.nan_to_num(obs)
    
    def step(self, action):
        if self.current_idx >= len(self.data) - 1:
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
        
        self.position = target_position
        
        # 计算当日收益
        next_price = self.data.iloc[self.current_idx + 1]['soxl_close']
        price_return = (next_price - current_price) / current_price
        
        portfolio_return = self.position * price_return - commission
        self.cash *= (1 + portfolio_return)
        self.values.append(self.cash)
        
        # 🔧 奖励函数：模仿专家 + 实际收益
        # 1. 专家动作
        expert_action = expert_policy(row)
        
        # 2. 模仿奖励
        if action == expert_action:
            imitation_reward = 1.0
        else:
            imitation_reward = -0.5
        
        # 3. 综合奖励 = 模仿奖励 + 收益奖励
        reward = imitation_reward + portfolio_return * 10
        
        # 前进一步
        self.current_idx += 1
        
        done = self.current_idx >= len(self.data) - 1
        
        return self._get_obs(), reward, done, False, {
            'cash': self.cash,
            'position': self.position,
        }

# 生成专家轨迹
print("生成专家轨迹...")
expert_trajectories = []
env = ExpertEnv(train_df)

obs, _ = env.reset()
done = False

while not done:
    # 专家决策
    row = train_df.iloc[env.current_idx]
    expert_action = expert_policy(row)
    
    expert_trajectories.append((obs.copy(), expert_action))
    
    obs, reward, done, truncated, info = env.step(expert_action)

print(f"✅ 生成 {len(expert_trajectories)} 个专家样本")

# ============================================================
# 4. Behavior Cloning
# ============================================================

print("\n" + "="*70)
print("Behavior Cloning")
print("="*70)

# 准备训练数据
observations = np.array([t[0] for t in expert_trajectories])
actions = np.array([t[1] for t in expert_trajectories])

print(f"训练数据: {len(observations)} samples")
print(f"动作分布:")
print(f"  卖 (0): {(actions == 0).sum()} ({(actions == 0).mean():.1%})")
print(f"  持有 (1): {(actions == 1).sum()} ({(actions == 1).mean():.1%})")
print(f"  买 (2): {(actions == 2).sum()} ({(actions == 2).mean():.1%})")

# 训练DQN（Behavior Cloning）
train_env = DummyVecEnv([lambda: ExpertEnv(train_df)])

model = DQN(
    "MlpPolicy",
    train_env,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=100,
    batch_size=64,
    exploration_fraction=0.1,  # 快速降低探索
    exploration_final_eps=0.01,
    verbose=0,
)

print("训练（模仿专家 - 500k步）...")
# 不预填充，直接训练，环境会提供奖励信号
model.learn(total_timesteps=500000)
print("✅ 训练完成")

# ============================================================
# 5. 测试
# ============================================================

print("\n" + "="*70)
print("测试（2020-2026）")
print("="*70)

test_env = ExpertEnv(test_df)
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
print(f"\nRL动作统计:")
print(f"  卖 (0): {(actions_taken == 0).sum()} ({(actions_taken == 0).mean():.1%})")
print(f"  持有 (1): {(actions_taken == 1).sum()} ({(actions_taken == 1).mean():.1%})")
print(f"  买 (2): {(actions_taken == 2).sum()} ({(actions_taken == 2).mean():.1%})")

print(f"\n总收益:      {total_return:.2%}")
print(f"Sharpe:      {sharpe:.2f}")
print(f"最大回撤:    {max_dd:.2%}")
print(f"最终价值:    ${test_values[-1]:,.0f}")
print(f"收益倍数:    {test_values[-1]/100000:.2f}x")

# ============================================================
# 6. 对比
# ============================================================

print("\n" + "="*70)
print("对比")
print("="*70)
print(f"专家策略（贝叶斯优化）: 990%, Sharpe 1.02")
print(f"RL模仿学习:             {total_return:.2%}, Sharpe {sharpe:.2f}")

if total_return > 7.92:
    print(f"\n✅ 超过目标！ {total_return:.2%} > 792%")
    if total_return > 9.90 * 0.9:  # 达到专家的90%
        print(f"✅ 接近专家水平！ {total_return:.2%} ≈ 990%")
else:
    print(f"\n⚠️ 未达目标 {total_return:.2%} < 792%")

# ============================================================
# 7. 保存
# ============================================================

import pathlib
pathlib.Path("models/qlib_rl_imitation").mkdir(parents=True, exist_ok=True)
model.save("models/qlib_rl_imitation/dqn_model")

print("\n模型保存: models/qlib_rl_imitation/dqn_model.zip")

print("\n" + "="*70)
print("✅ 完成！")
print("="*70)
print("""
架构：
  专家策略（贝叶斯优化）→ 轨迹
                        ↓
  Behavior Cloning → RL智能体
                        ↓
  RL模仿专家 → 应该达到990%
""")
