#!/usr/bin/env python3
"""
修复 Qlib RL - 调整奖励函数

问题：RL仓位太低（0.20 vs 0.48）
原因：回撤惩罚太强

修复：
  1. 降低回撤惩罚：30 → 10
  2. 增加收益奖励：150 → 200
  3. 移除风险调整
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from qlib.rl import Simulator, StateInterpreter, ActionInterpreter, Reward

print("="*70)
print("修复 Qlib RL - 调整奖励函数")
print("="*70)
print("""
问题：RL仓位太低（0.20 vs 0.48）
原因：回撤惩罚太强

修复：
  ✅ 降低回撤惩罚：30 → 10
  ✅ 增加收益奖励：150 → 200
  ✅ 移除风险调整
""")

# ============================================================
# 1. 修复奖励函数
# ============================================================

class QlibSOXLFixedSimulator:
    """修复版模拟器 - 调整奖励函数"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_cash: float = 100000,
        commission: float = 0.001,
        panic_threshold: float = -25,
        vix_panic: float = 22,
        vix_extreme: float = 26,
        vix_greed: float = 15,
    ):
        self.data = data
        self.initial_cash = initial_cash
        self.commission = commission
        self.panic_threshold = panic_threshold
        self.vix_panic = vix_panic
        self.vix_extreme = vix_extreme
        self.vix_greed = vix_greed
        
        self.reset()
    
    def reset(self):
        self.current_idx = 60
        self.cash = self.initial_cash
        self.position = 0.0
        self.total_value = self.initial_cash
        self.history = [self.initial_cash]
        self.returns = []
        self._done = False
        
        return self.get_state()
    
    def _get_rule_signal(self) -> float:
        row = self.data.iloc[self.current_idx]
        
        if row['panic_score'] <= self.panic_threshold:
            if row['vix'] > self.vix_extreme:
                return 1.0
            elif row['vix'] > self.vix_panic:
                return 0.9
            return 0.7
        elif row['rsi'] < 65 and row['ma5'] > row['ma20']:
            return 0.6
        elif row['vix'] < self.vix_greed:
            if row['rsi'] > 65:
                return 0.2
            return 0.4
        return 0.5
    
    def step(self, action: float) -> Tuple[Dict, float, bool, Dict]:
        if self._done:
            return {}, 0, True, {}
        
        rule_signal = self._get_rule_signal()
        rl_position = float(np.clip(action, 0, 1))
        target_position = rule_signal * rl_position
        target_position = float(np.clip(target_position, 0, 0.95))
        
        current_price = self.data.iloc[self.current_idx]['soxl_close']
        position_change = target_position - self.position
        trade_value = abs(position_change) * self.total_value
        commission_cost = trade_value * self.commission
        
        self.position = target_position
        
        self.current_idx += 1
        
        if self.current_idx >= len(self.data):
            self._done = True
            return self.get_state(), 0, True, {}
        
        next_price = self.data.iloc[self.current_idx]['soxl_close']
        price_return = (next_price - current_price) / current_price
        portfolio_return = self.position * price_return - commission_cost / self.total_value
        
        self.total_value *= (1 + portfolio_return)
        self.history.append(self.total_value)
        self.returns.append(portfolio_return)
        
        # 🔧 修复奖励函数
        reward = self._calculate_reward_fixed(portfolio_return)
        
        return self.get_state(), reward, self._done, {'total_value': self.total_value}
    
    def get_state(self) -> Dict[str, Any]:
        if self.current_idx >= len(self.data):
            return {}
        
        row = self.data.iloc[self.current_idx]
        
        if len(self.history) > 10:
            hist = np.array(self.history[-20:])
            rets = np.diff(hist) / hist[:-1]
            volatility = np.std(rets) * np.sqrt(252)
        else:
            volatility = 0
        
        return {
            'return': row.get('return', 0),
            'ma5_ma20_ratio': row['ma5'] / row['ma20'] if row['ma20'] > 0 else 1,
            'volatility': volatility,
            'rsi': row.get('rsi', 50) / 100,
            'vix_zscore': row.get('vix_zscore', 0),
            'panic_score': row.get('panic_score', 0) / 100,
            'position': self.position,
            'portfolio_return': (self.total_value - self.initial_cash) / self.initial_cash,
            'rule_signal': self._get_rule_signal(),
        }
    
    def _calculate_reward_fixed(self, portfolio_return: float) -> float:
        """
        🔧 修复版奖励函数
        
        修改：
          1. 增加收益奖励：150 → 200
          2. 降低回撤惩罚：30 → 10
          3. 移除风险调整（让RL专注收益）
        """
        # 收益奖励（增加）
        reward = portfolio_return * 200  # 之前 150
        
        # 🔧 移除风险调整（之前会导致RL太保守）
        # if len(self.returns) > 10:
        #     mean_ret = np.mean(self.returns[-20:])
        #     std_ret = np.std(self.returns[-20:]) + 1e-8
        #     reward += mean_ret / std_ret * 15
        
        # 🔧 降低回撤惩罚（30 → 10）
        if len(self.history) > 0:
            peak = max(self.history)
            dd = (peak - self.total_value) / peak
            if dd > 0.15:
                reward -= (dd - 0.15) * 10  # 之前 30
        
        return reward


class QlibSOXLFixedEnv(gym.Env):
    """修复版环境"""
    
    def __init__(self, data, initial_cash=100000, commission=0.001,
                 panic_threshold=-25, vix_panic=22, vix_extreme=26, vix_greed=15):
        super().__init__()
        
        self.simulator = QlibSOXLFixedSimulator(
            data, initial_cash, commission,
            panic_threshold, vix_panic, vix_extreme, vix_greed
        )
        
        self.action_space = spaces.Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32
        )
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        state_dict = self.simulator.reset()
        
        features = [
            state_dict.get('return', 0),
            state_dict.get('ma5_ma20_ratio', 1),
            state_dict.get('volatility', 0),
            state_dict.get('rsi', 0.5),
            state_dict.get('vix_zscore', 0),
            state_dict.get('panic_score', 0),
            state_dict.get('position', 0),
            state_dict.get('portfolio_return', 0),
            state_dict.get('rule_signal', 0.5),
        ]
        
        return np.array(features, dtype=np.float32), {}
    
    def step(self, action):
        target_position = float(np.clip(action[0], 0, 1))
        next_state_dict, reward, done, info = self.simulator.step(target_position)
        
        if not next_state_dict:
            return np.zeros(9, dtype=np.float32), reward, done, False, info
        
        features = [
            next_state_dict.get('return', 0),
            next_state_dict.get('ma5_ma20_ratio', 1),
            next_state_dict.get('volatility', 0),
            next_state_dict.get('rsi', 0.5),
            next_state_dict.get('vix_zscore', 0),
            next_state_dict.get('panic_score', 0),
            next_state_dict.get('position', 0),
            next_state_dict.get('portfolio_return', 0),
            next_state_dict.get('rule_signal', 0.5),
        ]
        
        return np.array(features, dtype=np.float32), reward, done, False, info


# ============================================================
# 2. 准备数据
# ============================================================

print("\n准备数据...")

df = pd.read_csv(
    "/Users/rziiiii/.openclaw/workspace/data/soxl_soxx_vix_full.csv",
    parse_dates=['date']
)
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
    np.where(df['return'] < 0, np.minimum(df['return'] / -0.05, 1) * -50, 0)
)

df = df.dropna()

train_df = df[df.index < '2020-01-01']
test_df = df[df.index >= '2020-01-01']

print(f"训练数据: {len(train_df)} records")
print(f"测试数据: {len(test_df)} records")

# ============================================================
# 3. 训练修复版
# ============================================================

print("\n训练修复版（500k步）...")

train_env = DummyVecEnv([lambda: QlibSOXLFixedEnv(
    train_df,
    panic_threshold=-25,
    vix_panic=22,
    vix_extreme=26,
    vix_greed=15,
)])

model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    verbose=0,
)

model.learn(total_timesteps=500000)
print("✅ 训练完成")

# ============================================================
# 4. 测试
# ============================================================

print("\n" + "="*70)
print("测试修复版（2020-2026）")
print("="*70)

test_env = QlibSOXLFixedEnv(
    test_df,
    panic_threshold=-25,
    vix_panic=22,
    vix_extreme=26,
    vix_greed=15,
)

obs, _ = test_env.reset()
test_values = [100000]
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = test_env.step(action)
    test_values.append(info.get('total_value', test_values[-1]))

test_values = np.array(test_values)
total_return = (test_values[-1] - 100000) / 100000
daily_returns = np.diff(test_values) / test_values[:-1]
sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
max_dd = np.max((np.maximum.accumulate(test_values) - test_values) / np.maximum.accumulate(test_values))

print(f"\n总收益:      {total_return:.2%}")
print(f"Sharpe:      {sharpe:.2f}")
print(f"最大回撤:    {max_dd:.2%}")
print(f"收益倍数:    {test_values[-1]/100000:.2f}x")

# ============================================================
# 5. 对比
# ============================================================

print("\n" + "="*70)
print("对比")
print("="*70)
print(f"目标（Pandas）:  792%, Sharpe 0.88")
print(f"Qlib纯规则:     990%, Sharpe 1.02")
print(f"Qlib RL旧版:    179%, Sharpe 0.65（RL太保守）")
print(f"Qlib RL修复版:  {total_return:.2%}, Sharpe {sharpe:.2f}")

if total_return > 7.92:
    print(f"\n✅ 超过目标！ {total_return:.2%} > 792%")
else:
    print(f"\n⚠️ 未达目标 {total_return:.2%} < 792%")

# ============================================================
# 6. 保存
# ============================================================

import pathlib
pathlib.Path("models/qlib_rl_fixed").mkdir(parents=True, exist_ok=True)
model.save("models/qlib_rl_fixed/ppo_model")

print("\n模型保存: models/qlib_rl_fixed/ppo_model.zip")
