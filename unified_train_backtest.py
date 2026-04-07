#!/usr/bin/env python3
"""
统一数据源 - 训练 + 回测

使用相同的数据源（CSV）进行：
1. DQN 训练
2. FinRL 风格回测
3. 对比结果
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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

print("="*70)
print("统一数据源 - 训练 + 回测")
print("="*70)

# 1. 加载统一数据（CSV）
print("\n1. 加载统一数据（CSV）...")

csv_path = '/Users/rziiiii/.openclaw/workspace/data/soxl_soxx_vix_full.csv'
df = pd.read_csv(csv_path, parse_dates=['date'])
df = df.set_index('date')

# 只用 SOXL 数据
data = df[['soxl_close', 'soxl_volume', 'vix']].copy()
data.columns = ['close', 'volume', 'vix']

print(f"   ✅ 数据加载完成: {len(data)} 条")
print(f"   时间范围: {data.index.min()} to {data.index.max()}")

# 2. 特征工程（与之前一致）
print("\n2. 特征工程...")

close = data['close']
vix = data['vix']

data['return'] = close.pct_change()
data['ma5'] = close.rolling(5).mean()
data['ma20'] = close.rolling(20).mean()

# RSI
delta = close.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
data['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))

# VIX
data['vix_zscore'] = (vix - vix.rolling(20).mean()) / (vix.rolling(20).std() + 1e-8)

# 恐慌得分
data['panic_score'] = (
    data['vix_zscore'] * 40 +
    vix.rolling(20).rank(pct=True) * 60 +
    np.where(data['return'] < 0, np.minimum(data['return'] / -0.05, 1) * (-50), 0)
)

data = data.dropna()

# 分割数据
train_data = data[data.index < '2020-01-01']
test_data = data[data.index >= '2020-01-01']

print(f"   ✅ 训练数据: {len(train_data)} 条")
print(f"   ✅ 测试数据: {len(test_data)} 条")

# 3. 创建 RL 环境（与之前一致）
print("\n3. 创建 RL 环境...")

class TradingEnv(gym.Env):
    """交易环境（17维观察，10档仓位）"""
    
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
        
        # 奖励（模仿学习）
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
        """专家策略（贝叶斯优化参数）"""
        # 恐慌买入
        if row['panic_score'] <= -25:
            if row['vix'] > 28:
                return 9
            elif row['vix'] > 22:
                return 8
            else:
                return 6
        
        # 趋势跟踪
        if row['rsi'] < 65 and row['ma5'] > row['ma20']:
            return 6
        
        # 贪婪卖出
        if row['vix'] < 15:
            if row['rsi'] > 70:
                return 0
            elif row['rsi'] > 65:
                return 1
            else:
                return 3
        
        return 5

# 4. 训练（300k步，与之前一致）
print("\n4. 训练 DQN（300k步）...")

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
print("   ✅ 训练完成")

# 5. 回测（FinRL 风格，统一数据）
print("\n5. 回测（统一数据源）...")

test_env = TradingEnv(test_data)
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

final_value = test_values[-1]
total_return = (final_value - 100000) / 100000 * 100

daily_returns = np.diff(test_values) / test_values[:-1]
sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)

peak = np.maximum.accumulate(test_values)
max_dd = np.max((peak - test_values) / peak)

# 仓位分布
print(f"\n仓位分布:")
for i in range(10):
    count = (positions == i * 0.1).sum()
    pct = (positions == i * 0.1).mean()
    if count > 0:
        print(f"   {i*10:2d}%: {count:4d} ({pct:.1%})")

print(f"\n" + "="*70)
print("回测结果（统一数据源）")
print("="*70)
print(f"最终资产: ${final_value:,.2f}")
print(f"总收益: {total_return:.2f}%")
print(f"Sharpe: {sharpe:.2f}")
print(f"MaxDD: {max_dd:.2%}")

# 6. 对比
print(f"\n" + "="*70)
print("对比")
print("="*70)

# 买入持有
buy_hold = (test_data['close'].iloc[-1] / test_data['close'].iloc[0] - 1) * 100
print(f"DRL 策略（统一数据）: {total_return:.2f}%")
print(f"买入持有: {buy_hold:.2f}%")
print(f"超额收益: {total_return - buy_hold:.2f}%")

print(f"\n与之前结果对比:")
print(f"  自定义回测（不同数据）: 1156.55%")
print(f"  官方回测（qlib binary）: 162.67%")
print(f"  统一数据回测（这次）: {total_return:.2f}%")

# 7. 绘图
print(f"\n7. 生成图表...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

dates = test_data.index[60:60+len(test_values)]
ax1.plot(dates, test_values / 1000, label='DRL Strategy (Unified Data)', linewidth=2.5, color='#2E86AB')
ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
ax1.fill_between(dates, 100, test_values / 1000, alpha=0.2, color='#2E86AB')
ax1.set_ylabel('Portfolio Value ($K)', fontsize=14, fontweight='bold')
ax1.set_title(f'DRL Strategy (Unified CSV Data) - {total_return:.1f}% Return', 
              fontsize=16, fontweight='bold')
ax1.legend(loc='upper left', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}K'))

ax2.fill_between(dates, positions, alpha=0.5, color='#28A745')
ax2.set_ylabel('Position', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 1.1)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig('unified_data_backtest.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"   ✅ 图表已保存: unified_data_backtest.png")

# 8. 保存
import pathlib
pathlib.Path("models/unified_drl").mkdir(parents=True, exist_ok=True)
model.save("models/unified_drl/dqn_model")
print(f"   ✅ 模型已保存: models/unified_drl/dqn_model.zip")

print("\n" + "="*70)
print("✅ 统一数据源训练+回测完成")
print("="*70)
