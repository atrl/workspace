#!/usr/bin/env python3
"""
Qlib 最优策略 v3 - 精选交叉特征 + 500k步训练

基于 v2 的 IC 分析，只保留有效交叉特征：
- rolling_te_vol_20 (IC=0.1139) ✅
- soxl_soxx_rel_5 (IC=-0.0724) ✅  
- soxx_mom_5 (IC=-0.0588) ✅
去掉：
- tracking_error (IC=-0.002) ❌
- cum_divergence_20 (IC=-0.015) ❌

观察空间: 17 → 20 维
训练步数: 300k → 500k
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import warnings
warnings.filterwarnings('ignore')

import qlib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CSV_PATH = '/Users/rziiiii/.openclaw/workspace/data/soxl_soxx_vix_full.csv'
QLIB_DATA_PATH = '/Users/rziiiii/.openclaw/workspace/qlib_soxl_binary'

# ============================================================
# Phase 1: RETRIEVE
# ============================================================
print("="*70)
print("Phase 1: RETRIEVE")
print("="*70)

if Path(QLIB_DATA_PATH).exists():
    qlib.init(provider_uri=QLIB_DATA_PATH)
    print("   ✅ qlib 初始化")

df = pd.read_csv(CSV_PATH, parse_dates=['date']).set_index('date')
print(f"   CSV: {len(df)} 条")

# ============================================================
# Phase 2: GENERATE
# ============================================================
print("\n" + "="*70)
print("Phase 2: GENERATE (20维 + 500k)")
print("="*70)

close = df['soxl_close']
volume = df['soxl_volume']
soxx_close = df['soxx_close']
vix = df['vix']

features = pd.DataFrame(index=df.index)
features['close'] = close
features['volume'] = volume
features['return'] = close.pct_change()
features['ma5'] = close.rolling(5).mean()
features['ma20'] = close.rolling(20).mean()
features['ma60'] = close.rolling(60).mean()
features['volatility'] = features['return'].rolling(20).std() * np.sqrt(252)
features['vix'] = vix
features['vix_zscore'] = (vix - vix.rolling(252).mean()) / (vix.rolling(252).std() + 1e-8)

# RSI
delta = close.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
features['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))

features['panic_score'] = (
    features['vix_zscore'] * 40 +
    features['vix'].rolling(20).rank(pct=True) * 60 +
    np.where(features['return'] < 0, np.minimum(features['return'] / -0.05, 1) * (-50), 0)
)
features['vol_pct'] = volume / volume.rolling(20).mean()

# 🆕 只保留 IC>0.05 的交叉特征（3个）
print("   计算精选交叉特征（IC>0.05）...")
soxx_return = soxx_close.pct_change()
soxl_return = features['return']

# 1. rolling_te_vol_20 (IC=0.1139) - 最强！
tracking_error = soxl_return - (3 * soxx_return)
features['rolling_te_vol_20'] = tracking_error.rolling(20).std()

# 2. soxl_soxx_rel_5 (IC=-0.0724)
features['soxl_soxx_rel_5'] = soxl_return.rolling(5).sum() - soxx_return.rolling(5).sum()

# 3. soxx_mom_5 (IC=-0.0588)
features['soxx_mom_5'] = soxx_return.rolling(5).sum()

features = features.dropna()
print(f"   ✅ 特征: {len(features)} 条, 精选3个交叉特征")

class TradingEnvV3(gym.Env):
    POSITION_MAP = {i: i * 0.1 for i in range(10)}
    
    def __init__(self, data, lookback=60):
        super().__init__()
        self.data = data
        self.lookback = lookback
        self.action_space = spaces.Discrete(10)
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
            row['return'],                                    # 0
            row['ma5'] / row['ma20'] - 1,                    # 1
            lb['return'].mean(),                              # 2
            lb['return'].std() * np.sqrt(252),               # 3
            row['vix'],                                       # 4
            row['vix_zscore'],                                # 5
            lb['vix'].mean(),                                 # 6
            row['rsi'] / 100,                                 # 7
            row['panic_score'] / 100,                         # 8
            np.log(row['vol_pct'] + 1e-8),                    # 9
            self.position,                                     # 10
            (self.cash - 100000) / 100000,                    # 11
            self.current_idx / len(self.data),                # 12
            lb['return'].skew() if len(lb) > 3 else 0,       # 13
            lb['return'].kurtosis() if len(lb) > 4 else 0,    # 14
            0, 0,                                             # 15-16
            # 精选交叉特征 (17-19)
            row['rolling_te_vol_20'] * 100,                    # 17: IC=0.1139
            row['soxl_soxx_rel_5'] * 10,                       # 18: IC=-0.0724
            row['soxx_mom_5'] * 100,                           # 19: IC=-0.0588
        ], dtype=np.float32))
        return obs
    
    def step(self, action):
        if self.current_idx >= len(self.data) - 1:
            return self._get_obs(), 0, True, False, {}
        action = int(action.item() if hasattr(action, 'item') else action)
        target_pos = self.POSITION_MAP[action]
        row = self.data.iloc[self.current_idx]
        commission = abs(target_pos - self.position) * 0.001
        self.position = target_pos
        next_price = self.data.iloc[self.current_idx + 1]['close']
        price_return = (next_price - row['close']) / row['close']
        portfolio_return = self.position * price_return - commission
        self.cash *= (1 + portfolio_return)
        self.values.append(self.cash)
        
        expert_action = self._expert_policy(row)
        imitation_reward = 1.0 if action == expert_action else -0.5
        return_reward = portfolio_return * 10
        reward = imitation_reward + return_reward
        
        self.current_idx += 1
        done = self.current_idx >= len(self.data) - 1
        return self._get_obs(), reward, done, False, {}
    
    def _expert_policy(self, row):
        if row['panic_score'] <= -25:
            if row['vix'] > 28: return 9
            elif row['vix'] > 22: return 8
            else: return 6
        # 交叉特征增强判断
        if row['rolling_te_vol_20'] > 0.02:  # 高偏离度波动 → 减仓
            return 3
        if row['soxx_mom_5'] > 0.05 and row['soxl_soxx_rel_5'] > 0:  # SOXX强势+SOXL跟随
            return 7
        if row['rsi'] < 65 and row['ma5'] > row['ma20']:
            return 6
        if row['vix'] < 15:
            if row['rsi'] > 70: return 0
            elif row['rsi'] > 65: return 1
            else: return 3
        return 5

# 分割
train_data = features[features.index < '2020-01-01']
test_data = features[features.index >= '2020-01-01']
print(f"   训练集: {len(train_data)}, 测试集: {len(test_data)}")

# 训练 500k
print(f"\n2.6 训练 DQN (500k步, 20维)...")
train_env = DummyVecEnv([lambda: TradingEnvV3(train_data)])
model = DQN("MlpPolicy", train_env, learning_rate=1e-3, buffer_size=50000,
            learning_starts=1000, batch_size=64, verbose=0)
model.learn(total_timesteps=500000)
print("   ✅ 训练完成")

# ============================================================
# Phase 3: EVALUATE
# ============================================================
print("\n" + "="*70)
print("Phase 3: EVALUATE")
print("="*70)

test_env = TradingEnvV3(test_data)
obs, _ = test_env.reset()
values = [100000]
positions = []
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = test_env.step(action)
    values.append(test_env.cash)
    positions.append(test_env.position)

values = np.array(values)
total_return = (values[-1] - 100000) / 100000 * 100
daily_returns = np.diff(values) / values[:-1]
sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
peak = np.maximum.accumulate(values)
max_dd = np.max((peak - values) / peak) * 100
buy_hold = (test_data['close'].iloc[-1] / test_data['close'].iloc[0] - 1) * 100

print(f"   最终资产: ${values[-1]:,.2f}")
print(f"   总收益: {total_return:.2f}%")
print(f"   Sharpe: {sharpe:.2f}")
print(f"   MaxDD: {max_dd:.2f}%")
print(f"   买入持有: {buy_hold:.2f}%")
print(f"   超额收益: {total_return - buy_hold:+.2f}%")

print(f"\n   仓位分布:")
for i in range(10):
    pct = (np.array(positions) == i * 0.1).mean()
    if pct > 0:
        print(f"   {i*10}%: {pct:.1%}")

# ============================================================
# Phase 4: VALIDATE
# ============================================================
print("\n" + "="*70)
print("Phase 4: VALIDATE")
print("="*70)
print(f"   Sharpe {sharpe:.2f} {'< 1.5 ✅' if sharpe < 1.5 else '> 1.5 ⚠️'}")
print(f"   MaxDD {max_dd:.1f}% {'< 70% ✅' if max_dd < 70 else '> 70% ⚠️'}")

# ============================================================
# Phase 5: DISTILL
# ============================================================
print("\n" + "="*70)
print("Phase 5: DISTILL")
print("="*70)

model_dir = Path('models/qlib_cross_feature_v3')
model_dir.mkdir(parents=True, exist_ok=True)
model.save(str(model_dir / 'dqn_model'))
print(f"   ✅ 模型: {model_dir / 'dqn_model.zip'}")

# 绘图
n = min(len(test_data.index[60:]), len(values), len(positions))
dates = test_data.index[60:60+n]
vals = values[:n]
pos = positions[:n]

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

ax1 = axes[0]
ax1.plot(dates, vals / 1000, label='DQN v3 (20dim+500k)', linewidth=2.5, color='#2E86AB')
ax1.fill_between(dates, 100, vals / 1000, alpha=0.2, color='#2E86AB')
ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
ax1.set_ylabel('Portfolio ($K)', fontsize=12)
ax1.set_title(f'Qlib 精选交叉特征策略（20维+500k步）\n策略: {total_return:.1f}% vs 基准: {buy_hold:.1f}% (超额: {total_return-buy_hold:+.1f}%)',
              fontsize=14, fontweight='bold')
ax1.legend(loc='upper left'); ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.fill_between(dates, pos, alpha=0.5, color='#28A745', label='SOXL仓位')
ax2.set_ylabel('仓位', fontsize=12); ax2.set_ylim(0, 1.1)
ax2.legend(loc='upper left'); ax2.grid(True, alpha=0.3)

ax3 = axes[2]
vix_s = test_data['vix'].iloc[60:60+n]
te_s = test_data['rolling_te_vol_20'].iloc[60:60+n]
ax3.plot(dates[:len(vix_s)], vix_s, color='red', alpha=0.7, label='VIX')
ax3b = ax3.twinx()
ax3b.plot(dates[:len(te_s)], te_s * 100, color='purple', alpha=0.5, label='TE Vol x100')
ax3b.set_ylabel('TE Vol x100', color='purple')
ax3.axhline(y=22, color='orange', linestyle='--', alpha=0.5)
ax3.set_ylabel('VIX', fontsize=12); ax3.set_xlabel('Date', fontsize=12)
ax3.legend(loc='upper left'); ax3b.legend(loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_locator(mdates.YearLocator())
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig('qlib_cross_feature_v3.png', dpi=300, bbox_inches='tight')
print(f"   ✅ 图表: qlib_cross_feature_v3.png")

# Git
import subprocess
subprocess.run(['git', 'add', 'qlib_cross_feature_v3.py', 'qlib_cross_feature_v3.png',
                'models/qlib_cross_feature_v3/'], cwd='/Users/rziiiii/.openclaw/workspace')
subprocess.run(['git', 'commit', '-m',
                f'feat: Qlib cross-feature v3 (20dim+500k) return {total_return:.0f}% sharpe {sharpe:.2f}'],
               cwd='/Users/rziiiii/.openclaw/workspace')
subprocess.run(['git', 'push'], cwd='/Users/rziiiii/.openclaw/workspace')
print("   ✅ Git 已推送")

print(f"""
对比总结:
  v1 (17dim, 300k): 878%, Sharpe 0.96, MaxDD 65.2%
  v2 (22dim, 300k): 337%, Sharpe 0.72, MaxDD 71.6%
  v3 (20dim, 500k): {total_return:.0f}%, Sharpe {sharpe:.2f}, MaxDD {max_dd:.1f}%
""")
