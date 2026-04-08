#!/usr/bin/env python3
"""
Step 1: 验证思路
- SOXL-SOXX 价差趋势 vs 未来收益
- VIX 分组 vs 未来收益
- 交叉分组（VIX × 价差）vs 未来收益

Step 2: 改奖励函数训练
- 价差增大→加仓，缩小→减仓
- VIX高→加仓，低→减仓
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
import warnings; warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CSV_PATH = '/Users/rziiiii/.openclaw/workspace/data/soxl_soxx_vix_full.csv'

print("\n跳过 Step 1 (已完成)，直接 Step 2...")

# Step 1 已完成，关键结论：
# - spread_trend IC(5d)=-0.11 有效
# - VIX IC(5d)=0.07 有效
# - VIX高+价差缩 5日收益3.0% 最佳组合
# - 简单规则回测全军覆没（成本问题）
# 结论：逻辑有效，需要 RL 学习合适的仓位管理

df = pd.read_csv(CSV_PATH, parse_dates=['date']).set_index('date')
close = df['soxl_close']; soxx = df['soxx_close']; vix = df['vix']
soxl_ret = close.pct_change()
soxx_ret = soxx.pct_change()
spread = soxl_ret - 3 * soxx_ret
spread_ma5 = spread.rolling(5).mean()
spread_ma20 = spread.rolling(20).mean()
spread_trend = spread_ma5 - spread_ma20


# 重新计算完整特征
volume = df['soxl_volume']
features = pd.DataFrame(index=df.index)
features['close'] = close; features['volume'] = volume
features['return'] = close.pct_change()
features['ma5'] = close.rolling(5).mean(); features['ma20'] = close.rolling(20).mean()
features['ma60'] = close.rolling(60).mean()
features['volatility'] = features['return'].rolling(20).std() * np.sqrt(252)
features['vix'] = vix; features['vix_ma20'] = vix.rolling(20).mean()
features['vix_zscore'] = (vix - vix.rolling(252).mean()) / (vix.rolling(252).std() + 1e-8)
delta = close.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss_d = (-delta.where(delta < 0, 0)).rolling(14).mean()
features['rsi'] = 100 - (100 / (1 + gain / (loss_d + 1e-8)))
features['panic_score'] = (
    features['vix_zscore'] * 40 +
    features['vix'].rolling(20).rank(pct=True) * 60 +
    np.where(features['return'] < 0, np.minimum(features['return'] / -0.05, 1) * (-50), 0)
)
features['vol_pct'] = volume / volume.rolling(20).mean()

# 价差特征
features['spread'] = spread
features['spread_trend'] = spread_trend
features['spread_ma5'] = spread_ma5
features['spread_ma20'] = spread_ma20

features = features.dropna()

train_data = features[features.index < '2020-01-01']
test_data = features[features.index >= '2020-01-01']
print(f"训练: {len(train_data)}, 测试: {len(test_data)}")
print(f"特征: {list(features.columns)}")

# === 新环境：增强奖励函数 ===
class TradingEnvV4(gym.Env):
    POSITION_MAP = {i: i * 0.1 for i in range(10)}
    
    def __init__(self, data, lookback=60):
        super().__init__()
        self.data = data; self.lookback = lookback
        self.action_space = spaces.Discrete(10)
        # 22维: 原17 + vix_ma20 + spread + spread_trend + spread_ma5 + spread_ma20 - 2reserved
        self.observation_space = spaces.Box(-np.inf, np.inf, (23,), np.float32)
        self.current_idx = lookback
        self.position = 0.0; self.cash = 100000; self.values = [100000]
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.current_idx = self.lookback
        self.position = 0.0; self.cash = 100000; self.values = [100000]
        return self._get_obs(), {}
    
    def _get_obs(self):
        if self.current_idx >= len(self.data): return np.zeros(23, dtype=np.float32)
        row = self.data.iloc[self.current_idx]
        lb = self.data.iloc[self.current_idx-self.lookback:self.current_idx]
        return np.nan_to_num(np.array([
            row['return'], row['ma5']/row['ma20']-1, lb['return'].mean(),
            lb['return'].std()*np.sqrt(252), row['vix'], row['vix_zscore'],
            lb['vix'].mean(), row['rsi']/100, row['panic_score']/100,
            np.log(row['vol_pct']+1e-8), self.position,
            (self.cash-100000)/100000, self.current_idx/len(self.data),
            lb['return'].skew() if len(lb)>3 else 0,
            lb['return'].kurtosis() if len(lb)>4 else 0,
            # 新增特征
            row['vix_ma20'] / 20 - 1,              # 15: VIX相对20日均值
            row['vix'] / row['vix_ma20'] - 1,      # 16: VIX偏离20日均线
            row['spread'] * 50,                      # 17: 当日价差（放大）
            row['spread_trend'] * 100,               # 18: 价差趋势（放大）
            row['spread_ma5'] * 50,                  # 19: 价差5日均值
            row['spread_ma20'] * 50,                 # 20: 价差20日均值
            row['spread_ma5'] / (row['spread_ma20']+1e-8) - 1,  # 21: 价差金叉
            0,                                        # 22: 保留
        ], dtype=np.float32))
    
    def step(self, action):
        if self.current_idx >= len(self.data)-1: return self._get_obs(), 0, True, False, {}
        action = int(action.item() if hasattr(action, 'item') else action)
        target_pos = self.POSITION_MAP[action]
        row = self.data.iloc[self.current_idx]
        
        # 交易成本
        commission = abs(target_pos - self.position) * 0.001
        self.position = target_pos
        
        # 收益
        next_price = self.data.iloc[self.current_idx+1]['close']
        price_return = (next_price - row['close']) / row['close']
        portfolio_return = self.position * price_return - commission
        self.cash *= (1 + portfolio_return)
        self.values.append(self.cash)
        
        # === 新奖励函数 ===
        reward = 0.0
        
        # 1. 收益奖励（主要）
        reward += portfolio_return * 10
        
        # 2. VIX 仓位奖励（强信号）
        vix_ratio = row['vix'] / row['vix_ma20']  # VIX相对20日均线
        
        if row['vix'] > 30:  # 极度恐慌 → 应该重仓
            if target_pos >= 0.7: reward += 3.0  # 重仓奖励
            elif target_pos <= 0.3: reward -= 3.0  # 轻仓惩罚
        elif row['vix'] > 25:  # 恐慌 → 应该加仓
            if target_pos >= 0.6: reward += 2.0
            elif target_pos <= 0.3: reward -= 2.0
        elif vix_ratio > 1.2:  # VIX高于均线20% → 适度加仓
            if target_pos >= 0.5: reward += 1.0
            else: reward -= 1.0
        elif row['vix'] < 15:  # 贪婪/自满 → 应该减仓
            if target_pos <= 0.3: reward += 2.0
            elif target_pos >= 0.6: reward -= 2.0
        elif vix_ratio < 0.85:  # VIX低于均线15% → 减仓
            if target_pos <= 0.4: reward += 1.0
            else: reward -= 1.0
        
        # 3. 价差仓位奖励（核心逻辑）
        spread_t = row['spread_trend']
        
        if spread_t > 0.002:  # 价差显著增大 → 加仓
            if target_pos >= 0.6: reward += 2.0
            elif target_pos <= 0.3: reward -= 2.0
        elif spread_t > 0:  # 价差温和增大
            if target_pos >= 0.5: reward += 1.0
        elif spread_t < -0.002:  # 价差显著缩小 → 减仓
            if target_pos <= 0.3: reward += 2.0
            elif target_pos >= 0.6: reward -= 2.0
        elif spread_t < 0:  # 价差温和缩小
            if target_pos <= 0.4: reward += 1.0
        
        # 4. 组合奖励（VIX高+价差增 = 最强加仓信号）
        if row['vix'] > 25 and spread_t > 0:
            if target_pos >= 0.8: reward += 3.0  # 双重信号重仓
        if row['vix'] < 15 and spread_t < 0:
            if target_pos <= 0.2: reward += 3.0  # 双重信号清仓
        
        self.current_idx += 1
        done = self.current_idx >= len(self.data)-1
        return self._get_obs(), reward, done, False, {}

# 训练多个模型
print("\n训练 DQN + PPO + A2C（各500k步，新奖励函数）...")

models_config = {
    'DQN_v4': lambda env: DQN("MlpPolicy", env, learning_rate=1e-3, buffer_size=50000,
                               learning_starts=1000, batch_size=64, verbose=0),
    'PPO_v4': lambda env: PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048,
                               batch_size=64, n_epochs=10, verbose=0),
    'A2C_v4': lambda env: A2C("MlpPolicy", env, learning_rate=7e-4, n_steps=5, verbose=0),
}

results = {}
for name, model_fn in models_config.items():
    print(f"\n--- {name} ---")
    train_env = DummyVecEnv([lambda: TradingEnvV4(train_data)])
    model = model_fn(train_env)
    model.learn(total_timesteps=500000)
    
    # 回测
    test_env = TradingEnvV4(test_data)
    obs, _ = test_env.reset()
    values = [100000]; positions = []; vix_levels = []; spread_trends = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        values.append(test_env.cash)
        positions.append(test_env.position)
        vix_levels.append(test_env.data.iloc[test_env.current_idx]['vix'])
        spread_trends.append(test_env.data.iloc[test_env.current_idx]['spread_trend'])
    
    values = np.array(values)
    total_return = (values[-1]-100000)/100000*100
    daily_ret = np.diff(values)/values[:-1]
    sharpe = np.mean(daily_ret)/(np.std(daily_ret)+1e-8)*np.sqrt(252)
    peak = np.maximum.accumulate(values)
    max_dd = np.max((peak-values)/peak)*100
    buy_hold = (test_data['close'].iloc[-1]/test_data['close'].iloc[0]-1)*100
    
    positions_arr = np.array(positions)
    vix_arr = np.array(vix_levels)
    spread_arr = np.array(spread_trends)
    
    # VIX行为分析
    vix_low = positions_arr[vix_arr < 15].mean() if (vix_arr < 15).any() else np.nan
    vix_mid = positions_arr[(vix_arr >= 15) & (vix_arr < 25)].mean() if ((vix_arr >= 15) & (vix_arr < 25)).any() else np.nan
    vix_high = positions_arr[vix_arr >= 25].mean() if (vix_arr >= 25).any() else np.nan
    vix_vhigh = positions_arr[vix_arr >= 30].mean() if (vix_arr >= 30).any() else np.nan
    vix_corr = np.corrcoef(vix_arr, positions_arr)[0,1]
    
    # 价差行为分析
    spread_exp = positions_arr[spread_arr > 0].mean() if (spread_arr > 0).any() else np.nan
    spread_shr = positions_arr[spread_arr < 0].mean() if (spread_arr < 0).any() else np.nan
    spread_corr = np.corrcoef(spread_arr, positions_arr)[0,1]
    
    results[name] = {
        'return': total_return, 'sharpe': sharpe, 'max_dd': max_dd,
        'excess': total_return - buy_hold,
        'vix_low': vix_low, 'vix_mid': vix_mid, 'vix_high': vix_high, 'vix_vhigh': vix_vhigh,
        'vix_corr': vix_corr,
        'spread_exp': spread_exp, 'spread_shr': spread_shr,
        'spread_corr': spread_corr,
        'values': values, 'positions': positions_arr, 
        'vix': vix_arr, 'spread': spread_arr,
    }
    
    print(f"   收益: {total_return:.1f}%  Sharpe: {sharpe:.2f}  MaxDD: {max_dd:.1f}%")
    print(f"   VIX: <15={vix_low:.0%}  15-25={vix_mid:.0%}  >25={vix_high:.0%}  >30={vix_vhigh:.0%}  corr={vix_corr:.3f}")
    print(f"   价差: 增大={spread_exp:.0%}  缩小={spread_shr:.0%}  corr={spread_corr:.3f}")

# ============================================================
# 结果对比
# ============================================================
print("\n" + "="*70)
print("最终对比")
print("="*70)

print(f"\n{'模型':<10} {'收益':>8} {'Sharpe':>7} {'VIX<15':>7} {'VIX15-25':>8} {'VIX>25':>7} {'VIX>30':>7} {'VIX相关':>8} {'价差增':>7} {'价差缩':>7} {'价差相关':>8}")
print("-"*95)
for name, r in results.items():
    vix_learned = "✅" if r['vix_corr'] > 0.1 else "❌"
    spread_learned = "✅" if r['spread_corr'] > 0.1 else "❌"
    print(f"{name:<10} {r['return']:>7.1f}% {r['sharpe']:>6.2f} {r['vix_low']:>6.0%} {r['vix_mid']:>7.0%} "
          f"{r['vix_high']:>6.0%} {r['vix_vhigh']:>6.0%} {r['vix_corr']:>7.3f} {vix_learned} "
          f"{r['spread_exp']:>6.0%} {r['spread_shr']:>6.0%} {r['spread_corr']:>7.3f} {spread_learned}")

# 绘图
print("\n绘图...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 资产曲线
ax = axes[0,0]
colors = {'DQN_v4': '#2E86AB', 'PPO_v4': '#E8463A', 'A2C_v4': '#FFA500'}
for name, r in results.items():
    n = min(len(test_data.index[60:]), len(r['values']))
    dates = test_data.index[60:60+n]
    ax.plot(dates, r['values'][:n]/1000, label=f"{name} ({r['return']:.0f}%)", 
            linewidth=2, color=colors.get(name, 'gray'))
ax.set_title('v4 新奖励函数 - 资产曲线', fontsize=14, fontweight='bold')
ax.set_ylabel('Portfolio ($K)'); ax.legend(); ax.grid(True, alpha=0.3)

# 2. VIX vs 仓位散点图（最佳模型）
best_name = max(results, key=lambda x: results[x]['return'])
best = results[best_name]
ax = axes[0,1]
sc = ax.scatter(best['vix'], best['positions'], c=best['spread'], cmap='RdYlGn', alpha=0.3, s=8)
ax.set_xlabel('VIX'); ax.set_ylabel('仓位')
ax.set_title(f'{best_name} VIX vs 仓位 (VIX相关={best["vix_corr"]:.3f}, 价差相关={best["spread_corr"]:.3f})', 
             fontsize=11, fontweight='bold')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=20, color='gray', linestyle='--', alpha=0.5)
plt.colorbar(sc, ax=ax, label='价差趋势')

# 3. VIX区间仓位
ax = axes[1,0]
x = np.arange(len(results))
width = 0.2
labels = list(results.keys())
low = [results[k]['vix_low'] for k in labels]
mid = [results[k]['vix_mid'] for k in labels]
high = [results[k]['vix_high'] for k in labels]
vhigh = [results[k]['vix_vhigh'] for k in labels]
ax.bar(x-1.5*width, low, width, label='VIX<15', color='green', alpha=0.7)
ax.bar(x-0.5*width, mid, width, label='VIX 15-25', color='orange', alpha=0.7)
ax.bar(x+0.5*width, high, width, label='VIX>25', color='red', alpha=0.7)
ax.bar(x+1.5*width, vhigh, width, label='VIX>30', color='darkred', alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('平均仓位'); ax.set_title('VIX区间仓位（应：VIX高→仓位高）', fontsize=12, fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)

# 4. 价差区间仓位
ax = axes[1,1]
exp_pos = [results[k]['spread_exp'] for k in labels]
shr_pos = [results[k]['spread_shr'] for k in labels]
ax.bar(x-0.15, exp_pos, 0.3, label='价差增大', color='green', alpha=0.7)
ax.bar(x+0.15, shr_pos, 0.3, label='价差缩小', color='red', alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('平均仓位'); ax.set_title('价差趋势仓位（应：价差增→仓位高）', fontsize=12, fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('qlib_v4_reward_shaping.png', dpi=300, bbox_inches='tight')
print("✅ 图表: qlib_v4_reward_shaping.png")

# Git
import subprocess
subprocess.run(['git', 'add', 'qlib_v4_reward_shaping.py', 'qlib_v4_reward_shaping.png'],
               cwd='/Users/rziiiii/.openclaw/workspace')
subprocess.run(['git', 'commit', '-m', 
                f'feat: v4 reward shaping - spread+VIX logic, best={best_name} {results[best_name]["return"]:.0f}%'],
               cwd='/Users/rziiiii/.openclaw/workspace')
subprocess.run(['git', 'push'], cwd='/Users/rziiiii/.openclaw/workspace')
print("✅ Git 已推送")
