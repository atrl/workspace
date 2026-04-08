#!/usr/bin/env python3
"""
v5: 加大价差奖励权重 + 减少换手惩罚 + A2C为主

基于 v4 经验：
- VIX逻辑已学到（A2C相关0.747）
- 价差逻辑没学到（相关0.087）
- 改进：价差奖励 ±2→±5, 加换手惩罚
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
print("="*70)
print("v5: 强化价差奖励 + 换手惩罚")
print("="*70)

df = pd.read_csv(CSV_PATH, parse_dates=['date']).set_index('date')
close = df['soxl_close']; volume = df['soxl_volume']
soxx = df['soxx_close']; vix = df['vix']

soxl_ret = close.pct_change(); soxx_ret = soxx.pct_change()
spread = soxl_ret - 3 * soxx_ret
spread_ma5 = spread.rolling(5).mean()
spread_ma20 = spread.rolling(20).mean()
spread_trend = spread_ma5 - spread_ma20

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
features['spread'] = spread
features['spread_trend'] = spread_trend
features['spread_ma5'] = spread_ma5
features['spread_ma20'] = spread_ma20
# 新增: 价差趋势的变化率（加速度）
features['spread_accel'] = spread_trend.diff(5)
features = features.dropna()

train_data = features[features.index < '2020-01-01']
test_data = features[features.index >= '2020-01-01']
print(f"训练: {len(train_data)}, 测试: {len(test_data)}")

class TradingEnvV5(gym.Env):
    POSITION_MAP = {i: i * 0.1 for i in range(10)}
    
    def __init__(self, data, lookback=60):
        super().__init__()
        self.data = data; self.lookback = lookback
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(-np.inf, np.inf, (25,), np.float32)
        self.current_idx = lookback
        self.position = 0.0; self.cash = 100000; self.values = [100000]
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.current_idx = self.lookback
        self.position = 0.0; self.cash = 100000; self.values = [100000]
        return self._get_obs(), {}
    
    def _get_obs(self):
        if self.current_idx >= len(self.data): return np.zeros(25, dtype=np.float32)
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
            row['vix_ma20']/20-1, row['vix']/row['vix_ma20']-1,
            row['spread']*50, row['spread_trend']*100,
            row['spread_ma5']*50, row['spread_ma20']*50,
            row['spread_ma5']/(abs(row['spread_ma20'])+1e-8)-1,
            row['spread_accel']*200,  # 加速度
            0, 0,
        ], dtype=np.float32))
    
    def step(self, action):
        if self.current_idx >= len(self.data)-1: return self._get_obs(), 0, True, False, {}
        action = int(action.item() if hasattr(action, 'item') else action)
        target_pos = self.POSITION_MAP[action]
        row = self.data.iloc[self.current_idx]
        
        # 换手惩罚
        turnover = abs(target_pos - self.position)
        commission = turnover * 0.001
        self.position = target_pos
        
        next_price = self.data.iloc[self.current_idx+1]['close']
        price_return = (next_price - row['close']) / row['close']
        portfolio_return = self.position * price_return - commission
        self.cash *= (1 + portfolio_return)
        self.values.append(self.cash)
        
        # === 奖励函数 v5 ===
        reward = 0.0
        
        # 1. 收益奖励
        reward += portfolio_return * 10
        
        # 2. VIX 仓位奖励（保持 v4 强度）
        vix_ratio = row['vix'] / row['vix_ma20']
        if row['vix'] > 30:
            reward += 5.0 if target_pos >= 0.7 else (-4.0 if target_pos <= 0.3 else 0)
        elif row['vix'] > 25:
            reward += 3.0 if target_pos >= 0.6 else (-3.0 if target_pos <= 0.3 else 0)
        elif vix_ratio > 1.2:
            reward += 1.5 if target_pos >= 0.5 else -1.0
        elif row['vix'] < 15:
            reward += 3.0 if target_pos <= 0.3 else (-3.0 if target_pos >= 0.6 else 0)
        elif vix_ratio < 0.85:
            reward += 1.5 if target_pos <= 0.4 else -1.0
        
        # 3. 价差仓位奖励（加大到 ±5.0）
        spread_t = row['spread_trend']
        spread_a = row['spread_accel']  # 加速度
        
        if spread_t > 0.003:  # 价差显著增大
            reward += 5.0 if target_pos >= 0.6 else (-4.0 if target_pos <= 0.3 else -1.0)
        elif spread_t > 0.001:  # 价差温和增大
            reward += 3.0 if target_pos >= 0.5 else 0
        elif spread_t < -0.003:  # 价差显著缩小
            reward += 5.0 if target_pos <= 0.3 else (-4.0 if target_pos >= 0.6 else -1.0)
        elif spread_t < -0.001:  # 价差温和缩小
            reward += 3.0 if target_pos <= 0.4 else 0
        
        # 4. 价差加速度奖励（新！）
        if spread_a > 0.001:  # 价差趋势在加速增大
            reward += 3.0 if target_pos >= 0.6 else 0
        elif spread_a < -0.001:  # 价差趋势在加速缩小
            reward += 3.0 if target_pos <= 0.3 else 0
        
        # 5. 组合奖励
        if row['vix'] > 25 and spread_t > 0:  # VIX高+价差增
            reward += 5.0 if target_pos >= 0.8 else 0
        if row['vix'] < 15 and spread_t < 0:  # VIX低+价差缩
            reward += 5.0 if target_pos <= 0.2 else 0
        if row['vix'] > 25 and spread_t < -0.002:  # VIX高+价差缩（反转信号）
            reward += 3.0 if target_pos >= 0.7 else 0
        
        # 6. 换手惩罚（减少过度交易）
        reward -= turnover * 2.0
        
        self.current_idx += 1
        done = self.current_idx >= len(self.data)-1
        return self._get_obs(), reward, done, False, {}

# 训练
print("\n训练 DQN + PPO + A2C (各500k)...")
results = {}

models_config = {
    'DQN_v5': lambda env: DQN("MlpPolicy", env, learning_rate=1e-3, buffer_size=50000,
                               learning_starts=1000, batch_size=64, verbose=0),
    'PPO_v5': lambda env: PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048,
                               batch_size=64, n_epochs=10, verbose=0),
    'A2C_v5': lambda env: A2C("MlpPolicy", env, learning_rate=7e-4, n_steps=5, verbose=0),
}

for name, model_fn in models_config.items():
    print(f"\n--- {name} ---")
    train_env = DummyVecEnv([lambda: TradingEnvV5(train_data)])
    model = model_fn(train_env)
    model.learn(total_timesteps=500000)
    
    test_env = TradingEnvV5(test_data)
    obs, _ = test_env.reset()
    values = [100000]; positions = []; vix_levels = []; spread_trends = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        values.append(test_env.cash)
        positions.append(test_env.position)
        if test_env.current_idx < len(test_env.data):
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
    n = min(len(positions_arr), len(vix_arr), len(spread_arr))
    positions_arr = positions_arr[:n]; vix_arr = vix_arr[:n]; spread_arr = spread_arr[:n]
    
    vix_low = positions_arr[vix_arr < 15].mean() if (vix_arr < 15).any() else np.nan
    vix_mid = positions_arr[(vix_arr >= 15) & (vix_arr < 25)].mean() if ((vix_arr >= 15) & (vix_arr < 25)).any() else np.nan
    vix_high = positions_arr[vix_arr >= 25].mean() if (vix_arr >= 25).any() else np.nan
    vix_vhigh = positions_arr[vix_arr >= 30].mean() if (vix_arr >= 30).any() else np.nan
    vix_corr = np.corrcoef(vix_arr, positions_arr)[0,1]
    
    spread_exp = positions_arr[spread_arr > 0.001].mean() if (spread_arr > 0.001).any() else np.nan
    spread_shr = positions_arr[spread_arr < -0.001].mean() if (spread_arr < -0.001).any() else np.nan
    spread_corr = np.corrcoef(spread_arr, positions_arr)[0,1]
    
    # 换手率
    pos_changes = np.abs(np.diff(positions_arr))
    turnover = pos_changes.mean() * 252
    
    results[name] = {
        'return': total_return, 'sharpe': sharpe, 'max_dd': max_dd,
        'excess': total_return - buy_hold, 'turnover': turnover,
        'vix_low': vix_low, 'vix_mid': vix_mid, 'vix_high': vix_high, 'vix_vhigh': vix_vhigh,
        'vix_corr': vix_corr,
        'spread_exp': spread_exp, 'spread_shr': spread_shr, 'spread_corr': spread_corr,
        'values': values, 'positions': positions_arr, 'vix': vix_arr, 'spread': spread_arr,
    }
    
    print(f"   收益: {total_return:.1f}%  Sharpe: {sharpe:.2f}  MaxDD: {max_dd:.1f}%  换手: {turnover:.1f}x")
    print(f"   VIX: <15={vix_low:.0%}  15-25={vix_mid:.0%}  >25={vix_high:.0%}  >30={vix_vhigh:.0%}  corr={vix_corr:.3f}")
    print(f"   价差: 增={spread_exp:.0%}  缩={spread_shr:.0%}  corr={spread_corr:.3f}")

# 对比
print(f"\n{'模型':<10} {'收益':>7} {'Sharpe':>7} {'MaxDD':>6} {'VIX<15':>6} {'VIX>25':>6} {'VIX>30':>6} {'VIX相关':>7} {'价差增':>6} {'价差缩':>6} {'价差相关':>7} {'换手':>5}")
print("-"*90)
for name, r in results.items():
    vix_ok = "✅" if r['vix_corr'] > 0.3 else "❌"
    sp_ok = "✅" if r['spread_corr'] > 0.1 else "❌"
    print(f"{name:<10} {r['return']:>6.0f}% {r['sharpe']:>6.2f} {r['max_dd']:>5.1f}% "
          f"{r['vix_low']:>5.0%} {r['vix_high']:>5.0%} {r['vix_vhigh']:>5.0%} {r['vix_corr']:>6.3f}{vix_ok} "
          f"{r['spread_exp']:>5.0%} {r['spread_shr']:>5.0%} {r['spread_corr']:>6.3f}{sp_ok} "
          f"{r['turnover']:>4.1f}x")

# 绘图
best_name = max(results, key=lambda x: results[x]['sharpe'])
best = results[best_name]
print(f"\n🏆 最佳(Sharpe): {best_name} ({best['return']:.0f}%, Sharpe {best['sharpe']:.2f})")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax = axes[0,0]
colors = {'DQN_v5': '#2E86AB', 'PPO_v5': '#E8463A', 'A2C_v5': '#FFA500'}
for name, r in results.items():
    n = min(len(test_data.index[60:]), len(r['values']))
    dates = test_data.index[60:60+n]
    ax.plot(dates, r['values'][:n]/1000, label=f"{name} ({r['return']:.0f}%)", 
            linewidth=2, color=colors.get(name, 'gray'))
ax.set_title('v5 强化价差+换手惩罚', fontsize=14, fontweight='bold')
ax.set_ylabel('Portfolio ($K)'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0,1]
sc = ax.scatter(best['vix'], best['positions'], c=best['spread'], cmap='RdYlGn', alpha=0.3, s=8)
ax.set_xlabel('VIX'); ax.set_ylabel('仓位')
ax.set_title(f'{best_name} VIX×仓位 (VIX相关={best["vix_corr"]:.3f}, 价差相关={best["spread_corr"]:.3f})', fontsize=11, fontweight='bold')
plt.colorbar(sc, ax=ax, label='价差趋势')

ax = axes[1,0]
labels = list(results.keys()); x = np.arange(len(labels)); w = 0.2
ax.bar(x-1.5*w, [results[k]['vix_low'] for k in labels], w, label='VIX<15', color='green', alpha=0.7)
ax.bar(x-.5*w, [results[k]['vix_mid'] for k in labels], w, label='VIX 15-25', color='orange', alpha=0.7)
ax.bar(x+.5*w, [results[k]['vix_high'] for k in labels], w, label='VIX>25', color='red', alpha=0.7)
ax.bar(x+1.5*w, [results[k]['vix_vhigh'] for k in labels], w, label='VIX>30', color='darkred', alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('仓位')
ax.set_title('VIX区间仓位（应：VIX高→仓位高）', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1,1]
ax.bar(x-0.15, [results[k]['spread_exp'] for k in labels], 0.3, label='价差增大', color='green', alpha=0.7)
ax.bar(x+0.15, [results[k]['spread_shr'] for k in labels], 0.3, label='价差缩小', color='red', alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('仓位')
ax.set_title('价差趋势仓位（应：增大→高, 缩小→低）', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('qlib_v5_spread_enhanced.png', dpi=300, bbox_inches='tight')
print("✅ 图表: qlib_v5_spread_enhanced.png")

import subprocess
subprocess.run(['git', 'add', 'qlib_v5_spread_enhanced.py', 'qlib_v5_spread_enhanced.png'],
               cwd='/Users/rziiiii/.openclaw/workspace')
subprocess.run(['git', 'commit', '-m',
                f'feat: v5 spread reward x5 + turnover penalty, best={best_name} {best["return"]:.0f}%'],
               cwd='/Users/rziiiii/.openclaw/workspace')
subprocess.run(['git', 'push'], cwd='/Users/rziiiii/.openclaw/workspace')
print("✅ Git 已推送")
