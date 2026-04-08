#!/usr/bin/env python3
"""
多模型对比 + VIX仓位行为分析

分析：模型是否学到了 VIX 高加仓、低减仓？
对比：DQN vs PPO vs SAC vs A2C
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
from stable_baselines3 import DQN, PPO, SAC, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CSV_PATH = '/Users/rziiiii/.openclaw/workspace/data/soxl_soxx_vix_full.csv'

# ============================================================
# 数据准备（与v1完全一致）
# ============================================================
print("="*70)
print("多模型对比 + VIX行为分析")
print("="*70)

df = pd.read_csv(CSV_PATH, parse_dates=['date']).set_index('date')
close = df['soxl_close']; volume = df['soxl_volume']
soxx_close = df['soxx_close']; vix = df['vix']

features = pd.DataFrame(index=df.index)
features['close'] = close; features['volume'] = volume
features['return'] = close.pct_change()
features['ma5'] = close.rolling(5).mean()
features['ma20'] = close.rolling(20).mean()
features['ma60'] = close.rolling(60).mean()
features['volatility'] = features['return'].rolling(20).std() * np.sqrt(252)
features['vix'] = vix
features['vix_zscore'] = (vix - vix.rolling(252).mean()) / (vix.rolling(252).std() + 1e-8)
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

# 精选交叉特征
soxx_return = soxx_close.pct_change()
soxl_return = features['return']
tracking_error = soxl_return - (3 * soxx_return)
features['rolling_te_vol_20'] = tracking_error.rolling(20).std()
features['soxl_soxx_rel_5'] = soxl_return.rolling(5).sum() - soxx_return.rolling(5).sum()
features['soxx_mom_5'] = soxx_return.rolling(5).sum()

features = features.dropna()

train_data = features[features.index < '2020-01-01']
test_data = features[features.index >= '2020-01-01']
print(f"训练: {len(train_data)}, 测试: {len(test_data)}")

# ============================================================
# 环境（20维，与v3一致）
# ============================================================
class TradingEnv(gym.Env):
    POSITION_MAP = {i: i * 0.1 for i in range(10)}
    
    def __init__(self, data, lookback=60):
        super().__init__()
        self.data = data; self.lookback = lookback
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(-np.inf, np.inf, (20,), np.float32)
        self.current_idx = lookback
        self.position = 0.0; self.cash = 100000; self.values = [100000]
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.current_idx = self.lookback
        self.position = 0.0; self.cash = 100000; self.values = [100000]
        return self._get_obs(), {}
    
    def _get_obs(self):
        if self.current_idx >= len(self.data): return np.zeros(20, dtype=np.float32)
        row = self.data.iloc[self.current_idx]
        lb = self.data.iloc[self.current_idx-self.lookback:self.current_idx]
        return np.nan_to_num(np.array([
            row['return'], row['ma5']/row['ma20']-1, lb['return'].mean(),
            lb['return'].std()*np.sqrt(252), row['vix'], row['vix_zscore'],
            lb['vix'].mean(), row['rsi']/100, row['panic_score']/100,
            np.log(row['vol_pct']+1e-8), self.position,
            (self.cash-100000)/100000, self.current_idx/len(self.data),
            lb['return'].skew() if len(lb)>3 else 0,
            lb['return'].kurtosis() if len(lb)>4 else 0, 0, 0,
            row['rolling_te_vol_20']*100, row['soxl_soxx_rel_5']*10, row['soxx_mom_5']*100,
        ], dtype=np.float32))
    
    def step(self, action):
        if self.current_idx >= len(self.data)-1: return self._get_obs(), 0, True, False, {}
        action = int(action.item() if hasattr(action, 'item') else action)
        target_pos = self.POSITION_MAP[action]
        row = self.data.iloc[self.current_idx]
        commission = abs(target_pos - self.position) * 0.001
        self.position = target_pos
        next_price = self.data.iloc[self.current_idx+1]['close']
        price_return = (next_price - row['close']) / row['close']
        portfolio_return = self.position * price_return - commission
        self.cash *= (1 + portfolio_return)
        self.values.append(self.cash)
        
        expert_action = self._expert_policy(row)
        imitation_reward = 1.0 if action == expert_action else -0.5
        reward = imitation_reward + portfolio_return * 10
        
        self.current_idx += 1
        return self._get_obs(), reward, self.current_idx >= len(self.data)-1, False, {}
    
    def _expert_policy(self, row):
        if row['panic_score'] <= -25:
            if row['vix'] > 28: return 9
            elif row['vix'] > 22: return 8
            else: return 6
        if row['rolling_te_vol_20'] > 0.02: return 3
        if row['soxx_mom_5'] > 0.05 and row['soxl_soxx_rel_5'] > 0: return 7
        if row['rsi'] < 65 and row['ma5'] > row['ma20']: return 6
        if row['vix'] < 15:
            if row['rsi'] > 70: return 0
            elif row['rsi'] > 65: return 1
            else: return 3
        return 5

# ============================================================
# 多模型训练 + 回测
# ============================================================
print("\n" + "="*70)
print("训练 4 个模型（各300k步）")
print("="*70)

models_config = {
    'DQN': lambda env: DQN("MlpPolicy", env, learning_rate=1e-3, buffer_size=50000, 
                           learning_starts=1000, batch_size=64, verbose=0),
    'PPO': lambda env: PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, 
                           batch_size=64, n_epochs=10, verbose=0),
    'A2C': lambda env: A2C("MlpPolicy", env, learning_rate=7e-4, n_steps=5, 
                           verbose=0),
}

# SAC 需要 Box action space，用 wrapper
class ContinuousWrapper(gym.ActionWrapper):
    """将 Box(1,) 连续动作映射到 10 档离散"""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
    
    def action(self, action):
        return int(np.clip(action[0] * 10, 0, 9))

results = {}

for name, model_fn in models_config.items():
    print(f"\n--- {name} ---")
    train_env = DummyVecEnv([lambda: TradingEnv(train_data)])
    model = model_fn(train_env)
    model.learn(total_timesteps=300000)
    
    # 回测
    test_env = TradingEnv(test_data)
    obs, _ = test_env.reset()
    values = [100000]; positions = []; vix_levels = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        values.append(test_env.cash)
        positions.append(test_env.position)
        vix_levels.append(test_env.data.iloc[test_env.current_idx]['vix'])
    
    values = np.array(values)
    total_return = (values[-1]-100000)/100000*100
    daily_ret = np.diff(values)/values[:-1]
    sharpe = np.mean(daily_ret)/(np.std(daily_ret)+1e-8)*np.sqrt(252)
    peak = np.maximum.accumulate(values)
    max_dd = np.max((peak-values)/peak)*100
    buy_hold = (test_data['close'].iloc[-1]/test_data['close'].iloc[0]-1)*100
    
    # VIX-仓位相关性分析
    positions_arr = np.array(positions)
    vix_arr = np.array(vix_levels)
    
    # 分VIX区间看平均仓位
    vix_low = positions_arr[vix_arr < 15].mean() if (vix_arr < 15).any() else 0
    vix_mid = positions_arr[(vix_arr >= 15) & (vix_arr < 25)].mean() if ((vix_arr >= 15) & (vix_arr < 25)).any() else 0
    vix_high = positions_arr[vix_arr >= 25].mean() if (vix_arr >= 25).any() else 0
    vix_corr = np.corrcoef(vix_arr, positions_arr)[0,1]
    
    results[name] = {
        'return': total_return, 'sharpe': sharpe, 'max_dd': max_dd,
        'excess': total_return - buy_hold,
        'vix_low_pos': vix_low, 'vix_mid_pos': vix_mid, 'vix_high_pos': vix_high,
        'vix_corr': vix_corr,
        'values': values, 'positions': positions_arr, 'vix': vix_arr,
    }
    
    print(f"   收益: {total_return:.1f}%  Sharpe: {sharpe:.2f}  MaxDD: {max_dd:.1f}%")
    print(f"   VIX<15 仓位: {vix_low:.0%}  |  VIX 15-25: {vix_mid:.0%}  |  VIX>25: {vix_high:.0%}")
    print(f"   VIX-仓位相关系数: {vix_corr:.3f}")

# SAC (连续动作)
print(f"\n--- SAC ---")
train_env = DummyVecEnv([lambda: ContinuousWrapper(TradingEnv(train_data))])
sac_model = SAC("MlpPolicy", train_env, learning_rate=3e-4, buffer_size=50000,
                learning_starts=1000, batch_size=64, verbose=0)
sac_model.learn(total_timesteps=300000)

test_env = TradingEnv(test_data)
obs, _ = test_env.reset()
values = [100000]; positions = []; vix_levels = []
done = False
while not done:
    # SAC 输出连续动作，手动映射
    cont_obs = obs.reshape(1, -1)
    action, _ = sac_model.predict(cont_obs, deterministic=True)
    action_discrete = int(np.clip(float(action.flatten()[0]) * 10, 0, 9))
    obs, reward, done, truncated, info = test_env.step(action_discrete)
    values.append(test_env.cash)
    positions.append(test_env.position)
    vix_levels.append(test_env.data.iloc[test_env.current_idx]['vix'])

values = np.array(values)
total_return = (values[-1]-100000)/100000*100
daily_ret = np.diff(values)/values[:-1]
sharpe = np.mean(daily_ret)/(np.std(daily_ret)+1e-8)*np.sqrt(252)
peak = np.maximum.accumulate(values)
max_dd = np.max((peak-values)/peak)*100

positions_arr = np.array(positions)
vix_arr = np.array(vix_levels)
vix_low = positions_arr[vix_arr < 15].mean() if (vix_arr < 15).any() else 0
vix_mid = positions_arr[(vix_arr >= 15) & (vix_arr < 25)].mean() if ((vix_arr >= 15) & (vix_arr < 25)).any() else 0
vix_high = positions_arr[vix_arr >= 25].mean() if (vix_arr >= 25).any() else 0
vix_corr = np.corrcoef(vix_arr, positions_arr)[0,1]

results['SAC'] = {
    'return': total_return, 'sharpe': sharpe, 'max_dd': max_dd,
    'excess': total_return - buy_hold,
    'vix_low_pos': vix_low, 'vix_mid_pos': vix_mid, 'vix_high_pos': vix_high,
    'vix_corr': vix_corr,
    'values': values, 'positions': positions_arr, 'vix': vix_arr,
}
print(f"   收益: {total_return:.1f}%  Sharpe: {sharpe:.2f}  MaxDD: {max_dd:.1f}%")
print(f"   VIX<15 仓位: {vix_low:.0%}  |  VIX 15-25: {vix_mid:.0%}  |  VIX>25: {vix_high:.0%}")
print(f"   VIX-仓位相关系数: {vix_corr:.3f}")

# ============================================================
# VIX行为分析
# ============================================================
print("\n" + "="*70)
print("VIX 行为分析")
print("="*70)
print(f"\n{'模型':<6} {'收益':>8} {'Sharpe':>8} {'VIX<15仓位':>10} {'VIX15-25':>10} {'VIX>25仓位':>10} {'VIX相关':>8} {'学到?':>6}")
print("-"*78)
for name, r in results.items():
    learned = "✅" if r['vix_high_pos'] > r['vix_low_pos'] + 0.1 else "❌"
    print(f"{name:<6} {r['return']:>7.1f}% {r['sharpe']:>7.2f} {r['vix_low_pos']:>9.0%} {r['vix_mid_pos']:>9.0%} {r['vix_high_pos']:>9.0%} {r['vix_corr']:>7.3f} {learned:>6}")

print(f"\n期望行为: VIX>25仓位高（恐慌买入）, VIX<15仓位低（贪婪卖出）")
print(f"VIX正相关 = VIX高时仓位高 = 学到了恐慌买入逻辑")

# ============================================================
# 绘图
# ============================================================
print("\n绘图...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 资产曲线对比
ax = axes[0,0]
colors = {'DQN': '#2E86AB', 'PPO': '#E8463A', 'SAC': '#28A745', 'A2C': '#FFA500'}
for name, r in results.items():
    n = min(len(test_data.index[60:]), len(r['values']))
    dates = test_data.index[60:60+n]
    ax.plot(dates, r['values'][:n]/1000, label=f"{name} ({r['return']:.0f}%)", 
            linewidth=2, color=colors.get(name, 'gray'))
ax.set_title('多模型资产曲线对比', fontsize=14, fontweight='bold')
ax.set_ylabel('Portfolio ($K)'); ax.legend(); ax.grid(True, alpha=0.3)

# 2. VIX vs 仓位散点图（最佳模型）
best_name = max(results, key=lambda x: results[x]['return'])
best = results[best_name]
ax = axes[0,1]
scatter = ax.scatter(best['vix'], best['positions'], c=best['vix'], cmap='RdYlGn_r', 
                     alpha=0.3, s=5)
ax.set_xlabel('VIX'); ax.set_ylabel('仓位')
ax.set_title(f'{best_name} VIX vs 仓位 (corr={best["vix_corr"]:.3f})', fontsize=12, fontweight='bold')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=20, color='gray', linestyle='--', alpha=0.5)
plt.colorbar(scatter, ax=ax, label='VIX')

# 3. VIX区间仓位柱状图
ax = axes[1,0]
x = np.arange(len(results))
width = 0.25
labels = list(results.keys())
low = [results[k]['vix_low_pos'] for k in labels]
mid = [results[k]['vix_mid_pos'] for k in labels]
high = [results[k]['vix_high_pos'] for k in labels]
ax.bar(x-width, low, width, label='VIX<15', color='green', alpha=0.7)
ax.bar(x, mid, width, label='VIX 15-25', color='orange', alpha=0.7)
ax.bar(x+width, high, width, label='VIX>25', color='red', alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('平均仓位'); ax.set_title('VIX区间平均仓位对比', fontsize=12, fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)

# 4. Sharpe/收益对比
ax = axes[1,1]
for i, name in enumerate(labels):
    r = results[name]
    ax.scatter(r['sharpe'], r['return'], s=200, color=colors.get(name, 'gray'), zorder=5)
    ax.annotate(name, (r['sharpe'], r['return']), fontsize=12, ha='center', va='bottom')
ax.set_xlabel('Sharpe'); ax.set_ylabel('Return (%)')
ax.set_title('Sharpe vs Return', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('qlib_multi_model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ 图表: qlib_multi_model_comparison.png")

# 保存最佳模型
best_name = max(results, key=lambda x: results[x]['return'])
print(f"\n🏆 最佳模型: {best_name} ({results[best_name]['return']:.1f}%)")

# Git
import subprocess
subprocess.run(['git', 'add', 'qlib_multi_model_comparison.py', 'qlib_multi_model_comparison.png'],
               cwd='/Users/rziiiii/.openclaw/workspace')
subprocess.run(['git', 'commit', '-m', 
                f'feat: Multi-model comparison DQN/PPO/SAC/A2C best={best_name}'],
               cwd='/Users/rziiiii/.openclaw/workspace')
subprocess.run(['git', 'push'], cwd='/Users/rziiiii/.openclaw/workspace')
print("✅ Git 已推送")
