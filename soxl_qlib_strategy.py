#!/usr/bin/env python3
"""
SOXL Strategy - qlib 框架修复版 v3

修复 v2 问题：
- 仓位只有 10%/80% 两档（等于满仓空仓）
- VIX/价差特征没有被利用
- 原因：每次新订单覆盖持仓 + 固定基准 → RL 观察不到真实仓位

修复：
- 用真实持仓（get_stock_amount + get_cash）计算账户价值
- 传入真实仓位给 RL 观察空间
- 增量调仓（delta = target - current）
- 降低交易间隔到2天
- 降低仓位阈值到0.05
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
import warnings; warnings.filterwarnings('ignore')

import qlib
from qlib.data import D
from qlib.strategy.base import BaseStrategy
from qlib.backtest.decision import TradeDecisionWO, Order
from qlib.backtest.executor import SimulatorExecutor
from qlib.backtest import backtest

import pandas as pd
import numpy as np
from pathlib import Path
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

QLIB_DATA_PATH = '/Users/rziiiii/.openclaw/workspace/qlib_soxl_binary'
CSV_PATH = '/Users/rziiiii/.openclaw/workspace/data/soxl_soxx_vix_full.csv'
ACCOUNT = 1000000

print("="*70)
print("SOXL Strategy - qlib v3 (增量调仓 + 真实持仓)")
print("="*70)

# ============================================================
# 1-2. qlib.init + D.features
# ============================================================
print("\n1. qlib.init() ...")
qlib.init(provider_uri=QLIB_DATA_PATH)
print("   ✅ qlib.init()")

print("\n2. D.features() ...")
qlib_data = D.features(instruments=['SOXL'], fields=['$close', '$volume'],
                       start_time='2010-01-01', end_time='2026-03-31')
print(f"   ✅ D.features(): {len(qlib_data)} 条")

csv_df = pd.read_csv(CSV_PATH, parse_dates=['date']).set_index('date')

# ============================================================
# 3. 特征工程
# ============================================================
print("\n3. 特征工程 ...")
close = qlib_data.xs('SOXL', level='instrument')['$close']
volume = qlib_data.xs('SOXL', level='instrument')['$volume']
soxx = csv_df['soxx_close'].reindex(close.index).ffill()
vix = csv_df['vix'].reindex(close.index).ffill()

soxl_ret = close.pct_change()
soxx_ret = soxx.pct_change()
spread = soxl_ret - 3 * soxx_ret
spread_ma5 = spread.rolling(5).mean()
spread_ma20 = spread.rolling(20).mean()
spread_trend = spread_ma5 - spread_ma20
spread_accel = spread_trend.diff(5)

ma5 = close.rolling(5).mean()
ma20 = close.rolling(20).mean()
volatility = soxl_ret.rolling(20).std() * np.sqrt(252)
vix_ma20 = vix.rolling(20).mean()
vix_zscore = (vix - vix.rolling(252).mean()) / (vix.rolling(252).std() + 1e-8)
delta = close.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss_d = (-delta.where(delta < 0, 0)).rolling(14).mean()
rsi = 100 - (100 / (1 + gain / (loss_d + 1e-8)))
panic_score = (vix_zscore * 40 + vix.rolling(20).rank(pct=True) * 60 +
               np.where(soxl_ret < 0, np.minimum(soxl_ret / -0.05, 1) * (-50), 0))
vol_pct = volume / volume.rolling(20).mean()

features = pd.DataFrame({
    'close': close, 'volume': volume, 'return': soxl_ret,
    'ma5': ma5, 'ma20': ma20, 'volatility': volatility,
    'vix': vix, 'vix_ma20': vix_ma20, 'vix_zscore': vix_zscore,
    'rsi': rsi, 'panic_score': panic_score, 'vol_pct': vol_pct,
    'spread': spread, 'spread_trend': spread_trend,
    'spread_ma5': spread_ma5, 'spread_ma20': spread_ma20,
    'spread_accel': spread_accel,
}).dropna()
print(f"   ✅ {len(features)} 条, {len(features.columns)} 列")

# ============================================================
# 4. RL 训练
# ============================================================
print("\n4. RL 训练 ...")
train_data = features[features.index < '2020-01-01']
test_data = features[features.index >= '2020-01-01']

class SOXLTradingEnv(gym.Env):
    POSITION_MAP = {i: i * 0.1 for i in range(10)}
    def __init__(self, data, lookback=60):
        super().__init__()
        self.data = data; self.lookback = lookback
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(-np.inf, np.inf, (25,), np.float32)
        self.current_idx = lookback; self.position = 0.0
        self.cash = 100000; self.values = [100000]
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.current_idx = self.lookback; self.position = 0.0
        self.cash = 100000; self.values = [100000]
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
            row['spread_accel']*200, 0, 0,
        ], dtype=np.float32))
    def step(self, action):
        if self.current_idx >= len(self.data)-1: return self._get_obs(), 0, True, False, {}
        action = int(action.item() if hasattr(action, 'item') else action)
        target_pos = self.POSITION_MAP[action]
        row = self.data.iloc[self.current_idx]
        turnover = abs(target_pos - self.position)
        self.position = target_pos
        next_price = self.data.iloc[self.current_idx+1]['close']
        pr = (next_price - row['close']) / row['close']
        port_ret = self.position * pr - turnover * 0.001
        self.cash *= (1 + port_ret); self.values.append(self.cash)
        reward = port_ret * 10
        vix_ratio = row['vix'] / row['vix_ma20']
        if row['vix'] > 30: reward += 5.0 if target_pos >= 0.7 else (-4.0 if target_pos <= 0.3 else 0)
        elif row['vix'] > 25: reward += 3.0 if target_pos >= 0.6 else (-3.0 if target_pos <= 0.3 else 0)
        elif vix_ratio > 1.2: reward += 1.5 if target_pos >= 0.5 else -1.0
        elif row['vix'] < 15: reward += 3.0 if target_pos <= 0.3 else (-3.0 if target_pos >= 0.6 else 0)
        elif vix_ratio < 0.85: reward += 1.5 if target_pos <= 0.4 else -1.0
        st = row['spread_trend']
        if st > 0.003: reward += 5.0 if target_pos >= 0.6 else (-4.0 if target_pos <= 0.3 else -1.0)
        elif st > 0.001: reward += 3.0 if target_pos >= 0.5 else 0
        elif st < -0.003: reward += 5.0 if target_pos <= 0.3 else (-4.0 if target_pos >= 0.6 else -1.0)
        elif st < -0.001: reward += 3.0 if target_pos <= 0.4 else 0
        sa = row['spread_accel']
        if sa > 0.001: reward += 3.0 if target_pos >= 0.6 else 0
        elif sa < -0.001: reward += 3.0 if target_pos <= 0.3 else 0
        if row['vix'] > 25 and st > 0: reward += 5.0 if target_pos >= 0.8 else 0
        if row['vix'] < 15 and st < 0: reward += 5.0 if target_pos <= 0.2 else 0
        reward -= turnover * 2.0
        self.current_idx += 1
        return self._get_obs(), reward, self.current_idx >= len(self.data)-1, False, {}

print(f"   训练: {len(train_data)}, 测试: {len(test_data)}")
train_env = DummyVecEnv([lambda: SOXLTradingEnv(train_data)])
rl_model = A2C("MlpPolicy", train_env, learning_rate=7e-4, n_steps=5, verbose=0)
rl_model.learn(total_timesteps=500000)
print("   ✅ A2C 训练完成")

# ============================================================
# 5-7. BaseStrategy + SimulatorExecutor + backtest
# ============================================================
print("\n5-7. qlib 官方回测（增量调仓版）...")

class SOXLRlStrategyV3(BaseStrategy):
    """增量调仓 + 真实持仓反馈"""
    def __init__(self, rl_model, df, lookback=60, **kwargs):
        super().__init__(**kwargs)
        self.rl_model = rl_model
        self.df = df
        self.lookback = lookback
        self.last_trade_step = -10
        self.trade_count = 0
        self.position_history = []
        self.vix_at_trade = []
        self.spread_at_trade = []
    
    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        total_steps = self.trade_calendar.get_trade_len()
        
        if trade_step >= total_steps - 2:
            return TradeDecisionWO([], self)
        
        trade_date = self.trade_calendar.get_step_time(trade_step)[0]
        next_date = self.trade_calendar.get_step_time(trade_step + 1)[0]
        
        if trade_date not in self.df.index:
            return TradeDecisionWO([], self)
        
        idx = self.df.index.get_loc(trade_date)
        if idx < self.lookback:
            return TradeDecisionWO([], self)
        
        row = self.df.iloc[idx]
        lb = self.df.iloc[idx-self.lookback:idx]
        
        # === 关键：获取真实持仓 ===
        pos = self.trade_position
        cash = pos.get_cash()
        stock = pos.get_stock_amount('SOXL') if pos.check_stock('SOXL') else 0
        price = row['close']
        account_value = cash + stock * price
        current_position = (stock * price) / account_value if account_value > 0 else 0
        
        # RL 观察（传入真实 current_position）
        obs = np.nan_to_num(np.array([
            row['return'], row['ma5']/row['ma20']-1, lb['return'].mean(),
            lb['return'].std()*np.sqrt(252), row['vix'], row['vix_zscore'],
            lb['vix'].mean(), row['rsi']/100, row['panic_score']/100,
            np.log(row['vol_pct']+1e-8), current_position,  # ← 真实仓位!
            (account_value - ACCOUNT) / ACCOUNT,             # ← 真实收益!
            idx/len(self.df),
            lb['return'].skew() if len(lb)>3 else 0,
            lb['return'].kurtosis() if len(lb)>4 else 0,
            row['vix_ma20']/20-1, row['vix']/row['vix_ma20']-1,
            row['spread']*50, row['spread_trend']*100,
            row['spread_ma5']*50, row['spread_ma20']*50,
            row['spread_ma5']/(abs(row['spread_ma20'])+1e-8)-1,
            row['spread_accel']*200, 0, 0,
        ], dtype=np.float32))
        
        action, _ = self.rl_model.predict(obs, deterministic=True)
        target_position = int(action.item() if hasattr(action, 'item') else action) * 0.1
        
        # 仓位变化太小 → 不交易
        if abs(target_position - current_position) < 0.05:
            self.position_history.append(current_position)
            return TradeDecisionWO([], self)
        
        # 交易间隔 ≥ 2天
        if trade_step - self.last_trade_step < 2:
            self.position_history.append(current_position)
            return TradeDecisionWO([], self)
        
        # === 增量调仓 ===
        target_value = account_value * target_position
        target_shares = int(target_value / price / 100) * 100
        delta_shares = target_shares - int(stock)
        
        orders = []
        if delta_shares >= 100:
            orders.append(Order(stock_id='SOXL', amount=delta_shares, direction=Order.BUY,
                               start_time=trade_date, end_time=next_date))
        elif delta_shares <= -100:
            orders.append(Order(stock_id='SOXL', amount=abs(delta_shares), direction=Order.SELL,
                               start_time=trade_date, end_time=next_date))
        
        if orders:
            self.last_trade_step = trade_step
            self.trade_count += 1
            self.vix_at_trade.append(row['vix'])
            self.spread_at_trade.append(row['spread_trend'])
        
        self.position_history.append(target_position if orders else current_position)
        return TradeDecisionWO(orders, self)

strategy = SOXLRlStrategyV3(rl_model=rl_model, df=test_data)

port, indicator = backtest(
    strategy=strategy,
    executor=SimulatorExecutor(time_per_step='day', generate_portfolio_metrics=True),
    start_time='2020-01-02', end_time='2026-03-31',
    account=ACCOUNT, benchmark='SOXL',
    exchange_kwargs={
        'freq': 'day', 'deal_price': 'close',
        'open_cost': 0.001, 'close_cost': 0.001, 'min_cost': 5,
    },
)

# 解析结果
pdf = port['1day'] if isinstance(port, dict) and '1day' in port else port
if isinstance(pdf, tuple): pdf = pdf[0]

final_value = pdf['account'].iloc[-1]
total_return = (final_value / ACCOUNT - 1) * 100
daily_ret = np.diff(pdf['account'].values) / pdf['account'].values[:-1]
sharpe = np.mean(daily_ret) / (np.std(daily_ret) + 1e-8) * np.sqrt(252)
peak = np.maximum.accumulate(pdf['account'].values)
max_dd = np.max((peak - pdf['account'].values) / peak) * 100
buy_hold = (test_data['close'].iloc[-1] / test_data['close'].iloc[0] - 1) * 100

print(f"\n📊 qlib 官方回测结果 (v3):")
print(f"   最终: ${final_value:,.0f}")
print(f"   收益: {total_return:.1f}%")
print(f"   Sharpe: {sharpe:.2f}")
print(f"   MaxDD: {max_dd:.1f}%")
print(f"   买入持有: {buy_hold:.1f}%")
print(f"   超额: {total_return - buy_hold:+.1f}%")
print(f"   交易次数: {strategy.trade_count}")

# 仓位分布
if strategy.position_history:
    pos_arr = np.array(strategy.position_history)
    print(f"\n   仓位分布:")
    for i in range(10):
        p = i * 0.1
        c = (pos_arr == p).sum()
        if c > 0: print(f"     {p:.0%}: {c:4d} ({c/len(pos_arr):.1%})")

# VIX/Spread 行为分析
if strategy.vix_at_trade:
    vix_arr = np.array(strategy.vix_at_trade)
    spread_arr = np.array(strategy.spread_at_trade)
    pos_at_trade = np.array(strategy.position_history[-len(vix_arr):])
    
    # 按 VIX 分组看仓位
    vix_low_mask = vix_arr < 15
    vix_mid_mask = (vix_arr >= 15) & (vix_arr < 25)
    vix_high_mask = vix_arr >= 25
    
    print(f"\n   VIX 行为分析:")
    if vix_low_mask.any(): print(f"     VIX<15: 仓位={pos_at_trade[vix_low_mask].mean():.0%} ({vix_low_mask.sum()}次)")
    if vix_mid_mask.any(): print(f"     VIX15-25: 仓位={pos_at_trade[vix_mid_mask].mean():.0%} ({vix_mid_mask.sum()}次)")
    if vix_high_mask.any(): print(f"     VIX>25: 仓位={pos_at_trade[vix_high_mask].mean():.0%} ({vix_high_mask.sum()}次)")
    
    # 按价差分组
    spread_exp = spread_arr > 0
    spread_shr = spread_arr < 0
    print(f"\n   价差行为分析:")
    if spread_exp.any(): print(f"     价差增大: 仓位={pos_at_trade[spread_exp].mean():.0%} ({spread_exp.sum()}次)")
    if spread_shr.any(): print(f"     价差缩小: 仓位={pos_at_trade[spread_shr].mean():.0%} ({spread_shr.sum()}次)")

# 绘图
print("\n绘图...")
fig, axes = plt.subplots(4, 1, figsize=(16, 18))

ax = axes[0]
ax.plot(pdf.index, pdf['account']/1000, linewidth=2.5, color='#2E86AB', label='A2C v3 (qlib)')
ax.fill_between(pdf.index, ACCOUNT/1000, pdf['account']/1000, alpha=0.15, color='#2E86AB')
ax.set_title(f'SOXL Strategy v3 (qlib + Incremental) | {total_return:.1f}% | Sharpe {sharpe:.2f} | MaxDD {max_dd:.0f}%',
            fontsize=14, fontweight='bold')
ax.set_ylabel('Account ($K)'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
if strategy.position_history:
    pos_dates = pdf.index[:len(strategy.position_history)]
    ax.fill_between(pos_dates, strategy.position_history, alpha=0.5, color='#28A745', label='Position')
    ax.set_ylabel('Position'); ax.set_ylim(0, 1.1)
    ax.set_title('Position (should vary with VIX & spread)', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[2]
vix_s = test_data['vix'].reindex(pdf.index).ffill()
ax.plot(pdf.index, vix_s, color='red', alpha=0.7, label='VIX')
ax.axhline(y=25, color='orange', linestyle='--', alpha=0.5, label='VIX=25')
ax.axhline(y=15, color='green', linestyle='--', alpha=0.5, label='VIX=15')
ax.set_ylabel('VIX'); ax.legend(); ax.grid(True, alpha=0.3)
ax.set_title('VIX (high → should increase position)', fontweight='bold')

ax = axes[3]
spread_s = test_data['spread_trend'].reindex(pdf.index).ffill()
ax.fill_between(pdf.index, 0, spread_s*100, where=spread_s>0, alpha=0.3, color='green', label='Spread Expanding')
ax.fill_between(pdf.index, 0, spread_s*100, where=spread_s<0, alpha=0.3, color='red', label='Spread Shrinking')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_ylabel('Spread Trend'); ax.set_xlabel('Date'); ax.legend(); ax.grid(True, alpha=0.3)
ax.set_title('SOXL-SOXX Spread Trend', fontweight='bold')
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig('soxl_qlib_strategy.png', dpi=300, bbox_inches='tight')
print("✅ soxl_qlib_strategy.png")

import subprocess
subprocess.run(['git', 'add', 'soxl_qlib_strategy.py', 'soxl_qlib_strategy.png'],
               cwd='/Users/rziiiii/.openclaw/workspace')
subprocess.run(['git', 'commit', '-m',
                f'feat: SOXL qlib v3 incremental + real position - {total_return:.0f}% sharpe {sharpe:.2f}'],
               cwd='/Users/rziiiii/.openclaw/workspace')
subprocess.run(['git', 'push'], cwd='/Users/rziiiii/.openclaw/workspace')
print("✅ Git 已推送")
