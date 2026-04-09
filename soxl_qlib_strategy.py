#!/usr/bin/env python3
"""
SOXL Strategy - 完整 qlib 框架版（修复版）

完整使用 qlib 5 大组件 + RL 模型
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

print("="*70)
print("SOXL Strategy - 完整 qlib 框架版")
print("="*70)

# ============================================================
# 1. qlib.init()
# ============================================================
print("\n1. qlib.init() ...")
qlib.init(provider_uri=QLIB_DATA_PATH)
print("   ✅ qlib.init()")

# ============================================================
# 2. D.features()
# ============================================================
print("\n2. D.features() ...")
qlib_data = D.features(
    instruments=['SOXL'],
    fields=['$close', '$volume'],
    start_time='2010-01-01', end_time='2026-03-31',
)
print(f"   ✅ D.features(): {len(qlib_data)} 条")

csv_df = pd.read_csv(CSV_PATH, parse_dates=['date']).set_index('date')
print(f"   ✅ CSV 补充: {len(csv_df)} 条")

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
# 5. BaseStrategy
# ============================================================
print("\n5. BaseStrategy + 6. SimulatorExecutor + 7. backtest.backtest() ...")

class SOXLRlStrategy(BaseStrategy):
    def __init__(self, rl_model, feature_data, lookback=60, **kwargs):
        super().__init__(**kwargs)
        self.rl_model = rl_model
        self.feature_data = feature_data
        self.lookback = lookback
        self._actions = {}
        self._precompute()
    
    def _precompute(self):
        dates = self.feature_data.index
        for i in range(self.lookback, len(dates)):
            date = dates[i]
            row = self.feature_data.iloc[i]
            lb = self.feature_data.iloc[i-self.lookback:i]
            obs = np.nan_to_num(np.array([
                row['return'], row['ma5']/row['ma20']-1, lb['return'].mean(),
                lb['return'].std()*np.sqrt(252), row['vix'], row['vix_zscore'],
                lb['vix'].mean(), row['rsi']/100, row['panic_score']/100,
                np.log(row['vol_pct']+1e-8), 0.5,
                0, i/len(dates),
                lb['return'].skew() if len(lb)>3 else 0,
                lb['return'].kurtosis() if len(lb)>4 else 0,
                row['vix_ma20']/20-1, row['vix']/row['vix_ma20']-1,
                row['spread']*50, row['spread_trend']*100,
                row['spread_ma5']*50, row['spread_ma20']*50,
                row['spread_ma5']/(abs(row['spread_ma20'])+1e-8)-1,
                row['spread_accel']*200, 0, 0,
            ], dtype=np.float32))
            action, _ = self.rl_model.predict(obs, deterministic=True)
            self._actions[date] = action * 0.1
        print(f"   预计算: {len(self._actions)} 个交易日")
    
    def generate_trade_decision(self, execute_result=None):
        step = self.trade_calendar.get_trade_step()
        step_time = self.trade_calendar.get_step_time()
        trade_date = step_time[0]
        
        # 正确获取持仓信息
        pos = self.trade_position
        cash = pos.get_cash()
        stock = pos.get_stock_amount('SOXL') if pos.check_stock('SOXL') else 0
        
        # 获取当前价格
        try:
            exchange = self.common_infra.get("trade_exchange")
            price = exchange.get_close(trade_date, 'SOXL')
            if price is None or np.isnan(float(price)):
                price = None
        except:
            price = None
        
        if price is None and trade_date in self.feature_data.index:
            price = float(self.feature_data.loc[trade_date, 'close'])
        
        if price is None or price <= 0:
            return TradeDecisionWO([], self)
        
        # 账户总值 = 现金 + 持仓市值
        account_value = cash + stock * price
        
        # RL 目标仓位
        target_pos = self._actions.get(trade_date, 0.5)
        target_value = account_value * target_pos
        target_stock = target_value / price
        delta = target_stock - stock
        
        # 生成订单（整数股）
        orders = []
        if abs(delta) >= 1:  # 至少 1 股
            if delta > 0:
                orders.append(Order(stock_id='SOXL', amount=int(delta), direction=Order.BUY,
                                   start_time=step_time[0], end_time=step_time[1]))
            else:
                orders.append(Order(stock_id='SOXL', amount=int(abs(delta)), direction=Order.SELL,
                                   start_time=step_time[0], end_time=step_time[1]))
        
        return TradeDecisionWO(orders, self)

# 运行 qlib 回测
strategy = SOXLRlStrategy(rl_model=rl_model, feature_data=test_data)

port, indicator = backtest(
    strategy=strategy,
    executor=SimulatorExecutor(time_per_step='day', generate_portfolio_metrics=True),
    start_time='2020-01-02',
    end_time='2026-03-31',
    account=100000,
    benchmark='SOXL',
    exchange_kwargs={
        'freq': 'day',
        'limit_threshold': None,
        'deal_price': 'close',
        'open_cost': 0.001,
        'close_cost': 0.001,
        'min_cost': 0,
    },
)

# 解析结果
port_df = port['1day'] if isinstance(port, dict) and '1day' in port else port
if isinstance(port_df, tuple): port_df = port_df[0]

print(f"\n📊 qlib 官方回测结果:")
print(f"   行数: {len(port_df)}")

total_return = (port_df['account'].iloc[-1] - 100000) / 100000 * 100
daily_ret = port_df['return'].dropna()
sharpe = daily_ret.mean() / (daily_ret.std() + 1e-8) * np.sqrt(252)
peak = port_df['account'].expanding().max()
max_dd = ((peak - port_df['account']) / peak).max() * 100
buy_hold = (test_data['close'].iloc[-1] / test_data['close'].iloc[0] - 1) * 100

print(f"   最终: ${port_df['account'].iloc[-1]:,.2f}")
print(f"   收益: {total_return:.1f}%")
print(f"   Sharpe: {sharpe:.2f}")
print(f"   MaxDD: {max_dd:.1f}%")
print(f"   买入持有: {buy_hold:.1f}%")
print(f"   超额: {total_return - buy_hold:+.1f}%")

# 交易统计
trade_count = (port_df['turnover'] > 0).sum()
print(f"   交易天数: {trade_count}")
print(f"   总换手: {port_df['turnover'].sum():.2f}")
print(f"   总成本: {port_df['cost'].sum():.2f}")

# ============================================================
# 绘图
# ============================================================
print("\n绘图...")
fig, axes = plt.subplots(3, 1, figsize=(16, 14))

ax = axes[0]
ax.plot(port_df.index, port_df['account']/1000, linewidth=2.5, color='#2E86AB', label='A2C Strategy')
ax.plot(port_df.index, port_df['bench']*1000+100000, linewidth=1.5, color='gray', alpha=0.5, label='Benchmark')
ax.set_title(f'SOXL Strategy (qlib framework)\nReturn: {total_return:.1f}% | Sharpe: {sharpe:.2f} | MaxDD: {max_dd:.0f}% | Buy&Hold: {buy_hold:.0f}%',
            fontsize=13, fontweight='bold')
ax.set_ylabel('Account ($K)'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.fill_between(port_df.index, port_df['value'], alpha=0.5, color='#28A745', label='Stock Value')
ax.plot(port_df.index, port_df['cash']/1000, linewidth=1, color='orange', alpha=0.7, label='Cash ($K)')
ax.set_ylabel('Value ($K)'); ax.set_title('Position (Stock Value + Cash)', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[2]
vix_s = test_data['vix'].reindex(port_df.index).ffill()
spread_s = test_data['spread_trend'].reindex(port_df.index).ffill()
ax.plot(port_df.index, vix_s, color='red', alpha=0.7, label='VIX')
ax2 = ax.twinx()
ax2.fill_between(port_df.index, 0, spread_s*100, where=spread_s>0, alpha=0.3, color='green', label='Spread+')
ax2.fill_between(port_df.index, 0, spread_s*100, where=spread_s<0, alpha=0.3, color='red', label='Spread-')
ax2.set_ylabel('Spread Trend', color='purple')
ax.set_ylabel('VIX'); ax.set_xlabel('Date')
ax.legend(loc='upper left'); ax2.legend(loc='upper right')
ax.set_title('VIX & Spread Trend', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig('soxl_qlib_strategy.png', dpi=300, bbox_inches='tight')
print("✅ soxl_qlib_strategy.png")

import subprocess
subprocess.run(['git', 'add', 'soxl_qlib_strategy.py', 'soxl_qlib_strategy.png'],
               cwd='/Users/rziiiii/.openclaw/workspace')
subprocess.run(['git', 'commit', '-m',
                f'feat: SOXL qlib framework fixed - {total_return:.0f}% sharpe {sharpe:.2f}'],
               cwd='/Users/rziiiii/.openclaw/workspace')
subprocess.run(['git', 'push'], cwd='/Users/rziiiii/.openclaw/workspace')
print("✅ Git 已推送")

print(f"""
======================================================================
✅ 完整 qlib 框架:

  1. qlib.init()                    ✅
  2. D.features()                   ✅ SOXL close/volume
  3. BaseStrategy                   ✅ generate_trade_decision()
  4. SimulatorExecutor              ✅ 0.1% 成本
  5. backtest.backtest()            ✅ 1569 步官方回测

  结果: {total_return:.1f}% | Sharpe {sharpe:.2f} | MaxDD {max_dd:.1f}%
======================================================================
""")
