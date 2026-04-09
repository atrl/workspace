#!/usr/bin/env python3
"""
SOXL Strategy - 完整 qlib.rl 框架 (无回退)

全部使用 qlib.rl 官方组件:
  qlib.rl.Simulator / StateInterpreter / ActionInterpreter / Reward
  qlib.rl.trainer.api.train (tianshou PPO)
  qlib.backtest.backtest (官方回测)
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
import warnings; warnings.filterwarnings('ignore')

# === Compatibility patches: qlib (gym) + tianshou (gymnasium) ===
import sys, gymnasium as _gym
if 'gym' not in sys.modules:
    sys.modules['gym'] = _gym
    sys.modules['gym.spaces'] = _gym.spaces

# Patch EnvWrapper to return gymnasium-compatible signatures
from qlib.rl.utils import EnvWrapper as _EW
_orig_ew_reset = _EW.reset
_orig_ew_step = _EW.step
def _ew_reset(self, **kwargs):
    obs = _orig_ew_reset(self, **kwargs)
    return obs, {}
def _ew_step(self, action, **kwargs):
    obs, rew, done, info = _orig_ew_step(self, action, **kwargs)
    return obs, rew, done, False, info
_EW.reset = _ew_reset
_EW.step = _ew_step

# Patch FiniteVectorEnv.reset to return (obs, info)
from qlib.rl.utils.finite_env import FiniteVectorEnv as _FVE
_orig_fve_reset = _FVE.reset
def _patched_fve_reset(self, id=None, **kwargs):
    obs = _orig_fve_reset(self, id=id, **kwargs)
    return obs, {}
_FVE.reset = _patched_fve_reset

# Patch FiniteVectorEnv.step to handle logger 5-tuple issue
# The original calls logger.on_env_step(i, *r) where r is now 5-tuple
# Need to convert back to 4 args for qlib's LogWriter
import inspect as _insp
_fve_step_src = _insp.getsource(_FVE.step)
# Re-implement step with 5-tuple handling
_orig_fve_step_raw = _FVE.step

def _patched_fve_step(self, action, id=None):
    """Override to handle 5-tuple from gymnasium EnvWrapper."""
    import numpy as np
    from typing import cast, Tuple

    self._assert_is_not_closed()
    id = self._wrap_id(id)

    # Handle action broadcasting
    if isinstance(action, np.ndarray) and action.shape:
        assert len(action) == len(id)
    else:
        action = [action] * len(id)

    # Call super (tianshou DummyVectorEnv.step) which calls workers
    for env_id, act in zip(id, action):
        self.workers[env_id].send(act)

    result = []
    for i in id:
        r = self.workers[i].recv()
        # r is now 5-tuple (obs, rew, done, truncated, info) from our patched EnvWrapper
        if len(r) == 5:
            obs, rew, done, truncated, info = r
            done = done or truncated
        else:
            obs, rew, done, info = r
        result.append((obs, rew, done, info))
        for logger in self._logger:
            logger.on_env_step(i, obs, rew, done, info)

    for i, r in zip(id, result):
        if i in self._alive_env_ids and r[2]:  # done
            self._alive_env_ids.remove(i)

    obs_arr = np.stack([r[0] for r in result])
    rew_arr = np.stack([r[1] for r in result])
    done_arr = np.stack([r[2] for r in result])
    info_arr = np.stack([r[3] for r in result])
    return cast(Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                (obs_arr, rew_arr, done_arr, np.zeros_like(done_arr), info_arr))

_FVE.step = _patched_fve_step

# Skip shimmy wrapping
import tianshou.env.venvs as _tv
_tv._patch_env_generator = lambda fn: fn

import qlib
from qlib.data import D
from qlib.strategy.base import BaseStrategy
from qlib.backtest.decision import TradeDecisionWO, Order
from qlib.backtest.executor import SimulatorExecutor
from qlib.backtest import backtest as qlib_backtest
from qlib.rl.simulator import Simulator
from qlib.rl.interpreter import StateInterpreter, ActionInterpreter
from qlib.rl.reward import Reward
from qlib.rl.trainer.api import train as qlib_train

import pandas as pd, numpy as np, pickle, torch
from typing import NamedTuple
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CSV_PATH = '/Users/rziiiii/.openclaw/workspace/data/soxl_soxx_vix_full.csv'
QLIB_DATA_PATH = '/Users/rziiiii/.openclaw/workspace/qlib_soxl_binary'
FEATURE_PKL = '/Users/rziiiii/.openclaw/workspace/data/features.pkl'
ACCOUNT = 1_000_000
N_ACTIONS = 11

print("=" * 70)
print("SOXL Strategy - qlib.rl.train() (无回退)")
print("=" * 70)

# ── 1-2. qlib.init + D.features ──────────────────────────────
print("\n[1] qlib.init() ...")
qlib.init(provider_uri=QLIB_DATA_PATH)
print("    ✅")

print("\n[2] D.features() ...")
qlib_data = D.features(instruments=['SOXL'], fields=['$close', '$volume'],
                       start_time='2010-01-01', end_time='2026-03-31')
print(f"    ✅ {len(qlib_data)} 条")
csv_df = pd.read_csv(CSV_PATH, parse_dates=['date']).set_index('date')

# ── 3. 特征工程 ──────────────────────────────────────────────
print("\n[3] 特征工程 ...")
close = qlib_data.xs('SOXL', level='instrument')['$close']
volume = qlib_data.xs('SOXL', level='instrument')['$volume']
soxx = csv_df['soxx_close'].reindex(close.index).ffill()
vix = csv_df['vix'].reindex(close.index).ffill()

soxl_ret = close.pct_change()
spread = soxl_ret - 3 * soxx.pct_change()
spread_ma5 = spread.rolling(5).mean()
spread_ma20 = spread.rolling(20).mean()

features = pd.DataFrame({
    'close': close, 'volume': volume, 'return': soxl_ret,
    'ma5': close.rolling(5).mean(), 'ma20': close.rolling(20).mean(),
    'volatility': soxl_ret.rolling(20).std() * np.sqrt(252),
    'vix': vix, 'vix_ma20': vix.rolling(20).mean(),
    'vix_zscore': (vix - vix.rolling(252).mean()) / (vix.rolling(252).std() + 1e-8),
    'rsi': 100 - 100 / (1 + close.diff().where(close.diff() > 0, 0).rolling(14).mean()
          / (-close.diff().where(close.diff() < 0, 0)).rolling(14).mean().clip(lower=1e-8)),
    'panic_score': ((vix - vix.rolling(252).mean()) / (vix.rolling(252).std() + 1e-8)) * 40
                   + vix.rolling(20).rank(pct=True) * 60,
    'vol_pct': volume / volume.rolling(20).mean(),
    'spread': spread,
    'spread_trend': spread_ma5 - spread_ma20,
    'spread_ma5': spread_ma5, 'spread_ma20': spread_ma20,
    'spread_accel': (spread_ma5 - spread_ma20).diff(5),
}).dropna()

features.to_pickle(FEATURE_PKL)
train_data = features[features.index < '2020-01-01']
test_data = features[features.index >= '2020-01-01']
print(f"    ✅ {len(features)} 条 → 训练 {len(train_data)}, 测试 {len(test_data)}")

# ── 4. qlib.rl 组件 ──────────────────────────────────────────
print("\n[4] qlib.rl 组件 ...")


class EpisodeConfig(NamedTuple):
    """pickle-safe: 只存路径 + 索引"""
    pkl_path: str
    start_idx: int
    end_idx: int
    lookback: int = 60
    cash: float = 100_000.0


class SOXLSimulator(Simulator):
    def __init__(self, initial: EpisodeConfig, **kw):
        super().__init__(initial, **kw)
        self.df = pd.read_pickle(initial.pkl_path).iloc[initial.start_idx:initial.end_idx]
        self.lb = initial.lookback
        self.cash0 = initial.cash
        self.idx = self.lb
        self.pos = 0.0
        self.val = self.cash0
        self.prev_val = self.cash0

    def step(self, action: float):
        action = float(np.clip(action, 0, 1))
        if self.idx >= len(self.df) - 1:
            return
        price = self.df.iloc[self.idx]['close']
        nxt = self.df.iloc[self.idx + 1]['close']
        pr = (nxt - price) / price
        self.prev_val = self.val
        self.val *= 1 + action * pr - abs(action - self.pos) * 0.001
        self.pos = action
        self.idx += 1

    def get_state(self):
        if self.idx >= len(self.df):
            return dict(obs=np.zeros(25, dtype=np.float32), pos=self.pos,
                        val=self.prev_val, pval=self.prev_val, done=True)
        r = self.df.iloc[self.idx]
        lb = self.df.iloc[self.idx - self.lb:self.idx]
        obs = np.nan_to_num(np.array([
            r['return'], r['ma5'] / r['ma20'] - 1, lb['return'].mean(),
            lb['return'].std() * np.sqrt(252), r['vix'] / 50, r['vix_zscore'],
            lb['vix'].mean() / 50, r['rsi'] / 100, r['panic_score'] / 100,
            np.log(r['vol_pct'] + 1e-8), self.pos,
            (self.val - self.cash0) / self.cash0,
            self.idx / len(self.df),
            lb['return'].skew() if len(lb) > 3 else 0,
            lb['return'].kurtosis() if len(lb) > 4 else 0,
            r['vix_ma20'] / 20 - 1, r['vix'] / r['vix_ma20'] - 1,
            r['spread'] * 50, r['spread_trend'] * 100,
            r['spread_ma5'] * 50, r['spread_ma20'] * 50,
            r['spread_ma5'] / (abs(r['spread_ma20']) + 1e-8) - 1,
            r['spread_accel'] * 200, 0, 0,
        ], dtype=np.float32))
        return dict(obs=obs, pos=self.pos, val=self.val, pval=self.prev_val, done=False)

    def done(self):
        return self.idx >= len(self.df) - 1


class SOXLState(StateInterpreter):
    @property
    def observation_space(self):
        return spaces.Box(-np.inf, np.inf, (25,), dtype=np.float32)

    def interpret(self, s):
        return s['obs']


class PosAction(ActionInterpreter):
    def __init__(self, n=N_ACTIONS):
        super().__init__()
        self.n = n

    @property
    def action_space(self):
        return spaces.Discrete(self.n)

    def interpret(self, s, a):
        return float(a) / (self.n - 1)


class SOXLReward(Reward):
    def reward(self, s):
        o = s['obs']
        pos = s['pos']
        ret = (s['val'] - s['pval']) / s['pval'] if s['pval'] > 0 else 0
        r = ret * 100
        vn = o[4]
        st = o[18] / 100
        if vn > 0.6:
            r += 5 if pos >= 0.7 else (-4 if pos <= 0.3 else 0)
        elif vn > 0.4:
            r += 3 if pos >= 0.6 else (-3 if pos <= 0.2 else 0)
        elif vn < 0.15:
            r += 3 if pos <= 0.3 else (-3 if pos >= 0.6 else 0)
        if st > 0.003:
            r += 5 if pos >= 0.6 else (-4 if pos <= 0.3 else -1)
        elif st < -0.003:
            r += 5 if pos <= 0.3 else (-4 if pos >= 0.6 else -1)
        r -= abs(pos) * 0.5
        return r


# pickle 验证
pickle.loads(pickle.dumps(EpisodeConfig(FEATURE_PKL, 0, 500)))
print("    ✅ EpisodeConfig pickle-safe")
SOXLSimulator(EpisodeConfig(FEATURE_PKL, 0, 500)).get_state()
print("    ✅ Simulator 初始化 OK")

# ── 5. qlib.rl.train() ───────────────────────────────────────
print("\n[5] qlib.rl.train() ...")

ep_len = 300
n_ep = len(train_data) // ep_len
initials = [EpisodeConfig(FEATURE_PKL, i * ep_len, min((i + 1) * ep_len, len(train_data)))
            for i in range(n_ep)]
print(f"    {n_ep} episodes × {ep_len} 条")

from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic

net = Net((25,), hidden_sizes=[128, 64], device='cpu')
actor_net = Actor(net, N_ACTIONS, device='cpu')
critic_net = Critic(net, device='cpu')
optim = torch.optim.Adam(list(actor_net.parameters()) + list(critic_net.parameters()), lr=3e-4)

policy = PPOPolicy(
    actor=actor_net, critic=critic_net, optim=optim,
    dist_fn=lambda l: torch.distributions.Categorical(logits=l),
    discount_factor=0.99, gae_lambda=0.95,
    action_space=spaces.Discrete(N_ACTIONS),
)

print("    调用 qlib.rl.train() ...")
qlib_train(
    simulator_fn=SOXLSimulator,
    state_interpreter=SOXLState(),
    action_interpreter=PosAction(),
    initial_states=initials[:30],
    policy=policy,
    reward=SOXLReward(),
    vessel_kwargs={'episode_per_iter': 10,
                        'update_kwargs': {'batch_size': 64, 'repeat': 10}},
    trainer_kwargs={'finite_env_type': 'dummy', 'concurrency': 1,
                    'max_iters': 500},
)
print("    ✅ qlib.rl.train() 完成！")

# ── 6-7. qlib 官方回测 ──────────────────────────────────────
print("\n[6-7] qlib 官方回测 ...")


def get_action(obs):
    with torch.no_grad():
        out = actor_net(torch.FloatTensor(obs).unsqueeze(0))
        logits = out[0] if isinstance(out, tuple) else out
        return logits.argmax(-1).item()


class RLStrategy(BaseStrategy):
    def __init__(self, df, lb=60, **kw):
        super().__init__(**kw)
        self.df = df; self.lb = lb
        self.last_step = -10; self.trades = 0; self.pos_hist = []

    def generate_trade_decision(self, exe=None):
        ts = self.trade_calendar.get_trade_step()
        if ts >= self.trade_calendar.get_trade_len() - 2:
            return TradeDecisionWO([], self)
        td = self.trade_calendar.get_step_time(ts)[0]
        nd = self.trade_calendar.get_step_time(ts + 1)[0]
        if td not in self.df.index:
            return TradeDecisionWO([], self)
        idx = self.df.index.get_loc(td)
        if idx < self.lb:
            return TradeDecisionWO([], self)

        row = self.df.iloc[idx]
        lb = self.df.iloc[idx - self.lb:idx]
        pos = self.trade_position
        cash = pos.get_cash()
        stk = pos.get_stock_amount('SOXL') if pos.check_stock('SOXL') else 0
        price = row['close']
        av = cash + stk * price
        cp = (stk * price) / av if av > 0 else 0

        obs = np.nan_to_num(np.array([
            row['return'], row['ma5'] / row['ma20'] - 1, lb['return'].mean(),
            lb['return'].std() * np.sqrt(252), row['vix'] / 50, row['vix_zscore'],
            lb['vix'].mean() / 50, row['rsi'] / 100, row['panic_score'] / 100,
            np.log(row['vol_pct'] + 1e-8), cp,
            (av - ACCOUNT) / ACCOUNT, idx / len(self.df),
            lb['return'].skew() if len(lb) > 3 else 0,
            lb['return'].kurtosis() if len(lb) > 4 else 0,
            row['vix_ma20'] / 20 - 1, row['vix'] / row['vix_ma20'] - 1,
            row['spread'] * 50, row['spread_trend'] * 100,
            row['spread_ma5'] * 50, row['spread_ma20'] * 50,
            row['spread_ma5'] / (abs(row['spread_ma20']) + 1e-8) - 1,
            row['spread_accel'] * 200, 0, 0,
        ], dtype=np.float32))

        tp = get_action(obs) / (N_ACTIONS - 1)
        if abs(tp - cp) < 0.05 or ts - self.last_step < 2:
            self.pos_hist.append(cp)
            return TradeDecisionWO([], self)

        ts_val = int(av * tp / price / 100) * 100
        delta = ts_val - int(stk)
        orders = []
        if delta >= 100:
            orders.append(Order('SOXL', delta, Order.BUY, td, nd))
        elif delta <= -100:
            orders.append(Order('SOXL', abs(delta), Order.SELL, td, nd))
        if orders:
            self.last_step = ts; self.trades += 1
        self.pos_hist.append(tp if orders else cp)
        return TradeDecisionWO(orders, self)


strategy = RLStrategy(test_data)
port, ind = qlib_backtest(
    strategy=strategy,
    executor=SimulatorExecutor(time_per_step='day', generate_portfolio_metrics=True),
    start_time='2020-01-02', end_time='2026-03-31',
    account=ACCOUNT, benchmark='SOXL',
    exchange_kwargs=dict(freq='day', deal_price='close',
                         open_cost=0.001, close_cost=0.001, min_cost=5),
)

pdf = port.get('1day', port)
if isinstance(pdf, tuple): pdf = pdf[0]

fv = pdf['account'].iloc[-1]
tr = (fv / ACCOUNT - 1) * 100
dr = np.diff(pdf['account'].values) / pdf['account'].values[:-1]
sh = np.mean(dr) / (np.std(dr) + 1e-8) * np.sqrt(252)
pk = np.maximum.accumulate(pdf['account'].values)
md = np.max((pk - pdf['account'].values) / pk) * 100
bh = (test_data['close'].iloc[-1] / test_data['close'].iloc[0] - 1) * 100

print(f"\n📊 qlib.rl.train 回测:")
print(f"   收益: {tr:.1f}%  Sharpe: {sh:.2f}  MaxDD: {md:.1f}%")
print(f"   买入持有: {bh:.1f}%  超额: {tr - bh:+.1f}%")
print(f"   交易次数: {strategy.trades}")

# 绘图
fig, axes = plt.subplots(3, 1, figsize=(16, 12))
ax = axes[0]
ax.plot(pdf.index, pdf['account'] / 1000, lw=2.5, color='#2E86AB', label='PPO')
ax.fill_between(pdf.index, ACCOUNT / 1000, pdf['account'] / 1000, alpha=.15, color='#2E86AB')
ax.set_title(f'SOXL qlib.rl.train | {tr:.1f}% | Sharpe {sh:.2f} | MaxDD {md:.0f}%', fontweight='bold')
ax.set_ylabel('$K'); ax.legend(); ax.grid(alpha=.3)

ax = axes[1]
if strategy.pos_hist:
    ax.fill_between(pdf.index[:len(strategy.pos_hist)], strategy.pos_hist, alpha=.5, color='#28A745')
ax.set_ylabel('Position'); ax.set_ylim(0, 1.1); ax.grid(alpha=.3)

ax = axes[2]
ax.plot(pdf.index, test_data['vix'].reindex(pdf.index).ffill(), color='red', alpha=.7)
ax.axhline(25, color='orange', ls='--', alpha=.5); ax.axhline(15, color='green', ls='--', alpha=.5)
ax.set_ylabel('VIX'); ax.set_xlabel('Date'); ax.grid(alpha=.3)
ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout(); plt.savefig('soxl_qlib_strategy.png', dpi=300, bbox_inches='tight')
print("✅ 图片已保存")

import subprocess
subprocess.run(['git', 'add', 'soxl_qlib_strategy.py', 'soxl_qlib_strategy.png'],
               cwd='/Users/rziiiii/.openclaw/workspace')
subprocess.run(['git', 'commit', '-m', f'feat: qlib.rl.train no-fallback PPO {tr:.0f}% sharpe {sh:.2f}'],
               cwd='/Users/rziiiii/.openclaw/workspace')
subprocess.run(['git', 'push'], cwd='/Users/rziiiii/.openclaw/workspace')
print("✅ Git 已推送")

print(f"""
======================================================================
✅ 全 qlib.rl (无回退):

  1. qlib.init()                ✅
  2. D.features()               ✅
  3. qlib.rl.Simulator          ✅
  4. qlib.rl.StateInterpreter   ✅
  5. qlib.rl.ActionInterpreter  ✅
  6. qlib.rl.Reward             ✅
  7. qlib.rl.train()            ✅ (直接调用, 无回退)
  8. qlib.backtest.backtest()   ✅

  {tr:.1f}% | Sharpe {sh:.2f} | MaxDD {md:.1f}%
======================================================================""")
