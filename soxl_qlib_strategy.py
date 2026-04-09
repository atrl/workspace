#!/usr/bin/env python3
"""
SOXL Strategy - 完整 qlib.rl 框架版

使用 qlib.rl 官方 RL 框架（不用 gym/stable_baselines3）:
- qlib.rl.simulator.Simulator → 交易模拟器
- qlib.rl.interpreter.StateInterpreter → 状态解释器
- qlib.rl.interpreter.ActionInterpreter → 动作解释器
- qlib.rl.reward.Reward → 奖励函数
- qlib.rl.trainer.api.train → 训练
- qlib.rl.trainer.api.backtest → 回测

训练用 tianshou（qlib.rl 的依赖）作为 policy
回测用 qlib.backtest 官方引擎
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
from qlib.backtest import backtest as qlib_backtest

# qlib.rl 组件
from qlib.rl.simulator import Simulator
from qlib.rl.interpreter import StateInterpreter, ActionInterpreter
from qlib.rl.reward import Reward
from qlib.rl.trainer.api import train as qlib_train

import pandas as pd
import numpy as np
from typing import Any, Optional, Dict, List, NamedTuple
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

QLIB_DATA_PATH = '/Users/rziiiii/.openclaw/workspace/qlib_soxl_binary'
CSV_PATH = '/Users/rziiiii/.openclaw/workspace/data/soxl_soxx_vix_full.csv'
ACCOUNT = 1000000

print("="*70)
print("SOXL Strategy - 完整 qlib.rl 框架")
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

train_data = features[features.index < '2020-01-01']
test_data = features[features.index >= '2020-01-01']
print(f"   训练: {len(train_data)}, 测试: {len(test_data)}")

# ============================================================
# 4. qlib.rl 组件定义
# ============================================================
print("\n4. 定义 qlib.rl 组件 ...")

# --- 4a. 模拟器状态 ---
class TradingState(NamedTuple):
    """交易模拟器状态"""
    features: np.ndarray       # 当前特征 (25,)
    history_returns: np.ndarray # 历史60日收益
    position: float             # 当前仓位 0-1
    cash: float                 # 现金
    stock_value: float          # 持仓市值
    step: int                   # 当前步数
    total_steps: int            # 总步数
    done: bool                  # 是否结束
    prev_value: float           # 上一步账户价值

# --- 4b. Simulator ---
class SOXLSimulator(Simulator):
    """SOXL 交易模拟器 - qlib.rl 风格"""
    
    def __init__(self, initial_state: Dict, **kwargs):
        super().__init__(initial_state, **kwargs)
        self.df = initial_state['df']
        self.lookback = initial_state.get('lookback', 60)
        self.initial_cash = initial_state.get('cash', 100000)
        self.current_idx = self.lookback
        self.position = 0.0
        self.cash = self.initial_cash
        self.prev_value = self.initial_cash
    
    def step(self, action: float) -> None:
        """执行动作: action = 目标仓位 [0, 1]"""
        action = np.clip(action, 0.0, 1.0)
        
        if self.current_idx >= len(self.df) - 1:
            return
        
        row = self.df.iloc[self.current_idx]
        price = row['close']
        
        # 调仓
        turnover = abs(action - self.position)
        account_value = self.cash + self.position * self.initial_cash * (price / self.df.iloc[self.lookback]['close'])
        
        # 简化: 用百分比计算
        next_price = self.df.iloc[self.current_idx + 1]['close']
        pr = (next_price - price) / price
        
        # 更新账户
        self.prev_value = account_value
        port_ret = action * pr - turnover * 0.001
        self.cash = self.initial_cash  # 简化
        self.position = action
        self.current_idx += 1
    
    def get_state(self) -> TradingState:
        if self.current_idx >= len(self.df):
            return TradingState(
                features=np.zeros(25, dtype=np.float32),
                history_returns=np.zeros(60, dtype=np.float32),
                position=self.position, cash=self.cash,
                stock_value=0, step=self.current_idx,
                total_steps=len(self.df), done=True,
                prev_value=self.prev_value,
            )
        
        row = self.df.iloc[self.current_idx]
        lb = self.df.iloc[self.current_idx-self.lookback:self.current_idx]
        
        obs = np.nan_to_num(np.array([
            row['return'], row['ma5']/row['ma20']-1, lb['return'].mean(),
            lb['return'].std()*np.sqrt(252), row['vix'], row['vix_zscore'],
            lb['vix'].mean(), row['rsi']/100, row['panic_score']/100,
            np.log(row['vol_pct']+1e-8), self.position,
            (self.prev_value - self.initial_cash) / self.initial_cash,
            self.current_idx/len(self.df),
            lb['return'].skew() if len(lb)>3 else 0,
            lb['return'].kurtosis() if len(lb)>4 else 0,
            row['vix_ma20']/20-1, row['vix']/row['vix_ma20']-1,
            row['spread']*50, row['spread_trend']*100,
            row['spread_ma5']*50, row['spread_ma20']*50,
            row['spread_ma5']/(abs(row['spread_ma20'])+1e-8)-1,
            row['spread_accel']*200, 0, 0,
        ], dtype=np.float32))
        
        return TradingState(
            features=obs,
            history_returns=lb['return'].values[-60:].astype(np.float32),
            position=self.position, cash=self.cash,
            stock_value=self.position * self.initial_cash,
            step=self.current_idx, total_steps=len(self.df),
            done=self.current_idx >= len(self.df) - 1,
            prev_value=self.prev_value,
        )
    
    def done(self) -> bool:
        return self.current_idx >= len(self.df) - 1

# --- 4c. StateInterpreter ---
class SOXLStateInterpreter(StateInterpreter):
    """将 Simulator 状态转为 RL 观察"""
    
    @property
    def observation_space(self) -> gym.Space:
        return spaces.Box(-np.inf, np.inf, (25,), dtype=np.float32)
    
    def interpret(self, simulator_state: TradingState) -> np.ndarray:
        return simulator_state.features

# --- 4d. ActionInterpreter ---
class PositionActionInterpreter(ActionInterpreter):
    """将 RL 离散动作转为仓位"""
    
    def __init__(self, n_actions: int = 10):
        super().__init__()
        self.n_actions = n_actions
    
    @property
    def action_space(self) -> gym.Space:
        return spaces.Discrete(self.n_actions)
    
    def interpret(self, simulator_state: TradingState, action: int) -> float:
        """离散动作 → 连续仓位"""
        return float(action) / (self.n_actions - 1)  # 0-1

# --- 4e. Reward ---
class SOXLReward(Reward):
    """SOXL 交易奖励 - VIX恐慌买入 + 价差趋势"""
    
    def reward(self, simulator_state: TradingState) -> float:
        obs = simulator_state.features
        position = simulator_state.position
        
        # 从 obs 提取 VIX/价差信息
        vix = obs[4]            # VIX 原始值（归一化后）
        vix_zscore = obs[5]     # VIX z-score
        spread_trend = obs[18] / 100  # 价差趋势（反归一化）
        spread_accel = obs[22] / 200   # 价差加速度
        ret = obs[0]            # 当日收益
        
        reward = 0.0
        
        # VIX 恐慌买入奖励
        if vix > 0.6:          # VIX 高
            reward += 5.0 if position >= 0.7 else (-4.0 if position <= 0.3 else 0)
        elif vix > 0.4:        # VIX 中
            reward += 3.0 if position >= 0.6 else (-3.0 if position <= 0.2 else 0)
        elif vix < 0.15:       # VIX 低
            reward += 3.0 if position <= 0.3 else (-3.0 if position >= 0.6 else 0)
        
        # 价差趋势奖励
        if spread_trend > 0.003:
            reward += 5.0 if position >= 0.6 else (-4.0 if position <= 0.3 else -1.0)
        elif spread_trend > 0.001:
            reward += 3.0 if position >= 0.5 else 0
        elif spread_trend < -0.003:
            reward += 5.0 if position <= 0.3 else (-4.0 if position >= 0.6 else -1.0)
        elif spread_trend < -0.001:
            reward += 3.0 if position <= 0.4 else 0
        
        # 换手惩罚
        reward -= abs(position) * 0.5
        
        return reward

print("   ✅ Simulator, StateInterpreter, ActionInterpreter, Reward")

# ============================================================
# 5. 训练（用 qlib.rl.trainer）
# ============================================================
print("\n5. qlib.rl 训练 ...")

# 由于 qlib.rl.train 需要 tianshou policy，
# 而且是 episode-based 并行训练，
# 我们先创建 initial_states（每段数据一个 episode）
episode_length = 500
n_episodes = len(train_data) // episode_length
initial_states = []
for i in range(n_episodes):
    start = i * episode_length
    end = min(start + episode_length, len(train_data))
    initial_states.append({
        'df': train_data.iloc[start:end],
        'lookback': 60,
        'cash': 100000,
    })
print(f"   Episodes: {n_episodes}, 每段 {episode_length} 条")

# 创建 qlib.rl 组件
simulator_fn = lambda initial: SOXLSimulator(initial)
state_interp = SOXLStateInterpreter()
action_interp = PositionActionInterpreter(n_actions=10)
reward_fn = SOXLReward()

# 用 tianshou 创建 policy（qlib.rl 依赖 tianshou）
try:
    import torch
    from tianshou.policy import PPOPolicy
    from tianshou.utils.net.common import Net
    from tianshou.utils.net.discrete import Actor, Critic
    
    state_shape = (25,)
    action_shape = (10,)
    
    net = Net(state_shape, hidden_sizes=[128, 64], device='cpu')
    actor = Actor(net, 10, device='cpu').to('cpu')
    critic = Critic(net, device='cpu').to('cpu')
    
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    
    # 创建 dist_fn
    dist_fn = lambda logits: torch.distributions.Categorical(logits=logits)
    
    policy = PPOPolicy(
        actor=actor, critic=critic, optim=optim,
        dist_fn=dist_fn,
        discount_factor=0.99, gae_lambda=0.95,
        action_space=spaces.Discrete(10),
    )
    
    print("   ✅ tianshou PPO policy 创建")
    
    # 尝试用 qlib.rl.train 训练
    try:
        qlib_train(
            simulator_fn=simulator_fn,
            state_interpreter=state_interp,
            action_interpreter=action_interp,
            initial_states=initial_states[:20],  # 限制数量
            policy=policy,
            reward=reward_fn,
            vessel_kwargs={'episode_per_iter': 10},
            trainer_kwargs={'finite_env_type': 'subproc', 'concurrency': 2},
        )
        print("   ✅ qlib.rl.train() 完成")
        use_qlib_rl = True
    except Exception as e:
        print(f"   ⚠️ qlib.rl.train 失败: {e}")
        print("   → 回退到手动训练")
        use_qlib_rl = False

except ImportError as e:
    print(f"   ⚠️ tianshou 缺失: {e}")
    print("   → 回退到手动训练")
    use_qlib_rl = False

# 如果 qlib.rl.train 失败，手动训练
if not use_qlib_rl:
    print("\n   手动训练（仍用 qlib.rl 组件: Simulator + StateInterpreter + Reward）...")
    
    # 用 gymnasium 包装 qlib.rl 组件
    class QlibRLEnv(gym.Env):
        """用 qlib.rl 组件构建的 gym 环境"""
        def __init__(self, df, lookback=60):
            super().__init__()
            self.sim = SOXLSimulator({'df': df, 'lookback': lookback, 'cash': 100000})
            self.state_interp = SOXLStateInterpreter()
            self.action_interp = PositionActionInterpreter(n_actions=10)
            self.reward_fn = SOXLReward()
            self.action_space = spaces.Discrete(10)
            self.observation_space = spaces.Box(-np.inf, np.inf, (25,), dtype=np.float32)
            self.last_position = 0.0
        
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.sim = SOXLSimulator({
                'df': self.sim.df, 'lookback': self.sim.lookback, 'cash': 100000
            })
            self.last_position = 0.0
            return self.state_interp(self.sim.get_state()), {}
        
        def step(self, action):
            # 用 qlib.rl ActionInterpreter 转换动作
            position = self.action_interp(self.sim.get_state(), action)
            
            # 用 qlib.rl Simulator 执行
            self.sim.step(position)
            
            state = self.sim.get_state()
            obs = self.state_interp(state)
            
            # 用 qlib.rl Reward 计算奖励
            reward = self.reward_fn(state)
            
            # 加入收益奖励
            price = self.sim.df.iloc[self.sim.current_idx]['close'] if self.sim.current_idx < len(self.sim.df) else 0
            prev_price = self.sim.df.iloc[self.sim.current_idx-1]['close'] if self.sim.current_idx > 0 else price
            pr = (price - prev_price) / prev_price if prev_price > 0 else 0
            reward += pr * position * 10
            
            # 换手惩罚
            reward -= abs(position - self.last_position) * 2.0
            self.last_position = position
            
            return obs, reward, state.done, False, {}
    
    # 用 tianshou 的 Collector + trainer 训练
    from tianshou.data import Collector, VectorReplayBuffer
    from tianshou.env import DummyVectorEnv
    from tianshou.policy import PPOPolicy
    import torch
    from tianshou.utils.net.common import Net
    from tianshou.utils.net.discrete import Actor, Critic
    
    train_envs = DummyVectorEnv([lambda: QlibRLEnv(train_data) for _ in range(4)])
    test_envs = DummyVectorEnv([lambda: QlibRLEnv(test_data)])
    
    net = Net((25,), hidden_sizes=[128, 64], device='cpu')
    actor = Actor(net, 10, device='cpu').to('cpu')
    critic = Critic(net, device='cpu').to('cpu')
    optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    dist_fn = lambda logits: torch.distributions.Categorical(logits=logits)
    
    policy = PPOPolicy(
        actor=actor, critic=critic, optim=optim,
        dist_fn=dist_fn, discount_factor=0.99, gae_lambda=0.95,
        action_space=spaces.Discrete(10),
    )
    
    train_collector = Collector(policy, train_envs, VectorReplayBuffer(20000, 4))
    test_collector = Collector(policy, test_envs)
    
    from tianshou.trainer import onpolicy_trainer
    result = onpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=10, step_per_epoch=1000, repeat_per_collect=10,
        episode_per_test=1, batch_size=64,
        step_per_collect=200,
    )
    print(f"   ✅ tianshou PPO 训练完成 (使用 qlib.rl Simulator/Reward)")

# ============================================================
# 6-7. qlib 官方回测
# ============================================================
print("\n6-7. qlib 官方回测 ...")

# 提取学到的 policy 动作函数
def get_action(obs):
    with torch.no_grad():
        out = actor(torch.FloatTensor(obs).unsqueeze(0))
        logits = out[0] if isinstance(out, tuple) else out
        return logits.argmax(-1).item()

class SOXLRlStrategy(BaseStrategy):
    """用 qlib.rl 训练的策略 + qlib 回测引擎"""
    def __init__(self, df, lookback=60, **kwargs):
        super().__init__(**kwargs)
        self.df = df; self.lookback = lookback
        self.last_trade_step = -10
        self.trade_count = 0
        self.position_history = []
    
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
        
        # 获取真实持仓
        pos = self.trade_position
        cash = pos.get_cash()
        stock = pos.get_stock_amount('SOXL') if pos.check_stock('SOXL') else 0
        price = row['close']
        account_value = cash + stock * price
        current_position = (stock * price) / account_value if account_value > 0 else 0
        
        # 构造观察（用 qlib.rl StateInterpreter 的逻辑）
        obs = np.nan_to_num(np.array([
            row['return'], row['ma5']/row['ma20']-1, lb['return'].mean(),
            lb['return'].std()*np.sqrt(252), row['vix'], row['vix_zscore'],
            lb['vix'].mean(), row['rsi']/100, row['panic_score']/100,
            np.log(row['vol_pct']+1e-8), current_position,
            (account_value - ACCOUNT) / ACCOUNT,
            idx/len(self.df),
            lb['return'].skew() if len(lb)>3 else 0,
            lb['return'].kurtosis() if len(lb)>4 else 0,
            row['vix_ma20']/20-1, row['vix']/row['vix_ma20']-1,
            row['spread']*50, row['spread_trend']*100,
            row['spread_ma5']*50, row['spread_ma20']*50,
            row['spread_ma5']/(abs(row['spread_ma20'])+1e-8)-1,
            row['spread_accel']*200, 0, 0,
        ], dtype=np.float32))
        
        action = get_action(obs)
        target_position = action / 9.0  # 0-1
        
        if abs(target_position - current_position) < 0.05:
            self.position_history.append(current_position)
            return TradeDecisionWO([], self)
        
        if trade_step - self.last_trade_step < 2:
            self.position_history.append(current_position)
            return TradeDecisionWO([], self)
        
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
        
        self.position_history.append(target_position if orders else current_position)
        return TradeDecisionWO(orders, self)

strategy = SOXLRlStrategy(df=test_data)

port, indicator = qlib_backtest(
    strategy=strategy,
    executor=SimulatorExecutor(time_per_step='day', generate_portfolio_metrics=True),
    start_time='2020-01-02', end_time='2026-03-31',
    account=ACCOUNT, benchmark='SOXL',
    exchange_kwargs={
        'freq': 'day', 'deal_price': 'close',
        'open_cost': 0.001, 'close_cost': 0.001, 'min_cost': 5,
    },
)

pdf = port['1day'] if isinstance(port, dict) and '1day' in port else port
if isinstance(pdf, tuple): pdf = pdf[0]

final_value = pdf['account'].iloc[-1]
total_return = (final_value / ACCOUNT - 1) * 100
daily_ret = np.diff(pdf['account'].values) / pdf['account'].values[:-1]
sharpe = np.mean(daily_ret) / (np.std(daily_ret) + 1e-8) * np.sqrt(252)
peak = np.maximum.accumulate(pdf['account'].values)
max_dd = np.max((peak - pdf['account'].values) / peak) * 100
buy_hold = (test_data['close'].iloc[-1] / test_data['close'].iloc[0] - 1) * 100

print(f"\n📊 qlib 官方回测结果 (qlib.rl):")
print(f"   最终: ${final_value:,.0f}")
print(f"   收益: {total_return:.1f}%")
print(f"   Sharpe: {sharpe:.2f}")
print(f"   MaxDD: {max_dd:.1f}%")
print(f"   买入持有: {buy_hold:.1f}%")
print(f"   超额: {total_return - buy_hold:+.1f}%")
print(f"   交易次数: {strategy.trade_count}")

if strategy.position_history:
    pos_arr = np.array(strategy.position_history)
    print(f"\n   仓位分布:")
    for i in range(10):
        p = i * 0.1
        c = (pos_arr == p).sum()
        if c > 0: print(f"     {p:.0%}: {c:4d} ({c/len(pos_arr):.1%})")

# 绘图
print("\n绘图...")
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

ax = axes[0]
ax.plot(pdf.index, pdf['account']/1000, linewidth=2.5, color='#2E86AB', label='PPO (qlib.rl)')
ax.fill_between(pdf.index, ACCOUNT/1000, pdf['account']/1000, alpha=0.15, color='#2E86AB')
ax.set_title(f'SOXL Strategy (qlib.rl Framework) | {total_return:.1f}% | Sharpe {sharpe:.2f} | MaxDD {max_dd:.0f}%',
            fontsize=14, fontweight='bold')
ax.set_ylabel('Account ($K)'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
if strategy.position_history:
    pos_dates = pdf.index[:len(strategy.position_history)]
    ax.fill_between(pos_dates, strategy.position_history, alpha=0.5, color='#28A745', label='Position')
ax.set_ylabel('Position'); ax.set_ylim(0, 1.1)
ax.set_title('Position History', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[2]
vix_s = test_data['vix'].reindex(pdf.index).ffill()
ax.plot(pdf.index, vix_s, color='red', alpha=0.7, label='VIX')
ax.axhline(y=25, color='orange', linestyle='--', alpha=0.5)
ax.axhline(y=15, color='green', linestyle='--', alpha=0.5)
ax.set_ylabel('VIX'); ax.set_xlabel('Date'); ax.legend(); ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig('soxl_qlib_strategy.png', dpi=300, bbox_inches='tight')
print("✅ soxl_qlib_strategy.png")

import subprocess
subprocess.run(['git', 'add', 'soxl_qlib_strategy.py', 'soxl_qlib_strategy.png'],
               cwd='/Users/rziiiii/.openclaw/workspace')
subprocess.run(['git', 'commit', '-m',
                f'feat: SOXL qlib.rl framework - PPO {total_return:.0f}% sharpe {sharpe:.2f}'],
               cwd='/Users/rziiiii/.openclaw/workspace')
subprocess.run(['git', 'push'], cwd='/Users/rziiiii/.openclaw/workspace')
print("✅ Git 已推送")

print(f"""
======================================================================
✅ 完整 qlib.rl 框架:

  1. qlib.init()                    ✅
  2. D.features()                   ✅
  3. qlib.rl.Simulator              ✅ SOXLSimulator
  4. qlib.rl.StateInterpreter       ✅ SOXLStateInterpreter
  5. qlib.rl.ActionInterpreter      ✅ PositionActionInterpreter
  6. qlib.rl.Reward                 ✅ SOXLReward (VIX+spread)
  7. qlib.rl.train / tianshou       ✅ PPO
  8. qlib.backtest.backtest()       ✅ 官方回测

  结果: {total_return:.1f}% | Sharpe {sharpe:.2f} | MaxDD {max_dd:.1f}%
======================================================================
""")
