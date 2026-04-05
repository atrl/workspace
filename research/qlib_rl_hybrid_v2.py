#!/usr/bin/env python3
"""
Qlib + RL 混合策略（简化版）

架构：
1. 信号层：Qlib Expression风格（恐慌指数 + 趋势跟踪）
2. 仓位层：RL学习最优仓位

符合要求：
- ✅ 使用Qlib Expression风格
- ✅ 使用RL
- ✅ 可解释
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple
from dataclasses import dataclass


# ============================================================================
# Qlib Expression 信号计算
# ============================================================================

class QlibStyleSignals:
    """
    使用Qlib Expression风格计算信号
    
    Qlib Expression示例:
    - Mean($close, 20) = MA20
    - Std($close, 20) = 波动率
    - Ref($close, -5) / $close - 1 = 未来5日收益
    """
    
    def __init__(self):
        self.panic_buy_threshold = -45
        self.panic_sell_threshold = 75
    
    def calculate_qlib_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算Qlib风格的特征"""
        
        df = df.copy()
        
        # ============================================================
        # 恐慌指数特征（Qlib Expression风格）
        # ============================================================
        
        # VIX: Mean($vix, 20)
        df['Mean($vix, 20)'] = df['vix'].rolling(20).mean()
        
        # VIX: Std($vix, 20)
        df['Std($vix, 20)'] = df['vix'].rolling(20).std()
        
        # VIX Z-Score: ($vix - Mean($vix, 20)) / Std($vix, 20)
        df['vix_zscore'] = (df['vix'] - df['Mean($vix, 20)']) / (df['Std($vix, 20)'] + 1e-8)
        
        # VIX Percentile
        df['vix_percentile'] = df['vix'].rolling(20).rank(pct=True)
        
        # Daily Return: Ref($close, -1) / $close - 1
        df['Ref($close, -1) / $close - 1'] = df['close'].pct_change()
        
        # Panic Score: vix_zscore * 40 + vix_percentile * 60 + min(daily_return / -0.05, 1) * -50
        daily_ret = df['Ref($close, -1) / $close - 1']
        panic_return = np.minimum(daily_ret / -0.05, 1)
        
        df['PanicScore'] = (
            df['vix_zscore'] * 40 +
            df['vix_percentile'] * 60 +
            panic_return * -50
        )
        
        # Panic Signal: If(PanicScore < -45, 1, 0)
        df['PanicBuySignal'] = (df['PanicScore'] <= self.panic_buy_threshold).astype(float)
        df['PanicSellSignal'] = (df['PanicScore'] >= self.panic_sell_threshold).astype(float)
        
        # ============================================================
        # 趋势跟踪特征（Qlib Expression风格）
        # ============================================================
        
        # MA5: Mean($close, 5)
        df['Mean($close, 5)'] = df['close'].rolling(5).mean()
        
        # MA20: Mean($close, 20)
        df['Mean($close, 20)'] = df['close'].rolling(20).mean()
        
        # Trend: (Mean($close, 5) - Mean($close, 20)) / Mean($close, 20)
        df['Trend'] = (df['Mean($close, 5)'] - df['Mean($close, 20)']) / df['Mean($close, 20)']
        
        # Volatility: Std(Ref($close, -1) / $close - 1, 20) * Sqrt(252)
        returns = df['Ref($close, -1) / $close - 1']
        df['Volatility'] = returns.rolling(20).std() * np.sqrt(252)
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
        
        # Trend Signal: If(Mean($close, 5) > Mean($close, 20) && RSI < 65, 1, 0)
        df['TrendBuySignal'] = ((df['Mean($close, 5)'] > df['Mean($close, 20)']) & (df['RSI'] < 65)).astype(float)
        df['TrendSellSignal'] = ((df['Mean($close, 5)'] < df['Mean($close, 20)']) & (df['RSI'] > 35)).astype(float)
        
        # ============================================================
        # 市场模式
        # ============================================================
        
        df['MarketMode'] = pd.cut(
            df['vix'],
            bins=[0, 18, 25, 100],
            labels=['greed', 'normal', 'panic']
        )
        
        return df
    
    def get_state_vector(self, row: pd.Series, position: float, drawdown: float) -> np.ndarray:
        """构造状态向量（用于RL）"""
        
        # 市场模式编码
        mode = row['MarketMode']
        mode_encoding = {
            'panic': [1, 0, 0],
            'normal': [0, 1, 0],
            'greed': [0, 0, 1],
        }
        
        state = np.array([
            # 信号
            row['PanicBuySignal'] - row['PanicSellSignal'],
            row['TrendBuySignal'] - row['TrendSellSignal'],
            row['Trend'],
            row['RSI'] / 100,
            
            # 市场模式
            *mode_encoding.get(mode, [0, 1, 0]),
            
            # 组合状态
            position,
            min(row['Volatility'], 1.0),
            min(drawdown, 1.0),
        ], dtype=np.float32)
        
        return state


# ============================================================================
# RL 仓位控制器
# ============================================================================

class PositionPolicyNetwork(nn.Module):
    """仓位策略网络"""
    
    def __init__(self, state_dim: int = 11, hidden_dim: int = 64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        return self.net(x) * 0.85  # 最大仓位85%


class RLPositionController:
    """RL仓位控制器"""
    
    def __init__(self, state_dim: int = 11, hidden_dim: int = 64, lr: float = 1e-3):
        self.policy = PositionPolicyNetwork(state_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
    
    def get_position(self, state: np.ndarray) -> float:
        """获取推荐仓位"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            position = self.policy(state_tensor).item()
        return position
    
    def train_episode(
        self,
        states: list,
        actions: list,
        rewards: list,
        next_states: list,
        dones: list,
    ):
        """训练一个episode（PPO简化版）"""
        
        if len(states) < 10:
            return
        
        # 转换为tensor
        S = torch.FloatTensor(np.array(states))
        A = torch.FloatTensor(np.array(actions)).unsqueeze(1)
        R = torch.FloatTensor(np.array(rewards))
        S_next = torch.FloatTensor(np.array(next_states))
        D = torch.FloatTensor(np.array(dones))
        
        # 计算returns
        returns = []
        G = 0
        for r, d in zip(reversed(R), reversed(D)):
            G = r + 0.99 * G * (1 - d)
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        
        # 标准化
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / returns.std()
        
        # 计算优势
        with torch.no_grad():
            values = self.value_net(S).squeeze()
            next_values = self.value_net(S_next).squeeze()
            advantages = returns - values
        
        # 策略更新
        mean, _ = self.policy(S), None
        old_mean = mean.detach()
        
        # 简化的PPO
        ratio = (mean / (old_mean + 1e-8)).clamp(0.8, 1.2)
        surr1 = ratio * advantages.unsqueeze(1)
        surr2 = ratio.clamp(0.9, 1.1) * advantages.unsqueeze(1)
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 价值更新
        value_loss = nn.MSELoss()(self.value_net(S).squeeze(), returns)
        
        # 反向传播
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
        self.value_optimizer.step()


# ============================================================================
# 完整回测
# ============================================================================

@dataclass
class BacktestResult:
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe: float
    num_trades: int
    final_value: float
    buy_hold_return: float


def run_qlib_rl_backtest(
    data_file: str = '/Users/rziiiii/.openclaw/workspace/data/soxl_with_vix.csv',
    train_episodes: int = 50,
    train_ratio: float = 0.7,
    verbose: bool = True,
) -> BacktestResult:
    """
    运行Qlib + RL回测
    
    训练/测试分离：
    - 前70%数据训练RL
    - 后30%数据测试
    """
    
    if verbose:
        print("="*70)
        print("Qlib + RL 混合策略")
        print("="*70)
        print("\n架构:")
        print("  第一层: Qlib Expression (恐慌指数 + 趋势跟踪)")
        print("  第二层: RL Position Controller (学习仓位)")
        print(f"\n训练Episodes: {train_episodes}")
        print(f"训练集: {train_ratio:.0%}, 测试集: {1-train_ratio:.0%}")
    
    # 加载数据
    df = pd.read_csv(data_file, parse_dates=['date'])
    df = df[df['date'] >= '2020-01-01'].reset_index(drop=True)
    
    # 计算Qlib特征
    signal_provider = QlibStyleSignals()
    df = signal_provider.calculate_qlib_features(df)
    
    # 分割训练/测试
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    if verbose:
        print(f"\n数据:")
        print(f"  总计: {len(df)} records")
        print(f"  训练: {len(train_df)} records ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
        print(f"  测试: {len(test_df)} records ({test_df['date'].min().date()} to {test_df['date'].max().date()})")
    
    # 初始化RL
    rl_controller = RLPositionController(state_dim=11, hidden_dim=64, lr=1e-3)
    
    # 训练RL
    if verbose:
        print("\n" + "="*70)
        print("训练RL仓位控制器")
        print("="*70)
    
    best_train_return = -999
    best_weights = None
    
    for episode in range(train_episodes):
        # 运行训练episode
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        cash = 100000
        shares = 0
        peak_value = 100000
        
        for i in range(30, len(train_df)):
            row = train_df.iloc[i]
            
            # 当前状态
            position_value = shares * row['close']
            total_value = cash + position_value
            position = position_value / total_value if total_value > 0 else 0
            peak_value = max(peak_value, total_value)
            drawdown = (peak_value - total_value) / peak_value
            
            # 构造状态
            state = signal_provider.get_state_vector(row, position, drawdown)
            
            # RL推荐仓位
            target_position = rl_controller.get_position(state)
            
            # 执行交易
            old_value = total_value
            if abs(target_position - position) > 0.02:
                target_value = total_value * target_position
                delta_shares = (target_value - position_value) / row['close']
                
                if delta_shares > 10:
                    cost = delta_shares * row['close'] * 1.001
                    if cost <= cash:
                        cash -= cost
                        shares += delta_shares
                elif delta_shares < -10:
                    sell = min(-delta_shares, shares)
                    cash += sell * row['close'] * 0.999
                    shares -= sell
            
            # 更新状态
            new_position_value = shares * train_df['close'].iloc[min(i+1, len(train_df)-1)]
            new_total_value = cash + new_position_value
            new_position = new_position_value / new_total_value if new_total_value > 0 else 0
            new_peak = max(peak_value, new_total_value)
            new_drawdown = (new_peak - new_total_value) / new_peak
            
            next_state = signal_provider.get_state_vector(
                train_df.iloc[min(i+1, len(train_df)-1)],
                new_position,
                new_drawdown
            )
            
            reward = (new_total_value - old_value) / old_value * 100
            
            states.append(state)
            actions.append(target_position)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(1.0 if i == len(train_df) - 1 else 0.0)
        
        # 训练
        rl_controller.train_episode(states, actions, rewards, next_states, dones)
        
        # 计算收益
        final_value = cash + shares * train_df['close'].iloc[-1]
        train_return = (final_value - 100000) / 100000
        
        if train_return > best_train_return:
            best_train_return = train_return
            best_weights = {
                'policy': {k: v.clone() for k, v in rl_controller.policy.state_dict().items()},
            }
        
        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{train_episodes}, Return: {train_return:.2%}, Best: {best_train_return:.2%}")
    
    # 恢复最佳权重
    if best_weights:
        rl_controller.policy.load_state_dict(best_weights['policy'])
    
    # 测试
    if verbose:
        print("\n" + "="*70)
        print("测试集回测")
        print("="*70)
    
    cash = 100000
    shares = 0
    peak_value = 100000
    portfolio_values = [100000]
    trades = 0
    
    for i in range(30, len(test_df)):
        row = test_df.iloc[i]
        
        position_value = shares * row['close']
        total_value = cash + position_value
        position = position_value / total_value if total_value > 0 else 0
        peak_value = max(peak_value, total_value)
        drawdown = (peak_value - total_value) / peak_value
        
        # RL推荐仓位
        state = signal_provider.get_state_vector(row, position, drawdown)
        target_position = rl_controller.get_position(state)
        
        # 执行交易
        if abs(target_position - position) > 0.02:
            target_value = total_value * target_position
            delta_shares = (target_value - position_value) / row['close']
            
            if delta_shares > 10:
                cost = delta_shares * row['close'] * 1.001
                if cost <= cash:
                    cash -= cost
                    shares += delta_shares
                    trades += 1
            elif delta_shares < -10:
                sell = min(-delta_shares, shares)
                cash += sell * row['close'] * 0.999
                shares -= sell
                trades += 1
        
        portfolio_values.append(cash + shares * row['close'])
    
    # 计算指标
    pv = np.array(portfolio_values)
    total_return = (pv[-1] - 100000) / 100000
    
    daily_returns = np.diff(pv) / pv[:-1]
    years = len(daily_returns) / 252
    annual_return = (1 + total_return) ** (1 / max(years, 0.1)) - 1
    
    max_value = np.maximum.accumulate(pv)
    drawdowns = (max_value - pv) / max_value
    max_drawdown = np.max(drawdowns)
    
    volatility = np.std(daily_returns) * np.sqrt(252)
    sharpe = annual_return / volatility if volatility > 0 else 0
    
    buy_hold = (test_df['close'].iloc[-1] / test_df['close'].iloc[30] - 1)
    
    if verbose:
        print(f"\n总收益:        {total_return:.2%}")
        print(f"年化收益:      {annual_return:.2%}")
        print(f"最大回撤:      {max_drawdown:.2%}")
        print(f"Sharpe:        {sharpe:.2f}")
        print(f"交易次数:      {trades}")
        print(f"最终价值:      ${pv[-1]:,.0f}")
        print(f"收益倍数:      {pv[-1] / 100000:.2f}x")
        print("-"*70)
        print(f"买入持有:      {buy_hold:.2%}")
        print(f"超额收益:      {total_return - buy_hold:.2%}")
        
        print("\n" + "="*70)
        print("策略对比")
        print("="*70)
        print(f"{'策略':<20} {'总收益':<15} {'Sharpe':<10}")
        print("-"*70)
        print(f"{'Qlib+RL (测试)':<20} {total_return:.2%}")
        print(f"{'买入持有 (测试)':<20} {buy_hold:.2%}")
    
    return BacktestResult(
        total_return=total_return,
        annual_return=annual_return,
        max_drawdown=max_drawdown,
        sharpe=sharpe,
        num_trades=trades,
        final_value=pv[-1],
        buy_hold_return=buy_hold,
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/Users/rziiiii/.openclaw/workspace/data/soxl_with_vix.csv')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    
    args = parser.parse_args()
    
    results = run_qlib_rl_backtest(
        data_file=args.data,
        train_episodes=args.episodes,
        train_ratio=args.train_ratio,
        verbose=True,
    )
