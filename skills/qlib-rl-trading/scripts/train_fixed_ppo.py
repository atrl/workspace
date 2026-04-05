#!/usr/bin/env python3
"""
Fixed PPO Training - 修复版训练脚本

修复：
1. 仓位比例计算错误
2. 奖励函数优化
3. 更稳定的训练
"""
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class TradingState:
    """交易状态"""
    features: np.ndarray
    position_value: float  # 持仓市值
    cash: float            # 现金
    portfolio_value: float # 总资产
    step: int
    max_steps: int
    
    def to_tensor(self) -> np.ndarray:
        """转换为神经网络输入"""
        # 归一化的组合特征
        portfolio_features = np.array([
            self.position_value / max(self.portfolio_value, 1),  # 仓位比例 [0, 1]
            self.cash / max(self.portfolio_value, 1),            # 现金比例 [0, 1]
            self.step / max(self.max_steps, 1),                  # 时间进度 [0, 1]
        ], dtype=np.float32)
        
        return np.concatenate([self.features, portfolio_features])


@dataclass
class TradingAction:
    """交易动作"""
    position_delta: float  # [-1, 1]
    
    def to_target_ratio(self) -> float:
        """映射到目标仓位比例 [0, 0.95]"""
        normalized = (self.position_delta + 1) / 2  # [-1, 1] -> [0, 1]
        return normalized * 0.95  # 最大95%仓位


class FixedTradingSimulator:
    """修复版交易模拟器"""
    
    def __init__(self, data: pd.DataFrame, feature_cols: List[str], norm_stats: Dict):
        self.data = data.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.norm_stats = norm_stats
        
        # 配置
        self.initial_cash = 100000
        self.transaction_cost = 0.001  # 10bp
        self.max_steps = 240
        self.stop_loss = 0.15
        self.take_profit = 0.20
        
        # 状态
        self.reset()
        
    def reset(self, start_idx: Optional[int] = None) -> TradingState:
        """重置"""
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = 0.0
        self.avg_entry_price = 0.0
        
        if start_idx is None:
            max_start = len(self.data) - self.max_steps - 30
            start_idx = np.random.randint(30, max(max_start, 31))
        
        self.start_idx = start_idx
        self.current_idx = start_idx
        
        return self._get_state()
    
    def step(self, action: TradingAction) -> Tuple[TradingState, float, bool, Dict]:
        """执行一步"""
        old_value = self.portfolio_value
        
        # 风控
        action = self._apply_risk_controls(action)
        
        # 执行动作
        trade_executed = self._execute_action(action)
        
        # 前进时间
        self.current_idx += 1
        self.current_step += 1
        
        # 获取新状态
        new_state = self._get_state()
        
        # 计算奖励
        reward = self._calculate_reward(old_value, trade_executed)
        
        # 检查终止
        done = self.current_step >= self.max_steps
        
        info = {
            'portfolio_value': self.portfolio_value,
            'shares': self.shares,
            'cash': self.cash,
            'trade_executed': trade_executed,
        }
        
        return new_state, reward, done, info
    
    @property
    def portfolio_value(self) -> float:
        """总资产"""
        return self.cash + self.shares * self._get_current_price()
    
    @property
    def position_value(self) -> float:
        """持仓市值"""
        return self.shares * self._get_current_price()
    
    def _apply_risk_controls(self, action: TradingAction) -> TradingAction:
        """风控"""
        if self.shares <= 0:
            return action
        
        current_price = self._get_current_price()
        pnl = (current_price - self.avg_entry_price) / self.avg_entry_price
        
        # 止损
        if pnl <= -self.stop_loss:
            return TradingAction(position_delta=-1.0)  # 全部卖出
        
        # 止盈
        if pnl >= self.take_profit:
            return TradingAction(position_delta=0.0)  # 保持当前仓位
        
        return action
    
    def _execute_action(self, action: TradingAction) -> bool:
        """执行交易"""
        current_price = self._get_current_price()
        total_value = self.portfolio_value
        
        # 计算目标仓位
        target_ratio = action.to_target_ratio()
        target_value = total_value * target_ratio
        target_shares = target_value / current_price
        
        # 计算需要交易的股数
        delta_shares = target_shares - self.shares
        trade_executed = False
        
        # 最小交易阈值（避免噪音交易）
        if abs(delta_shares) < 10:  # 少于10股不交易
            return False
        
        if delta_shares > 0:  # 买入
            # 计算最大可买股数
            max_shares = self.cash / (current_price * (1 + self.transaction_cost))
            shares_to_buy = min(delta_shares, max_shares)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                self.cash -= cost
                
                # 更新平均成本
                total_cost = self.avg_entry_price * self.shares + current_price * shares_to_buy
                self.shares += shares_to_buy
                self.avg_entry_price = total_cost / self.shares if self.shares > 0 else 0
                
                trade_executed = True
        
        else:  # 卖出
            shares_to_sell = min(abs(delta_shares), self.shares)
            
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price * (1 - self.transaction_cost)
                self.cash += revenue
                self.shares -= shares_to_sell
                
                trade_executed = True
        
        return trade_executed
    
    def _get_state(self) -> TradingState:
        """获取状态"""
        # 提取特征
        features = []
        for col in self.feature_cols:
            if col in self.data.columns:
                val = self.data[col].iloc[self.current_idx]
                if pd.isna(val) or np.isinf(val):
                    val = 0.0
                features.append(val)
            else:
                features.append(0.0)
        
        features = np.array(features, dtype=np.float32)
        
        # 归一化特征
        for i, col in enumerate(self.feature_cols):
            if col in self.norm_stats:
                mean = self.norm_stats[col]['mean']
                std = self.norm_stats[col]['std']
                if std > 1e-8:
                    features[i] = (features[i] - mean) / std
                features[i] = np.clip(features[i], -3.0, 3.0)
        
        return TradingState(
            features=features,
            position_value=self.position_value,
            cash=self.cash,
            portfolio_value=self.portfolio_value,
            step=self.current_step,
            max_steps=self.max_steps,
        )
    
    def _calculate_reward(self, old_value: float, trade_executed: bool) -> float:
        """计算奖励"""
        # 步收益
        step_return = (self.portfolio_value - old_value) / old_value
        step_return = np.clip(step_return, -0.1, 0.1)  # 裁剪到±10%
        
        # 回撤惩罚
        peak = max(self.portfolio_value, self.initial_cash)
        drawdown = (peak - self.portfolio_value) / peak
        risk_penalty = drawdown ** 2 * 0.5
        
        # 交易成本惩罚
        trade_penalty = 0.005 if trade_executed else 0.0
        
        # 总奖励（放大步收益）
        reward = step_return * 100 - risk_penalty - trade_penalty
        reward = np.clip(reward, -1.0, 1.0)
        
        return reward
    
    def _get_current_price(self) -> float:
        """获取当前价格"""
        return float(self.data['close'].iloc[self.current_idx])
    
    def get_metrics(self) -> Dict:
        """获取指标"""
        total_return = (self.portfolio_value - self.initial_cash) / self.initial_cash
        
        return {
            'total_return': total_return,
            'final_value': self.portfolio_value,
            'final_shares': self.shares,
            'final_cash': self.cash,
        }


class SimplePolicy(nn.Module):
    """简化策略网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )
        
        self.log_std = nn.Parameter(torch.ones(1) * -1.0)
    
    def forward(self, x):
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        
        mean = self.net(x)
        std = torch.exp(self.log_std)
        std = torch.clamp(std, 1e-3, 1.0)
        
        return mean, std
    
    def sample(self, x):
        mean, std = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob


def train_fixed_ppo(data_file: str, output_dir: str, timesteps: int = 10000, lr: float = 3e-4) -> str:
    """训练"""
    
    print(f"Loading data from {data_file}...")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    df = data['df']
    feature_cols = data['feature_cols']
    norm_stats = data['norm_stats']
    
    print(f"Data loaded: {len(df)} records, {len(feature_cols)} features")
    
    # 创建模拟器
    simulator = FixedTradingSimulator(df, feature_cols, norm_stats)
    
    # 创建策略
    input_dim = len(feature_cols) + 3  # 特征数 + 3个组合特征
    policy = SimplePolicy(input_dim=input_dim, hidden_dim=64)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    print(f"Training for {timesteps} timesteps...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    episode = 0
    total_steps = 0
    best_return = -999
    episode_returns = []
    
    while total_steps < timesteps:
        states, actions, rewards, log_probs = [], [], [], []
        
        state = simulator.reset()
        done = False
        
        while not done and total_steps < timesteps:
            state_tensor = torch.FloatTensor(state.to_tensor()).unsqueeze(0)
            
            with torch.no_grad():
                action_tensor, log_prob = policy.sample(state_tensor)
            
            action = TradingAction(position_delta=action_tensor.item())
            next_state, reward, done, info = simulator.step(action)
            
            states.append(state.to_tensor())
            actions.append(action_tensor.item())
            rewards.append(reward)
            log_probs.append(log_prob.item())
            
            state = next_state
            total_steps += 1
        
        # PPO更新
        if len(states) > 10:
            states_tensor = torch.FloatTensor(np.array(states))
            actions_tensor = torch.FloatTensor(np.array(actions)).unsqueeze(1)
            old_log_probs = torch.FloatTensor(np.array(log_probs))
            
            # Returns
            returns = torch.FloatTensor(np.array(rewards))
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # Forward
            mean, std = policy(states_tensor)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions_tensor).squeeze()
            
            # PPO loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            ratio = torch.clamp(ratio, 0.1, 10.0)
            
            clip_range = 0.1
            surr1 = ratio * returns
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * returns
            loss = -torch.min(surr1, surr2).mean()
            
            if not torch.isnan(loss):
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.1)
                optimizer.step()
        
        episode += 1
        metrics = simulator.get_metrics()
        episode_returns.append(metrics['total_return'])
        
        if metrics['total_return'] > best_return:
            best_return = metrics['total_return']
        
        if episode % 20 == 0:
            avg_return = np.mean(episode_returns[-20:])
            print(f"Episode {episode}, Steps {total_steps}/{timesteps}, Return: {metrics['total_return']:.2%}, Avg: {avg_return:.2%}, Best: {best_return:.2%}")
    
    # 保存
    model_file = output_path / 'ppo_model.zip'
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'feature_cols': feature_cols,
        'norm_stats': norm_stats,
        'metrics': {
            'best_return': best_return,
            'avg_return': np.mean(episode_returns),
            'episodes': episode,
            'timesteps': total_steps,
        }
    }, model_file)
    
    print(f"\nTraining complete!")
    print(f"Best Return: {best_return:.2%}")
    print(f"Avg Return: {np.mean(episode_returns):.2%}")
    print(f"Model saved to: {model_file}")
    
    return str(model_file)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output-dir', default='models')
    parser.add_argument('--timesteps', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=3e-4)
    
    args = parser.parse_args()
    
    train_fixed_ppo(
        data_file=args.data,
        output_dir=args.output_dir,
        timesteps=args.timesteps,
        lr=args.lr,
    )
