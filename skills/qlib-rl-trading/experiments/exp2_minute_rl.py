#!/usr/bin/env python3
"""
Experiment 2: Minute-Level RL Strategy

使用分钟数据训练高频交易模型
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
class MinuteState:
    features: np.ndarray
    position_value: float
    cash: float
    portfolio_value: float
    step: int
    max_steps: int
    
    def to_tensor(self) -> np.ndarray:
        portfolio_features = np.array([
            self.position_value / max(self.portfolio_value, 1),
            self.cash / max(self.portfolio_value, 1),
            self.step / max(self.max_steps, 1),
        ], dtype=np.float32)
        return np.concatenate([self.features, portfolio_features])


@dataclass
class MinuteAction:
    position_delta: float
    
    def to_target_ratio(self) -> float:
        return (self.position_delta + 1) / 2 * 0.95


class MinuteTradingSimulator:
    """分钟级交易模拟器"""
    
    def __init__(self, data: pd.DataFrame, feature_cols: List[str]):
        self.data = data.reset_index(drop=True)
        self.feature_cols = feature_cols
        
        # 配置 - 分钟级特殊参数
        self.initial_cash = 100000
        self.transaction_cost = 0.0005  # 5bp - 分钟级需要更低成本
        self.max_steps = 390  # 约1个交易日（6.5小时 * 60分钟）
        
        # 状态
        self.reset()
    
    def reset(self, start_idx: Optional[int] = None) -> MinuteState:
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = 0.0
        self.avg_entry_price = 0.0
        
        if start_idx is None:
            max_start = len(self.data) - self.max_steps - 100
            start_idx = np.random.randint(100, max(max_start, 101))
        
        self.start_idx = start_idx
        self.current_idx = start_idx
        
        return self._get_state()
    
    @property
    def portfolio_value(self) -> float:
        return self.cash + self.shares * self._get_current_price()
    
    @property
    def position_value(self) -> float:
        return self.shares * self._get_current_price()
    
    def step(self, action: MinuteAction) -> Tuple[MinuteState, float, bool, Dict]:
        old_value = self.portfolio_value
        
        # 执行动作
        trade_executed = self._execute_action(action)
        
        # 前进时间
        self.current_idx += 1
        self.current_step += 1
        
        new_state = self._get_state()
        
        # 计算奖励
        reward = self._calculate_reward(old_value, trade_executed)
        
        done = self.current_step >= self.max_steps
        
        info = {
            'portfolio_value': self.portfolio_value,
            'trade_executed': trade_executed,
        }
        
        return new_state, reward, done, info
    
    def _execute_action(self, action: MinuteAction) -> bool:
        current_price = self._get_current_price()
        total_value = self.portfolio_value
        
        target_ratio = action.to_target_ratio()
        target_value = total_value * target_ratio
        target_shares = target_value / current_price
        
        delta_shares = target_shares - self.shares
        trade_executed = False
        
        # 分钟级最小交易阈值 - 更小
        if abs(delta_shares) < 5:  # 5股
            return False
        
        if delta_shares > 0:  # 买入
            max_shares = self.cash / (current_price * (1 + self.transaction_cost))
            shares_to_buy = min(delta_shares, max_shares)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                self.cash -= cost
                
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
    
    def _get_state(self) -> MinuteState:
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
        
        # 归一化（使用简单clip）
        features = np.clip(features, -3.0, 3.0)
        
        return MinuteState(
            features=features,
            position_value=self.position_value,
            cash=self.cash,
            portfolio_value=self.portfolio_value,
            step=self.current_step,
            max_steps=self.max_steps,
        )
    
    def _calculate_reward(self, old_value: float, trade_executed: bool) -> float:
        step_return = (self.portfolio_value - old_value) / old_value
        step_return = np.clip(step_return, -0.05, 0.05)  # 分钟级更小范围
        
        peak = max(self.portfolio_value, self.initial_cash)
        drawdown = (peak - self.portfolio_value) / peak
        risk_penalty = drawdown ** 2 * 0.3
        
        # 交易惩罚 - 分钟级需要更严格
        trade_penalty = 0.02 if trade_executed else 0.0
        
        reward = step_return * 50 - risk_penalty - trade_penalty
        reward = np.clip(reward, -1.0, 1.0)
        
        return reward
    
    def _get_current_price(self) -> float:
        return float(self.data['close'].iloc[self.current_idx])
    
    def get_metrics(self) -> Dict:
        total_return = (self.portfolio_value - self.initial_cash) / self.initial_cash
        return {'total_return': total_return}


class MinutePolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
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


def train_minute_rl(data_file: str, output_dir: str, timesteps: int = 20000):
    """训练分钟级RL模型"""
    
    print("="*60)
    print("Experiment 2: Minute-Level RL")
    print("="*60)
    
    # 加载分钟数据
    print(f"Loading minute data from {data_file}...")
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"Data: {len(df)} records")
    print(f"Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # 特征列
    feature_cols = [
        'returns_1m', 'returns_5m', 'returns_15m',
        'volatility_20m', 'ma_20',
    ]
    
    # 创建模拟器
    simulator = MinuteTradingSimulator(df, feature_cols)
    
    # 创建策略
    input_dim = len(feature_cols) + 3
    policy = MinutePolicy(input_dim=input_dim, hidden_dim=128)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    
    print(f"\nTraining for {timesteps} timesteps...")
    
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
            
            action = MinuteAction(position_delta=action_tensor.item())
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
            
            returns = torch.FloatTensor(np.array(rewards))
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            mean, std = policy(states_tensor)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions_tensor).squeeze()
            
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
        
        if episode % 50 == 0:
            avg_return = np.mean(episode_returns[-50:])
            print(f"Episode {episode}, Steps {total_steps}/{timesteps}, Return: {metrics['total_return']:.2%}, Avg: {avg_return:.2%}, Best: {best_return:.2%}")
    
    # 保存
    model_file = output_path / 'minute_ppo_model.zip'
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'feature_cols': feature_cols,
        'metrics': {
            'best_return': best_return,
            'avg_return': np.mean(episode_returns),
            'episodes': episode,
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
    parser.add_argument('--timesteps', type=int, default=20000)
    
    args = parser.parse_args()
    
    train_minute_rl(
        data_file=args.data,
        output_dir=args.output_dir,
        timesteps=args.timesteps,
    )
