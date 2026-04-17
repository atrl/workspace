#!/usr/bin/env python3
"""
Experiment 3: Improved Reward Function

改进奖励函数设计，平衡收益/风险/交易成本
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
class EnhancedState:
    features: np.ndarray
    position_value: float
    cash: float
    portfolio_value: float
    step: int
    max_steps: int
    peak_value: float  # 用于计算回撤
    
    def to_tensor(self) -> np.ndarray:
        portfolio_features = np.array([
            self.position_value / max(self.portfolio_value, 1),
            self.cash / max(self.portfolio_value, 1),
            self.step / max(self.max_steps, 1),
            (self.peak_value - self.portfolio_value) / max(self.peak_value, 1),  # 当前回撤
        ], dtype=np.float32)
        return np.concatenate([self.features, portfolio_features])


@dataclass
class EnhancedAction:
    position_delta: float
    
    def to_target_ratio(self) -> float:
        return (self.position_delta + 1) / 2 * 0.95


class EnhancedRewardCalculator:
    """增强版奖励计算器"""
    
    def __init__(self):
        self.trade_history = []
        self.return_history = []
        self.peak_value = 100000
    
    def calculate(
        self,
        old_value: float,
        new_value: float,
        trade_executed: bool,
        action_delta: float,
        position_ratio: float,
        step: int,
        max_steps: int,
    ) -> float:
        """
        多维度奖励函数
        
        组成部分：
        1. 步收益奖励（放大）
        2. 回撤惩罚（指数增长）
        3. 交易成本惩罚
        4. 换手率惩罚
        5. Sharpe奖励（滚动）
        6. 持仓奖励（鼓励低频交易）
        """
        
        # 1. 步收益（放大100倍，分钟级数据信号弱）
        step_return = (new_value - old_value) / old_value
        step_return = np.clip(step_return, -0.05, 0.05)
        return_reward = step_return * 100
        
        # 2. 回撤惩罚（指数增长）
        self.peak_value = max(self.peak_value, new_value)
        drawdown = (self.peak_value - new_value) / self.peak_value
        drawdown_penalty = drawdown ** 2 * 5  # 平方惩罚
        
        # 3. 交易成本惩罚（高频更严厉）
        trade_penalty = 0.05 if trade_executed else 0.0
        
        # 4. 动作幅度惩罚（大幅调仓要惩罚）
        action_penalty = abs(action_delta) * 0.1
        
        # 5. 滚动Sharpe奖励（每10步计算一次）
        self.return_history.append(step_return)
        sharpe_reward = 0.0
        if len(self.return_history) >= 20:
            recent_returns = np.array(self.return_history[-20:])
            if recent_returns.std() > 0:
                rolling_sharpe = recent_returns.mean() / recent_returns.std()
                sharpe_reward = rolling_sharpe * 0.5  # 奖励稳定性
        
        # 6. 持仓稳定性奖励（鼓励持有）
        if not trade_executed:
            holding_bonus = 0.01  # 不交易有奖励
        else:
            holding_bonus = 0.0
        
        # 7. 仓位极端惩罚（不满仓也不空仓）
        if position_ratio > 0.9:
            position_penalty = (position_ratio - 0.9) * 2  # 满仓惩罚
        elif position_ratio < 0.1:
            position_penalty = (0.1 - position_ratio) * 1  # 空仓惩罚
        else:
            position_penalty = 0.0
        
        # 总奖励
        total_reward = (
            return_reward           # 步收益
            - drawdown_penalty      # 回撤惩罚
            - trade_penalty         # 交易成本
            - action_penalty        # 动作幅度
            + sharpe_reward         # Sharpe奖励
            + holding_bonus         # 持仓奖励
            - position_penalty      # 仓位极端惩罚
        )
        
        # 裁剪到[-1, 1]
        total_reward = np.clip(total_reward, -1.0, 1.0)
        
        return total_reward


class EnhancedSimulator:
    """增强版模拟器"""
    
    def __init__(self, data: pd.DataFrame, feature_cols: List[str], norm_stats: Dict):
        self.data = data.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.norm_stats = norm_stats
        
        self.initial_cash = 100000
        self.transaction_cost = 0.001
        self.max_steps = 240
        
        self.reward_calc = EnhancedRewardCalculator()
        self.reset()
    
    def reset(self, start_idx: Optional[int] = None) -> EnhancedState:
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = 0.0
        self.avg_entry_price = 0.0
        
        self.reward_calc = EnhancedRewardCalculator()
        
        if start_idx is None:
            max_start = len(self.data) - self.max_steps - 30
            start_idx = np.random.randint(30, max(max_start, 31))
        
        self.start_idx = start_idx
        self.current_idx = start_idx
        
        return self._get_state()
    
    @property
    def portfolio_value(self) -> float:
        return self.cash + self.shares * self._get_current_price()
    
    @property
    def position_value(self) -> float:
        return self.shares * self._get_current_price()
    
    @property
    def position_ratio(self) -> float:
        return self.position_value / self.portfolio_value if self.portfolio_value > 0 else 0
    
    def step(self, action: EnhancedAction) -> Tuple[EnhancedState, float, bool, Dict]:
        old_value = self.portfolio_value
        old_position = self.position_ratio
        
        trade_executed = self._execute_action(action)
        
        self.current_idx += 1
        self.current_step += 1
        
        new_state = self._get_state()
        
        # 使用增强版奖励
        reward = self.reward_calc.calculate(
            old_value=old_value,
            new_value=self.portfolio_value,
            trade_executed=trade_executed,
            action_delta=action.position_delta,
            position_ratio=self.position_ratio,
            step=self.current_step,
            max_steps=self.max_steps,
        )
        
        done = self.current_step >= self.max_steps
        
        info = {
            'portfolio_value': self.portfolio_value,
            'trade_executed': trade_executed,
        }
        
        return new_state, reward, done, info
    
    def _execute_action(self, action: EnhancedAction) -> bool:
        current_price = self._get_current_price()
        total_value = self.portfolio_value
        
        target_ratio = action.to_target_ratio()
        target_value = total_value * target_ratio
        target_shares = target_value / current_price
        
        delta_shares = target_shares - self.shares
        trade_executed = False
        
        if abs(delta_shares) < 10:
            return False
        
        if delta_shares > 0:
            max_shares = self.cash / (current_price * (1 + self.transaction_cost))
            shares_to_buy = min(delta_shares, max_shares)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                self.cash -= cost
                
                total_cost = self.avg_entry_price * self.shares + current_price * shares_to_buy
                self.shares += shares_to_buy
                self.avg_entry_price = total_cost / self.shares if self.shares > 0 else 0
                
                trade_executed = True
        
        else:
            shares_to_sell = min(abs(delta_shares), self.shares)
            
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price * (1 - self.transaction_cost)
                self.cash += revenue
                self.shares -= shares_to_sell
                
                trade_executed = True
        
        return trade_executed
    
    def _get_state(self) -> EnhancedState:
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
        
        for i, col in enumerate(self.feature_cols):
            if col in self.norm_stats:
                mean = self.norm_stats[col]['mean']
                std = self.norm_stats[col]['std']
                if std > 1e-8:
                    features[i] = (features[i] - mean) / std
                features[i] = np.clip(features[i], -3.0, 3.0)
        
        return EnhancedState(
            features=features,
            position_value=self.position_value,
            cash=self.cash,
            portfolio_value=self.portfolio_value,
            step=self.current_step,
            max_steps=self.max_steps,
            peak_value=self.reward_calc.peak_value,
        )
    
    def _get_current_price(self) -> float:
        return float(self.data['close'].iloc[self.current_idx])
    
    def get_metrics(self) -> Dict:
        return {
            'total_return': (self.portfolio_value - self.initial_cash) / self.initial_cash
        }


class EnhancedPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
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


def train_enhanced_rl(data_file: str, output_dir: str, timesteps: int = 30000):
    """训练增强版RL模型"""
    
    print("="*60)
    print("Experiment 3: Enhanced Reward Function")
    print("="*60)
    
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    df = data['df']
    feature_cols = data['feature_cols']
    norm_stats = data['norm_stats']
    
    print(f"Data: {len(df)} records")
    
    # 创建模拟器
    simulator = EnhancedSimulator(df, feature_cols, norm_stats)
    
    # 创建策略
    input_dim = len(feature_cols) + 4  # 增加1维（当前回撤）
    policy = EnhancedPolicy(input_dim=input_dim, hidden_dim=128)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    
    print(f"\nTraining for {timesteps} timesteps...")
    print("Enhanced reward components:")
    print("  - Step return (100x)")
    print("  - Drawdown penalty (^2)")
    print("  - Trade cost penalty (5%)")
    print("  - Action magnitude penalty")
    print("  - Rolling Sharpe reward")
    print("  - Holding bonus (1%)")
    print("  - Position extreme penalty")
    
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
            
            action = EnhancedAction(position_delta=action_tensor.item())
            next_state, reward, done, info = simulator.step(action)
            
            states.append(state.to_tensor())
            actions.append(action_tensor.item())
            rewards.append(reward)
            log_probs.append(log_prob.item())
            
            state = next_state
            total_steps += 1
        
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
        
        if episode % 20 == 0:
            avg_return = np.mean(episode_returns[-20:])
            print(f"Episode {episode}, Steps {total_steps}/{timesteps}, Return: {metrics['total_return']:.2%}, Avg: {avg_return:.2%}, Best: {best_return:.2%}")
    
    # 保存
    model_file = output_path / 'enhanced_ppo_model.zip'
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'feature_cols': feature_cols,
        'norm_stats': norm_stats,
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
    parser.add_argument('--timesteps', type=int, default=30000)
    
    args = parser.parse_args()
    
    train_enhanced_rl(
        data_file=args.data,
        output_dir=args.output_dir,
        timesteps=args.timesteps,
    )
