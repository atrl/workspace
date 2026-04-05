#!/usr/bin/env python3
"""
Train PPO - Qlib Native

使用PPO算法训练RL策略，包含：
1. 数值稳定性保护（NaN检测、梯度裁剪）
2. 成本感知动作空间
3. 极端行情风控
4. TensorBoard监控
"""
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import yaml

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# TensorBoard 可选
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


# ============= 状态、动作、奖励定义 =============

@dataclass
class TradingState:
    """交易状态"""
    features: np.ndarray          # 特征向量
    position_ratio: float         # 当前仓位比例
    unrealized_pnl: float         # 未实现盈亏
    cash_ratio: float             # 现金比例
    step: int                     # 当前步数
    max_steps: int                # 最大步数
    
    def to_tensor(self) -> np.ndarray:
        """转换为神经网络输入"""
        portfolio_features = np.array([
            self.position_ratio,
            np.clip(self.unrealized_pnl, -1, 1),
            self.cash_ratio,
            self.step / self.max_steps,
        ], dtype=np.float32)
        
        return np.concatenate([self.features, portfolio_features])


@dataclass  
class TradingAction:
    """交易动作"""
    position_delta: float  # [-1, 1] -> [0%, 95%]
    
    def to_target_ratio(self, min_pos: float = 0.0, max_pos: float = 0.95) -> float:
        """映射到目标仓位比例"""
        normalized = (self.position_delta + 1) / 2  # [-1, 1] -> [0, 1]
        return min_pos + normalized * (max_pos - min_pos)


@dataclass
class TradingReward:
    """交易奖励"""
    step_return: float
    risk_penalty: float
    action_cost: float
    total: float


# ============= 模拟器 =============

class SafeTradingSimulator:
    """
    安全交易模拟器
    
    特性：
    1. 数值稳定性保护
    2. 极端行情风控
    3. 成本感知
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        norm_stats: Dict,
        config: Dict,
    ):
        self.data = data.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.norm_stats = norm_stats
        self.config = config
        
        # 模拟器参数
        self.initial_cash = config.get('initial_cash', 100000)
        self.transaction_cost = config.get('transaction_cost', 0.001)
        self.max_steps = config.get('max_steps', 240)
        
        # 风控参数
        self.stop_loss = config.get('stop_loss', 0.15)
        self.take_profit = config.get('take_profit', 0.20)
        self.max_position = config.get('max_position', 0.95)
        self.min_position = config.get('min_position', 0.0)
        
        # 状态变量
        self.current_step = 0
        self.cash = self.initial_cash
        self.position = 0.0
        self.avg_entry_price = 0.0
        self.portfolio_value = self.initial_cash
        
        # 添加from dataclasses import dataclass
        from dataclasses import dataclass
        
    def reset(self, start_idx: Optional[int] = None) -> TradingState:
        """重置环境"""
        self.current_step = 0
        self.cash = self.initial_cash
        self.position = 0.0
        self.avg_entry_price = 0.0
        self.portfolio_value = self.initial_cash
        
        # 随机起始点
        if start_idx is None:
            max_start = len(self.data) - self.max_steps - 30
            start_idx = np.random.randint(30, max(max_start, 31))
        
        self.start_idx = start_idx
        self.current_idx = start_idx
        
        return self._get_state()
    
    def step(self, action: TradingAction) -> Tuple[TradingState, TradingReward, bool, Dict]:
        """执行一步"""
        old_value = self.portfolio_value
        
        # 1. 极端行情风控
        action = self._apply_risk_controls(action)
        
        # 2. 执行动作
        trade_executed = self._execute_action(action)
        
        # 3. 前进时间
        self.current_idx += 1
        self.current_step += 1
        
        # 4. 获取新状态
        new_state = self._get_state()
        
        # 5. 计算奖励
        reward = self._calculate_reward(old_value, trade_executed, action)
        
        # 6. 检查终止
        done = self.current_step >= self.max_steps
        
        # 7. 信息
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'trade_executed': trade_executed,
        }
        
        return new_state, reward, done, info
    
    def _apply_risk_controls(self, action: TradingAction) -> TradingAction:
        """应用风控规则"""
        current_price = self._get_current_price()
        
        # 1. 止损检查
        if self.position > 0 and self.avg_entry_price > 0:
            unrealized_loss = (current_price - self.avg_entry_price) / self.avg_entry_price
            if unrealized_loss <= -self.stop_loss:
                # 强制平仓
                return TradingAction(position_delta=-1.0)
        
        # 2. 止盈检查
        if self.position > 0 and self.avg_entry_price > 0:
            unrealized_gain = (current_price - self.avg_entry_price) / self.avg_entry_price
            if unrealized_gain >= self.take_profit:
                # 部分止盈（卖出50%）
                current_ratio = self.position * current_price / self.portfolio_value
                target_ratio = current_ratio * 0.5
                delta = target_ratio * 2 - 1  # 映射回[-1, 1]
                return TradingAction(position_delta=delta)
        
        # 3. VIX极端值检查
        if 'vix_zscore' in self.data.columns:
            vix_zscore = self.data['vix_zscore'].iloc[self.current_idx]
            if not np.isnan(vix_zscore) and vix_zscore > 2.0:  # VIX极端高
                # 限制最大仓位到50%
                target = action.to_target_ratio(self.min_position, self.max_position)
                target = min(target, 0.5)
                delta = target * 2 - 1
                return TradingAction(position_delta=delta)
        
        return action
    
    def _execute_action(self, action: TradingAction) -> bool:
        """执行交易动作"""
        current_price = self._get_current_price()
        
        # 计算目标仓位
        target_ratio = action.to_target_ratio(self.min_position, self.max_position)
        target_position_value = self.portfolio_value * target_ratio
        target_shares = target_position_value / current_price
        
        # 计算差值
        delta_shares = target_shares - self.position
        
        trade_executed = False
        
        if abs(delta_shares) > 1.0:  # 最小交易阈值
            if delta_shares > 0:  # 买入
                shares_to_buy = min(
                    delta_shares,
                    self.cash / (current_price * (1 + self.transaction_cost))
                )
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price * (1 + self.transaction_cost)
                    self.cash -= cost
                    
                    # 更新平均成本
                    total_cost = self.avg_entry_price * self.position + current_price * shares_to_buy
                    self.position += shares_to_buy
                    self.avg_entry_price = total_cost / self.position if self.position > 0 else 0
                    
                    trade_executed = True
            
            else:  # 卖出
                shares_to_sell = min(abs(delta_shares), self.position)
                if shares_to_sell > 0:
                    revenue = shares_to_sell * current_price * (1 - self.transaction_cost)
                    self.cash += revenue
                    self.position -= shares_to_sell
                    
                    trade_executed = True
        
        # 更新组合价值
        self.portfolio_value = self.cash + self.position * current_price
        
        return trade_executed
    
    def _get_state(self) -> TradingState:
        """获取当前状态"""
        # 提取特征
        features = []
        for col in self.feature_cols:
            if col in self.data.columns:
                val = self.data[col].iloc[self.current_idx]
                if np.isnan(val) or np.isinf(val):
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
        
        # 计算未实现盈亏
        current_price = self._get_current_price()
        unrealized_pnl = 0.0
        if self.position > 0 and self.avg_entry_price > 0:
            unrealized_pnl = (current_price - self.avg_entry_price) / self.avg_entry_price
        
        # 仓位比例
        position_ratio = (self.position * current_price) / self.portfolio_value if self.portfolio_value > 0 else 0.0
        
        return TradingState(
            features=features,
            position_ratio=position_ratio,
            unrealized_pnl=unrealized_pnl,
            cash_ratio=self.cash / self.initial_cash,
            step=self.current_step,
            max_steps=self.max_steps,
        )
    
    def _calculate_reward(self, old_value: float, trade_executed: bool, action: TradingAction) -> TradingReward:
        """计算奖励"""
        # 1. 步收益
        step_return = (self.portfolio_value - old_value) / old_value
        step_return = np.clip(step_return, -0.5, 0.5)  # 裁剪
        
        # 2. 风险惩罚（回撤）
        peak_value = max(self.portfolio_value, self.initial_cash)
        drawdown = (peak_value - self.portfolio_value) / peak_value
        risk_penalty = drawdown ** 2 * 0.5
        
        # 3. 动作成本
        action_cost = abs(action.position_delta) * self.transaction_cost if trade_executed else 0.0
        
        # 4. 总奖励（归一化到[-1, 1]）
        total = step_return * 10 - risk_penalty - action_cost
        total = np.clip(total, -1.0, 1.0)
        
        return TradingReward(
            step_return=step_return,
            risk_penalty=risk_penalty,
            action_cost=action_cost,
            total=total,
        )
    
    def _get_current_price(self) -> float:
        """获取当前价格"""
        return float(self.data['close'].iloc[self.current_idx])
    
    def get_metrics(self) -> Dict:
        """获取回测指标"""
        total_return = (self.portfolio_value - self.initial_cash) / self.initial_cash
        
        return {
            'total_return': total_return,
            'final_value': self.portfolio_value,
            'final_position': self.position,
            'final_cash': self.cash,
        }


# ============= PPO策略网络 =============

class PPOPolicy(nn.Module):
    """PPO策略网络（带NaN保护）"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        
        # 特征提取器
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 策略头
        self.mean_head = nn.Linear(prev_dim, 1)
        self.log_std = nn.Parameter(torch.ones(1) * -1.0)
        
        # 初始化（小权重提高稳定性）
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播（带NaN检查）"""
        # NaN检查
        if torch.isnan(state).any():
            state = torch.nan_to_num(state, nan=0.0)
        
        features = self.feature_extractor(state)
        
        # NaN检查
        if torch.isnan(features).any():
            features = torch.zeros_like(features)
        
        mean = torch.tanh(self.mean_head(features))  # [-1, 1]
        std = torch.exp(self.log_std)
        std = torch.clamp(std, 1e-3, 1.0)
        
        return mean, std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样动作"""
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob


# ============= 训练函数 =============

def train_ppo(
    data_file: str,
    output_dir: str,
    config_file: Optional[str] = None,
    timesteps: int = 100000,
    lr: float = 3e-4,
    batch_size: int = 64,
    clip_range: float = 0.1,
) -> str:
    """训练PPO模型"""
    
    # 1. 加载数据
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    df = data['df']
    feature_cols = data['feature_cols']
    norm_stats = data['norm_stats']
    
    # 2. 加载配置
    if config_file:
        with open(config_file) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'initial_cash': 100000,
            'transaction_cost': 0.001,
            'max_steps': 240,
            'stop_loss': 0.15,
            'take_profit': 0.20,
            'max_position': 0.95,
            'min_position': 0.0,
        }
    
    # 3. 创建模拟器
    simulator = SafeTradingSimulator(df, feature_cols, norm_stats, config)
    
    # 4. 创建策略
    input_dim = len(feature_cols) + 4  # 特征数 + 组合特征
    policy = PPOPolicy(input_dim=input_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    # 5. TensorBoard (可选)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    writer = None
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(output_path / 'logs')
    
    # 6. 训练循环
    print(f"Training PPO for {timesteps} timesteps...")
    
    episode = 0
    total_steps = 0
    
    while total_steps < timesteps:
        # 收集episode
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
            rewards.append(reward.total)
            log_probs.append(log_prob.item())
            
            state = next_state
            total_steps += 1
        
        # PPO更新
        if len(states) > 10:  # 最少10步才更新
            states_tensor = torch.FloatTensor(np.array(states))
            actions_tensor = torch.FloatTensor(np.array(actions)).unsqueeze(1)
            old_log_probs = torch.FloatTensor(np.array(log_probs))
            
            # 计算returns（简化版）
            returns = torch.FloatTensor(np.array(rewards))
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # 前向传播
            mean, std = policy(states_tensor)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions_tensor).squeeze()
            
            # PPO loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            ratio = torch.clamp(ratio, 0.1, 10.0)
            
            surr1 = ratio * returns
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * returns
            loss = -torch.min(surr1, surr2).mean()
            
            # 检查loss
            if not torch.isnan(loss) and not torch.isinf(loss):
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.1)
                
                optimizer.step()
                
                # 记录（如果 TensorBoard 可用）
                if writer:
                    writer.add_scalar('loss', loss.item(), total_steps)
        
        episode += 1
        
        if episode % 10 == 0:
            metrics = simulator.get_metrics()
            print(f"Episode {episode}, Steps {total_steps}, Return: {metrics['total_return']:.2%}")
            if writer:
                writer.add_scalar('return', metrics['total_return'], episode)
    
    # 7. 保存模型
    model_file = output_path / 'ppo_model.zip'
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'config': config,
        'feature_cols': feature_cols,
        'norm_stats': norm_stats,
    }, model_file)
    
    print(f"\nTraining complete! Model saved to {model_file}")
    if writer:
        writer.close()
    
    return str(model_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train PPO model")
    parser.add_argument('--data', required=True, help="Data file (pkl)")
    parser.add_argument('--output-dir', default='models', help="Output directory")
    parser.add_argument('--config', help="Config file (yaml)")
    parser.add_argument('--timesteps', type=int, default=100000, help="Total timesteps")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size")
    parser.add_argument('--clip-range', type=float, default=0.1, help="PPO clip range")
    
    args = parser.parse_args()
    
    model_file = train_ppo(
        data_file=args.data,
        output_dir=args.output_dir,
        config_file=args.config,
        timesteps=args.timesteps,
        lr=args.lr,
        batch_size=args.batch_size,
        clip_range=args.clip_range,
    )
    
    print(f"\nModel saved to: {model_file}")
