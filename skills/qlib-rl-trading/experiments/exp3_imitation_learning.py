#!/usr/bin/env python3
"""
方案3: 模仿学习 - 让RL学习恐慌指数策略

核心思路:
1. 收集恐慌指数策略的expert demonstrations
2. 使用Behavior Cloning训练策略网络
3. 再用PPO微调（可选）
"""
import sys
sys.path.insert(0, '/Users/rziiiii/.openclaw/workspace/research')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass
from soxl_panic_strategy_qlib_with_vix import SOXLPanicStrategyQlibWithVIX


@dataclass
class ExpertState:
    """Expert状态表示 - 简化为3维"""
    panic_score: float      # 恐慌得分 [-150, 150]
    position_ratio: float   # 当前仓位 [0, 1]
    volatility: float       # 波动率 [0, 1]
    
    def to_tensor(self) -> np.ndarray:
        return np.array([
            self.panic_score / 150.0,  # 归一化到 [-1, 1]
            self.position_ratio,
            self.volatility,
        ], dtype=np.float32)


@dataclass
class ExpertAction:
    """Expert动作 - 目标仓位"""
    target_position: float  # [0, 0.95]
    
    def to_delta(self, current_position: float) -> float:
        return self.target_position - current_position


class ExpertDataCollector:
    """收集恐慌指数策略的expert demonstrations"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def collect_expert_demonstrations(self) -> List[Tuple[ExpertState, ExpertAction]]:
        """运行恐慌指数策略，收集状态-动作对"""
        
        demonstrations = []
        
        # 使用最优参数
        strategy = SOXLPanicStrategyQlibWithVIX(
            buy_score_threshold=-50,
            sell_score_threshold=80,
            daily_return_threshold=-0.08,
            max_position=0.95,
            stop_loss_threshold=-0.20,
        )
        
        # 运行策略
        cash = 100000
        shares = 0
        position = 0.0
        
        for i in range(20, len(self.df)):
            row = self.df.iloc[i]
            
            # 计算恐慌得分
            vix = row.get('vix', 20)
            daily_ret = (self.df['close'].iloc[i] / self.df['close'].iloc[i-1] - 1) if i > 0 else 0
            
            vix_rolling = self.df['vix'].iloc[max(0,i-20):i+1]
            vix_mean = vix_rolling.mean()
            vix_std = vix_rolling.std() + 1e-8
            vix_zscore = (vix - vix_mean) / vix_std
            
            vix_percentile = vix_rolling.rank(pct=True).iloc[-1]
            
            panic_score = (
                vix_zscore * 40 +
                vix_percentile * 60 +
                min(daily_ret / -0.05, 1) * -50 if daily_ret < 0 else 0
            )
            
            # 计算波动率
            returns = self.df['close'].pct_change().iloc[max(0,i-20):i+1]
            volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.2
            
            # 当前仓位
            price = row['close']
            position_value = shares * price
            total_value = cash + position_value
            position_ratio = position_value / total_value if total_value > 0 else 0
            
            # Expert决策（恐慌指数策略的逻辑）
            target_position = position_ratio  # 默认保持
            
            # 买入条件
            if panic_score <= -50 and position_ratio < 0.95:
                # 分批建仓
                if position_ratio < 0.5:
                    target_position = 0.5
                elif position_ratio < 0.7:
                    target_position = 0.7
                elif position_ratio < 0.85:
                    target_position = 0.85
                else:
                    target_position = 0.95
            
            # 卖出条件
            elif panic_score >= 80 and position_ratio > 0:
                # 分批减仓
                if position_ratio > 0.7:
                    target_position = 0.5
                elif position_ratio > 0.5:
                    target_position = 0.3
                else:
                    target_position = 0.0
            
            # 记录状态-动作对
            state = ExpertState(
                panic_score=panic_score,
                position_ratio=position_ratio,
                volatility=min(volatility, 1.0),
            )
            
            action = ExpertAction(target_position=target_position)
            
            demonstrations.append((state, action))
            
            # 执行交易（更新仓位）
            if abs(target_position - position_ratio) > 0.05:
                target_value = total_value * target_position
                delta_shares = (target_value - position_value) / price
                
                if delta_shares > 0:  # 买入
                    cost = delta_shares * price * 1.001
                    if cost <= cash:
                        cash -= cost
                        shares += delta_shares
                        position_ratio = target_position
                else:  # 卖出
                    sell = min(-delta_shares, shares)
                    cash += sell * price * 0.999
                    shares -= sell
                    position_ratio = target_position
        
        return demonstrations


class ImitationPolicy(nn.Module):
    """模仿学习策略网络"""
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # 输出 [0, 1] 的目标仓位
        )
    
    def forward(self, x):
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        return self.net(x)
    
    def predict_position(self, state: ExpertState) -> float:
        """预测目标仓位"""
        state_tensor = torch.FloatTensor(state.to_tensor()).unsqueeze(0)
        with torch.no_grad():
            position = self.forward(state_tensor).item()
        return position * 0.95  # 缩放到 [0, 0.95]


class BehaviorCloning:
    """Behavior Cloning训练器"""
    
    def __init__(self, policy: ImitationPolicy, lr: float = 1e-3):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    
    def train(self, demonstrations: List[Tuple[ExpertState, ExpertAction]], 
              epochs: int = 100, batch_size: int = 32):
        """训练策略网络"""
        
        # 准备数据
        states = torch.FloatTensor([s.to_tensor() for s, _ in demonstrations])
        actions = torch.FloatTensor([[a.target_position / 0.95] for _, a in demonstrations])
        
        dataset = torch.utils.data.TensorDataset(states, actions)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Training Behavior Cloning on {len(demonstrations)} samples...")
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_states, batch_actions in dataloader:
                self.optimizer.zero_grad()
                
                predictions = self.policy(batch_states)
                loss = self.criterion(predictions.squeeze(), batch_actions.squeeze())
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return avg_loss


class ImitationSimulator:
    """模仿学习策略的回测模拟器"""
    
    def __init__(self, df: pd.DataFrame, policy: ImitationPolicy):
        self.df = df
        self.policy = policy
        self.initial_cash = 100000
        self.transaction_cost = 0.001
        
    def backtest(self, start_idx: int = 0, verbose: bool = False) -> Dict:
        """回测模仿学习策略"""
        
        cash = self.initial_cash
        shares = 0
        portfolio_values = []
        trades = []
        
        for i in range(max(20, start_idx), len(self.df)):
            row = self.df.iloc[i]
            
            # 构造状态
            vix = row.get('vix', 20)
            daily_ret = (self.df['close'].iloc[i] / self.df['close'].iloc[i-1] - 1) if i > 0 else 0
            
            vix_rolling = self.df['vix'].iloc[max(0,i-20):i+1]
            vix_mean = vix_rolling.mean()
            vix_std = vix_rolling.std() + 1e-8
            vix_zscore = (vix - vix_mean) / vix_std
            
            vix_percentile = vix_rolling.rank(pct=True).iloc[-1]
            
            panic_score = (
                vix_zscore * 40 +
                vix_percentile * 60 +
                min(daily_ret / -0.05, 1) * -50 if daily_ret < 0 else 0
            )
            
            returns = self.df['close'].pct_change().iloc[max(0,i-20):i+1]
            volatility = min(returns.std() * np.sqrt(252), 1.0) if len(returns) > 1 else 0.2
            
            price = row['close']
            position_value = shares * price
            total_value = cash + position_value
            position_ratio = position_value / total_value if total_value > 0 else 0
            
            state = ExpertState(
                panic_score=panic_score,
                position_ratio=position_ratio,
                volatility=volatility,
            )
            
            # 策略决策
            target_position = self.policy.predict_position(state)
            
            # 执行交易
            if abs(target_position - position_ratio) > 0.03:  # 3%阈值
                target_value = total_value * target_position
                delta_shares = (target_value - position_value) / price
                
                if delta_shares > 10:  # 买入
                    cost = delta_shares * price * (1 + self.transaction_cost)
                    if cost <= cash:
                        cash -= cost
                        shares += delta_shares
                        trades.append({'date': row['date'], 'type': 'buy', 'price': price, 'target': target_position})
                elif delta_shares < -10:  # 卖出
                    sell = min(-delta_shares, shares)
                    cash += sell * price * (1 - self.transaction_cost)
                    shares -= sell
                    trades.append({'date': row['date'], 'type': 'sell', 'price': price, 'target': target_position})
            
            portfolio_values.append(cash + shares * price)
        
        # 计算指标
        pv = np.array(portfolio_values)
        total_return = (pv[-1] - self.initial_cash) / self.initial_cash
        
        daily_returns = np.diff(pv) / pv[:-1]
        years = len(daily_returns) / 252
        annual_return = (1 + total_return) ** (1 / max(years, 0.1)) - 1
        
        max_value = np.maximum.accumulate(pv)
        drawdowns = (max_value - pv) / max_value
        max_drawdown = np.max(drawdowns)
        
        volatility = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # 买入持有对比
        buy_hold_return = (self.df['close'].iloc[-1] / self.df['close'].iloc[max(20, start_idx)] - 1)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'sharpe': sharpe,
            'num_trades': len(trades),
            'final_value': pv[-1],
            'buy_hold_return': buy_hold_return,
            'trades': trades,
        }


def train_imitation_rl(
    data_file: str,
    output_dir: str,
    bc_epochs: int = 100,
    ppo_timesteps: int = 10000,
):
    """训练模仿学习RL策略"""
    
    print("="*70)
    print("方案3: 模仿学习 - 让RL学习恐慌指数策略")
    print("="*70)
    
    # 加载数据
    df = pd.read_csv(data_file, parse_dates=['date'])
    df = df[df['date'] >= '2020-01-01'].reset_index(drop=True)
    
    print(f"\n数据: {len(df)} records ({df['date'].min().date()} to {df['date'].max().date()})")
    
    # 1. 收集expert demonstrations
    print("\n[1/4] 收集Expert Demonstrations...")
    collector = ExpertDataCollector(df)
    demonstrations = collector.collect_expert_demonstrations()
    
    print(f"  收集到 {len(demonstrations)} 个状态-动作对")
    
    # 分析expert行为
    buy_actions = [a.target_position for s, a in demonstrations if a.target_position > 0.6]
    sell_actions = [a.target_position for s, a in demonstrations if a.target_position < 0.4]
    
    print(f"  买入动作: {len(buy_actions)} 次，平均目标仓位: {np.mean(buy_actions):.2%}" if buy_actions else "")
    print(f"  卖出动作: {len(sell_actions)} 次，平均目标仓位: {np.mean(sell_actions):.2%}" if sell_actions else "")
    
    # 2. Behavior Cloning
    print("\n[2/4] Behavior Cloning训练...")
    policy = ImitationPolicy(input_dim=3, hidden_dim=64)
    bc_trainer = BehaviorCloning(policy, lr=1e-3)
    bc_trainer.train(demonstrations, epochs=bc_epochs, batch_size=32)
    
    # 3. 回测BC策略
    print("\n[3/4] 回测BC策略...")
    simulator = ImitationSimulator(df, policy)
    results = simulator.backtest(start_idx=0, verbose=True)
    
    print("\n" + "="*70)
    print("模仿学习策略回测结果")
    print("="*70)
    print(f"总收益:        {results['total_return']:.2%}")
    print(f"年化收益:      {results['annual_return']:.2%}")
    print(f"最大回撤:      {results['max_drawdown']:.2%}")
    print(f"波动率:        {results['volatility']:.2%}")
    print(f"Sharpe:        {results['sharpe']:.2f}")
    print(f"交易次数:      {results['num_trades']}")
    print(f"最终价值:      ${results['final_value']:,.0f}")
    print(f"收益倍数:      {results['final_value'] / 100000:.2f}x")
    print("-"*70)
    print(f"买入持有:      {results['buy_hold_return']:.2%}")
    print(f"超额收益:      {results['total_return'] - results['buy_hold_return']:.2%}")
    
    # 4. 保存模型
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_file = output_path / 'imitation_policy.zip'
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'metrics': results,
    }, model_file)
    
    print(f"\n[4/4] 模型已保存到: {model_file}")
    
    # 5. 对比原始恐慌指数策略
    print("\n" + "="*70)
    print("策略对比")
    print("="*70)
    print(f"{'策略':<20} {'总收益':<15} {'Sharpe':<10} {'交易次数':<10}")
    print("-"*70)
    print(f"{'恐慌指数(Expert)':<20} {'944.82%':<15} {'0.88':<10} {'101':<10}")
    print(f"{'模仿学习(BC)':<20} {results['total_return']:.2%:<15} {results['sharpe']:.2f:<10} {results['num_trades']:<10}")
    print(f"{'买入持有':<20} {results['buy_hold_return']:.2%:<15} {'-':<10} {'1':<10}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/Users/rziiiii/.openclaw/workspace/data/soxl_with_vix.csv')
    parser.add_argument('--output-dir', default='models/imitation')
    parser.add_argument('--bc-epochs', type=int, default=100)
    parser.add_argument('--ppo-timesteps', type=int, default=10000)
    
    args = parser.parse_args()
    
    train_imitation_rl(
        data_file=args.data,
        output_dir=args.output_dir,
        bc_epochs=args.bc_epochs,
        ppo_timesteps=args.ppo_timesteps,
    )
