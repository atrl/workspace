#!/usr/bin/env python3
"""
Backtest RL Strategy

回测RL策略并输出完整指标
"""
import argparse
import pickle
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
import torch

from train_ppo import SafeTradingSimulator, TradingAction, PPOPolicy


def backtest_rl(
    model_file: str,
    data_file: str,
    start: str = None,
    end: str = None,
    verbose: bool = True,
) -> Dict:
    """回测RL策略"""
    
    # 1. 加载模型
    checkpoint = torch.load(model_file)
    config = checkpoint['config']
    feature_cols = checkpoint['feature_cols']
    norm_stats = checkpoint['norm_stats']
    
    policy = PPOPolicy(input_dim=len(feature_cols) + 4)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    # 2. 加载数据
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    df = data['df']
    if start:
        df = df[df['date'] >= start]
    if end:
        df = df[df['date'] <= end]
    
    # 3. 创建模拟器
    simulator = SafeTradingSimulator(df, feature_cols, norm_stats, config)
    
    # 4. 回测
    if verbose:
        print(f"Backtesting from {df['date'].iloc[0]} to {df['date'].iloc[-1]}...")
    
    state = simulator.reset(start_idx=0)  # 从头开始
    done = False
    
    portfolio_values = []
    positions = []
    trades = []
    
    while not done:
        state_tensor = torch.FloatTensor(state.to_tensor()).unsqueeze(0)
        
        with torch.no_grad():
            mean, std = policy(state_tensor)
            action_tensor = mean  # 确定性策略
        
        action = TradingAction(position_delta=action_tensor.item())
        next_state, reward, done, info = simulator.step(action)
        
        portfolio_values.append(info['portfolio_value'])
        positions.append(info['position'])
        
        if info['trade_executed']:
            trades.append({
                'step': simulator.current_step,
                'action': action.position_delta,
                'position': info['position'],
            })
        
        state = next_state
    
    # 5. 计算指标
    metrics = calculate_metrics(
        portfolio_values=np.array(portfolio_values),
        positions=np.array(positions),
        trades=trades,
        initial_cash=config['initial_cash'],
    )
    
    if verbose:
        print_metrics(metrics)
    
    return metrics


def calculate_metrics(
    portfolio_values: np.ndarray,
    positions: np.ndarray,
    trades: list,
    initial_cash: float,
) -> Dict:
    """计算回测指标"""
    
    # 1. 收益指标
    total_return = (portfolio_values[-1] - initial_cash) / initial_cash
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    annual_return = (1 + total_return) ** (252 / len(daily_returns)) - 1 if len(daily_returns) > 0 else 0
    
    # 2. 风险指标
    max_value = np.maximum.accumulate(portfolio_values)
    drawdowns = (max_value - portfolio_values) / max_value
    max_drawdown = np.max(drawdowns)
    
    volatility = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0
    
    # 3. 风险调整指标
    sharpe = annual_return / volatility if volatility > 0 else 0
    sortino = annual_return / (np.std(daily_returns[daily_returns < 0]) * np.sqrt(252)) if len(daily_returns[daily_returns < 0]) > 1 else sharpe
    calmar = annual_return / max_drawdown if max_drawdown > 0 else sharpe
    
    # 4. 交易指标
    num_trades = len(trades)
    avg_position = np.mean(positions)
    turnover_rate = num_trades / (len(positions) / 252)  # 年换手率
    
    return {
        'return': {
            'total_return': float(total_return),
            'annual_return': float(annual_return),
        },
        'risk': {
            'max_drawdown': float(max_drawdown),
            'volatility': float(volatility),
        },
        'risk_adjusted': {
            'sharpe': float(sharpe),
            'sortino': float(sortino),
            'calmar': float(calmar),
        },
        'trading': {
            'num_trades': int(num_trades),
            'avg_position': float(avg_position),
            'turnover_rate': float(turnover_rate),
        },
        'portfolio': {
            'final_value': float(portfolio_values[-1]),
            'final_position': float(positions[-1]),
        }
    }


def print_metrics(metrics: Dict):
    """打印指标"""
    print("\n" + "="*60)
    print("Backtest Results")
    print("="*60)
    
    print("\nReturns:")
    print(f"  Total Return:      {metrics['return']['total_return']:.2%}")
    print(f"  Annual Return:     {metrics['return']['annual_return']:.2%}")
    
    print("\nRisk:")
    print(f"  Max Drawdown:      {metrics['risk']['max_drawdown']:.2%}")
    print(f"  Volatility:        {metrics['risk']['volatility']:.2%}")
    
    print("\nRisk-Adjusted:")
    print(f"  Sharpe Ratio:      {metrics['risk_adjusted']['sharpe']:.2f}")
    print(f"  Sortino Ratio:     {metrics['risk_adjusted']['sortino']:.2f}")
    print(f"  Calmar Ratio:      {metrics['risk_adjusted']['calmar']:.2f}")
    
    print("\nTrading:")
    print(f"  Num Trades:        {metrics['trading']['num_trades']}")
    print(f"  Avg Position:      {metrics['trading']['avg_position']:.2%}")
    print(f"  Turnover Rate:     {metrics['trading']['turnover_rate']:.1f}x")
    
    print("\nPortfolio:")
    print(f"  Final Value:       ${metrics['portfolio']['final_value']:,.0f}")
    print(f"  Final Position:    {metrics['portfolio']['final_position']:.2%}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Backtest RL strategy")
    parser.add_argument('--model', required=True, help="Model file (zip)")
    parser.add_argument('--data', required=True, help="Data file (pkl)")
    parser.add_argument('--start', help="Start date")
    parser.add_argument('--end', help="End date")
    
    args = parser.parse_args()
    
    metrics = backtest_rl(
        model_file=args.model,
        data_file=args.data,
        start=args.start,
        end=args.end,
    )
