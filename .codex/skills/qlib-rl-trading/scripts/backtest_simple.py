#!/usr/bin/env python3
"""
Simple Backtest - 简化版回测
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# 导入训练脚本中的类
from train_simple_ppo import SimpleTradingSimulator, TradingAction


def backtest(model_file: str, data_file: str, verbose: bool = True):
    """回测模型"""
    
    # 加载模型
    checkpoint = torch.load(model_file, weights_only=False)
    feature_cols = checkpoint['feature_cols']
    norm_stats = checkpoint['norm_stats']
    
    # 加载数据
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    df = data['df']
    
    # 创建简单策略
    from train_simple_ppo import SimplePolicy
    input_dim = len(feature_cols) + 4
    policy = SimplePolicy(input_dim=input_dim)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    # 创建模拟器
    simulator = SimpleTradingSimulator(df, feature_cols, norm_stats)
    
    if verbose:
        print(f"Backtesting from {df['date'].iloc[0]} to {df['date'].iloc[-1]}...")
    
    # 回测（从头开始）
    state = simulator.reset(start_idx=0)
    done = False
    
    portfolio_values = []
    positions = []
    trades = 0
    
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
            trades += 1
        
        state = next_state
    
    # 计算指标
    metrics = calculate_metrics(
        portfolio_values=np.array(portfolio_values),
        positions=np.array(positions),
        trades=trades,
        initial_cash=100000,
    )
    
    if verbose:
        print_metrics(metrics)
    
    return metrics


def calculate_metrics(portfolio_values, positions, trades, initial_cash):
    """计算指标"""
    
    # 收益
    total_return = (portfolio_values[-1] - initial_cash) / initial_cash
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    annual_return = (1 + total_return) ** (252 / len(daily_returns)) - 1 if len(daily_returns) > 0 else 0
    
    # 风险
    max_value = np.maximum.accumulate(portfolio_values)
    drawdowns = (max_value - portfolio_values) / max_value
    max_drawdown = np.max(drawdowns)
    
    volatility = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0
    
    # 风险调整
    sharpe = annual_return / volatility if volatility > 0 else 0
    calmar = annual_return / max_drawdown if max_drawdown > 0 else sharpe
    
    # 交易
    avg_position = np.mean(positions)
    turnover = trades / (len(positions) / 252)
    
    return {
        'total_return': float(total_return),
        'annual_return': float(annual_return),
        'max_drawdown': float(max_drawdown),
        'volatility': float(volatility),
        'sharpe': float(sharpe),
        'calmar': float(calmar),
        'num_trades': int(trades),
        'avg_position': float(avg_position),
        'turnover': float(turnover),
        'final_value': float(portfolio_values[-1]),
    }


def print_metrics(metrics):
    """打印指标"""
    print("\n" + "="*60)
    print("Backtest Results")
    print("="*60)
    print(f"Total Return:      {metrics['total_return']:.2%}")
    print(f"Annual Return:     {metrics['annual_return']:.2%}")
    print(f"Max Drawdown:      {metrics['max_drawdown']:.2%}")
    print(f"Volatility:        {metrics['volatility']:.2%}")
    print(f"Sharpe Ratio:      {metrics['sharpe']:.2f}")
    print(f"Calmar Ratio:      {metrics['calmar']:.2f}")
    print(f"Num Trades:        {metrics['num_trades']}")
    print(f"Avg Position:      {metrics['avg_position']:.2%}")
    print(f"Turnover Rate:     {metrics['turnover']:.1f}x")
    print(f"Final Value:       ${metrics['final_value']:,.0f}")
    print("="*60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    
    args = parser.parse_args()
    
    backtest(args.model, args.data)
