#!/usr/bin/env python3
"""
运行优化后的策略并对比结果
"""
import sys
sys.path.insert(0, 'research')

from strategy_optimization import (
    EnhancedFeatures,
    EnhancedHybridStrategy,
    EnhancedStrategyParams,
    EnhancedOptimizer,
)
import pandas as pd
import numpy as np
from datetime import datetime


def run_optimized_strategy(
    data_file: str = '/Users/rziiiii/.openclaw/workspace/data/soxl_with_vix.csv',
    optimize: bool = True,
    verbose: bool = True,
):
    """运行优化后的策略"""
    
    print("="*70)
    print("优化后的策略")
    print("="*70)
    
    # 1. 加载数据
    print("\n加载数据...")
    df = pd.read_csv(data_file, parse_dates=['date'])
    df = df[df['date'] >= '2020-01-01'].reset_index(drop=True)
    df = df.set_index('date')
    print(f"  加载 {len(df)} 条记录")
    
    # 2. 计算增强特征
    print("\n计算增强特征...")
    df = EnhancedFeatures.calculate_all(df)
    print(f"  ✅ 特征数量: {len(df.columns)}")
    
    # 3. 参数优化
    if optimize:
        print("\n" + "="*70)
        print("运行参数优化...")
        print("="*70)
        
        optimizer = EnhancedOptimizer(population_size=50, generations=20)
        params = optimizer.optimize(df, metric='sharpe', verbose=True)
        
        print("\n" + "="*70)
        print("优化后的参数")
        print("="*70)
        print(f"  恐慌买入阈值:   {params.panic_buy_threshold:.1f}")
        print(f"  恐慌基础仓位:   {params.panic_position_base:.2%}")
        print(f"  恐慌最大仓位:   {params.panic_position_max:.2%}")
        print(f"  趋势基础仓位:   {params.trend_position_base:.2%}")
        print(f"  趋势最大仓位:   {params.trend_position_max:.2%}")
        print(f"  VIX恐慌阈值:    {params.vix_panic:.1f}")
        print(f"  VIX极端阈值:    {params.vix_extreme:.1f}")
        print(f"  VIX贪婪阈值:    {params.vix_greed:.1f}")
        print(f"  硬止损:         {params.stop_loss:.2%}")
        print(f"  移动止损:       {params.trailing_stop:.2%}")
    else:
        params = EnhancedStrategyParams()
    
    # 4. 运行回测
    print("\n" + "="*70)
    print("回测优化后的策略")
    print("="*70)
    
    strategy = EnhancedHybridStrategy(params)
    strategy.reset()
    
    cash = 100000
    shares = 0
    avg_price = 0
    
    portfolio_values = [100000]
    dates = [df.index[29]]
    trades = []
    
    panic_count = 0
    trend_count = 0
    stop_loss_count = 0
    
    for i in range(30, len(df)):
        row = df.iloc[i]
        
        position_value = shares * row['close']
        total_value = cash + position_value
        current_position = position_value / total_value if total_value > 0 else 0
        
        # 获取目标仓位
        target_position, strategy_name, confidence = strategy.get_target_position(
            row, current_position, total_value
        )
        
        # 判断是否交易
        if strategy.should_trade(current_position, target_position):
            target_value = total_value * target_position
            delta_shares = (target_value - position_value) / row['close']
            
            if delta_shares > 10:
                cost = delta_shares * row['close'] * 1.001
                if cost <= cash:
                    total_cost = avg_price * shares + row['close'] * delta_shares
                    shares += delta_shares
                    avg_price = total_cost / shares
                    cash -= cost
                    
                    trades.append({
                        'date': df.index[i],
                        'type': 'buy',
                        'strategy': strategy_name,
                        'confidence': confidence,
                    })
                    
                    if 'panic' in strategy_name:
                        panic_count += 1
                    elif 'trend' in strategy_name:
                        trend_count += 1
                    
                    strategy.last_trade_day = strategy.current_day
            
            elif delta_shares < -10:
                sell = min(-delta_shares, shares)
                cash += sell * row['close'] * 0.999
                shares -= sell
                
                trades.append({
                    'date': df.index[i],
                    'type': 'sell',
                    'strategy': strategy_name,
                    'confidence': confidence,
                })
                
                if 'stop' in strategy_name:
                    stop_loss_count += 1
                elif 'panic' in strategy_name:
                    panic_count += 1
                elif 'trend' in strategy_name:
                    trend_count += 1
                
                strategy.last_trade_day = strategy.current_day
        
        portfolio_values.append(cash + shares * row['close'])
        dates.append(df.index[i])
    
    # 5. 计算结果
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
    calmar = annual_return / max_drawdown if max_drawdown > 0 else sharpe
    
    buy_hold = (df['close'].iloc[-1] / df['close'].iloc[30] - 1)
    
    print("\n" + "="*70)
    print("优化后策略回测结果")
    print("="*70)
    print(f"总收益:        {total_return:.2%}")
    print(f"年化收益:      {annual_return:.2%}")
    print(f"最大回撤:      {max_drawdown:.2%}")
    print(f"波动率:        {volatility:.2%}")
    print(f"Sharpe:        {sharpe:.2f}")
    print(f"Calmar:        {calmar:.2f}")
    print("-"*70)
    print(f"交易次数:      {len(trades)}")
    print(f"  恐慌策略:    {panic_count}")
    print(f"  趋势策略:    {trend_count}")
    print(f"  止损触发:    {stop_loss_count}")
    print("-"*70)
    print(f"最终价值:      ${pv[-1]:,.0f}")
    print(f"收益倍数:      {pv[-1] / 100000:.2f}x")
    print("-"*70)
    print(f"买入持有:      {buy_hold:.2%}")
    print(f"超额收益:      {total_return - buy_hold:.2%}")
    
    # 6. 对比
    print("\n" + "="*70)
    print("优化前后对比")
    print("="*70)
    print(f"{'指标':<20} {'优化前':<15} {'优化后':<15} {'变化':<15}")
    print("-"*70)
    print(f"{'总收益':<20} {'146.64%':<15} {total_return:.2%}{'':<10} {total_return - 1.4664:+.2%}")
    print(f"{'最大回撤':<20} {'49.40%':<15} {max_drawdown:.2%}{'':<10} {max_drawdown - 0.4940:+.2%}")
    print(f"{'Sharpe':<20} {'0.44':<15} {sharpe:.2f}{'':<11} {sharpe - 0.44:+.2f}")
    print(f"{'交易次数':<20} {'405':<15} {len(trades):<15} {len(trades) - 405:+d}")
    print(f"{'收益倍数':<20} {'2.47x':<15} {pv[-1] / 100000:.2f}x{'':<11} {pv[-1] / 100000 - 2.47:+.2f}x")
    
    return {
        'params': params,
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'trades': trades,
    }


if __name__ == '__main__':
    results = run_optimized_strategy(optimize=True, verbose=True)
