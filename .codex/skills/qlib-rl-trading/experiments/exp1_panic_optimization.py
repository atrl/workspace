#!/usr/bin/env python3
"""
Experiment 1: Panic Strategy Parameter Optimization

网格搜索最优参数组合
"""
import sys
sys.path.insert(0, '/Users/rziiiii/.openclaw/workspace/research')

import pandas as pd
import numpy as np
from itertools import product
from soxl_panic_strategy_qlib_with_vix import SOXLPanicStrategyQlibWithVIX

# 加载数据
df = pd.read_csv('/Users/rziiiii/.openclaw/workspace/data/soxl_with_vix.csv', parse_dates=['date'])
df = df[df['date'] >= '2020-01-01'].reset_index(drop=True)

print("="*60)
print("Experiment 1: Panic Strategy Parameter Optimization")
print("="*60)
print(f"Data: {len(df)} records, {df['date'].min().date()} to {df['date'].max().date()}")
print()

# 参数网格
param_grid = {
    'buy_score_threshold': [-30, -40, -50, -60],
    'sell_score_threshold': [75, 80, 85, 90],
    'daily_return_threshold': [-0.05, -0.08, -0.10],
    'max_position': [0.80, 0.90, 0.95],
    'stop_loss_threshold': [-0.10, -0.15, -0.20],
}

# 网格搜索
best_result = {'return': -999, 'sharpe': -999}
all_results = []

print("Running grid search...")
total_combinations = 1
for vals in param_grid.values():
    total_combinations *= len(vals)

print(f"Total combinations: {total_combinations}")

count = 0
for buy_thresh, sell_thresh, daily_ret, max_pos, stop_loss in product(
    param_grid['buy_score_threshold'],
    param_grid['sell_score_threshold'],
    param_grid['daily_return_threshold'],
    param_grid['max_position'],
    param_grid['stop_loss_threshold']
):
    count += 1
    
    strategy = SOXLPanicStrategyQlibWithVIX(
        buy_score_threshold=buy_thresh,
        sell_score_threshold=sell_thresh,
        daily_return_threshold=daily_ret,
        max_position=max_pos,
        stop_loss_threshold=stop_loss,
    )
    
    results = strategy.run_backtest(df, verbose=False)
    
    all_results.append({
        'buy_thresh': buy_thresh,
        'sell_thresh': sell_thresh,
        'daily_ret': daily_ret,
        'max_pos': max_pos,
        'stop_loss': stop_loss,
        'total_return': results['total_return'],
        'sharpe': results['sharpe'],
        'max_drawdown': results['max_drawdown'],
    })
    
    if results['total_return'] > best_result['return']:
        best_result = {
            'return': results['total_return'],
            'sharpe': results['sharpe'],
            'max_dd': results['max_drawdown'],
            'params': {
                'buy_score_threshold': buy_thresh,
                'sell_score_threshold': sell_thresh,
                'daily_return_threshold': daily_ret,
                'max_position': max_pos,
                'stop_loss_threshold': stop_loss,
            }
        }
    
    if count % 50 == 0:
        print(f"  Progress: {count}/{total_combinations} ({count/total_combinations*100:.1f}%)")

# 排序结果
df_results = pd.DataFrame(all_results)
df_results = df_results.sort_values('total_return', ascending=False)

print("\n" + "="*60)
print("Top 5 Parameter Combinations")
print("="*60)
print(df_results.head(5).to_string())

print("\n" + "="*60)
print("Best Parameters")
print("="*60)
print(f"Best Return: {best_result['return']:.2%}")
print(f"Best Sharpe: {best_result['sharpe']:.2f}")
print(f"Best Max DD: {best_result['max_dd']:.2%}")
print(f"\nParameters:")
for k, v in best_result['params'].items():
    print(f"  {k}: {v}")
