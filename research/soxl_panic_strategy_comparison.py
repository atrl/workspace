#!/usr/bin/env python3
"""
SOXL Panic Strategy - Comparison Script

对比原始策略和Qlib版本

Author: clawQuant
Date: 2026-04-02
"""

import pandas as pd
import sys

# 导入两个版本
from soxl_panic_strategy_original_v2 import OriginalPanicStrategy
from soxl_panic_strategy_qlib_v2 import SOXLPanicStrategyQlibV2


def load_data_from_csv(csv_path: str) -> pd.DataFrame:
    """从CSV加载数据"""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def print_comparison(original_results: dict, qlib_results: dict):
    """打印对比结果"""
    print("\n" + "="*80)
    print("策略对比结果")
    print("="*80)
    
    # 表头
    print(f"{'指标':<20} {'原始策略':>20} {'Qlib版本':>20} {'差异':>15}")
    print("-"*80)
    
    # 对比项
    metrics = [
        ('初始资金', 'initial_capital', '${:,.0f}', False),
        ('最终资金', 'final_value', '${:,.0f}', False),
        ('总收益率', 'total_return', '{:.2%}', True),
        ('年化收益率', 'annual_return', '{:.2%}', True),
        ('Sharpe比率', 'sharpe', '{:.2f}', True),
        ('最大回撤', 'max_drawdown', '{:.2%}', True),
        ('总交易次数', 'total_trades', '{}', False),
        ('最终仓位', 'final_position', '{:.2%}', False),
    ]
    
    for label, key, fmt, is_percent in metrics:
        orig_val = original_results[key]
        qlib_val = qlib_results[key]
        
        if is_percent:
            diff = qlib_val - orig_val
            diff_str = f"{diff:+.2%}"
        else:
            if isinstance(orig_val, (int, float)) and key not in ['total_trades']:
                diff = qlib_val - orig_val
                diff_str = f"{diff:+,.0f}"
            else:
                diff_str = f"{int(qlib_val - orig_val):+d}"
        
        print(f"{label:<20} {fmt.format(orig_val):>20} {fmt.format(qlib_val):>20} {diff_str:>15}")
    
    print("="*80)
    
    # 分析
    print("\n分析:")
    if qlib_results['total_return'] > original_results['total_return']:
        print(f"✓ Qlib版本总收益率更高 (+{(qlib_results['total_return'] - original_results['total_return']):.2%})")
    else:
        print(f"✗ Qlib版本总收益率更低 ({(qlib_results['total_return'] - original_results['total_return']):.2%})")
    
    if qlib_results['sharpe'] > original_results['sharpe']:
        print(f"✓ Qlib版本Sharpe比率更高 ({qlib_results['sharpe']:.2f} vs {original_results['sharpe']:.2f})")
    else:
        print(f"✗ Qlib版本Sharpe比率更低 ({qlib_results['sharpe']:.2f} vs {original_results['sharpe']:.2f})")
    
    if qlib_results['max_drawdown'] < original_results['max_drawdown']:
        print(f"✓ Qlib版本最大回撤更小 ({qlib_results['max_drawdown']:.2%} vs {original_results['max_drawdown']:.2%})")
    else:
        print(f"✗ Qlib版本最大回撤更大 ({qlib_results['max_drawdown']:.2%} vs {original_results['max_drawdown']:.2%})")


if __name__ == '__main__':
    # 加载数据
    csv_path = '/Users/rziiiii/.openclaw/workspace/data/symbols/SOXL.csv'
    df = load_data_from_csv(csv_path)
    
    # 过滤日期
    df = df[df['date'] >= '2020-01-01']
    
    print("="*80)
    print("SOXL Panic Strategy - 策略对比")
    print("="*80)
    print(f"数据: {len(df)} 条记录")
    print(f"日期范围: {df['date'].min().strftime('%Y-%m-%d')} 到 {df['date'].max().strftime('%Y-%m-%d')}")
    print()
    
    # 运行原始策略
    print("运行原始策略...")
    original_strategy = OriginalPanicStrategy()
    original_results = original_strategy.run_backtest(df, verbose=False)
    
    # 运行Qlib版本
    print("运行Qlib版本...")
    qlib_strategy = SOXLPanicStrategyQlibV2()
    qlib_results = qlib_strategy.run_backtest(df, verbose=False)
    
    # 打印对比
    print_comparison(original_results, qlib_results)
