#!/usr/bin/env python3
"""
扩大训练数据：使用完整的 2020-2026 数据

当前问题：
  - SOXL 数据从 2020-01-02 开始（1571条记录）
  - 没有找到 2012 年的数据
  
解决方案：
  1. 使用全部 2020-2026 数据（6年+）
  2. 调整训练/测试比例（80%/20%）
  3. 增加数据增强
  
注意：
  - SOXL ETF 成立于 2010-03-11
  - 要获取 2012-2019 数据需要从外部 API 获取
"""
import yfinance as yf
import pandas as pd
from datetime import datetime

print("="*70)
print("检查 SOXL 历史数据")
print("="*70)

# 1. 检查本地数据
local_file = '/Users/rziiiii/.openclaw/workspace/data/soxl_with_vix.csv'
df_local = pd.read_csv(local_file, parse_dates=['date'])

print(f"\n本地数据:")
print(f"  开始日期: {df_local['date'].min()}")
print(f"  结束日期: {df_local['date'].max()}")
print(f"  记录数:   {len(df_local)}")

# 2. 尝试从 Yahoo Finance 获取更多历史数据
print("\n尝试从 Yahoo Finance 获取 SOXL 历史...")
try:
    ticker = yf.Ticker("SOXL")
    df_yahoo = ticker.history(start="2010-03-11", end="2026-04-05")
    
    print(f"\nYahoo Finance 数据:")
    print(f"  开始日期: {df_yahoo.index.min()}")
    print(f"  结束日期: {df_yahoo.index.max()}")
    print(f"  记录数:   {len(df_yahoo)}")
    
    if len(df_yahoo) > len(df_local):
        print(f"\n✅ Yahoo Finance 有更多数据 ({len(df_yahoo)} vs {len(df_local)})")
        
        # 保存
        df_yahoo.to_csv('/Users/rziiiii/.openclaw/workspace/data/soxl_yahoo.csv')
        print(f"已保存: data/soxl_yahoo.csv")
        
        # 显示早期数据样本
        print(f"\n早期数据样本（2010-2012）:")
        early = df_yahoo[df_yahoo.index < '2012-01-01']
        if len(early) > 0:
            print(early.head(10))
        else:
            print("  无数据（可能尚未交易）")
    else:
        print(f"\n⚠️ Yahoo Finance 数据与本地相同或更少")
        
except Exception as e:
    print(f"\n❌ 无法从 Yahoo Finance 获取数据: {e}")
    print("\n可能原因:")
    print("  1. 需要安装 yfinance: pip install yfinance")
    print("  2. 网络问题")
    print("  3. API 限制")

print("\n" + "="*70)
print("当前可用数据范围")
print("="*70)
print(f"本地数据: 2020-01-02 到 2026-04-01 (1571 records)")
print(f"训练集建议: 2020-01-02 到 2025-04-01 (80%)")
print(f"测试集建议: 2025-04-02 到 2026-04-01 (20%)")
print()
print("扩大数据方案:")
print("  1. 使用全部 2020-2026 数据（当前可用）")
print("  2. 从 Yahoo Finance 获取 2010-2019 数据（需要 yfinance）")
print("  3. 从其他数据源获取（Alpha Vantage, Quandl 等）")
