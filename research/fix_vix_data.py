#!/usr/bin/env python3
"""
修复 VIX 数据并重新保存
"""
import yfinance as yf
import pandas as pd

print("下载完整数据...")

# 下载 SOXL
soxl = yf.Ticker("SOXL")
df_soxl = soxl.history(start="2010-03-11", end="2026-04-05", auto_adjust=False)

# 下载 VIX
vix = yf.Ticker("^VIX")
df_vix = vix.history(start="2010-03-11", end="2026-04-05", auto_adjust=False)

# 准备 SOXL 数据
df = df_soxl[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
df.columns = ['open', 'high', 'low', 'close', 'volume']

# 准备 VIX 数据
df_vix_clean = df_vix[['Close']].copy()
df_vix_clean.columns = ['vix']

# 统一时区（移除时区信息）
df.index = df.index.tz_localize(None)
df_vix_clean.index = df_vix_clean.index.tz_localize(None)

# 合并
df = df.join(df_vix_clean, how='left')

# 前向填充 VIX
df['vix'] = df['vix'].ffill()

# 重置索引
df = df.reset_index()
df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'vix']

# 保存
output_file = '/Users/rziiiii/.openclaw/workspace/data/soxl_with_vix_full.csv'
df.to_csv(output_file, index=False)

print(f"✅ 已保存: {output_file}")
print(f"   总记录数: {len(df)}")
print(f"   日期范围: {df['date'].min()} 到 {df['date'].max()}")
print(f"   VIX 缺失: {df['vix'].isna().sum()}")

# 显示前5行和后5行
print("\n前5行:")
print(df.head())
print("\n后5行:")
print(df.tail())
