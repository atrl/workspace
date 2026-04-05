#!/usr/bin/env python3
"""
准备完整的 SOXL + VIX 数据（2010-2026）

数据来源：
  - SOXL: Yahoo Finance (2010-03-11 到 2026-04-02, 4041条)
  - VIX: Yahoo Finance (^VIX)
  
输出：
  - data/soxl_with_vix_full.csv (2010-2026)
"""
import yfinance as yf
import pandas as pd

print("="*70)
print("下载完整 SOXL + VIX 数据")
print("="*70)

# 1. 下载 SOXL
print("\n下载 SOXL...")
soxl = yf.Ticker("SOXL")
df_soxl = soxl.history(start="2010-03-11", end="2026-04-05")
print(f"SOXL: {len(df_soxl)} records ({df_soxl.index.min()} to {df_soxl.index.max()})")

# 2. 下载 VIX
print("\n下载 VIX...")
vix = yf.Ticker("^VIX")
df_vix = vix.history(start="2010-03-11", end="2026-04-05")
print(f"VIX:   {len(df_vix)} records ({df_vix.index.min()} to {df_vix.index.max()})")

# 3. 合并
print("\n合并数据...")
df = df_soxl[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
df.columns = ['open', 'high', 'low', 'close', 'volume']

# 添加 VIX
df_vix_clean = df_vix[['Close']].copy()
df_vix_clean.columns = ['vix']

# 按日期合并
df = df.join(df_vix_clean, how='left')

# 前向填充 VIX 缺失值
df['vix'] = df['vix'].fillna(method='ffill')

# 重置索引
df = df.reset_index()
df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'vix']

# 4. 保存
output_file = '/Users/rziiiii/.openclaw/workspace/data/soxl_with_vix_full.csv'
df.to_csv(output_file, index=False)

print(f"\n✅ 已保存: {output_file}")
print(f"   总记录数: {len(df)}")
print(f"   日期范围: {df['date'].min()} 到 {df['date'].max()}")
print(f"   VIX 缺失: {df['vix'].isna().sum()}")

# 5. 显示统计
print("\n" + "="*70)
print("数据统计")
print("="*70)
print(f"{'年份':<8} {'记录数':>8} {'SOXL范围':>30}")
print("-"*70)

for year in range(2010, 2027):
    year_data = df[df['date'].dt.year == year]
    if len(year_data) > 0:
        print(f"{year:<8} {len(year_data):>8} ${year_data['close'].min():.2f} - ${year_data['close'].max():.2f}")

print("="*70)

# 6. 对比新旧数据
print("\n对比新旧数据:")
old_file = '/Users/rziiiii/.openclaw/workspace/data/soxl_with_vix.csv'
df_old = pd.read_csv(old_file, parse_dates=['date'])

print(f"旧数据: {len(df_old)} records ({df_old['date'].min()} to {df_old['date'].max()})")
print(f"新数据: {len(df)} records ({df['date'].min()} to {df['date'].max()})")
print(f"增加:   {len(df) - len(df_old)} records (+{(len(df) / len(df_old) - 1) * 100:.1f}%)")

print("\n✅ 数据准备完成！")
