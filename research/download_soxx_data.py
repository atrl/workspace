#!/usr/bin/env python3
"""
下载 SOXX + VIX 完整数据

数据：
  - SOXX: 1x 半导体ETF（基准）
  - SOXL: 3x 杠杆半导体ETF（交易标的）
  - VIX:  恐慌指数（风险指标）
"""
import yfinance as yf
import pandas as pd

print("="*70)
print("下载 SOXX + SOXL + VIX 数据")
print("="*70)

# 1. 下载 SOXX
print("\n下载 SOXX...")
soxx = yf.Ticker("SOXX")
df_soxx = soxx.history(start="2010-03-11", end="2026-04-05", auto_adjust=False)
print(f"SOXX: {len(df_soxx)} records ({df_soxx.index.min()} to {df_soxx.index.max()})")

# 2. 下载 SOXL
print("\n下载 SOXL...")
soxl = yf.Ticker("SOXL")
df_soxl = soxl.history(start="2010-03-11", end="2026-04-05", auto_adjust=False)
print(f"SOXL: {len(df_soxl)} records ({df_soxl.index.min()} to {df_soxl.index.max()})")

# 3. 下载 VIX
print("\n下载 VIX...")
vix = yf.Ticker("^VIX")
df_vix = vix.history(start="2010-03-11", end="2026-04-05", auto_adjust=False)
print(f"VIX:   {len(df_vix)} records ({df_vix.index.min()} to {df_vix.index.max()})")

# 4. 合并数据
print("\n合并数据...")

# 准备 SOXL 数据（主要交易标的）
df = df_soxl[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
df.columns = ['soxl_open', 'soxl_high', 'soxl_low', 'soxl_close', 'soxl_volume']

# 添加 SOXX 数据（基准）
df_soxx_clean = df_soxx[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
df_soxx_clean.columns = ['soxx_open', 'soxx_high', 'soxx_low', 'soxx_close', 'soxx_volume']

# 添加 VIX 数据
df_vix_clean = df_vix[['Close']].copy()
df_vix_clean.columns = ['vix']

# 统一时区
df.index = df.index.tz_localize(None)
df_soxx_clean.index = df_soxx_clean.index.tz_localize(None)
df_vix_clean.index = df_vix_clean.index.tz_localize(None)

# 合并
df = df.join(df_soxx_clean, how='left')
df = df.join(df_vix_clean, how='left')

# 前向填充
df = df.ffill()

# 重置索引
df = df.reset_index()
df.columns = ['date', 
              'soxl_open', 'soxl_high', 'soxl_low', 'soxl_close', 'soxl_volume',
              'soxx_open', 'soxx_high', 'soxx_low', 'soxx_close', 'soxx_volume',
              'vix']

# 5. 保存
output_file = '/Users/rziiiii/.openclaw/workspace/data/soxl_soxx_vix_full.csv'
df.to_csv(output_file, index=False)

print(f"\n✅ 已保存: {output_file}")
print(f"   总记录数: {len(df)}")
print(f"   日期范围: {df['date'].min()} 到 {df['date'].max()}")
print(f"   缺失值:")
print(f"     SOXL: {df['soxl_close'].isna().sum()}")
print(f"     SOXX: {df['soxx_close'].isna().sum()}")
print(f"     VIX:  {df['vix'].isna().sum()}")

# 6. 显示前5行
print("\n前5行:")
print(df.head())

# 7. 显示统计
print("\n" + "="*70)
print("数据统计")
print("="*70)

print(f"\n{'年份':<8} {'记录数':>8} {'SOXL范围':>25} {'SOXX范围':>25}")
print("-"*70)

for year in range(2010, 2027):
    year_data = df[df['date'].dt.year == year]
    if len(year_data) > 0:
        soxl_min = year_data['soxl_close'].min()
        soxl_max = year_data['soxl_close'].max()
        soxx_min = year_data['soxx_close'].min()
        soxx_max = year_data['soxx_close'].max()
        print(f"{year:<8} {len(year_data):>8} ${soxl_min:>6.2f} - ${soxl_max:>6.2f}   ${soxx_min:>6.2f} - ${soxx_max:>6.2f}")

print("="*70)

# 8. 计算相关性
print("\n" + "="*70)
print("相关性分析")
print("="*70)

correlation = df['soxl_close'].corr(df['soxx_close'])
print(f"SOXL vs SOXX 相关性: {correlation:.4f}")

correlation_vix = df['soxl_close'].corr(df['vix'])
print(f"SOXL vs VIX 相关性:  {correlation_vix:.4f}")

print("\n✅ 数据准备完成！")
