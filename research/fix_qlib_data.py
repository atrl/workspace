#!/usr/bin/env python3
"""
修复 Qlib binary 数据格式

问题：之前转换的数据中有异常值（1.030844e-39）
原因：可能是不支持的数据类型或字节序问题
"""
import pandas as pd
import numpy as np
from pathlib import Path
import struct

print("="*70)
print("修复 Qlib 数据")
print("="*70)

DATA_DIR = Path("/Users/rziiiii/.openclaw/workspace/qlib_data")
SOURCE_FILE = "/Users/rziiiii/.openclaw/workspace/data/soxl_soxx_vix_full.csv"

# 读取源数据
df = pd.read_csv(SOURCE_FILE, parse_dates=['date'])
df = df.set_index('date')

print(f"源数据: {len(df)} records")

# 检查数据质量
print("\n数据质量检查:")
print(f"  SOXL Close - min: {df['soxl_close'].min():.2f}, max: {df['soxl_close'].max():.2f}")
print(f"  VIX - min: {df['vix'].min():.2f}, max: {df['vix'].max():.2f}")
print(f"  NaN数量: {df.isna().sum().sum()}")

# 重新转换为 Qlib binary 格式
print("\n重新转换 Qlib binary 格式...")

def to_qlib_binary_v2(series: pd.Series, output_file: Path):
    """
    转换为 Qlib binary 格式（修复版）
    
    使用 float64 而非 float32，避免精度问题
    """
    dates = []
    values = []
    
    for idx, value in zip(series.index, series.values):
        # 跳过 NaN
        if pd.isna(value):
            continue
        
        # 确保值有效
        if not np.isfinite(value):
            continue
        
        dates.append(idx.toordinal())
        values.append(float(value))
    
    # 写入二进制
    with open(output_file, 'wb') as f:
        for date_ord, value in zip(dates, values):
            # 使用 int32 + float64
            f.write(struct.pack('<i', int(date_ord)))   # little-endian int32
            f.write(struct.pack('<d', float(value)))    # little-endian float64
    
    return len(dates)

# 转换特征
features = {
    'close': df['soxl_close'],
    'open': df['soxl_open'],
    'high': df['soxl_high'],
    'low': df['soxl_low'],
    'volume': df['soxl_volume'],
    'vix': df['vix'],
}

for name, series in features.items():
    output_file = DATA_DIR / "features" / "SOXL" / f"{name}.day.bin"
    count = to_qlib_binary_v2(series, output_file)
    print(f"  ✅ {name}.day.bin: {count} records")

# 验证数据
print("\n验证修复后的数据...")

def read_qlib_binary_v2(file_path: Path, n=5):
    """读取前N条记录验证"""
    data = []
    with open(file_path, 'rb') as f:
        for i in range(n):
            date_bytes = f.read(4)  # int32
            value_bytes = f.read(8)  # float64
            
            if not date_bytes or len(date_bytes) < 4:
                break
            
            date_ord = struct.unpack('<i', date_bytes)[0]
            value = struct.unpack('<d', value_bytes)[0]
            date = pd.Timestamp.fromordinal(date_ord)
            
            data.append((date, value))
    
    return data

# 验证 close 数据
close_file = DATA_DIR / "features" / "SOXL" / "close.day.bin"
sample_data = read_qlib_binary_v2(close_file, 5)

print("\n样本数据 (close.day.bin - 修复版):")
for date, value in sample_data:
    print(f"  {date.strftime('%Y-%m-%d')}: ${value:.2f}")

# 测试 Qlib API
print("\n测试 Qlib API...")

import qlib
from qlib.data import D

# 重新初始化
qlib.init(provider_uri=str(DATA_DIR))

# 加载数据
test_df = D.features(
    instruments=['SOXL'],
    fields=['$close', '$volume', '$vix'],
    start_time='2020-01-01',
    end_time='2020-01-10'
)

print("\n加载的数据（前5天）:")
print(test_df.head())

# 检查是否还有异常值
print("\n数据质量检查:")
close_values = test_df['$close'].values
print(f"  Min: {close_values.min():.2f}")
print(f"  Max: {close_values.max():.2f}")
print(f"  Mean: {close_values.mean():.2f}")

if close_values.min() < 1e-30:
    print("  ⚠️ 仍有异常小值！")
else:
    print("  ✅ 数据正常")

print("\n" + "="*70)
print("✅ Qlib 数据修复完成！")
print("="*70)
