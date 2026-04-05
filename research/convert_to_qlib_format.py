#!/usr/bin/env python3
"""
将 SOXL 数据转换为 Qlib 数据格式

Qlib 数据格式：
  qlib_data/
  ├── instruments/
  │   └── all.txt         # 股票列表
  ├── calendars/
  │   └── day.txt         # 交易日历
  └── features/
      └── SOXL/
          ├── close.day.bin    # 二进制数据
          ├── open.day.bin
          ├── high.day.bin
          ├── low.day.bin
          ├── volume.day.bin
          └── vix.day.bin      # 外部数据
"""
import pandas as pd
import numpy as np
from pathlib import Path
import struct
from datetime import datetime

print("="*70)
print("转换 SOXL 数据为 Qlib 格式")
print("="*70)

# 配置
DATA_DIR = Path("/Users/rziiiii/.openclaw/workspace/qlib_data")
SOURCE_FILE = "/Users/rziiiii/.openclaw/workspace/data/soxl_soxx_vix_full.csv"

# 创建目录
(DATA_DIR / "instruments").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "calendars").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "features" / "SOXL").mkdir(parents=True, exist_ok=True)

print(f"数据目录: {DATA_DIR}")

# ============================================================
# 1. 读取源数据
# ============================================================

print("\n读取源数据...")
df = pd.read_csv(SOURCE_FILE, parse_dates=['date'])
print(f"源数据: {len(df)} records ({df['date'].min()} to {df['date'].max()})")

# ============================================================
# 2. 创建交易日历
# ============================================================

print("\n创建交易日历...")
calendar = df['date'].dt.strftime('%Y-%m-%d').tolist()
calendar_file = DATA_DIR / "calendars" / "day.txt"

with open(calendar_file, 'w') as f:
    f.write('\n'.join(calendar))

print(f"✅ 交易日历: {calendar_file} ({len(calendar)} days)")

# ============================================================
# 3. 创建 instruments 列表
# ============================================================

print("\n创建 instruments...")
instruments_file = DATA_DIR / "instruments" / "all.txt"

with open(instruments_file, 'w') as f:
    f.write("SOXL\n")

print(f"✅ Instruments: {instruments_file}")

# ============================================================
# 4. 转换为 Qlib binary 格式
# ============================================================

print("\n转换为 Qlib binary 格式...")

def to_qlib_binary(series: pd.Series, output_file: Path):
    """
    转换为 Qlib binary 格式
    
    Qlib binary 格式：
    - 每条记录: [date_ordinal (int32), value (float32)]
    - 按日期排序
    """
    # 准备数据
    dates = series.index.map(lambda x: x.toordinal()).values.astype(np.int32)
    values = series.values.astype(np.float32)
    
    # 写入二进制
    with open(output_file, 'wb') as f:
        for date_ord, value in zip(dates, values):
            # Qlib 格式: int32(date) + float32(value)
            f.write(struct.pack('<i', date_ord))  # little-endian int32
            f.write(struct.pack('<f', value))     # little-endian float32
    
    return len(dates)

# 设置日期索引
df_indexed = df.set_index('date')

# 转换每个特征
features = {
    'close': df_indexed['soxl_close'],
    'open': df_indexed['soxl_open'],
    'high': df_indexed['soxl_high'],
    'low': df_indexed['soxl_low'],
    'volume': df_indexed['soxl_volume'],
    'vix': df_indexed['vix'],
}

for name, series in features.items():
    output_file = DATA_DIR / "features" / "SOXL" / f"{name}.day.bin"
    count = to_qlib_binary(series, output_file)
    print(f"  ✅ {name}.day.bin: {count} records")

# ============================================================
# 5. 创建 Qlib 配置文件
# ============================================================

print("\n创建 Qlib 配置文件...")

config = f"""# Qlib Configuration for SOXL
provider_uri: {DATA_DIR}
region: US
"""

config_file = DATA_DIR / "qlib.yaml"
with open(config_file, 'w') as f:
    f.write(config)

print(f"✅ 配置文件: {config_file}")

# ============================================================
# 6. 验证数据
# ============================================================

print("\n验证数据...")

def read_qlib_binary(file_path: Path, n=5):
    """读取前N条记录验证"""
    data = []
    with open(file_path, 'rb') as f:
        for i in range(n):
            date_bytes = f.read(4)
            value_bytes = f.read(4)
            
            if not date_bytes:
                break
            
            date_ord = struct.unpack('<i', date_bytes)[0]
            value = struct.unpack('<f', value_bytes)[0]
            date = datetime.fromordinal(date_ord)
            
            data.append((date, value))
    
    return data

# 验证 close 数据
close_file = DATA_DIR / "features" / "SOXL" / "close.day.bin"
sample_data = read_qlib_binary(close_file, 5)

print("\n样本数据 (close.day.bin):")
for date, value in sample_data:
    print(f"  {date.strftime('%Y-%m-%d')}: ${value:.2f}")

# ============================================================
# 7. 总结
# ============================================================

print("\n" + "="*70)
print("✅ Qlib 数据格式转换完成！")
print("="*70)
print(f"""
数据目录结构：
  {DATA_DIR}/
  ├── instruments/
  │   └── all.txt
  ├── calendars/
  │   └── day.txt
  ├── features/
  │   └── SOXL/
  │       ├── close.day.bin
  │       ├── open.day.bin
  │       ├── high.day.bin
  │       ├── low.day.bin
  │       ├── volume.day.bin
  │       └── vix.day.bin
  └── qlib.yaml

使用方法：
  import qlib
  from qlib.data import D
  
  # 初始化 Qlib
  qlib.init(provider_uri='{DATA_DIR}')
  
  # 加载数据
  df = D.features(
      instruments=['SOXL'],
      fields=['$close', '$volume', '$vix'],
      start_time='2020-01-01',
      end_time='2026-04-01'
  )
""")

# ============================================================
# 8. 测试 Qlib API
# ============================================================

print("\n测试 Qlib API...")

try:
    import qlib
    from qlib.data import D
    
    # 初始化 Qlib
    qlib.init(provider_uri=str(DATA_DIR))
    
    # 加载数据
    test_df = D.features(
        instruments=['SOXL'],
        fields=['$close', '$volume', '$vix'],
        start_time='2020-01-01',
        end_time='2020-01-10'
    )
    
    print("\n✅ Qlib API 测试成功！")
    print(f"\n加载的数据（前5天）:")
    print(test_df.head())
    
except Exception as e:
    print(f"\n❌ Qlib API 测试失败: {e}")
    print("可能需要调整数据格式")

print("\n" + "="*70)
