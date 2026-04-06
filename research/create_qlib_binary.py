#!/usr/bin/env python3
"""
创建正确的 Qlib Binary 格式数据

Qlib binary 格式：
- 小端字节序（<）
- float32（f）
- 第一个值：起始索引
- 后续值：数据
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("创建 Qlib Binary 格式数据")
print("="*70)

# 加载数据
df = pd.read_csv(
    "/Users/rziiiii/.openclaw/workspace/data/soxl_soxx_vix_full.csv",
    parse_dates=['date']
)
df = df.set_index('date').sort_index()

print(f"✅ 数据: {len(df)} 条 ({df.index.min()} to {df.index.max()})")

# 目录
QLIB_DIR = Path("/Users/rziiiii/.openclaw/workspace/qlib_soxl_binary")
(QLIB_DIR / "calendars").mkdir(parents=True, exist_ok=True)
(QLIB_DIR / "instruments").mkdir(parents=True, exist_ok=True)
(QLIB_DIR / "features/SOXL").mkdir(parents=True, exist_ok=True)

# 1. 日历
calendar = df.index.strftime('%Y-%m-%d').tolist()
with open(QLIB_DIR / "calendars" / "day.txt", 'w') as f:
    f.write('\n'.join(calendar))
print(f"✅ 日历: {len(calendar)} 天")

# 2. Instruments
with open(QLIB_DIR / "instruments" / "all.txt", 'w') as f:
    f.write(f"SOXL\t{df.index[0].strftime('%Y-%m-%d')}\t{df.index[-1].strftime('%Y-%m-%d')}\n")
print("✅ Instruments: SOXL")

# 3. 特征 - Binary 格式
print("\n创建 Binary 特征...")

features_map = {
    'close': 'soxl_close',
    'open': 'soxl_open',
    'high': 'soxl_high',
    'low': 'soxl_low',
    'volume': 'soxl_volume',
    'vix': 'vix',
}

for qlib_name, csv_name in features_map.items():
    # 获取数据
    values = df[csv_name].values
    
    # 创建 binary 数据
    # 格式：[起始索引, value1, value2, ...]
    # 起始索引 = 0（从第0天开始）
    binary_data = np.hstack([[0], values]).astype('<f')  # 小端 float32
    
    # 写入 binary 文件
    bin_file = QLIB_DIR / "features/SOXL" / f"{qlib_name}.day.bin"
    with open(bin_file, 'wb') as f:
        binary_data.tofile(f)
    
    print(f"  ✅ {qlib_name}: {len(values)} values, {bin_file.stat().st_size} bytes")

# 4. 配置文件
config_yaml = f"""# Qlib 配置
provider_uri: {QLIB_DIR}
redis:
  host: null
  port: null
market:
  market: SOXL
  market_type: stock
"""

with open(QLIB_DIR / "qlib.yaml", 'w') as f:
    f.write(config_yaml)

print(f"✅ 配置: qlib.yaml")

# 5. 测试
print("\n" + "="*70)
print("测试 Qlib D.features()")
print("="*70)

try:
    import qlib
    from qlib.data import D
    
    qlib.init(provider_uri=str(QLIB_DIR))
    print("✅ Qlib 初始化成功")
    
    test_df = D.features(
        instruments=['SOXL'],
        fields=['$close', '$volume', '$vix'],
        start_time='2020-01-01',
        end_time='2020-01-10'
    )
    
    print("\n✅ D.features() 成功:")
    print(test_df)
    
    if not test_df.empty:
        print(f"\n✅ 数据正常，行数: {len(test_df)}")
        
        # 检查值
        close_values = test_df['$close'].dropna().values
        if len(close_values) > 0:
            print(f"   Min: ${close_values.min():.2f}")
            print(f"   Max: ${close_values.max():.2f}")
    else:
        print("\n⚠️ 数据为空")

except Exception as e:
    print(f"\n❌ 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("完成！")
print("="*70)
