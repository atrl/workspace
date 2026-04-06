#!/usr/bin/env python3
"""
将 SOXL 数据转换为 Qlib 格式

步骤：
1. 创建 Qlib 目录结构
2. 转换 CSV 为 Qlib binary 格式
3. 创建配置文件
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("转换 SOXL 数据为 Qlib 格式")
print("="*70)

# ============================================================
# 1. 准备目录
# ============================================================

QLIB_DIR = Path("/Users/rziiiii/.openclaw/workspace/qlib_soxl_data")
(QLIB_DIR / "instruments").mkdir(parents=True, exist_ok=True)
(QLIB_DIR / "calendars").mkdir(parents=True, exist_ok=True)
(QLIB_DIR / "features").mkdir(parents=True, exist_ok=True)

print(f"✅ Qlib 数据目录: {QLIB_DIR}")

# ============================================================
# 2. 加载源数据
# ============================================================

print("\n加载源数据...")
df = pd.read_csv(
    "/Users/rziiiii/.openclaw/workspace/data/soxl_soxx_vix_full.csv",
    parse_dates=['date']
)
df = df.sort_values('date')
print(f"✅ 加载 {len(df)} 条记录 ({df['date'].min()} to {df['date'].max()})")

# ============================================================
# 3. 创建日历
# ============================================================

print("\n创建 Qlib 日历...")
calendar = df['date'].dt.strftime('%Y-%m-%d').tolist()
calendar_file = QLIB_DIR / "calendars" / "day.txt"
with open(calendar_file, 'w') as f:
    f.write('\n'.join(calendar))
print(f"✅ 日历: {len(calendar)} 天")

# ============================================================
# 4. 创建 instruments
# ============================================================

print("\n创建 Qlib instruments...")
instruments_file = QLIB_DIR / "instruments" / "all.txt"
with open(instruments_file, 'w') as f:
    f.write("SOXL\n")
print(f"✅ Instruments: SOXL")

# ============================================================
# 5. 转换特征数据为 Qlib binary 格式
# ============================================================

print("\n转换特征数据...")

# Qlib binary 格式说明：
# - 小端字节序
# - 每个值是 float64 (8 bytes)
# - 按 instrument 分文件
# - 文件名: features/SOXL/close.day.bin

features = {
    'close': df['soxl_close'].values,
    'open': df['soxl_open'].values,
    'high': df['soxl_high'].values,
    'low': df['soxl_low'].values,
    'volume': df['soxl_volume'].values,
    'vix': df['vix'].values,
}

for feature_name, values in features.items():
    # 转换为 float64
    values = values.astype(np.float64)
    
    # 写入 binary 文件
    bin_file = QLIB_DIR / "features" / "SOXL" / f"{feature_name}.day.bin"
    bin_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(bin_file, 'wb') as f:
        values.tofile(f)
    
    print(f"  ✅ {feature_name}: {len(values)} values")

# ============================================================
# 6. 创建 Qlib 配置文件
# ============================================================

print("\n创建 Qlib 配置文件...")

config_yaml = f"""# Qlib 配置 - SOXL 数据
provider_uri: {QLIB_DIR}
redis:
  host: null
  port: null
market:
  market: SOXL
  market_type: stock
"""

config_file = QLIB_DIR / "qlib.yaml"
with open(config_file, 'w') as f:
    f.write(config_yaml)

print(f"✅ 配置: {config_file}")

# ============================================================
# 7. 测试加载
# ============================================================

print("\n" + "="*70)
print("测试 Qlib 数据加载")
print("="*70)

try:
    import qlib
    from qlib.data import D
    
    # 初始化 Qlib
    qlib.init(provider_uri=str(QLIB_DIR))
    print("✅ Qlib 初始化成功")
    
    # 测试加载
    test_df = D.features(
        instruments=['SOXL'],
        fields=['$close', '$volume', '$vix'],
        start_time='2020-01-01',
        end_time='2020-01-10'
    )
    
    print("\n✅ D.features() 加载成功:")
    print(test_df.head(10))
    
    # 检查数据质量
    if len(test_df) > 0:
        close_values = test_df['$close'].values
        
        if np.all(close_values > 0) and np.all(close_values < 1e6):
            print(f"\n✅ 数据正常")
            print(f"   Min: ${close_values.min():.2f}")
            print(f"   Max: ${close_values.max():.2f}")
            print(f"   Mean: ${close_values.mean():.2f}")
        else:
            print(f"\n⚠️ 数据异常")
            print(f"   Min: {close_values.min()}")
            print(f"   Max: {close_values.max()}")
    else:
        print("\n⚠️ 数据为空")
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("完成！")
print("="*70)
print(f"""
Qlib 数据目录: {QLIB_DIR}

下一步：
1. 在 RL 策略中使用 qlib.init(provider_uri='{QLIB_DIR}')
2. 使用 D.features() 加载数据
3. 使用 Qlib Expression 计算特征
""")
