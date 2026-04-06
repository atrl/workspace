#!/usr/bin/env python3
"""
使用 Qlib FileStorage Provider

FileStorage 更简单，直接用 CSV 文件
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

print("="*70)
print("创建 Qlib FileStorage 数据")
print("="*70)

# ============================================================
# 1. 准备目录
# ============================================================

QLIB_DIR = Path("/Users/rziiiii/.openclaw/workspace/qlib_filestorage")
QLIB_DATA_DIR = QLIB_DIR / "soxl_data"
(QLIB_DATA_DIR / "instruments").mkdir(parents=True, exist_ok=True)
(QLIB_DATA_DIR / "calendars").mkdir(parents=True, exist_ok=True)
(QLIB_DATA_DIR / "features").mkdir(parents=True, exist_ok=True)

print(f"✅ Qlib FileStorage 目录: {QLIB_DIR}")

# ============================================================
# 2. 加载源数据
# ============================================================

print("\n加载源数据...")
df = pd.read_csv(
    "/Users/rziiiii/.openclaw/workspace/data/soxl_soxx_vix_full.csv",
    parse_dates=['date']
)
df = df.set_index('date')

print(f"✅ 加载 {len(df)} 条记录 ({df.index.min()} to {df.index.max()})")

# ============================================================
# 3. 创建日历
# ============================================================

print("\n创建日历...")
calendar = df.index.strftime('%Y-%m-%d').tolist()
calendar_file = QLIB_DATA_DIR / "calendars" / "day.txt"
with open(calendar_file, 'w') as f:
    f.write('\n'.join(calendar))
print(f"✅ 日历: {len(calendar)} 天")

# ============================================================
# 4. 创建 instruments
# ============================================================

print("\n创建 instruments...")
# Qlib instruments 格式：instrument\tstart_date\tend_date
instruments_file = QLIB_DATA_DIR / "instruments" / "all.txt"
start_date = df.index[0].strftime('%Y-%m-%d')
end_date = df.index[-1].strftime('%Y-%m-%d')
with open(instruments_file, 'w') as f:
    f.write(f"SOXL\t{start_date}\t{end_date}\n")
print(f"✅ Instruments: SOXL ({start_date} to {end_date})")

# ============================================================
# 5. 创建特征文件（CSV格式，FileStorage用）
# ============================================================

print("\n创建特征文件...")

# FileStorage 格式：每只股票一个目录，每个特征一个CSV
soxl_dir = QLIB_DATA_DIR / "features" / "SOXL"
soxl_dir.mkdir(parents=True, exist_ok=True)

# 转换特征
features_map = {
    'close': 'soxl_close',
    'open': 'soxl_open',
    'high': 'soxl_high',
    'low': 'soxl_low',
    'volume': 'soxl_volume',
    'vix': 'vix',
}

for qlib_name, csv_name in features_map.items():
    # Qlib FileStorage 格式：date,value
    feature_df = pd.DataFrame({
        'date': df.index,
        'value': df[csv_name].values
    })
    feature_df = feature_df.set_index('date')
    
    # 保存
    feature_file = soxl_dir / f"{qlib_name}.day.data"
    feature_df.to_csv(feature_file)
    print(f"  ✅ {qlib_name}: {len(feature_df)} values")

# ============================================================
# 6. 创建 Qlib 配置
# ============================================================

print("\n创建 Qlib 配置...")

config_content = f"""# Qlib FileStorage 配置
provider_uri: {QLIB_DATA_DIR}
redis:
  host: null
  port: null
"""

config_file = QLIB_DIR / "qlib.yaml"
with open(config_file, 'w') as f:
    f.write(config_content)

print(f"✅ 配置: {config_file}")

# ============================================================
# 7. 测试 Qlib 加载
# ============================================================

print("\n" + "="*70)
print("测试 Qlib FileStorage")
print("="*70)

try:
    import qlib
    from qlib.data import D
    
    # 初始化
    qlib.init(conf_path=str(config_file))
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
    if len(test_df) > 0 and not test_df.empty:
        close_values = test_df['$close'].dropna().values
        
        if len(close_values) > 0:
            if np.all(close_values > 0) and np.all(close_values < 1e6):
                print(f"\n✅ 数据正常")
                print(f"   Min: ${close_values.min():.2f}")
                print(f"   Max: ${close_values.max():.2f}")
                print(f"   Mean: ${close_values.mean():.2f}")
            else:
                print(f"\n⚠️ 数据异常")
        else:
            print(f"\n⚠️ 无有效数据")
    else:
        print(f"\n⚠️ 数据为空")
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("完成！")
print("="*70)
print(f"""
Qlib FileStorage 目录: {QLIB_DATA_DIR}

如果测试成功，下一步：
1. 使用 qlib.init(conf_path='{config_file}')
2. 使用 D.features() 加载数据
3. 使用 Qlib Expression 计算特征

示例：
    import qlib
    from qlib.data import D
    
    qlib.init(conf_path='{config_file}')
    
    df = D.features(
        instruments=['SOXL'],
        fields=['$close', '$volume', 'Mean($close, 20)', 'RSI($close, 14)'],
        start_time='2010-01-01',
        end_time='2026-04-01'
    )
""")
