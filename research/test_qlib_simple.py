#!/usr/bin/env python3
"""
创建并测试 Qlib 自定义 Provider
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Iterable

print("="*70)
print("创建 Qlib 自定义 Provider")
print("="*70)

# 加载源数据
df = pd.read_csv(
    "/Users/rziiiii/.openclaw/workspace/data/soxl_soxx_vix_full.csv",
    parse_dates=['date']
)
df = df.set_index('date')
print(f"✅ 加载数据: {len(df)} 条")

# 创建目录
QLIB_DIR = Path("/Users/rziiiii/.openclaw/workspace/qlib_soxl_data")
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

# 3. 特征
for field, col in [('$close', 'soxl_close'), ('$volume', 'soxl_volume'), ('$vix', 'vix')]:
    feature_df = pd.DataFrame({'value': df[col].values}, index=df.index)
    feature_df.to_csv(QLIB_DIR / "features/SOXL" / f"{field.lstrip('$')}.day.data")
print("✅ 特征: 3 个字段")

# 4. 测试 Qlib
print("\n" + "="*70)
print("测试 Qlib D.features()")
print("="*70)

try:
    import qlib
    from qlib.data import D
    
    # 初始化
    qlib.init(provider_uri=str(QLIB_DIR))
    print("✅ Qlib 初始化成功")
    
    # 测试加载
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
    else:
        print("\n⚠️ 数据为空")
    
except Exception as e:
    print(f"\n❌ 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
