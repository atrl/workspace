#!/usr/bin/env python3
"""
紧急恢复脚本：重新生成 Qlib 数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
import subprocess

print("=" * 60)
print("Qlib 数据恢复")
print("=" * 60)

workspace = Path('/Users/rziiiii/.openclaw/workspace')

# 检查是否有备份的 CSV 数据
csv_backup_paths = [
    workspace / 'data' / 'symbols' / 'SOXL.csv',
    Path.home() / '.qlib' / 'qlib_data' / 'csv' / 'soxl.csv',
]

csv_path = None
for path in csv_backup_paths:
    if path.exists():
        csv_path = path
        print(f"✅ 找到 CSV 数据: {csv_path}")
        break

if csv_path is None:
    print("❌ 未找到 CSV 数据，需要重新下载")
    print("\n建议：从 yfinance 下载")
    print("运行: python3 scripts/fetch_soxl_data.py")
    exit(1)

# 显示 CSV 信息
df = pd.read_csv(csv_path)
print(f"   行数: {len(df)}")
print(f"   列: {df.columns.tolist()}")
print(f"   时间范围: {df.iloc[0, 0]} 至 {df.iloc[-1, 0]}")

print("\n" + "=" * 60)
print("请手动运行 Qlib 数据转换")
print("=" * 60)
print("\n参考命令:")
print(f"python3 -m qlib/scripts/dump_bin.py dump_all \\")
print(f"  --csv_path {csv_path.parent} \\")
print(f"  --qlib_dir {workspace / 'qlib_data'} \\")
print(f"  --symbol_field_name symbol \\")
print(f"  --date_field_name date \\")
print(f"  --include_fields open,high,low,close,volume")
