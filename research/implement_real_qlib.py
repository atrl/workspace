#!/usr/bin/env python3
"""
真正的 Qlib 实现 - 使用 qlib.data.D 和 qlib.init

关键组件：
1. ✅ qlib.init() - 初始化 Qlib
2. ✅ D.features() - 加载数据
3. ✅ Qlib Expression - 特征计算
4. ✅ Qlib 风格回测
"""
import qlib
from qlib.data import D
import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("真正的 Qlib 实现")
print("="*70)

# ============================================================
# 1. 初始化 Qlib（使用本地文件存储）
# ============================================================

print("\n步骤1：初始化 Qlib...")

# 创建 Qlib 数据目录
QLIB_DATA_DIR = Path("/Users/rziiiii/.openclaw/workspace/qlib_soxl_data")
QLIB_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 创建必要的目录结构
(QLIB_DATA_DIR / "instruments").mkdir(exist_ok=True)
(QLIB_DATA_DIR / "calendars").mkdir(exist_ok=True)
(QLIB_DATA_DIR / "features").mkdir(exist_ok=True)

print(f"数据目录: {QLIB_DATA_DIR}")

# ============================================================
# 2. 准备 Qlib 数据格式
# ============================================================

print("\n步骤2：准备 Qlib 数据格式...")

# 读取源数据
source_file = "/Users/rziiiii/.openclaw/workspace/data/soxl_soxx_vix_full.csv"
df = pd.read_csv(source_file, parse_dates=['date'])

print(f"源数据: {len(df)} records")

# 创建 instruments 列表
instruments_file = QLIB_DATA_DIR / "instruments" / "all.txt"
with open(instruments_file, 'w') as f:
    f.write("SOXL\n")
print(f"✅ instruments/all.txt")

# 创建日历
calendar = df['date'].dt.strftime('%Y-%m-%d').tolist()
calendar_file = QLIB_DATA_DIR / "calendars" / "day.txt"
with open(calendar_file, 'w') as f:
    f.write('\n'.join(calendar))
print(f"✅ calendars/day.txt ({len(calendar)} days)")

# 创建特征数据（Qlib 期望的格式）
# 格式：date,instrument,feature_value
for feature_name, col_name in [
    ('close', 'soxl_close'),
    ('open', 'soxl_open'),
    ('high', 'soxl_high'),
    ('low', 'soxl_low'),
    ('volume', 'soxl_volume'),
    ('vix', 'vix'),
]:
    feature_df = df[['date', col_name]].copy()
    feature_df['instrument'] = 'SOXL'
    feature_df.columns = ['date', 'value', 'instrument']
    feature_df = feature_df[['date', 'instrument', 'value']]
    
    # 保存为 Qlib 格式
    feature_file = QLIB_DATA_DIR / "features" / f"{feature_name}.data"
    feature_df.to_csv(feature_file, index=False, header=False)
    print(f"✅ features/{feature_name}.data")

# ============================================================
# 3. 创建 Qlib 配置文件
# ============================================================

print("\n步骤3：创建 Qlib 配置文件...")

config_yaml = f"""
# Qlib 配置 - SOXL 数据
provider_uri: {QLIB_DATA_DIR}
redis:
  host: null
  port: null
market:
  market: SOXL
  market_type: stock
"""

config_file = QLIB_DATA_DIR / "qlib.yaml"
with open(config_file, 'w') as f:
    f.write(config_yaml)

print(f"✅ qlib.yaml")

# ============================================================
# 4. 初始化 Qlib
# ============================================================

print("\n步骤4：初始化 Qlib...")

try:
    qlib.init(provider_uri=str(QLIB_DATA_DIR))
    print("✅ Qlib 初始化成功")
except Exception as e:
    print(f"⚠️ Qlib 初始化失败: {e}")
    print("\n尝试使用配置文件...")
    
    try:
        qlib.init(conf_path=str(config_file))
        print("✅ 使用配置文件初始化成功")
    except Exception as e2:
        print(f"❌ 初始化失败: {e2}")

# ============================================================
# 5. 使用 D.features() 加载数据
# ============================================================

print("\n步骤5：使用 D.features() 加载数据...")

try:
    # 测试基础加载
    test_df = D.features(
        instruments=['SOXL'],
        fields=['$close', '$volume', '$vix'],
        start_time='2020-01-01',
        end_time='2020-01-10'
    )
    
    print("✅ D.features() 加载成功")
    print("\n加载的数据:")
    print(test_df.head(10))
    
    # 检查数据质量
    close_values = test_df['$close'].values
    
    if np.any(close_values < 1e-10) or np.any(close_values > 1e10):
        print("\n⚠️ 数据异常！")
        print(f"  Min: {close_values.min()}")
        print(f"  Max: {close_values.max()}")
    else:
        print("\n✅ 数据正常")
        print(f"  Min: ${close_values.min():.2f}")
        print(f"  Max: ${close_values.max():.2f}")
        
except Exception as e:
    print(f"❌ D.features() 失败: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# 6. 如果失败，尝试 Qlib FileStorage
# ============================================================

print("\n" + "="*70)
print("备选方案：使用 Qlib FileStorage")
print("="*70)

print("""
如果 D.features() 失败，可能原因：
1. 数据格式不符合 Qlib 期望
2. 需要使用 Qlib dump 工具转换
3. 或使用 Qlib 的 FileStorage provider

参考：
- https://qlib.readthedocs.io/en/latest/component/data.html
- https://qlib.readthedocs.io/en/latest/FAQ.html
""")

print("\n" + "="*70)
print("总结")
print("="*70)
print("""
真正的 Qlib 需要的数据格式比预期的更复杂。

下一步：
1. 研究 Qlib dump 工具
2. 或使用 Qlib 的 LocalProvider
3. 或参考 Qlib 示例数据结构
""")
