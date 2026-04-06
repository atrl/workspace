#!/usr/bin/env python3
"""
使用 Qlib Storage API 创建自定义数据 Provider

参考：
https://qlib.readthedocs.io/en/latest/component/data.html#storage-api
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Iterable
import re

print("="*70)
print("创建 Qlib 自定义 Storage Provider")
print("="*70)

# ============================================================
# 1. 实现 Qlib Storage 接口
# ============================================================

from qlib.data.storage import CalendarStorage, InstrumentStorage, FeatureStorage, CalVT, InstVT, InstKT

class SimpleCalendarStorage(CalendarStorage):
    """自定义日历存储"""
    
    def __init__(self, freq: str, future: bool, **kwargs):
        super().__init__(freq, future, **kwargs)
        self._data = None
    
    @property
    def data(self) -> Iterable[CalVT]:
        """返回日历数据"""
        if self._data is None:
            # 从文件加载
            calendar_file = Path("qlib_soxl_data/calendars/day.txt")
            if calendar_file.exists():
                with open(calendar_file) as f:
                    self._data = [line.strip() for line in f if line.strip()]
            else:
                raise ValueError(f"Calendar file not found: {calendar_file}")
        return self._data


class SimpleInstrumentStorage(InstrumentStorage):
    """自定义 Instrument 存储"""
    
    def __init__(self, market: str, **kwargs):
        super().__init__(**kwargs)
        self.market = market
        self._data = None
    
    @property
    def data(self) -> Dict[InstKT, InstVT]:
        """返回 instruments 数据"""
        if self._data is None:
            # 从文件加载
            inst_file = Path("qlib_soxl_data/instruments/all.txt")
            if inst_file.exists():
                with open(inst_file) as f:
                    lines = f.readlines()
                    self._data = {}
                    for line in lines:
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            self._data[parts[0]] = [(parts[1], parts[2])]
                        elif len(parts) == 1:
                            # 默认范围
                            self._data[parts[0]] = [('2010-01-01', '2026-12-31')]
            else:
                # 默认数据
                self._data = {'SOXL': [('2010-03-11', '2026-04-02')]}
        return self._data


class SimpleFeatureStorage(FeatureStorage):
    """自定义特征存储"""
    
    def __init__(self, instrument: str, field: str, freq: str, **kwargs):
        super().__init__(**kwargs)
        self.instrument = instrument
        self.field = field
        self.freq = freq
        self._data = None
        self._index = None
    
    def _load_data(self):
        """加载数据"""
        if self._data is not None:
            return
        
        # 尝试从文件加载
        feature_file = Path(f"qlib_soxl_data/features/{self.instrument}/{self.field}.day.data")
        
        if feature_file.exists():
            # 读取 CSV 格式
            df = pd.read_csv(feature_file, index_col=0, parse_dates=True)
            self._data = df['value'].values.astype(np.float32)
            self._index = df.index
        else:
            # 从源数据加载
            source_file = Path("data/soxl_soxx_vix_full.csv")
            if not source_file.exists():
                raise ValueError(f"No feature data for {self.instrument}.{self.field}")
            
            df = pd.read_csv(source_file, parse_dates=['date'])
            df = df.set_index('date')
            
            # 映射字段
            field_map = {
                '$close': 'soxl_close',
                '$open': 'soxl_open',
                '$high': 'soxl_high',
                '$low': 'soxl_low',
                '$volume': 'soxl_volume',
                '$vix': 'vix',
            }
            
            col_name = field_map.get(self.field, self.field.lstrip('$'))
            
            if col_name in df.columns:
                self._data = df[col_name].values.astype(np.float32)
                self._index = df.index
            else:
                raise ValueError(f"Field {self.field} not found in data")
    
    def __getitem__(self, s: slice) -> pd.Series:
        """获取数据切片"""
        self._load_data()
        
        if self._data is None:
            return pd.Series(dtype=np.float32)
        
        values = self._data[s]
        start = s.start if s.start is not None else 0
        index = pd.RangeIndex(start, start + len(values))
        
        return pd.Series(values, index=index)


# ============================================================
# 2. 创建自定义 Provider
# ============================================================

from qlib.data.data import LocalProvider

class SOXLProvider(LocalProvider):
    """SOXL 自定义 Provider"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 设置 storage 类
        self._calendar_storage_class = SimpleCalendarStorage
        self._instrument_storage_class = SimpleInstrumentStorage
        self._feature_storage_class = SimpleFeatureStorage


# ============================================================
# 3. 准备数据文件
# ============================================================

print("\n准备数据文件...")

# 确保数据目录存在
QLIB_DIR = Path("qlib_soxl_data")
(QLIB_DIR / "calendars").mkdir(parents=True, exist_ok=True)
(QLIB_DIR / "instruments").mkdir(parents=True, exist_ok=True)
(QLIB_DIR / "features" / "SOXL").mkdir(parents=True, exist_ok=True)

# 加载源数据
df = pd.read_csv("data/soxl_soxx_vix_full.csv", parse_dates=['date'])
df = df.set_index('date')

# 创建日历
calendar = df.index.strftime('%Y-%m-%d').tolist()
with open(QLIB_DIR / "calendars" / "day.txt", 'w') as f:
    f.write('\n'.join(calendar))
print(f"✅ 日历: {len(calendar)} 天")

# 创建 instruments
with open(QLIB_DIR / "instruments" / "all.txt", 'w') as f:
    f.write(f"SOXL\t{df.index[0].strftime('%Y-%m-%d')}\t{df.index[-1].strftime('%Y-%m-%d')}\n")
print(f"✅ Instruments: SOXL")

# 创建特征文件
for field, col in [('$close', 'soxl_close'), ('$open', 'soxl_open'), 
                   ('$high', 'soxl_high'), ('$low', 'soxl_low'),
                   ('$volume', 'soxl_volume'), ('$vix', 'vix')]:
    feature_file = QLIB_DIR / "features" / "SOXL" / f"{field.lstrip('$')}.day.data"
    feature_df = pd.DataFrame({'value': df[col].values}, index=df.index)
    feature_df.to_csv(feature_file)

print(f"✅ 特征: 6 个字段")

# ============================================================
# 4. 测试 Qlib 加载
# ============================================================

print("\n" + "="*70)
print("测试 Qlib Provider")
print("="*70)

try:
    import qlib
    from qlib.data import D
    
    # 使用自定义 Provider 初始化
    provider = SOXLProvider()
    
    # 初始化 Qlib
    qlib.init(provider=provider)
    print("✅ Qlib 初始化成功（自定义 Provider）")
    
    # 测试加载
    print("\n测试 D.features()...")
    test_df = D.features(
        instruments=['SOXL'],
        fields=['$close', '$volume', '$vix'],
        start_time='2020-01-01',
        end_time='2020-01-10'
    )
    
    print("\n✅ D.features() 加载成功:")
    print(test_df.head(10))
    
    if len(test_df) > 0:
        close_values = test_df['$close'].dropna().values
        if len(close_values) > 0:
            print(f"\n✅ 数据正常")
            print(f"   Min: ${close_values.min():.2f}")
            print(f"   Max: ${close_values.max():.2f}")
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("完成！")
print("="*70)
