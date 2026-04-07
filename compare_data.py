#!/usr/bin/env python3
"""
对比不同回测的数据差异

对比：
1. FinRL 回测（yfinance 数据）
2. qlib 官方回测（qlib binary 数据）
3. 自定义回测（CSV 数据）
"""
import pandas as pd
import numpy as np
import yfinance as yf
import qlib
from qlib.data import D

print("="*70)
print("数据差异对比")
print("="*70)

# 1. yfinance 数据（FinRL 使用）
print("\n1. yfinance 数据（FinRL）:")
yf_data = yf.download("SOXL", start="2020-01-01", end="2020-03-31", progress=False)
if isinstance(yf_data.columns, pd.MultiIndex):
    yf_data.columns = yf_data.columns.get_level_values(0)
print(f"   数据条数: {len(yf_data)}")
print(f"   开始日期: {yf_data.index[0]}")
print(f"   结束日期: {yf_data.index[-1]}")
print(f"   起始价格: ${yf_data['Close'].iloc[0]:.2f}")
print(f"   结束价格: ${yf_data['Close'].iloc[-1]:.2f}")
yf_return = (yf_data['Close'].iloc[-1] / yf_data['Close'].iloc[0] - 1) * 100
print(f"   区间收益: {yf_return:.2f}%")

# 2. qlib 数据
print("\n2. qlib 数据:")
qlib.init(provider_uri="/Users/rziiiii/.openclaw/workspace/qlib_soxl_binary")

qlib_data = D.features(
    instruments=['SOXL'],
    fields=['$close'],
    start_time='2020-01-01',
    end_time='2020-03-31'
)

qlib_data = qlib_data.reset_index()
qlib_data['datetime'] = pd.to_datetime(qlib_data['datetime'])
qlib_data = qlib_data.set_index('datetime')

if 'instrument' in qlib_data.columns:
    qlib_data = qlib_data.drop('instrument', axis=1)

print(f"   数据条数: {len(qlib_data)}")
print(f"   开始日期: {qlib_data.index[0]}")
print(f"   结束日期: {qlib_data.index[-1]}")
print(f"   起始价格: ${qlib_data['$close'].iloc[0]:.2f}")
print(f"   结束价格: ${qlib_data['$close'].iloc[-1]:.2f}")
qlib_return = (qlib_data['$close'].iloc[-1] / qlib_data['$close'].iloc[0] - 1) * 100
print(f"   区间收益: {qlib_return:.2f}%")

# 3. CSV 数据
print("\n3. CSV 数据:")
csv_data = pd.read_csv('/Users/rziiiii/.openclaw/workspace/data/soxl_soxx_vix_full.csv', parse_dates=['date'])
csv_data = csv_data.set_index('date')
csv_test = csv_data[(csv_data.index >= '2020-01-01') & (csv_data.index <= '2020-03-31')]
print(f"   数据条数: {len(csv_test)}")
print(f"   开始日期: {csv_test.index[0]}")
print(f"   结束日期: {csv_test.index[-1]}")
print(f"   起始价格: ${csv_test['soxl_close'].iloc[0]:.2f}")
print(f"   结束价格: ${csv_test['soxl_close'].iloc[-1]:.2f}")
csv_return = (csv_test['soxl_close'].iloc[-1] / csv_test['soxl_close'].iloc[0] - 1) * 100
print(f"   区间收益: {csv_return:.2f}%")

# 4. 详细对比
print("\n" + "="*70)
print("详细对比（前10天）")
print("="*70)

print(f"\n{'日期':<12} {'yfinance':<12} {'qlib':<12} {'CSV':<12} {'差异%':<10}")
print("-"*70)

for i in range(min(10, len(yf_data), len(qlib_data), len(csv_test))):
    date = yf_data.index[i]
    yf_close = yf_data['Close'].iloc[i]
    
    # 找到对应日期的 qlib 数据
    if date in qlib_data.index:
        qlib_close = qlib_data.loc[date, '$close']
    else:
        qlib_close = np.nan
    
    # 找到对应日期的 CSV 数据
    if date in csv_test.index:
        csv_close = csv_test.loc[date, 'soxl_close']
    else:
        csv_close = np.nan
    
    if not np.isnan(qlib_close) and not np.isnan(csv_close):
        diff = (qlib_close - csv_close) / csv_close * 100
    else:
        diff = np.nan
    
    print(f"{date.strftime('%Y-%m-%d'):<12} ${yf_close:<11.2f} ${qlib_close:<11.2f} ${csv_close:<11.2f} {diff:<10.2f}%")

# 5. 总结
print("\n" + "="*70)
print("总结")
print("="*70)
print(f"""
数据源对比（2020-01-01 to 2020-03-31）:

1. yfinance（FinRL 使用）:
   - 区间收益: {yf_return:.2f}%
   - 数据源: Yahoo Finance API

2. qlib（官方回测使用）:
   - 区间收益: {qlib_return:.2f}%
   - 数据源: 自建 binary 数据

3. CSV（自定义回测使用）:
   - 区间收益: {csv_return:.2f}%
   - 数据源: 本地 CSV 文件

差异分析:
""")

if abs(yf_return - qlib_return) > 5:
    print(f"   ⚠️ yfinance vs qlib 差异: {abs(yf_return - qlib_return):.2f}%")
else:
    print(f"   ✅ yfinance vs qlib 差异: {abs(yf_return - qlib_return):.2f}%")

if abs(yf_return - csv_return) > 5:
    print(f"   ⚠️ yfinance vs CSV 差异: {abs(yf_return - csv_return):.2f}%")
else:
    print(f"   ✅ yfinance vs CSV 差异: {abs(yf_return - csv_return):.2f}%")

if abs(qlib_return - csv_return) > 5:
    print(f"   ⚠️ qlib vs CSV 差异: {abs(qlib_return - csv_return):.2f}%")
else:
    print(f"   ✅ qlib vs CSV 差异: {abs(qlib_return - csv_return):.2f}%")

print("""
可能原因：
1. 复权处理：前复权 vs 后复权 vs 不复权
2. 数据源：Yahoo Finance vs 其他
3. 日历差异：交易日 vs 自然日
4. 时间戳：收盘价时间不同
""")
