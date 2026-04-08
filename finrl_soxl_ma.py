#!/usr/bin/env python3
"""
FinRL SOXL + Cash 简单策略（MA）

目标: SOXL + 现金
策略: 移动平均
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

print("="*70)
print("FinRL SOXL + Cash 策略（MA Crossover）")
print("="*70)

# 1. 数据
print("\n1. 获取 SOXL 数据...")
data = yf.download('SOXL', start='2010-01-01', end='2026-03-31', progress=False)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
data.columns = [str(c) for c in data.columns]

print(f"   数据范围: {len(data)}")
print(f"   时间: {data.index[0]} to {data.index[-1]}")

# 2. 特征
print("\n2. 计算特征...")
data['return'] = data['Close'].pct_change()
data['ma5'] = data['Close'].rolling(5).mean()
data['ma20'] = data['Close'].rolling(20).mean()

data = data.dropna()

train_data = data[data.index < '2020-01-01'].copy()
test_data = data[data.index >= '2020-01-01'].copy()

print(f"   训练: {len(train_data)}")
print(f"   测试: {len(test_data)}")

# 3. 策略
print("\n3. 运行 MA Crossover 策略...")

initial_capital = 100000
cash = initial_capital
shares = 0
position = 0.0

values = []
positions = []

test_close = test_data['Close'].values
test_ma5 = test_data['ma5'].values
test_ma20 = test_data['ma20'].values

for i in range(len(test_data)):
    close = test_close[i]
    
    # 信号： MA5 > MA20 则持有，否则现金
    if test_ma5[i] > test_ma20[i]:
        # 买入信号
        if position < 1.0:
            # 买入
            shares = cash / close
            cash = 0
            position = 1.0
    else:
        # 卖出信号
        if position > 0.0:
            # 卖出
            cash = shares * close
            shares = 0
            position = 0.0
    
    # 记录账户价值
    portfolio_value = cash + shares * close
    values.append(portfolio_value)
    positions.append(position)

# 4. 结果
values = np.array(values)
final_value = values[-1]
total_return = (final_value - initial_capital) / initial_capital * 100

daily_returns = np.diff(values) / values[:-1]
sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)

peak = np.maximum.accumulate(values)
drawdown = (peak - values) / peak
max_dd = np.max(drawdown) * 100

print("\n" + "="*70)
print("回测结果")
print("="*70)
print(f"最终资产: ${final_value:,.2f}")
print(f"总收益: {total_return:.2f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.2f}")

# 仓位分布
print("\n仓位分布:")
position_changes = np.diff(positions)
for i in range(2):
    pct = (np.array(positions) == i).sum()
    print(f"   {i*100}%: {pct} ({pct/len(positions):.1%})")

# 对比
buy_hold_return = (test_data['Close'].iloc[-1] / test_data['Close'].iloc[0] - 1) * 100
print(f"\n对比:")
print(f"  MA 策略: {total_return:.2f}%")
print(f"  买入持有: {buy_hold_return:.2f}%")
print(f"  超额收益: {total_return - buy_hold_return:.2f}%")

# 5. 绘图
print("\n5. 绘图...")
dates = test_data.index
plt.figure(figsize=(14, 8))

ax1 = plt.subplot(2, 1, 1)
ax1.plot(dates, values / 1000, label='MA Strategy', linewidth=2.5, color='#2E86AB')
ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
ax1.fill_between(dates, 100, values / 1000, alpha=0.2, color='#2E86AB')
ax1.set_ylabel('Portfolio Value ($K)', fontsize=12)
ax1.set_title(f'FinRL SOXL+Cash MA Strategy - {total_return:.1f}% Return', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(2, 1, 2)
ax2.fill_between(dates, positions, alpha=0.5, color='#28A745')
ax2.set_ylabel('Position (%)', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylim(0, 1.1)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('finrl_soxl_cash_ma.png', dpi=300, bbox_inches='tight')
print(f"   ✅ 图表: finrl_soxl_cash_ma.png")

print("\n" + "="*70)
print("✅ 完成")
print("="*70)
