#!/usr/bin/env python3
"""
FinRL Trading SOXL + Cash - 恐慌买入策略（最终版）

使用正确的 VIX 方向：恐慌时买入
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf

print("="*70)
print("FinRL Trading SOXL + Cash - 恐慌买入策略")
print("="*70)

# 1. 下载数据
print("\n1. 下载市场数据...")

def download_data(symbol, start='2010-01-01'):
    data = yf.download(symbol, start=start, end='2026-04-08', progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.reset_index()
    data.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    data['date'] = pd.to_datetime(data['date'])
    return data.set_index('date')

soxl = download_data('SOXL')
shy = download_data('SHY')
vix_data = download_data('^VIX')

print(f"   SOXL: {len(soxl)} 条")
print(f"   SHY: {len(shy)} 条")
print(f"   VIX: {len(vix_data)} 条")

# 2. 计算特征
print("\n2. 计算特征（恐慌得分）...")

common_dates = soxl.index.intersection(shy.index).intersection(vix_data.index)
soxl_test = soxl.loc[common_dates]
shy_test = shy.loc[common_dates]
vix_test = vix_data.loc[common_dates]

vix = vix_test['close']
close = soxl_test['close']

# VIX Z-score
vix_zscore = (vix - vix.rolling(252).mean()) / (vix.rolling(252).std() + 1e-8)

# 恐慌得分
panic_score = (
    vix_zscore * 40 +
    vix.rolling(20).rank(pct=True) * 60 +
    np.where(close.pct_change() < 0,
             np.minimum(close.pct_change() / -0.05, 1) * (-50), 0)
)

print(f"   ✅ 特征计算完成")

# 3. 回测
print("\n3. 运行回测（恐慌买入策略）...")

test_start = '2020-01-01'
test_end = '2026-03-31'

test_dates = soxl_test.index[(soxl_test.index >= test_start) & (soxl_test.index <= test_end)]

initial_capital = 100000
cash = initial_capital
shares_soxl = 0
shares_shy = 0

values = []
positions_soxl = []
signals = []

for date in test_dates:
    vix_val = vix.loc[date]
    panic = panic_score.loc[date]
    
    # 恐慌时买入
    if panic <= -25:
        if vix_val > 28:
            target_soxl = 0.9
            signals.append('extreme_panic_buy')
        elif vix_val > 22:
            target_soxl = 0.8
            signals.append('panic_buy')
        else:
            target_soxl = 0.6
            signals.append('mild_panic_buy')
    
    # 贪婪时卖出
    elif vix_val < 15:
        target_soxl = 0.0
        signals.append('greedy_sell')
    
    # 中性
    else:
        target_soxl = 0.5
        signals.append('neutral')
    
    target_shy = 1.0 - target_soxl
    
    # 计算当前总资产
    soxl_price = soxl_test.loc[date, 'close']
    shy_price = shy_test.loc[date, 'close']
    total_value = cash + shares_soxl * soxl_price + shares_shy * shy_price
    
    # 调整仓位
    target_soxl_value = total_value * target_soxl
    target_shy_value = total_value * target_shy
    
    # 交易成本
    trade_value_soxl = abs(target_soxl_value - shares_soxl * soxl_price)
    trade_value_shy = abs(target_shy_value - shares_shy * shy_price)
    cost = (trade_value_soxl + trade_value_shy) * 0.001
    
    # 调整仓位
    shares_soxl = target_soxl_value / soxl_price
    shares_shy = target_shy_value / shy_price
    
    # 更新现金
    cash = total_value - shares_soxl * soxl_price - shares_shy * shy_price - cost
    
    # 记录
    final_value = cash + shares_soxl * soxl_price + shares_shy * shy_price
    values.append(final_value)
    positions_soxl.append(target_soxl)

# 4. 结果
print("\n" + "="*70)
print("回测结果（恐慌买入策略）")
print("="*70)

values = np.array(values)
final_value = values[-1]
total_return = (final_value - initial_capital) / initial_capital * 100

daily_returns = np.diff(values) / values[:-1]
sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)

peak = np.maximum.accumulate(values)
drawdown = (peak - values) / peak
max_dd = np.max(drawdown) * 100

print(f"最终资产: ${final_value:,.2f}")
print(f"总收益: {total_return:.2f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.2f}%")

# 信号分布
signal_counts = pd.Series(signals).value_counts()
print(f"\n信号分布:")
for signal, count in signal_counts.items():
    print(f"   {signal}: {count} ({count/len(signals):.1%})")

# 平均仓位
avg_soxl = np.mean(positions_soxl)
print(f"\n平均仓位:")
print(f"   SOXL: {avg_soxl:.1%}")
print(f"   SHY (现金): {1-avg_soxl:.1%}")

# 对比
test_soxl_data = soxl_test.loc[test_start:test_end]
buy_hold = (test_soxl_data['close'].iloc[-1] / test_soxl_data['close'].iloc[0] - 1) * 100
print(f"\n对比:")
print(f"  恐慌买入策略: {total_return:.2f}%")
print(f"  买入持有 SOXL: {buy_hold:.2f}%")
print(f"  超额收益: {total_return - buy_hold:.2f}%")

# 5. 绘图
print("\n5. 绘制图表...")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

dates = test_dates

ax1 = axes[0]
ax1.plot(dates, values / 1000, label='恐慌买入策略', linewidth=2.5, color='#2E86AB')
ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
ax1.fill_between(dates, 100, values / 1000, alpha=0.2, color='#2E86AB')
ax1.set_ylabel('Portfolio Value ($K)', fontsize=12)
ax1.set_title(f'FinRL Trading SOXL+Cash 恐慌买入策略\nReturn: {total_return:.1f}%, Sharpe: {sharpe:.2f}, MaxDD: {max_dd:.1f}%', 
              fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.fill_between(dates, positions_soxl, alpha=0.5, color='#28A745', label='SOXL 仓位')
ax2.set_ylabel('SOXL 仓位 (%)', fontsize=12)
ax2.set_ylim(0, 1.1)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left')

ax3 = axes[2]
ax3.plot(dates, vix.loc[test_dates], color='red', alpha=0.7, label='VIX')
ax3.axhline(y=22, color='orange', linestyle='--', alpha=0.5, label='恐慌阈值 (22)')
ax3.axhline(y=28, color='red', linestyle='--', alpha=0.5, label='极度恐慌 (28)')
ax3.set_ylabel('VIX', fontsize=12)
ax3.set_xlabel('Date', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper left')
ax3.xaxis.set_major_locator(mdates.YearLocator())
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig('finrl_panic_buy_final.png', dpi=300, bbox_inches='tight')
print(f"   ✅ 图表: finrl_panic_buy_final.png")

print("\n" + "="*70)
print("✅ 完成")
print("="*70)
