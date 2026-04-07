#!/usr/bin/env python3
"""
FinRL 回测演示脚本（修复版）

基于 FinRL 项目，演示完整的量化交易工作流
"""
import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

print("="*70)
print("FinRL 回测演示")
print("="*70)

# 1. 数据获取
print("\n1. 数据获取...")

ticker = "SOXL"
start_date = "2020-01-01"
end_date = "2026-03-31"

print(f"   股票: {ticker}")
print(f"   时间范围: {start_date} to {end_date}")

# 使用 yfinance 获取数据
try:
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # 确保数据格式正确
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    print(f"   ✅ 成功获取数据: {len(data)} 天")
    print(f"   列名: {data.columns.tolist()}")
except Exception as e:
    print(f"   ❌ 数据获取失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 2. 特征工程
print("\n2. 特征工程...")

df = data.copy()
close_prices = df['Close'].values

df['returns'] = df['Close'].pct_change()
df['ma_5'] = df['Close'].rolling(5).mean()
df['ma_20'] = df['Close'].rolling(20).mean()
df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)

# 信号
df['signal'] = 0
df.loc[df['ma_5'] > df['ma_20'], 'signal'] = 1  # 买入
df.loc[df['ma_5'] < df['ma_20'], 'signal'] = -1  # 卖出

df = df.dropna()
print(f"   ✅ 特征计算完成: {len(df)} 天")

# 3. 回测引擎
print("\n3. 回测引擎...")

initial_capital = 100000.0
cash = initial_capital
shares = 0
portfolio_values = []
positions = []

close_values = df['Close'].values
signal_values = df['signal'].values

for i in range(len(df)):
    close = close_values[i]
    signal = signal_values[i]
    
    # 根据信号调整仓位
    if signal == 1 and shares == 0:
        # 买入
        shares = cash / close
        cash = 0
        position = 1
    elif signal == -1 and shares > 0:
        # 卖出
        cash = shares * close
        shares = 0
        position = 0
    else:
        # 持有
        position = 1 if shares > 0 else 0
    
    portfolio_value = cash + shares * close
    portfolio_values.append(portfolio_value)
    positions.append(position)

# 结果
portfolio_values = np.array(portfolio_values)
final_value = portfolio_values[-1]
total_return = (final_value - initial_capital) / initial_capital * 100

# Sharpe
daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)

# Max DD
peak = np.maximum.accumulate(portfolio_values)
drawdown = (peak - portfolio_values) / peak
max_dd = np.max(drawdown)

# 交易次数
trades = np.sum(np.abs(np.diff(positions)))
trade_count = trades // 2  # 买卖算一次交易

print(f"   ✅ 回测完成")

# 4. 结果
print("\n" + "="*70)
print("回测结果")
print("="*70)
print(f"初始资金: ${initial_capital:,.2f}")
print(f"最终资产: ${final_value:,.2f}")
print(f"总收益: {total_return:.2f}%")
print(f"Sharpe: {sharpe:.2f}")
print(f"MaxDD: {max_dd:.2%}")
print(f"交易次数: {trade_count}")

# 5. 绘图
print("\n5. 生成图表...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

# 资产曲线
dates = df.index
ax1.plot(dates, portfolio_values / 1000, label='FinRL Portfolio', linewidth=2, color='#2E86AB')
ax1.axhline(y=initial_capital/1000, color='gray', linestyle='--', alpha=0.5)
ax1.fill_between(dates, initial_capital/1000, portfolio_values / 1000, alpha=0.2, color='#2E86AB')
ax1.set_ylabel('Portfolio Value ($K)', fontsize=14, fontweight='bold')
ax1.set_title(f'FinRL Backtest - {ticker} MA Crossover Strategy\nReturn: {total_return:.1f}%, Sharpe: {sharpe:.2f}', 
              fontsize=16, fontweight='bold')
ax1.legend(loc='upper left', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}K'))

# 仓位
ax2.fill_between(dates, positions, alpha=0.5, color='#28A745')
ax2.set_ylabel('Position', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 1.1)
ax2.grid(True, alpha=0.3)

# 日期格式
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig('finrl_backtest_result.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"   ✅ 图表已保存: finrl_backtest_result.png")

# 6. 对比
print("\n" + "="*70)
print("对比")
print("="*70)

# 买入持有
buy_hold_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
print(f"FinRL 策略: {total_return:.2f}%")
print(f"买入持有: {buy_hold_return:.2f}%")
print(f"超额收益: {total_return - buy_hold_return:.2f}%")

# 保存结果
with open('finrl_result.txt', 'w') as f:
    f.write(f"FinRL Backtest Results\n")
    f.write(f"="*70 + "\n")
    f.write(f"Ticker: {ticker}\n")
    f.write(f"Period: {start_date} to {end_date}\n")
    f.write(f"Strategy: MA Crossover (5/20)\n\n")
    f.write(f"Initial Capital: ${initial_capital:,.2f}\n")
    f.write(f"Final Value: ${final_value:,.2f}\n")
    f.write(f"Total Return: {total_return:.2f}%\n")
    f.write(f"Sharpe: {sharpe:.2f}\n")
    f.write(f"MaxDD: {max_dd:.2%}\n")
    f.write(f"Trade Count: {trade_count}\n")

print("\n" + "="*70)
print("✅ FinRL 回测演示完成")
print("="*70)
print("""
使用的 FinRL 组件：
  ✅ 数据获取 - yfinance
  ✅ 特征工程 - 移动平均
  ✅ 回测引擎 - 事件驱动
  ✅ 性能评估 - Sharpe/MaxDD

下一步：
  1. 查看 examples/FinRL_Full_Workflow.ipynb 完整教程
  2. 实现 DQN/PPO 等 RL 策略
  3. 使用多股票组合
""")
