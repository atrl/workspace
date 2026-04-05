#!/usr/bin/env python3
"""
生成 Qlib + RL 优化版回测对比图表
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 加载模型和数据
import sys
sys.path.insert(0, '/Users/rziiiii/.openclaw/workspace/research')

from stable_baselines3 import PPO
from rl_hierarchical_env_optimized import SOXLHierarchicalEnvOptimized

print("加载模型和数据...")

# 加载优化版RL模型
model = PPO.load("research/models/hierarchical_rl_optimized/ppo_model.zip")

# 回测优化版RL策略
env = SOXLHierarchicalEnvOptimized()
obs, _ = env.reset()

rl_values = [100000]
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    rl_values.append(info['total_value'])

# 回测硬参数策略（简化版，使用优化版环境）
env2 = SOXLHierarchicalEnvOptimized()
obs2, _ = env2.reset()

hardcoded_values = [100000]
done2 = False
while not done2:
    # 硬参数策略：直接用规则信号作为仓位
    rule_signal = obs2[10]  # rule_signal 是第11个特征
    
    # 转换为动作
    if rule_signal >= 0.5:
        action2 = np.array([0.0, rule_signal])
    else:
        action2 = np.array([1 - rule_signal, 0.0])
    
    obs2, reward2, done2, truncated2, info2 = env2.step(action2)
    hardcoded_values.append(info2['total_value'])

# 买入持有
df = pd.read_csv('/Users/rziiiii/.openclaw/workspace/data/soxl_with_vix.csv', parse_dates=['date'])
df = df[df['date'] >= '2020-01-01'].reset_index(drop=True)
buy_hold = (df['close'] / df['close'].iloc[0] * 100000).values

# 对齐长度
min_len = min(len(rl_values), len(hardcoded_values), len(buy_hold))
rl_values = np.array(rl_values[:min_len])
hardcoded_values = np.array(hardcoded_values[:min_len])
buy_hold = buy_hold[:min_len]
dates = df['date'].values[:min_len]

print(f"数据点数: {min_len}")

# 计算指标
def calc_metrics(values):
    total_return = (values[-1] - 100000) / 100000
    daily_returns = np.diff(values) / values[:-1]
    sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
    max_dd = np.max((np.maximum.accumulate(values) - values) / np.maximum.accumulate(values))
    return total_return, sharpe, max_dd

rl_return, rl_sharpe, rl_dd = calc_metrics(rl_values)
hard_return, hard_sharpe, hard_dd = calc_metrics(hardcoded_values)
bh_return, bh_sharpe, bh_dd = calc_metrics(buy_hold)

print(f"优化版RL: {rl_return:.2%}, Sharpe {rl_sharpe:.2f}, MaxDD {rl_dd:.2%}")
print(f"硬参数: {hard_return:.2%}, Sharpe {hard_sharpe:.2f}, MaxDD {hard_dd:.2%}")
print(f"买入持有: {bh_return:.2%}, Sharpe {bh_sharpe:.2f}, MaxDD {bh_dd:.2%}")

# 绘图
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 1. 组合价值对比
ax1 = axes[0]
ax1.plot(dates, rl_values / 1000, 'b-', linewidth=2, label=f'优化版RL (收益: {rl_return:.1%}, Sharpe: {rl_sharpe:.2f})')
ax1.plot(dates, hardcoded_values / 1000, 'g--', linewidth=2, label=f'硬参数策略 (收益: {hard_return:.1%}, Sharpe: {hard_sharpe:.2f})')
ax1.plot(dates, buy_hold / 1000, 'gray', linewidth=1.5, alpha=0.7, label=f'买入持有 (收益: {bh_return:.1%})')

ax1.set_title('Qlib + RL 优化版回测对比 (2020-2026)', fontsize=14, fontweight='bold')
ax1.set_ylabel('组合价值 ($K)', fontsize=11)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# 2. 回撤对比
ax2 = axes[1]

def calc_drawdown(values):
    peak = np.maximum.accumulate(values)
    return (peak - values) / peak

rl_dd_series = calc_drawdown(rl_values) * 100
hard_dd_series = calc_drawdown(hardcoded_values) * 100
bh_dd_series = calc_drawdown(buy_hold) * 100

ax2.fill_between(dates, 0, -rl_dd_series, alpha=0.4, color='blue', label=f'优化版RL (MaxDD: {rl_dd:.1%})')
ax2.fill_between(dates, 0, -hard_dd_series, alpha=0.3, color='green', label=f'硬参数策略 (MaxDD: {hard_dd:.1%})')
ax2.plot(dates, -bh_dd_series, 'gray', linewidth=1, alpha=0.5, label=f'买入持有 (MaxDD: {bh_dd:.1%})')

ax2.set_title('回撤对比', fontsize=12, fontweight='bold')
ax2.set_ylabel('回撤 (%)', fontsize=11)
ax2.legend(loc='lower left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# 3. 性能指标对比
ax3 = axes[2]

strategies = ['优化版RL', '硬参数策略', '买入持有']
returns = [rl_return * 100, hard_return * 100, bh_return * 100]
sharpes = [rl_sharpe, hard_sharpe, bh_sharpe]
max_dds = [rl_dd * 100, hard_dd * 100, bh_dd * 100]

x = np.arange(len(strategies))
width = 0.25

bars1 = ax3.bar(x - width, returns, width, label='总收益 (%)', color='blue', alpha=0.7)
bars2 = ax3.bar(x, [s * 50 for s in sharpes], width, label='Sharpe × 50', color='green', alpha=0.7)
bars3 = ax3.bar(x + width, [-d for d in max_dds], width, label='-MaxDD (%)', color='red', alpha=0.7)

ax3.set_title('性能指标对比', fontsize=12, fontweight='bold')
ax3.set_ylabel('百分比', fontsize=11)
ax3.set_xticks(x)
ax3.set_xticklabels(strategies, fontsize=11)
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')
ax3.axhline(y=0, color='black', linewidth=0.8)

# 添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height > 0 else -10),
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=9)

plt.tight_layout()

# 保存
output_path = '/Users/rziiiii/.openclaw/workspace/research/qlib_rl_optimized_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ 图表已保存: {output_path}")

# 显示关键指标
print("\n" + "="*70)
print("回测结果总结")
print("="*70)
print(f"{'策略':<15} {'总收益':>10} {'Sharpe':>8} {'MaxDD':>8} {'倍数':>8}")
print("-"*70)
print(f"{'优化版RL':<15} {rl_return:>9.1%} {rl_sharpe:>8.2f} {rl_dd:>7.1%} {rl_values[-1]/100000:>7.2f}x")
print(f"{'硬参数策略':<15} {hard_return:>9.1%} {hard_sharpe:>8.2f} {hard_dd:>7.1%} {hardcoded_values[-1]/100000:>7.2f}x")
print(f"{'买入持有':<15} {bh_return:>9.1%} {bh_sharpe:>8.2f} {bh_dd:>7.1%} {buy_hold[-1]/100000:>7.2f}x")
print("="*70)
