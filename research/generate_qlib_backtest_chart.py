#!/usr/bin/env python3
"""
生成 Qlib RL 回测对比图表

对比：
1. 训练集 (2010-2019)
2. 测试集 (2020-2026) - 完全未见过的6年数据
3. 买入持有基准
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
sys.path.insert(0, '/Users/rziiiii/.openclaw/workspace/research')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

# 导入环境
from train_2019_test_2020 import SOXLQlibHierarchicalEnvSplit

print("="*70)
print("生成 Qlib RL 回测图表")
print("="*70)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载模型
print("\n加载模型...")
model = PPO.load("models/qlib_rl_2019/ppo_model.zip")

# 2. 回测训练集 (2010-2019)
print("回测训练集 (2010-2019)...")
train_env = SOXLQlibHierarchicalEnvSplit(mode='train')
obs_train, _ = train_env.reset()

train_values = [100000]
done_train = False
while not done_train:
    action, _ = model.predict(obs_train, deterministic=True)
    obs_train, reward_train, done_train, truncated_train, info_train = train_env.step(action)
    train_values.append(info_train['total_value'])

train_values = np.array(train_values)

# 3. 回测测试集 (2020-2026)
print("回测测试集 (2020-2026)...")
test_env = SOXLQlibHierarchicalEnvSplit(mode='test')
obs_test, _ = test_env.reset()

test_values = [100000]
test_positions = []
done_test = False
while not done_test:
    action, _ = model.predict(obs_test, deterministic=True)
    obs_test, reward_test, done_test, truncated_test, info_test = test_env.step(action)
    test_values.append(info_test['total_value'])
    test_positions.append(info_test['rl_position'])

test_values = np.array(test_values)
test_pos = np.array(test_positions)

# 4. 买入持有基准
print("计算买入持有基准...")
df_full = pd.read_csv('/Users/rziiiii/.openclaw/workspace/data/soxl_with_vix_full.csv', parse_dates=['date'])
df_test = df_full[df_full['date'] >= '2020-01-01'].reset_index(drop=True)
buy_hold = (df_test['close'] / df_test['close'].iloc[0] * 100000).values

# 5. 计算指标
def calc_metrics(values):
    total_return = (values[-1] - 100000) / 100000
    daily_returns = np.diff(values) / values[:-1]
    sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
    max_dd = np.max((np.maximum.accumulate(values) - values) / np.maximum.accumulate(values))
    return total_return, sharpe, max_dd

train_return, train_sharpe, train_dd = calc_metrics(train_values)
test_return, test_sharpe, test_dd = calc_metrics(test_values)
bh_return, bh_sharpe, bh_dd = calc_metrics(buy_hold)

print(f"\n训练集: {train_return:.2%}, Sharpe {train_sharpe:.2f}")
print(f"测试集: {test_return:.2%}, Sharpe {test_sharpe:.2f}")
print(f"买入持有: {bh_return:.2%}")

# 6. 绘图
print("\n生成图表...")
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 图1: 组合价值对比
ax1 = axes[0]

# 训练集 (2010-2019)
ax1.plot(range(len(train_values)), train_values / 1000, 'b-', linewidth=2, 
         label=f'训练集 (2010-2019): {train_return:.1%}, Sharpe {train_sharpe:.2f}')

# 测试集 (2020-2026)
ax1.plot(range(len(train_values), len(train_values) + len(test_values)), 
         test_values / 1000, 'g-', linewidth=2.5,
         label=f'测试集 (2020-2026): {test_return:.1%}, Sharpe {test_sharpe:.2f}')

# 买入持有
ax1.plot(range(len(buy_hold)), buy_hold / 1000, 'gray', linewidth=1.5, alpha=0.7,
         label=f'买入持有 (2020-2026): {bh_return:.1%}')

# 分割线
ax1.axvline(x=len(train_values)-1, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
ax1.text(len(train_values)-1, ax1.get_ylim()[1]*0.95, '  训练/测试分割', 
         fontsize=10, color='red', alpha=0.7)

ax1.set_title('Qlib + RL 分层策略回测 (训练: 2010-2019 → 测试: 2020-2026)', 
              fontsize=14, fontweight='bold')
ax1.set_ylabel('组合价值 ($K)', fontsize=11)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# 图2: 回撤对比
ax2 = axes[1]

def calc_drawdown(values):
    peak = np.maximum.accumulate(values)
    return (peak - values) / peak * 100

train_dd_series = calc_drawdown(train_values)
test_dd_series = calc_drawdown(test_values)
bh_dd_series = calc_drawdown(buy_hold)

ax2.fill_between(range(len(train_dd_series)), 0, -train_dd_series, 
                 alpha=0.4, color='blue', label=f'训练集 MaxDD: {train_dd:.1%}')
ax2.fill_between(range(len(train_values), len(train_values) + len(test_dd_series)), 
                 0, -test_dd_series, alpha=0.4, color='green', 
                 label=f'测试集 MaxDD: {test_dd:.1%}')
ax2.plot(range(len(bh_dd_series)), -bh_dd_series, 'gray', linewidth=1, alpha=0.5,
         label=f'买入持有 MaxDD: {bh_dd:.1%}')

ax2.axvline(x=len(train_values)-1, color='red', linestyle='--', linewidth=1.5, alpha=0.5)

ax2.set_title('回撤对比', fontsize=12, fontweight='bold')
ax2.set_ylabel('回撤 (%)', fontsize=11)
ax2.legend(loc='lower left', fontsize=10)
ax2.grid(True, alpha=0.3)

# 图3: 性能指标对比
ax3 = axes[2]

metrics = ['总收益 (%)', 'Sharpe', 'MaxDD (%)', '收益倍数']
train_metrics = [train_return * 100, train_sharpe * 20, -train_dd * 100, train_values[-1] / 100000 * 10]
test_metrics = [test_return * 100, test_sharpe * 20, -test_dd * 100, test_values[-1] / 100000 * 10]
bh_metrics = [bh_return * 100, bh_sharpe * 20, -bh_dd * 100, buy_hold[-1] / 100000 * 10]

x = np.arange(len(metrics))
width = 0.25

bars1 = ax3.bar(x - width, train_metrics, width, label='训练集 (2010-2019)', color='blue', alpha=0.7)
bars2 = ax3.bar(x, test_metrics, width, label='测试集 (2020-2026)', color='green', alpha=0.7)
bars3 = ax3.bar(x + width, bh_metrics, width, label='买入持有', color='gray', alpha=0.5)

ax3.set_title('性能指标对比', fontsize=12, fontweight='bold')
ax3.set_ylabel('标准化值', fontsize=11)
ax3.set_xticks(x)
ax3.set_xticklabels(metrics, fontsize=11)
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')
ax3.axhline(y=0, color='black', linewidth=0.8)

# 添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if abs(height) > 1:
            ax3.annotate(f'{height:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height > 0 else -10),
                        textcoords="offset points",
                        ha='center', va='bottom' if height > 0 else 'top',
                        fontsize=8)

plt.tight_layout()

# 7. 保存
output_path = '/Users/rziiiii/.openclaw/workspace/research/qlib_rl_2019_backtest.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')

print(f"\n✅ 图表已保存: {output_path}")

# 8. 显示结果总结
print("\n" + "="*70)
print("回测结果总结")
print("="*70)
print(f"{'指标':<20} {'训练集 (2010-2019)':>20} {'测试集 (2020-2026)':>20}")
print("-"*70)
print(f"{'总收益':<20} {train_return:>18.2%} {test_return:>18.2%}")
print(f"{'Sharpe':<20} {train_sharpe:>18.2f} {test_sharpe:>18.2f}")
print(f"{'MaxDD':<20} {train_dd:>18.2%} {test_dd:>18.2%}")
print(f"{'最终价值':<20} ${train_values[-1]:>16,.0f} ${test_values[-1]:>16,.0f}")
print(f"{'收益倍数':<20} {train_values[-1]/100000:>18.2f}x {test_values[-1]/100000:>18.2f}x")
print("="*70)

print(f"\n过拟合检查:")
print(f"  Sharpe比 (测试/训练): {test_sharpe/train_sharpe:.2f}")
print(f"  收益比 (测试/训练):   {test_return/train_return:.2f}")

if test_sharpe/train_sharpe >= 0.8:
    print("  ✅ 泛化能力良好")
else:
    print("  ⚠️ 可能存在过拟合")

print(f"\n对比买入持有 (2020-2026):")
print(f"  RL策略:  {test_return:.2%} (${test_values[-1]:,.0f})")
print(f"  买入持有: {bh_return:.2%} (${buy_hold[-1]:,.0f})")
print(f"  超额收益: {(test_return - bh_return):.2%}")

print("\n" + "="*70)
print("✅ 完成！")
print("="*70)
