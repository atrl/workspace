#!/usr/bin/env python3
"""
生成 Qlib RL 模仿学习回测图表
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stable_baselines3 import DQN
import sys

print("="*70)
print("生成回测图表")
print("="*70)

# ============================================================
# 1. 加载模型和数据
# ============================================================

print("\n准备数据...")

df = pd.read_csv(
    "/Users/rziiiii/.openclaw/workspace/data/soxl_soxx_vix_full.csv",
    parse_dates=['date']
)
df = df.set_index('date')

# 计算特征
close = df['soxl_close']
vix = df['vix']

df['return'] = close.pct_change()
df['ma5'] = close.rolling(5).mean()
df['ma20'] = close.rolling(20).mean()

delta = close.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))

df['vix_zscore'] = (vix - vix.rolling(20).mean()) / (vix.rolling(20).std() + 1e-8)

df['panic_score'] = (
    df['vix_zscore'] * 40 +
    vix.rolling(20).rank(pct=True) * 60 +
    np.where(df['return'] < 0, np.minimum(df['return'] / -0.05, 1) * (-50), 0)
)

df = df.dropna()

# 测试数据
test_df = df[df.index >= '2020-01-01']
print(f"测试数据: {len(test_df)} records ({test_df.index.min()} to {test_df.index.max()})")

# ============================================================
# 2. 加载模型并回测
# ============================================================

print("\n加载 RL 模型...")
sys.path.insert(0, '/Users/rziiiii/.openclaw/workspace/research')
from train_qlib_rl_imitation import ExpertEnv

model = DQN.load("models/qlib_rl_imitation/dqn_model.zip")

test_env = ExpertEnv(test_df)
obs, _ = test_env.reset()

test_values = [100000]
positions = [0.0]
dates = [test_df.index[test_env.current_idx]]
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = test_env.step(action)
    test_values.append(info.get('cash', test_values[-1]))
    positions.append(info.get('position', positions[-1]))
    dates.append(test_df.index[test_env.current_idx] if test_env.current_idx < len(test_df) else dates[-1])

test_values = np.array(test_values)
positions = np.array(positions)

print(f"✅ 回测完成")

# ============================================================
# 3. 计算指标
# ============================================================

total_return = (test_values[-1] - 100000) / 100000
daily_returns = np.diff(test_values) / test_values[:-1]
sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
max_dd = np.max((np.maximum.accumulate(test_values) - test_values) / np.maximum.accumulate(test_values))

# 买入持有基准
buy_hold = test_df['soxl_close']
buy_hold_values = 100000 * (buy_hold / buy_hold.iloc[0])
buy_hold_return = (buy_hold_values.iloc[-1] - 100000) / 100000

print(f"\nRL策略:")
print(f"  总收益: {total_return:.2%}")
print(f"  Sharpe: {sharpe:.2f}")
print(f"  最大回撤: {max_dd:.2%}")

print(f"\n买入持有:")
print(f"  总收益: {buy_hold_return:.2%}")

# ============================================================
# 4. 绘制图表
# ============================================================

print("\n绘制图表...")

fig, axes = plt.subplots(4, 1, figsize=(16, 14))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 图1: 资金曲线
ax1 = axes[0]
ax1.plot(dates, test_values, label='RL Strategy', linewidth=2, color='#2E86AB')
ax1.plot(buy_hold.index, buy_hold_values, label='Buy & Hold', linewidth=2, color='#A23B72', alpha=0.7)
ax1.axhline(y=100000, color='gray', linestyle='--', alpha=0.5)
ax1.fill_between(dates, 100000, test_values, alpha=0.3, color='#2E86AB')
ax1.set_title(f'Qlib RL Strategy vs Buy & Hold (2020-2026)\n'
              f'RL: {total_return:.1%} ({test_values[-1]/100000:.2f}x) | '
              f'B&H: {buy_hold_return:.1%} ({buy_hold_values.iloc[-1]/100000:.2f}x)', 
              fontsize=14, fontweight='bold')
ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
ax1.legend(loc='upper left', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

# 图2: 仓位变化
ax2 = axes[1]
ax2.fill_between(dates, positions, alpha=0.5, color='#F18F01')
ax2.plot(dates, positions, color='#F18F01', linewidth=1.5)
ax2.set_title('Position Size (0-0.8)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Position', fontsize=12)
ax2.set_ylim(0, 1)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

# 图3: SOXL价格和VIX
ax3 = axes[2]
ax3_twin = ax3.twinx()

ax3.plot(test_df.index, test_df['soxl_close'], label='SOXL Price', color='#2E86AB', linewidth=1.5)
ax3.set_ylabel('SOXL Price ($)', color='#2E86AB', fontsize=12)
ax3.tick_params(axis='y', labelcolor='#2E86AB')

ax3_twin.plot(test_df.index, test_df['vix'], label='VIX', color='#E63946', linewidth=1.5, alpha=0.7)
ax3_twin.set_ylabel('VIX', color='#E63946', fontsize=12)
ax3_twin.tick_params(axis='y', labelcolor='#E63946')

ax3.set_title('SOXL Price & VIX', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

# 合并图例
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

# 图4: 回撤
ax4 = axes[3]
cummax = np.maximum.accumulate(test_values)
drawdowns = (cummax - test_values) / cummax
ax4.fill_between(dates, 0, -drawdowns, alpha=0.5, color='#E63946')
ax4.plot(dates, -drawdowns, color='#E63946', linewidth=1.5)
ax4.set_title(f'Drawdown (Max: {max_dd:.2%})', fontsize=14, fontweight='bold')
ax4.set_ylabel('Drawdown', fontsize=12)
ax4.set_xlabel('Date', fontsize=12)
ax4.grid(True, alpha=0.3)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

plt.tight_layout()

# 保存
output_path = "research/qlib_rl_backtest_chart.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ 图表保存: {output_path}")

# ============================================================
# 5. 添加水印和总结
# ============================================================

from PIL import Image, ImageDraw, ImageFont

img = Image.open(output_path)
draw = ImageDraw.Draw(img)

# 添加总结文字
summary_text = [
    f"Qlib RL Strategy Performance",
    f"",
    f"Total Return: {total_return:.2%}",
    f"Sharpe Ratio: {sharpe:.2f}",
    f"Max Drawdown: {max_dd:.2%}",
    f"Final Value: ${test_values[-1]:,.0f}",
    f"",
    f"Architecture: Imitation Learning",
    f"Model: DQN (Deep Q-Network)",
    f"Framework: Qlib + stable-baselines3",
]

try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
except:
    font = ImageFont.load_default()

y_offset = 50
for line in summary_text:
    draw.text((50, y_offset), line, fill=(255, 255, 255), font=font)
    y_offset += 25

# 添加背景框
from PIL import ImageFilter
overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
overlay_draw = ImageDraw.Draw(overlay)
overlay_draw.rectangle([(40, 40), (400, y_offset + 10)], fill=(0, 0, 0, 180))
overlay = overlay.filter(ImageFilter.BLUR)
img = img.convert('RGBA')
img = Image.alpha_composite(img, overlay)
img = img.convert('RGB')

img.save(output_path)
print(f"✅ 添加水印完成")

print("\n" + "="*70)
print("完成！")
print("="*70)
print(f"\n图表路径: {output_path}")
