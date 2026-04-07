#!/usr/bin/env python3
"""
SOXL 大中小单特征工程

基于 OHLCV 数据估算资金流向和大单特征
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("SOXL 大中小单特征工程")
print("="*70)

# 加载数据
csv_path = '/Users/rziiiii/.openclaw/workspace/data/soxl_soxx_vix_full.csv'
df = pd.read_csv(csv_path, parse_dates=['date'])
df = df.set_index('date')

print(f"\n数据时间范围: {df.index.min()} to {df.index.max()}")
print(f"数据条数: {len(df)}")

# ============================================================
# 特征1: 成交量分类（基于分位数）
# ============================================================
print("\n" + "="*70)
print("特征1: 成交量分类（大/中/小单）")
print("="*70)

volume = df['soxl_volume']

# 方法1: 基于分位数
df['vol_pct'] = volume / volume.rolling(20).mean()  # 相对成交量

# 分类
df['vol_class'] = pd.cut(
    df['vol_pct'],
    bins=[0, 0.7, 1.3, np.inf],
    labels=['小单', '中单', '大单']
)

# 统计
print(f"\n成交量分类统计:")
print(df['vol_class'].value_counts())

# ============================================================
# 特征2: 价格-成交量关系（估算主动买入/卖出）
# ============================================================
print("\n" + "="*70)
print("特征2: 价格-成交量关系（主动买/卖）")
print("="*70)

close = df['soxl_close']
returns = close.pct_change()

# 价格涨 + 成交量大 = 主动买入
# 价格跌 + 成交量大 = 主动卖出
df['price_change'] = returns
df['vol_change'] = volume.pct_change()

# 主动买入强度
df['buy_strength'] = np.where(
    (returns > 0) & (df['vol_pct'] > 1.2),
    df['vol_pct'] * returns,  # 放大
    0
)

# 主动卖出强度
df['sell_strength'] = np.where(
    (returns < 0) & (df['vol_pct'] > 1.2),
    df['vol_pct'] * abs(returns),  # 放大
    0
)

# 净买入强度
df['net_buy_strength'] = df['buy_strength'] - df['sell_strength']

print(f"\n主动买入强度（前5）:")
print(df[df['buy_strength'] > 0]['buy_strength'].sort_values(ascending=False).head())

print(f"\n主动卖出强度（前5）:")
print(df[df['sell_strength'] > 0]['sell_strength'].sort_values(ascending=False).head())

# ============================================================
# 特征3: 成交量加权价格（VWAP）
# ============================================================
print("\n" + "="*70)
print("特征3: VWAP 偏离度（判断大单方向）")
print("="*70)

# 典型价格
df['typical_price'] = (df['soxl_close'] + df['soxl_close'] + df['soxl_close']) / 3  # 简化

# VWAP
df['vwap'] = (df['typical_price'] * volume).rolling(20).sum() / volume.rolling(20).sum()

# VWAP 偏离度
df['vwap_deviation'] = (close - df['vwap']) / df['vwap']

print(f"\nVWAP 偏离度统计:")
print(df['vwap_deviation'].describe())

# 解读
df['vwap_signal'] = pd.cut(
    df['vwap_deviation'],
    bins=[-np.inf, -0.02, 0.02, np.inf],
    labels=['低于VWAP（可能大单卖出）', '中性', '高于VWAP（可能大单买入）']
)

print(f"\nVWAP 信号分布:")
print(df['vwap_signal'].value_counts())

# ============================================================
# 特征4: 成交量冲击（大单冲击成本）
# ============================================================
print("\n" + "="*70)
print("特征4: 成交量冲击（大单冲击）")
print("="*70)

# Amihud 非流动性指标
df['illiquidity'] = abs(returns) / (volume + 1e-8)

# 成交量冲击
df['volume_impact'] = df['vol_pct'] * df['illiquidity']

print(f"\n成交量冲击（前5）:")
print(df['volume_impact'].sort_values(ascending=False).head())

# ============================================================
# 特征5: 资金流向估算（基于价量关系）
# ============================================================
print("\n" + "="*70)
print("特征5: 资金流向估算")
print("="*70)

# 简单的资金流向模型
# 假设: 涨幅 * 成交量 = 流入, 跌幅 * 成交量 = 流出
df['money_flow_in'] = np.where(returns > 0, returns * volume, 0)
df['money_flow_out'] = np.where(returns < 0, abs(returns) * volume, 0)

# 累计净流入
df['net_money_flow'] = (df['money_flow_in'] - df['money_flow_out']).rolling(20).sum()

# 资金流向强度
df['mfi'] = df['money_flow_in'].rolling(14).sum() / (
    df['money_flow_in'].rolling(14).sum() + df['money_flow_out'].rolling(14).sum() + 1e-8
) * 100

print(f"\n资金流向指标 (MFI) 统计:")
print(df['mfi'].describe())

# MFI 分类
df['mfi_signal'] = pd.cut(
    df['mfi'],
    bins=[0, 20, 80, 100],
    labels=['超卖', '中性', '超买']
)

print(f"\nMFI 信号分布:")
print(df['mfi_signal'].value_counts())

# ============================================================
# 特征6: 大单估算（基于成交量异常）
# ============================================================
print("\n" + "="*70)
print("特征6: 大单估算（基于成交量异常检测）")
print("="*70)

# Z-score 成交量
df['vol_zscore'] = (volume - volume.rolling(60).mean()) / (volume.rolling(60).std() + 1e-8)

# 大单判定（Z-score > 2）
df['is_large_order'] = df['vol_zscore'] > 2

# 超大单判定（Z-score > 3）
df['is_huge_order'] = df['vol_zscore'] > 3

print(f"\n大单天数: {df['is_large_order'].sum()} ({df['is_large_order'].mean():.1%})")
print(f"超大单天数: {df['is_huge_order'].sum()} ({df['is_huge_order'].mean():.1%})")

# 大单日的平均收益
large_order_returns = df[df['is_large_order']]['price_change']
normal_order_returns = df[~df['is_large_order']]['price_change']

print(f"\n大单日平均收益: {large_order_returns.mean():.2%}")
print(f"普通日平均收益: {normal_order_returns.mean():.2%}")

# ============================================================
# 综合大单特征
# ============================================================
print("\n" + "="*70)
print("综合大单特征")
print("="*70)

# 大单强度（0-100）
df['large_order_score'] = (
    df['vol_zscore'].clip(0, 3) / 3 * 40 +  # 成交量异常
    df['net_buy_strength'].clip(-1, 1).abs() * 30 +  # 买卖强度
    df['vwap_deviation'].abs().clip(0, 0.05) / 0.05 * 30  # VWAP偏离
)

# 大单方向（-1 到 1）
df['large_order_direction'] = np.sign(df['net_buy_strength'] + df['vwap_deviation'] * 100)

print(f"\n大单强度统计:")
print(df['large_order_score'].describe())

print(f"\n大单方向统计:")
print(df['large_order_direction'].describe())

# ============================================================
# 可视化
# ============================================================
print("\n" + "="*70)
print("生成可视化...")
print("="*70)

fig, axes = plt.subplots(4, 1, figsize=(14, 16))

# 选择最近2年数据
recent = df.last('730D')

# 1. 价格 + 成交量分类
ax1 = axes[0]
ax1.plot(recent.index, recent['soxl_close'], label='SOXL Close', linewidth=2)
ax1.set_ylabel('Price ($)', fontsize=12)
ax1.set_title('SOXL Price & Volume Classification', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

ax1_twin = ax1.twinx()
colors = {'大单': 'red', '中单': 'orange', '小单': 'green'}
for vol_class in ['小单', '中单', '大单']:
    mask = recent['vol_class'] == vol_class
    ax1_twin.bar(recent.index[mask], recent.loc[mask, 'soxl_volume'],
                 alpha=0.5, label=vol_class, color=colors[vol_class])
ax1_twin.set_ylabel('Volume', fontsize=12)
ax1_twin.legend(loc='upper right')

# 2. VWAP
ax2 = axes[1]
ax2.plot(recent.index, recent['soxl_close'], label='Close', linewidth=2)
ax2.plot(recent.index, recent['vwap'], label='VWAP', linewidth=2, linestyle='--')
ax2.fill_between(recent.index, recent['soxl_close'], recent['vwap'],
                  where=recent['soxl_close'] > recent['vwap'],
                  alpha=0.3, color='green', label='Above VWAP')
ax2.fill_between(recent.index, recent['soxl_close'], recent['vwap'],
                  where=recent['soxl_close'] < recent['vwap'],
                  alpha=0.3, color='red', label='Below VWAP')
ax2.set_ylabel('Price ($)', fontsize=12)
ax2.set_title('VWAP Analysis', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# 3. 资金流向
ax3 = axes[2]
ax3.bar(recent.index, recent['net_money_flow'] / 1e6,
        color=np.where(recent['net_money_flow'] > 0, 'green', 'red'), alpha=0.5)
ax3.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax3.set_ylabel('Net Money Flow ($M)', fontsize=12)
ax3.set_title('Estimated Money Flow', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. 大单强度和方向
ax4 = axes[3]
ax4.bar(recent.index, recent['large_order_score'] * recent['large_order_direction'],
        color=np.where(recent['large_order_direction'] > 0, 'green', 'red'), alpha=0.5)
ax4.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax4.set_ylabel('Large Order Score', fontsize=12)
ax4.set_xlabel('Date', fontsize=12)
ax4.set_title('Large Order Analysis', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('soxl_large_order_features.png', dpi=300, bbox_inches='tight')
print(f"✅ 图表已保存: soxl_large_order_features.png")

# ============================================================
# 保存特征
# ============================================================
print("\n" + "="*70)
print("保存特征数据")
print("="*70)

features = df[[
    'soxl_close', 'soxl_volume',
    'vol_pct', 'vol_class', 'vol_zscore',
    'buy_strength', 'sell_strength', 'net_buy_strength',
    'vwap', 'vwap_deviation',
    'volume_impact',
    'mfi', 'mfi_signal',
    'is_large_order', 'is_huge_order',
    'large_order_score', 'large_order_direction'
]].copy()

features.to_csv('soxl_large_order_features.csv')
print(f"✅ 特征已保存: soxl_large_order_features.csv")

# ============================================================
# 总结
# ============================================================
print("\n" + "="*70)
print("大单特征总结")
print("="*70)

print(f"""
基于 OHLCV 数据估算的大单特征：

1. 成交量分类
   - 小单: {(df['vol_class'] == '小单').sum()} 天 ({(df['vol_class'] == '小单').mean():.1%})
   - 中单: {(df['vol_class'] == '中单').sum()} 天 ({(df['vol_class'] == '中单').mean():.1%})
   - 大单: {(df['vol_class'] == '大单').sum()} 天 ({(df['vol_class'] == '大单').mean():.1%})

2. 资金流向指标 (MFI)
   - 超卖: {(df['mfi_signal'] == '超卖').sum()} 天
   - 中性: {(df['mfi_signal'] == '中性').sum()} 天
   - 超买: {(df['mfi_signal'] == '超买').sum()} 天

3. 大单异常
   - 大单天数: {df['is_large_order'].sum()} ({df['is_large_order'].mean():.1%})
   - 超大单天数: {df['is_huge_order'].sum()} ({df['is_huge_order'].mean():.1%})

4. 大单日表现
   - 大单日平均收益: {large_order_returns.mean():.2%}
   - 普通日平均收益: {normal_order_returns.mean():.2%}
   - 差异: {(large_order_returns.mean() - normal_order_returns.mean()):.2%}

这些特征可以用于：
- 识别大资金进出
- 判断主力意图
- 增强策略信号
""")

print("="*70)
print("✅ 完成")
print("="*70)
