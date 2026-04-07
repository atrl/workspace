#!/usr/bin/env python3
"""
数据来源说明 - 大单特征是怎么计算的

区分：
1. 真实大单数据（需要富途/L2行情）
2. 估算特征（基于 OHLCV 衍生）
"""
import pandas as pd
import numpy as np

print("="*70)
print("数据来源说明")
print("="*70)

# 加载原始数据
csv_path = '/Users/rziiiii/.openclaw/workspace/data/soxl_soxx_vix_full.csv'
df = pd.read_csv(csv_path, parse_dates=['date'])
df = df.set_index('date')

print("\n" + "="*70)
print("1. 原始数据（真实数据）")
print("="*70)

print("""
原始数据来源：CSV 文件
  - soxl_close: 收盘价（真实）
  - soxl_volume: 成交量（真实）
  - vix: VIX 指数（真实）

这些都是真实的市场数据，不是假的。
""")

print("示例（前5天）:")
print(df[['soxl_close', 'soxl_volume']].head())

print("\n" + "="*70)
print("2. 衍生特征（计算得出）")
print("="*70)

print("""
基于原始数据计算的特征：

例如：vol_pct（相对成交量）
  计算 = 当日成交量 / 过去20日平均成交量
  
  示例：
    2020-01-02: volume = 1,234,567
    过去20日平均 = 1,000,000
    vol_pct = 1,234,567 / 1,000,000 = 1.23
    
  这不是假数据，而是计算出来的衍生指标。
""")

# 计算 vol_pct
volume = df['soxl_volume']
df['vol_pct'] = volume / volume.rolling(20).mean()

print("vol_pct 计算示例:")
print(df[['soxl_volume', 'vol_pct']].head(25).tail(5))

print("\n" + "="*70)
print("3. 真实大单数据 vs 估算特征")
print("="*70)

print("""
┌─────────────────────┬──────────────────┬──────────────────┐
│                     │ 真实大单数据      │ 估算特征         │
├─────────────────────┼──────────────────┼──────────────────┤
│ 数据源              │ 富途/L2逐笔数据   │ OHLCV数据        │
│ 包含信息            │ 每笔交易金额     │ 只有日频汇总     │
│ 大单判定            │ 精确（>50万）    │ 粗略（估算）     │
│ 主力识别            │ 准确             │ 间接推测         │
│ 成本                │ 付费/需要账户    │ 免费             │
│ 数据频率            │ 实时/逐笔        │ 日频             │
│ 可获得性            │ 难               │ 易               │
└─────────────────────┴──────────────────┴──────────────────┘
""")

print("\n" + "="*70)
print("4. 我生成的特征是什么？")
print("="*70)

print("""
生成的特征属于：估算特征（基于 OHLCV 衍生）

不是"假数据"，而是：
  ✅ 基于真实价格和成交量计算
  ✅ 使用标准的金融指标公式
  ✅ 可以作为真实数据的代理（proxy）
  
但需要注意：
  ⚠️ 不是逐笔交易数据
  ⚠️ 无法精确识别大单
  ⚠️ 是间接推测，不是直接观测
  
可以理解为：
  "如果无法获得真实大单数据，用价量关系来估算"
""")

print("\n" + "="*70)
print("5. 具体计算逻辑")
print("="*70)

# 示例：主动买入强度
close = df['soxl_close']
returns = close.pct_change()
df['vol_pct'] = volume / volume.rolling(20).mean()

# 主动买入强度
df['buy_strength'] = np.where(
    (returns > 0) & (df['vol_pct'] > 1.2),
    df['vol_pct'] * returns,
    0
)

print("""
主动买入强度的计算逻辑：

步骤1：计算收益率
  returns = (今日收盘 - 昨日收盘) / 昨日收盘

步骤2：计算相对成交量
  vol_pct = 今日成交量 / 过去20日平均成交量

步骤3：判定主动买入
  如果 (returns > 0) 且 (vol_pct > 1.2):
    说明：价格上涨 + 成交量放大 = 可能有主动买入
    buy_strength = vol_pct * returns
  否则:
    buy_strength = 0

示例计算：
""")

# 显示一个具体例子
example_date = '2020-03-24'
example = df.loc[example_date]

print(f"日期: {example_date}")
print(f"  收盘价: ${example['soxl_close']:.2f}")
print(f"  成交量: {example['soxl_volume']:,.0f}")
print(f"  相对成交量: {example['vol_pct']:.2f}")
print(f"  收益率: {returns.loc[example_date]:.2%}")

if example['buy_strength'] > 0:
    print(f"  主动买入强度: {example['buy_strength']:.4f}")
    print(f"  解读: 价格上涨 + 成交量放大 = 估算为主动买入")
else:
    print(f"  主动买入强度: 0")
    print(f"  解读: 不满足主动买入条件")

print("\n" + "="*70)
print("6. 这些估算特征有用吗？")
print("="*70)

print("""
有用，但需要理解局限性：

✅ 有用的方面：
  1. 可以识别成交量异常（相对历史）
  2. 可以判断价量关系（涨放量/跌放量）
  3. 可以作为策略信号增强
  4. 免费且易于计算

⚠️ 局限性：
  1. 无法区分真实的机构/散户
  2. 无法识别具体的买卖方
  3. 是间接推测，可能误判
  4. 只能用于趋势判断，不能用于精确计量

建议：
  - 如果只是做趋势判断：估算特征够用
  - 如果需要精确识别主力：需要富途/L2数据
  - 可以将估算特征作为辅助指标
""")

print("\n" + "="*70)
print("7. 数据文件对比")
print("="*70)

# 显示原始数据 vs 衍生特征
print("\n原始数据（真实）:")
print(df[['soxl_close', 'soxl_volume']].head())

print("\n衍生特征（计算）:")
print(df[['vol_pct', 'buy_strength']].head())

print("\n" + "="*70)
print("总结")
print("="*70)

print("""
我生成的数据不是"假数据"，而是：

1. 基于真实的 OHLCV 数据（收盘价、成交量等）
2. 使用标准的金融指标公式计算
3. 是真实数据的衍生/代理指标

但这些是估算特征，不是：
  ❌ 逐笔交易数据
  ❌ 真实的大单分类（>50万、>100万等）
  ❌ 精确的主力识别

如果需要真实大单数据：
  - 富途 API（需要账户+OpenD）
  - Level 2 行情（付费）
  - 第三方数据源（付费）

当前生成的估算特征：
  ✅ 可以免费使用
  ✅ 可以作为策略辅助
  ✅ 可以识别异常和趋势
  ⚠️ 但精度有限
""")

print("="*70)
