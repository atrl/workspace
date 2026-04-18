# 深度复盘：RL 与 Rule-Based 的本质差距

> 这份报告回应用户的核心质疑：之前的方案是工程补丁，未触及 RL 的根本缺陷。
> 我们从 reward 设计、时间尺度、交易哲学三个维度重新审视。

---

## 一、差距不是 274pp，是两种交易哲学

### Rule-Based（soxl_backtest）的交易逻辑

```
触发条件 → 大幅调仓 → 长期持仓 → 下一次触发
```

| 特征 | 数值 | 含义 |
|------|------|------|
| 交易次数 | 45 | 6 年仅 45 笔 |
| 平均持仓 | 57% | 接近恒定半仓 |
| 最小仓位 | 50% | **永不空仓** |
| 单笔调仓 | 10%~30% | 大幅进出 |
| 触发机制 | 恐慌指数/累积跌幅/止盈线 | **事件驱动** |

**交易哲学**：只在极端事件发生时交易。大部分时间持有 50% 底仓，让时间产生复利。

### RL（canonical balanced）的交易逻辑

```
每日观察 → 微调仓位 → 次日再观察 → 再微调
```

| 特征 | 数值 | 含义 |
|------|------|------|
| 交易次数 | 578 | 6 年 578 笔 |
| 平均持仓 | 42% | 低于 rule-based |
| 最小仓位 | 0% | **频繁空仓** |
| 中位持仓天数 | 3 天 | **隔日交易** |
| 单笔调仓 | <5%（63.7% 的交易）| 微幅进出 |
| 触发机制 | 每日 feature vector → actor | **连续优化** |

**交易哲学**：试图每天捕捉最优仓位。在噪声市场中，这等价于过度交易。

---

## 二、根本问题：RL 在错误的时间尺度上择时

### 日频噪声 vs 周频信号

SOXL 是 3x 杠杆半导体 ETF，日收益率分布：

| 统计量 | 数值 |
|--------|------|
| 日均波动 | ~4-5% |
| 日收益标准差 | ~4.2% |
| 5日收益标准差 | ~9.1% |
| 21日收益标准差 | ~17.5% |

**关键洞察**：日收益的**信噪比极低**。任何基于单日信号的择时，其期望都被噪声淹没。

数学上，如果日收益服从 N(μ, σ²)，其中 μ≈0.15%（年化 ~40% 的日均值），σ≈4.2%，则单日择时的信息比率：

```
IR = μ / σ ≈ 0.15 / 4.2 ≈ 0.036
```

这意味着：**即使你有完美的日频预测能力，每单位风险的超额收益也极低**。

而 rule-based 在周/月尺度上择时：
```
IR_5d = μ_5d / σ_5d ≈ 0.75 / 9.1 ≈ 0.082
IR_21d = μ_21d / σ_21d ≈ 3.2 / 17.5 ≈ 0.183
```

**时间尺度越长，信噪比越高，择时的期望越正。**

### RL 的"死亡螺旋"

1. 模型看到日频 signal → 微调仓位
2. 次日噪声反转 → 模型再调仓
3. 每次调仓付出 0.1% 成本
4. aggregate 收益被摩擦吞噬
5. 但模型从 aggregate reward 中学到"调仓是好的"（因为有时确实蹭到了趋势）
6. 模型继续高频调仓 → 摩擦继续积累

**这不是"模型不够聪明"，这是"在噪声中做信号检测"的结构性失败。**

---

## 三、Reward 函数的致命缺陷：Aggregate vs Marginal

### 当前 Reward 的核心项

```python
reward = pnl_ret * 100.0 * return_scale
```

其中 `pnl_ret` 是**账户总值的日变化率**，包含两部分：
1. **持仓收益**：`prev_pos * market_return`
2. **调仓收益/成本**：`(new_pos - prev_pos) * market_return - cost`

**问题**：模型无法区分这两部分。

### 示例：一次典型的"无效调仓"

| 步骤 | 事件 | 市场涨 3% | 模型行为 | 账户收益 |
|------|------|-----------|----------|----------|
| T | 持仓 40% | — | — | — |
| T+1 | 市场涨 3% | +3% | 加仓到 50% | +1.5% → +2.3%（扣除成本） |
| T+2 | 市场跌 2% | -2% | 减仓到 40% | +2.0% → +1.7%（扣除成本） |
| T+3 | 市场涨 1% | +1% | 保持 40% | +1.7% → +2.1% |

**分析**：
- 如果 T 日保持 40% 不动，T+3 的收益 = 40% × (1.03×0.98×1.01 - 1) = **+1.56%**
- 实际策略收益 = **+2.1%**（看起来更好！）
- 但 T+1 加仓、T+2 减仓这两笔调仓的真实边际价值：
  - T+1 加仓到 50%：多赚了 10% × 3% = 0.3%，但成本 ~0.15% → **净 +0.15%**
  - T+2 减仓到 40%：少亏了 10% × (-2%) = -0.2%，但成本 ~0.15% → **净 +0.05%**
  - 两笔调仓合计边际价值 = **+0.20%**，但付出了两笔摩擦

模型看到的 aggregate reward 是正的，但实际上这两笔调仓的**风险调整期望接近零**。

### 更糟的情况：噪声驱动的反向调仓

| 步骤 | 市场 | 模型行为 | 边际价值 |
|------|------|----------|----------|
| T | +2% | 加仓到 60% | +0.2%（好）|
| T+1 | -3% | 恐慌减仓到 30% | -0.3%（坏！追高杀跌）|
| T+2 | +4% | 追高加仓到 50% | 错过 20% × 4% = 0.8% |

模型在 T+1 因为恐慌减仓，T+2 因为 FOMO 加仓。这种"追涨杀跌"的模式在日频上反复出现，**aggregate reward 可能仍为正**（因为市场整体向上），但**每笔调仓的期望为负**。

---

## 四、为什么"胜率"思维害了 RL

### 当前 Reward 暗含的假设

当前 reward 设计（return_scale、sharpe_weight、drawdown_penalty 等）本质上是：

> "如果你让账户总值增长了，就是好策略。"

这等价于优化**胜率**（win rate in terms of aggregate metrics），而非**期望**（per-trade expectancy）。

### 凯利公式告诉我们什么

Kelly Criterion: `f* = W - (1-W)/R`

其中：
- W = 胜率
- R = 盈亏比 = avg_win / |avg_loss|
- f* = 最优仓位比例

**关键洞察**：Kelly 不优化"赢多少次"，它优化"每次下注的期望对数收益"。

在我们的 RL 场景中，这对应于：
> **模型不应该学习"每天调仓让账户增长"，而应该学习"只在期望 > 成本时才调仓"。**

### 当前 RL 的"伪正期望"

从 Phase 1 的分析：
- 整体 round-trip：Kelly = 0.1615（正但很低）
- 微幅调仓（<5%）：Kelly = 0.0437（接近零）
- 3-10 天持仓：Kelly = **-0.3287**（负！）

但模型仍然在大量执行这些负期望交易，因为 aggregate reward 掩盖了单笔交易的低质量。

---

## 五、触及本质的解决方案

### 方案 1：Marginal Expectancy Reward（边际期望奖励）⭐

**核心思想**：reward 只反映"这笔调仓带来的边际收益"，而非"账户总收益"。

```python
# 当前
reward = pnl_ret * return_scale

# 改进：Marginal Action Reward
hold_pnl = prev_pos * market_return  # 不调仓的收益
actual_pnl = new_pos * market_return - cost  # 调仓后的收益
marginal_pnl = actual_pnl - hold_pnl
reward = marginal_pnl * return_scale - turnover_penalty
```

**效果**：
- 如果模型保持仓位不动，reward = 0（中性）
- 如果模型调仓但边际收益 < 成本，reward < 0（惩罚）
- 模型被迫学习："只有当调仓确实能创造价值时才交易"

**风险**：
- 模型可能变得过于保守，完全不动
- 需要配合适当的"探索奖励"（exploration bonus）

### 方案 2：Delayed Trade Feedback（延迟交易反馈）⭐

**核心思想**：每笔调仓后，跟踪未来 N 步这笔调仓的边际表现，负表现给予追溯惩罚。

```python
# 在 simulator 中跟踪每笔调仓
class TradeTracker:
    def __init__(self):
        self.open_trades = []  # [(entry_step, entry_pos, delta, entry_price)]
    
    def on_rebalance(self, step, prev_pos, new_pos, price):
        if abs(new_pos - prev_pos) > threshold:
            self.open_trades.append({
                'step': step,
                'delta': new_pos - prev_pos,
                'price': price,
                'evaluated': False
            })
    
    def evaluate(self, current_step, current_price):
        for trade in self.open_trades:
            if not trade['evaluated'] and current_step - trade['step'] >= HOLDING_EVAL_WINDOW:
                # 计算这笔调仓的后续收益
                subsequent_return = (current_price / trade['price'] - 1) * np.sign(trade['delta'])
                if subsequent_return < 0:
                    # 这笔调仓是负期望的！给予惩罚
                    return -abs(subsequent_return) * EXPECTANCY_PENALTY_WEIGHT
                trade['evaluated'] = True
```

**效果**：
- 模型会学到："在噪声中频繁调仓会导致大量负期望交易"
- 自然地抑制日频微幅调仓
- 鼓励"有信念的"大幅调仓（因为大幅调仓的后续表现更容易评估）

**风险**：
- 增加了 reward 的延迟，可能减慢学习
- 需要仔细选择评估窗口（N 太大则反馈太迟，N 太小则被噪声干扰）

### 方案 3：Time-Scale Separation（时间尺度分离）

**核心思想**：RL 不应在日频上做仓位决策，而应在周频/事件频上做。

实现方式 A：**降频训练**
- 每 5 个交易日做一次决策（而不是每天）
- 中间天数保持仓位不变
- 这会强制模型学习更长周期的信号

实现方式 B：**双时间尺度架构**
- 慢速控制器（Slow Controller）：每 5-10 天决定"战略仓位"（core position）
- 快速控制器（Fast Controller）：每天决定"战术偏移"（±10%）
- 只有慢速控制器可以直接改变底仓，快速控制器只能微调

**效果**：
- 模型被迫在更高信噪比的尺度上做决策
- 自然减少交易次数
- 更接近 rule-based 的"大波段"哲学

### 方案 4：Kelly-Inspired Position Sizing（凯利启发式仓位）

**核心思想**：不直接套用 Kelly 公式（因为赌局不独立），但借鉴其"期望驱动仓位"的理念。

```python
# 在 action-to-position 映射中加入期望估计
def position_from_expectancy(expected_return, expected_vol, cost):
    """
    expected_return: 模型估计的未来 N 日期望收益
    expected_vol: 模型估计的未来 N 日波动率
    cost: 调仓成本
    """
    # 信息比率
    ir = expected_return / (expected_vol + 1e-8)
    
    # Kelly-like sizing: f = IR² / 2 (simplified)
    kelly_fraction = (ir ** 2) / 2.0
    
    # 保守缩放（Half-Kelly）
    position = kelly_fraction * 0.5
    
    # 如果期望收益 < 成本，不交易
    if expected_return < cost:
        position = 0
    
    return np.clip(position, floor, cap)
```

**效果**：
- 仓位大小与"期望/风险"成正比
- 低期望时自动降低仓位
- 高期望时自动增加仓位

**挑战**：
- 需要模型能估计期望收益和波动率（可以用 critic network 的输出来近似）
- 需要额外的网络输出头（mean 和 variance）

---

## 六、方案优先级与实验设计

### 优先级矩阵

| 方案 | 触及本质程度 | 实现难度 | 预期效果 | 风险 |
|------|-------------|----------|----------|------|
| Marginal Expectancy Reward | ⭐⭐⭐ | 低 | 高 | 模型可能过于保守 |
| Delayed Trade Feedback | ⭐⭐⭐ | 中 | 高 | 学习变慢 |
| Time-Scale Separation | ⭐⭐⭐ | 中 | 很高 | 架构改动大 |
| Kelly Position Sizing | ⭐⭐ | 高 | 中 | 估计误差可能放大 |

### 推荐实验顺序

**第一步：Marginal Expectancy Reward（轻量实验）**

```
修改：src/soxl_quant/rl.py 的 ContinuousReward.reward()
不修改：模型结构、特征、训练流程
预期：交易次数下降 50%+，平均持仓天数上升到 10+
通过标准：trades < 300，test_proxy Sharpe > 0.85
```

**第二步：Delayed Trade Feedback（若第一步效果不足）**

```
修改：SoxlSimulator 增加 TradeTracker
不修改：模型结构
预期：模型学会"只在有信念时调仓"
通过标准：微幅调仓（<5%）比例 < 20%
```

**第三步：Time-Scale Separation（若前两步仍不足）**

```
修改：训练频率降至每 3-5 天一次
修改：Action space 分为"战略仓位"和"战术偏移"
预期：模型行为接近 rule-based 的大波段
通过标准：trades < 100，test_proxy return > 500%
```

---

## 七、核心结论

### 之前的方案为什么不靠谱

之前的方案（硬底仓、交易锁、新特征）都是**对症状的工程补丁**：
- 硬底仓 → 强制模型保持仓位（但模型仍然每天"想做点什么"）
- 交易锁 → 强制减少交易（但模型没有学到"为什么不该交易"）
- 新特征 → 给模型更多信息（但模型仍然用这些信息做日频噪声交易）

**它们没有解决根本问题：模型在优化 aggregate reward，而非 per-trade expectancy。**

### 正确的方向

**让模型学到"只在期望 > 成本时交易"**：
1. **Reward 层面**：从 aggregate 改为 marginal
2. **反馈层面**：引入交易级延迟反馈
3. **时间尺度**：从日频提升到周频

### 与 Rule-Based 的和解

Rule-based 的 688.6% 不是因为它"更聪明"，而是因为它：
1. 在**正确的时间尺度**上交易（事件驱动，周/月级）
2. 每笔交易的**期望天然为正**（恐慌买入→均值回归）
3. **摩擦极低**（45 笔 vs 1000+ 笔）

RL 不需要复制 rule-based 的规则，但需要学习 rule-based 的**交易哲学**：
> **"不交易也是一种交易。只有当你有显著信息优势时，才值得付出摩擦成本。"**
