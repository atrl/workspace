# 模型调整方向评估与实验路线图

## 诊断摘要

基于 Phase 1-4 的并行分析，当前 SOXL PPO 策略的退化可归因于三个结构性问题：

| 问题 | 证据 | 量化影响 |
|------|------|----------|
| **1. 微幅调仓过度** | 109/171 笔交易仓位变化<5%，Kelly 仅 0.0437 | 摩擦成本 ~57.8%，微幅交易 PnL 贡献仅 23.65% |
| **2. 核心持仓缺失** | 跑输季度平均仓位 47.43%；2023-Q1 超额 -63.65% | 2020-2025 中 13 个季度超额 < -5%，全部是仓位过低 |
| **3. 特征盲区** | 现有 48 维特征缺少趋势通道位置和 VIX 期限结构 | donchian_position 21 日领先相关 +0.725；vix_term_slope 领先 -0.429 |

---

## 调整方向优先级排序

### 🔴 方向 A：硬底仓 + RL 战术叠加层（最高优先级）

**问题针对性**：直接解决"核心持仓缺失"和"2024-25 跑输 B&H"。

**证据**：
- rule-based 策略固定 50% 底仓，SOXL 6 年回报 688.6%，远超 RL 的 414%。
- Phase 2 显示：13 个跑输季度的平均仓位仅 47.43%，若强制底仓 50%，这些季度的回报将显著改善。
- 2023-Q1（AI 牛市启动）策略仓位 42.33%，错失 SOXL +95.81% 的涨幅。

**实现方案**：
1. 修改 action space 映射：
   - 当前：`action -> position in [floor, cap]`（如 0.15 ~ 0.90）
   - 新方案：`position = 0.50 + (action / max_action) * 0.40`，即范围压缩到 [0.50, 0.90]
2. 保留 `drawdown_position_cap` 作为危机时突破底仓的唯一机制。
3. 或更保守的方案：`position = max(0.50, action_mapped)`，允许危机时跌破 50%，但常态不低于 50%。

**预期效果**：
- 平均仓位从 ~48% 提升到 ~65%
- 2023-2024 回报提升 +50~100pp
- 2022 maxDD 可能从 33% 上升到 45%（接近 rule-based 的 69.9%），需通过 crisis overlay 控制

**风险**：
- 2022 年无法快速清仓，maxDD 恶化。
- 需要配套更强的 crisis detection（如 vix_term_slope > 1.2 时允许跌破底仓）。

**实验设计（MVE）**：
- 修改 `src/soxl_quant/rl.py` 中的 action-to-position 映射
- 创建 `balanced_core50` profile
- 不修改特征、reward、model 结构，只改 position mapping
- 通过标准：`test_2023_2024 return > 80%` 且 `test_2022 maxDD < 45%`

---

### 🔴 方向 B：交易频率硬约束（最高优先级）

**问题针对性**：直接解决"微幅调仓过度"和"摩擦成本"。

**证据**：
- Phase 1：微幅调仓（<5%）占 63.7%，但 Kelly 仅 0.0437，Expectancy 0.38%。
- 摩擦成本 57.8%（0.1% 单边），若去掉微幅调仓可节省 ~21.8% 摩擦。
- rule-based 仅 45 笔交易 vs RL 1000+ 笔。
- 3-10 天持仓的交易 Kelly 为 **-0.3287**（负期望！）。

**实现方案**：
1. **提高再平衡阈值**：
   - 当前：`base_threshold = 0.015`
   - 新值：`base_threshold = 0.05`（5% 仓位变化才触发交易）
2. **交易锁定期**：
   - 任意交易后，强制 3-5 天内不允许再次交易。
3. **最小交易幅度**：
   - 单次仓位变化必须 >= 5%，否则 action 被截断到 0（无交易）。
4. **Action 空间粗化（可选）**：
   - 从 11 级压缩到 5 级：0%, 30%, 50%, 75%, 100%（相对于可交易区间）。

**预期效果**：
- 交易次数从 ~1000 降至 ~200-300
- 摩擦成本从 ~57.8% 降至 ~15-20%
- 整体回报提升 +50~100pp

**风险**：
- 可能错过 1-2 天的快速反转（如 2020 年 3 月暴跌后的 V 型反弹）。
- 需要验证 crisis 期间的反应速度是否足够。

**实验设计（MVE）**：
- 修改 `src/soxl_quant/settings.py` 中的 ExecutionConfig
- 创建 `balanced_trade_lock` profile：threshold=0.05, trade_lock_days=3, min_trade_turnover=0.05
- 不修改 model，只改 execution overlay
- 通过标准：`trades < 400` 且 `test_proxy Sharpe > 0.80`

---

### 🟠 方向 C：新特征工程（高优先级）

**问题针对性**：提升趋势识别能力和危机预警精度。

**证据**：
- `donchian_position` 21 日领先相关 **+0.725**（最强特征！）。
- `vix_term_slope` 21 日领先相关 **-0.429**（危机预警）。
- `trend_alignment` 21 日领先相关 **+0.625**（趋势确认）。
- 现有特征缺少"我在趋势中的什么位置"的 explicit 度量。

**实现方案**：
1. **CTA 趋势特征**（零外部依赖）：
   - `donchian_position_20`: (close - 20日low) / (20日high - 20日low)
   - `trend_alignment`: MA5>MA21 + MA21>MA63 的计数（0-2）
   - `atr_position_14`: (close - 14日low) / (14日high - 14日low)
2. **VIX 期限结构特征**（CBOE 数据已获取）：
   - `vix_term_slope`: VIX / VIX3M
   - `vix_term_spread`: VIX3M - VIX
   - `vix_short_term_slope`: VIX9D / VIX
3. **VIX 衍生品特征**：
   - `vvix_zscore_21`: VVIX 的 21 日 Z-score（短期波动 regime 预警）
   - `skew_zscore_21`: SKEW 的 21 日 Z-score（尾部风险定价）

**预期效果**：
- 趋势市中的仓位决策更果断（减少"要不要加仓"的犹豫）
- 危机预警提前 1-5 天（vix_term_slope 领先 -0.429）
- 2022 maxDD 可能改善 3-5pp

**风险**：
- 特征过多可能导致过拟合（现有 48 维 + 新增 ~6 维）。
- 需要与现有特征做相关性检验，去除 r > 0.8 的重复特征。

**实验设计（MVE）**：
- 修改 `src/soxl_quant/data.py` 的 `FEATURE_COLUMNS`
- 创建 `balanced_new_features` profile
- 特征增加但 model 结构不变（48 -> 54 维）
- 通过标准：`valid score > 65` 且 `test_2022 maxDD < 35%`

---

### 🟡 方向 D：MoE Gate 监督 + Expert 再训练（中优先级）

**问题针对性**：提升 gate 选择 expert 的精度。

**证据**：
- Expert 2 单独 404.9% > natural gate 383.3%（+21.6pp）。
- gate 在 extreme regime 仅选 Expert 2 17.9% 的时间，而 Expert 2 在 stress 下表现最好。

**实现方案**：
1. 添加 `regime_routing_loss`：
   - regime_risk < 0.35 → 目标 Expert 0
   - regime_risk > 0.65 → 目标 Expert 2
   - 交叉熵损失，权重 0.1
2. 或更激进的方案：hard regime routing（直接根据 regime_risk 选择 expert，不使用 gate softmax）。

**预期效果**：
- +5-10% return， modest drawdown improvement
- 但不解决根本的"核心持仓缺失"和"交易过度"问题

**风险**：
- 单独实施 impact 有限（+5-10%），应作为方向 A/B 的配套措施。

---

### 🟡 方向 E：时段自适应 Reward（中优先级）

**问题针对性**：解决牛市中 reward 对"持仓"惩罚过重的问题。

**证据**：
- Phase 2 显示策略在 2023-Q1、2024-Q1 等牛市中仓位过低。
- 当前 reward 可能过度惩罚 drawdown，导致模型宁愿空仓也不承担趋势风险。

**实现方案**：
1. 添加 `trend_bonus`：当 `trend_alignment == 2`（多头排列）时，持仓 reward 系数提升 1.5x。
2. 添加 `missed_bull_penalty`：当 SOXL 5 日涨幅 > 10% 且仓位 < 50% 时，给予额外惩罚。

**风险**：
- 难以校准，可能导致 2022 年过度持仓。
- 不如方向 A（硬底仓）来得直接和稳定。

---

## 推荐执行顺序

```
Week 1:
  ├── 并行实验 A1: hard floor=50% + action overlay [0.50, 0.90]
  └── 并行实验 B1: threshold=0.05 + trade_lock=3 + min_turnover=0.05

Week 2:
  ├── 评估 A1/B1 结果，选胜者进入 Week 3
  └── 若 A1/B1 均失败，分析原因，考虑组合 A+B

Week 3:
  ├── 组合实验: A + B（硬底仓 + 交易锁）
  └── 并行实验 C: 新增 donchian_position + vix_term_slope + trend_alignment

Week 4:
  ├── 若 A+B 通过，叠加 C（新特征）
  ├── 若 gate 仍不理想，叠加 D（MoE 监督）
  └── 最终评估 vs rule-based 和 canonical balanced
```

---

## 通过/失败标准

| 实验 | 必须通过 | 加分项 |
|------|----------|--------|
| A（硬底仓） | `test_2023_2024 return > 80%` | `test_proxy return > 500%` |
| B（交易锁） | `trades < 400`, `test_proxy Sharpe > 0.80` | `friction < 20%` |
| C（新特征） | `valid score > 65`, `test_2022 maxDD < 35%` | `test_proxy Sharpe > 0.90` |
| A+B+C 组合 | `test_proxy return > 550%`, `test_2022 maxDD < 40%` | 超越 rule-based |

---

## 立即行动清单

1. [ ] 创建 `balanced_core50` profile（方向 A）
2. [ ] 创建 `balanced_trade_lock` profile（方向 B）
3. [ ] 修改 `src/soxl_quant/data.py` 添加新特征（方向 C）
4. [ ] 运行 smoke test 验证三个方向各自的 impact
5. [ ] 基于 smoke test 结果，选择 1-2 个方向进入完整训练
