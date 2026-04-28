# balanced_macro_v1 实验笔记

## 实验设计

**目标**：验证宏观领先指标（macro leading indicators）对 SOXL 3x ETF 策略泛化能力的改善效果。

**Baseline**：`balanced` (GRU-64 + MLP[128,64], 无 macro features)
- Fixed-split: 1729.3% / Sharpe 1.123 / MaxDD 50.1% (修复数据泄露后)
- Walk-forward: wf_2020 +125.8% / Sharpe 2.30, wf_2022 -25.8% / Sharpe -0.91, wf_2023_2024 -15.2% / Sharpe -0.22

**Changed layer**：`feature` — 新增 macro_signals feature pack（10 个宏观特征）

**Hypothesis**：
当前特征栈几乎全部是 technical/microstructure 指标（价格动量、波动率、回撤、VIX 等）。这些指标在 bear market 中通常是**同步或滞后**的——当 regime_risk 已经飙升到 0.8 时，市场已经在暴跌了。宏观领先指标（credit spread、yield curve slope、DXY）往往**领先** equity bear market 数月，为策略提供更早的防御信号，从而改善 walk-forward 在 2022 等 stress window 的表现。

**Mechanism delta**：
- 新增 10 个宏观特征（来自 FRED 免费数据）
- Credit spread (BAA - DGS10): 信用利差扩大 = 经济恶化信号
- Yield curve slopes (10Y-2Y, 10Y-3M): 倒挂 = recession 预测信号
- DXY z-score: 美元走强 = 风险偏好下降
- Oil z-score: 油价波动 = 通胀/增长预期变化

**External reference**：
- FRED (Federal Reserve Economic Data)
- BAA: Moody's Seasoned Baa Corporate Bond Yield
- DGS10: 10-Year Treasury Constant Maturity Rate
- T10Y2Y / T10Y3M: Yield curve spreads
- DTWEXBGS: Trade Weighted U.S. Dollar Index
- DCOILWTICO: WTI Crude Oil Price

**Accepted trade-off**：
- Macro features 月度数据居多（如 BAA），通过 forward-fill 对齐到日度，可能引入一定 staleness
- DXY 数据从 2006 年开始，2010 年前的训练数据部分为 0（no coverage），但 `_align_external` 会自动 fillna(0.0)
- 特征维度从 ~60 增加到 ~70，obs_dim 从 ~75 增加到 ~85，网络容量不变

**Target window**：
- Fixed-split: test_proxy, test_2020, test_2022, test_2023_2024
- Walk-forward: wf_2020, wf_2022, wf_2023_2024

**Comparison-table row key**：`balanced_macro_v1`

## 实验状态

- [x] 数据下载（FRED macro series）
- [x] data.py 接入 macro_signals feature pack
- [x] Profile 创建（balanced_macro_v1）
- [x] Registry 更新
- [x] Smoke / 快速验证训练（20 iters / 5 epi）
- [ ] 完整 80-iteration 训练（后台运行中）
- [ ] Fixed-split 回测对比
- [ ] Walk-forward 验证
- [ ] 结果分析与 knowledge.md 更新

## Quick Test 结果（20 iters / 5 epi / seed 42）

| Metric | balanced_macro_v1 (20it) | balanced baseline (80it) |
|--------|--------------------------|--------------------------|
| Return | 423.4% | 461.5% |
| Sharpe | 0.77 | 0.80 |
| MaxDD | 58.9% | 58.6% |
| Trades | 315 | 328 |

**注意**：20-iteration 结果不可与 80-iteration baseline 直接比较。此结果仅验证：
1. Macro feature pack 数据加载和训练流程无崩溃
2. 模型能学到有意义的策略（trade 数量与 baseline 接近）
3. 初步指标在同一数量级，未出现明显退化

## 完整训练结果（80 iters / 10 epi / seed 42）

**Status**: ✅ 完成

### 关键指标

| Metric | balanced_macro_v1 | balanced baseline (seed 20260411) | Delta |
|--------|-------------------|-----------------------------------|-------|
| **Valid** | | | |
| Return | 60.6% | 60.9% | -0.3pp |
| Sharpe | 1.47 | 1.63 | -0.16 |
| MaxDD | 32.7% | 27.5% | +5.2pp |
| Score | 86.15 | 90.20 | -4.05 |
| **Test Proxy** | | | |
| Return | 450.3% | 521.2% | -70.9pp |
| Sharpe | 0.78 | 0.85 | -0.07 |
| MaxDD | 58.3% | 53.9% | +4.4pp |
| **test_2020** | | | |
| Return | **144.4%** | 112.4% | **+32.0pp** |
| Sharpe | **2.26** | 2.17 | **+0.09** |
| MaxDD | 25.3% | 22.3% | +3.0pp |
| **test_2022** | | | |
| Return | **-35.5%** | -28.2% | **-7.3pp** |
| Sharpe | **-1.00** | -0.87 | **-0.13** |
| MaxDD | **39.5%** | 36.1% | **+3.4pp** |
| AvgPos | 0.37 | 0.31 | +0.06 |
| **test_2023_2024** | | | |
| Return | **82.5%** | 48.5% | **+34.0pp** |
| Sharpe | **0.87** | 0.70 | **+0.17** |
| MaxDD | 38.6% | 38.3% | +0.3pp |

### 结论

1. **macro features 改善了 bull/recovery 期间的表现**：2020 (+32pp) 和 2023-2024 (+34pp) 均有显著提升
2. **macro features 恶化了 bear market 期间的表现**：2022 (-7.3pp, MaxDD +3.4pp) — 这与假设相反
3. **整体 test proxy 略差**：450% vs 521%，但 slice 层面呈现明显的 regime-dependent 差异
4. **Valid 表现下降**：score 86.15 vs 90.20，macro features 在 2015-2018 期间可能引入了 noise

### 假设失败的可能原因

1. **Seed 差异**：baseline seed=20260411, macro_v1 seed=42。需要相同 seed 的 baseline 才能排除 random variance。
2. **Macro 信号 staleness**：credit spread 是月度数据，forward-fill 到日度可能在 fast-moving bear market（如 2022）中反应过慢。
3. **Signal-to-noise 问题**：宏观指标在 recovery 期提供了有价值的 context，但在 sustained bear 期间可能因为滞后性导致模型加仓过早或减仓过晚。
4. **特征缩放问题**：macro features 的数值范围（如 credit spread 0-6%）与 technical features（ mostly z-score / percentile）不同，可能对 GRU+MLP 的 learning 造成干扰。

### Ablations 进行中

- [x] balanced_macro_v1 (seed 42, 80 iters) — 完成
- [ ] balanced baseline (seed 42, 80 iters) — 后台运行中 (task: bash-6wadn9i6)
- [ ] Walk-forward 验证 — 待 baseline ablation 完成后决定

### 下一步

1. **等待 baseline seed 42 完成**，进行 direct fixed-split 对比
2. 如果 2022 差异确认（非 seed variance），则分析 macro feature staleness 问题
3. 考虑 walk-forward 验证（如果 fixed-split 差异有意义）
4. 特征工程改进方向：
   - 对 macro features 做 regime-conditioned normalization
   - 尝试 macro + technical 的交互特征（如 credit_spread × regime_risk）
   - 减少宏观特征数量，只保留最有效的 2-3 个

## 数据覆盖

| 文件 | 来源 | 频率 | 范围 | 行数 |
|------|------|------|------|------|
| credit_spread_baa.csv | BAA - DGS10 | 月度 (ffill) | 1962-02 ~ 2025-12 | 491 |
| yield_curve_10y2y.csv | FRED T10Y2Y | 日度 | 1976-06 ~ 2026-04 | 12472 |
| yield_curve_10y3m.csv | FRED T10Y3M | 日度 | 1982-01 ~ 2026-04 | 11081 |
| dxy_broad.csv | FRED DTWEXBGS | 日度 | 2006-01 ~ 2026-04 | 5092 |
| oil_wti.csv | FRED DCOILWTICO | 日度 | 1986-01 ~ 2026-04 | 10143 |
