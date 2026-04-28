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

## 完整训练（80 iters / 10 epi / seed 42）

- 后台任务已启动，预计 1-2 小时完成
- Log: `runtime/model/card/balanced_macro_v1/full_train.log`
- 完成后将自动生成 backtest summary、dashboard 更新、actor checkpoint

## 数据覆盖

| 文件 | 来源 | 频率 | 范围 | 行数 |
|------|------|------|------|------|
| credit_spread_baa.csv | BAA - DGS10 | 月度 (ffill) | 1962-02 ~ 2025-12 | 491 |
| yield_curve_10y2y.csv | FRED T10Y2Y | 日度 | 1976-06 ~ 2026-04 | 12472 |
| yield_curve_10y3m.csv | FRED T10Y3M | 日度 | 1982-01 ~ 2026-04 | 11081 |
| dxy_broad.csv | FRED DTWEXBGS | 日度 | 2006-01 ~ 2026-04 | 5092 |
| oil_wti.csv | FRED DCOILWTICO | 日度 | 1986-01 ~ 2026-04 | 10143 |
