# DATA Knowledge

> 最后更新: 2026-04-24
> 该方向负责数据管道、qlib store、外部数据接入和数据质量监控。

## 数据配置索引

| 配置 | 类别 | 状态 | 关键指标 | 卡片 |
|------|------|------|----------|------|
| SOXL Data Pipeline | data_pipeline | ✅ Operational | 2010-2025, 78 features | [card](cards/soxl_data_pipeline/card.md) |

## 核心结论

1. **SOXL 数据覆盖**: 2010-01-01 至 2025-12-31，包含 OHLCV + VIX。
2. **外部数据**: VIX3M, VIX9D, SKEW, VVIX 已接入（qlib store）。
3. **Feature 稳定性**: 50 core features + 14 rulebase + 7 academic + 7 transition。
4. **VIX-free 替代**: regime_risk_no_vix 在 VIX 缺失时可用。

## 50 Core Features 主题分类

| 主题 | 特征数 | 关键驱动 |
|------|--------|----------|
| Trend / Anchor | 6 | 长期趋势锚定 |
| Momentum | 7 | 短期方向 |
| Volatility | 6 | 风险体制 |
| Drawdown | 7 | 压力深度与持续时间 |
| Structure | 4 | 结构性弱势 |
| Downside | 4 | 下行不对称性 |
| VIX | 4 | 市场恐惧指标 |
| Tracking Error | 4 | LETF 特有摩擦 |
| Velocity | 5 | 体制变化速率 |
| Proxy | 4 | 衍生绩效指标 |
| Composite | 3 | 多信号 stress 指数 |

## 数据泄露修复（2026-04-27）

### 问题
`engineer_feature_frame()` 中的 `regime_risk`、`vol_stress_score` 等核心特征使用了**全序列** `.rank(pct=True)`。由于 feature frame 跨越 train+valid+test（`start_time=train_seg[0], end_time=test_seg[1]`），训练数据获得了未来信息。

### 修复
- 8 处 `.rank(pct=True)` → `_rolling_percentile(..., 252)`
- `jump_variation_5` 的 full-series std fallback → `expanding().std()`

### 影响评估
- `balanced` test_proxy Return 下降 ~12%（1958.9% → 1729.3%）
- 泄露确实存在但非灾难性
- 修复后 MaxDD 反而改善（-52.9% → -50.1%）

### 教训
- 所有 percentile/rank 操作必须用滚动窗口
- 已有 `_rolling_percentile()` 安全函数，但未全面应用
- 未来添加新特征时必须强制审查是否有 look-ahead

## 待验证方向

- [ ] 多资产联合数据管道（TQQQ/UPRO/SPXL/TECL）
- [ ] 实时数据延迟模拟
- [ ] 修复泄露后重新训练子模型
