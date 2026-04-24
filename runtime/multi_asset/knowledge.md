# MULTI_ASSET Knowledge

> 最后更新: 2026-04-24
> 该方向负责跨 3x ETF 泛化测试、联合训练和多资产策略设计。

## 多资产索引

| 配置 | 类别 | 状态 | 关键指标 | 卡片 |
|------|------|------|----------|------|
| SOXL Baseline | single_asset_baseline | ✅ Baseline | SOXL only | [card](cards/soxl_baseline/card.md) |
| Zero-Shot Migration v1 | zero_shot_transfer | ✅ Completed | 5 ETFs tested | [card](cards/zero_shot_migration_v1/card.md) |

## 核心结论

1. **当前仅 SOXL 单标训练**，多资产泛化尚未启动。
2. **Risk-budget 框架** 天然支持多资产配置（仓位预算分配）。
3. **Feature 通用性**: core features（除 VIX）适用于任何 3x ETF。

## 待验证方向

- [ ] TQQQ/UPRO/SPXL/TECL 迁移测试
- [ ] 共享 backbone + asset-specific head
- [ ] 联合训练 vs 迁移学习对比
