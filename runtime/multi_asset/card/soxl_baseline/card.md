# SOXL Baseline (Single Asset)

**方向**: multi_asset
**类别**: single_asset_baseline
**状态**: ✅ Baseline

## 假设

当前仅 SOXL 单标训练。多资产泛化尚未启动，但 risk-budget 框架天然支持多资产配置。

## 方法/配置

- 单标训练：SOXL 3x LETF
- Feature 通用性：core features（除 VIX）适用于任何 3x ETF
- Risk-budget 框架：支持仓位预算分配

## 过程

1. SOXL 单标训练
2. 固定切分验证
3. Walk-forward 验证（2020/2022/2023-2024）

## 结果

| 指标 | 数值 |
|------|------|
| 训练标的 | SOXL |
| 泛化测试 | 未开始 |
| 适用特征 | 50 core (VIX-free 可用) |

## 结论

SOXL 单标策略已达到一定水平，但多资产泛化是下一步关键。Risk-budget 框架已预留多资产接口。

## 与基线/相关实验的差异

这是单资产基线。未来需要对比：
- 联合训练 vs 迁移学习
- 共享 backbone + asset-specific head

## 下一步

- [ ] TQQQ/UPRO/SPXL/TECL 迁移测试
- [ ] 共享 backbone + asset-specific head
- [ ] 联合训练 vs 迁移学习对比
