# SOXL Data Pipeline

**方向**: data
**类别**: data_pipeline
**状态**: ✅ Operational

## 假设

标准化、稳定的数据管道是 RL 训练和回测的基础。数据覆盖 2010-2025，包含 OHLCV + VIX。

## 方法/配置

- Qlib store 标准化
- 外部数据：VIX3M, VIX9D, SKEW, VVIX
- ffill + 0-fill 对齐策略
- 50 core features + rulebase + academic + transition

## 过程

1. 原始 CSV → Qlib store (`scripts/prepare_qlib_data.py`)
2. 外部数据接入 (`src/soxl_quant/data.py`)
3. 特征工程 (`engineer_feature_frame()`)
4. 训练/回测/运行时的数据供给

## 结果

| 指标 | 数值 |
|------|------|
| 数据起始 | 2010-01-01 |
| 数据结束 | 2025-12-31 |
| Core features | 50 |
| Rulebase signals | 14 |
| Academic signals | 7 |
| Transition signals | 7 |
| VIX-free proxy | regime_risk_no_vix |

## 结论

数据管道稳定运行。所有训练均使用同一套特征工程流程，保证一致性。VIX-free 替代方案在 VIX 缺失时可用。

## 与基线/相关实验的差异

这是唯一的数据管道配置。多资产数据管道待建设。

## 下一步

- [ ] 多资产联合数据管道（TQQQ/UPRO/SPXL/TECL）
- [ ] 实时数据延迟模拟
