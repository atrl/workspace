# risk_budget_usage_v3

**方向**: model
**类别**: risk_budget
**状态**: 🟡 Experimental

## 假设

在 v2 基础上，让 Actor 直接观察到当前 mode 信息（one-hot），从而学习 mode-specific 的使用策略，降低样本复杂度。

## 方法/配置

Profile: `risk_budget_usage_v3`
网络: GRU-64 + [128,64] MLP

## 架构

继承 v2 的完整架构，新增：

```
Observation (~50 dims) + Mode One-Hot (4 dims)
    │
    └──→ GRU-64 → MLP 128→64 → 5-action
```

Mode One-Hot 编码：
- `[1, 0, 0, 0]` = trend_hold
- `[0, 1, 0, 0]` = defensive_hold
- `[0, 0, 1, 0]` = crash_rebuild
- `[0, 0, 0, 1]` = flat_wait

## 与 v2 的差异

| 维度 | v2 | v3 |
|------|-----|-----|
| Mode reward | ✅ | ✅ |
| Dynamic bands | ✅ | ✅ |
| Mode-conditioned input | ❌ | ✅ Actor 直接接收 mode one-hot |
| Observation dim | ~50 | ~54 (+4 mode one-hot) |

## 结果

### Fixed-Split Backtest (Saved Model)

| 切片 | Return | Sharpe | MaxDD | Trades |
|------|--------|--------|-------|--------|
| test (full) | 434.1% | 0.87 | 47.5% | 295 |

> ⚠️ Proxy metrics (Sharpe 1.074, MaxDD 21.4%) were overly optimistic vs actual backtest.

### Walk-Forward Validation

| Window | Return | Sharpe | MaxDD | Trades |
|--------|--------|--------|-------|--------|
| wf_2020 | 19.3% | 0.29 | 19.5% | 36 |
| wf_2022 | 59.9% | 0.39 | **59.0%** | 32 |
| wf_2023_2024 | 13.0% | 0.22 | 23.4% | 40 |

### Promotion Gate

- Fixed-split Sharpe: 0.87 (proxy inflated) ❌
- Fixed-split MaxDD: 47.5% ❌
- wf_2022 MaxDD: 59.0% ❌
- wf_2023_2024 Sharpe: 0.22 ❌

**❌ PROMOTION BLOCKED.**

## 与 v1/v2 对比

| 指标 | v1 | v2 | v3 |
|------|-----|-----|-----|
| proxy Sharpe | 0.942 | 0.884 | **1.074** |
| proxy MaxDD | 32.5% | 20.6% | 21.4% |
| proxy Return | 274% | 158% | 275% |
| 2022 MaxDD | 18.7% | 7.4% | 12.0% |
| 2022 avg_pos | 0.127 | 0.070 | 0.143 |

## 结论

**❌ PROMOTION BLOCKED — Walk-forward generalization failed.**

v3's mode-conditioned actor input improved proxy metrics (Sharpe 1.074) but:
- **Proxy-to-backtest gap**: Actual backtest Sharpe 0.87, MaxDD 47.5% (vs proxy 21.4%)
- **Walk-forward collapse**: wf_2022 MaxDD 59.0%, wf_2023_2024 Sharpe 0.22
- **Root cause**: Model overfit to the fixed test period; mode-conditioned obs does not generalize to novel regimes

## 实验原始记录

详细配置、动机和迭代历史见：
- `experiments/candidate/risk_budget_usage_v3/notes.md`
- `experiments/candidate/risk_budget_usage_v3/expectations.md`
- `experiments/candidate/risk_budget_usage_v3/profile.yaml`

## 下一步

1. **诊断 mode 信息泄漏**: 验证 mode classifier 是否使用了未来信息
2. **诊断 budget_usage 过拟合**: 测试更严格的 usage 映射或正则化
3. **在 proven baseline 上测试**: 先在 generalizing baseline (如 balanced_rule_residual_v19) 上验证 mode-conditioned input
4. **考虑 v4 方向**: 更强的正则化 + 更保守的 usage 范围 + mode 输入
