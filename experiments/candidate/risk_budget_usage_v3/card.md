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

| 切片 | Return | Sharpe | MaxDD | AvgPos | StressAvgPos |
|------|--------|--------|-------|--------|-------------|
| **test_proxy** | **274.7%** | **1.074** | 21.4% | 0.208 | 0.131 |
| test_2020 | 29.3% | 1.430 | 18.5% | 0.233 | 0.158 |
| test_2022 | -9.7% | -0.601 | **12.0%** | **0.143** | 0.120 |
| test_2023_2024 | 60.5% | 1.378 | 16.0% | 0.215 | 0.134 |
| valid | 4.8% | 0.377 | 17.0% | 0.272 | 0.131 |

### Promotion Gate

- test_2022 MaxDD: 12.0% < 20.0% ✅
- test_2022 avg_pos: 0.143 < 0.15 ✅
- test_proxy Sharpe: **1.074 > 0.95** ✅

**Fixed Split Gate: PASSED!** 🎉

## 与 v1/v2 对比

| 指标 | v1 | v2 | v3 |
|------|-----|-----|-----|
| proxy Sharpe | 0.942 | 0.884 | **1.074** |
| proxy MaxDD | 32.5% | 20.6% | 21.4% |
| proxy Return | 274% | 158% | 275% |
| 2022 MaxDD | 18.7% | 7.4% | 12.0% |
| 2022 avg_pos | 0.127 | 0.070 | 0.143 |

## 结论

**✅ PROMOTE to best candidate — first risk_budget_usage to pass the gate.**

v3 的 mode-conditioned actor input 带来质的飞跃：
- proxy Sharpe 首次突破 1.0（0.942 → **1.074**）
- 通过 fixed-split promotion gate（2022 MaxDD=12%, avg_pos=0.143）
- 关键洞察：让 Actor 直接"看到"mode 比 reward shaping 间接引导更有效

## 实验原始记录

详细配置、动机和迭代历史见：
- `experiments/candidate/risk_budget_usage_v3/notes.md`
- `experiments/candidate/risk_budget_usage_v3/expectations.md`
- `experiments/candidate/risk_budget_usage_v3/profile.yaml`

## 下一步

- 运行 walk-forward 验证
- 考虑将 v3 提升为 canonical 基线
