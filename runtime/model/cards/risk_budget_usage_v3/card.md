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

(TBD — 训练完成后填写)

## 实验原始记录

详细配置、动机和迭代历史见：
- `experiments/candidate/risk_budget_usage_v3/notes.md`
- `experiments/candidate/risk_budget_usage_v3/expectations.md`
- `experiments/candidate/risk_budget_usage_v3/profile.yaml`

## 下一步

- 运行训练并验证指标
- 对比 v2 的收敛速度和最终表现
