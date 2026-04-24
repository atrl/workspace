# risk_budget_usage_v1

**方向**: model
**类别**: risk_budget
**状态**: 🟡 Good

## 假设

通过风险预算分配机制，限制单一标的风险暴露。

## 方法/配置

Profile: `risk_budget_usage_v1`
网络: GRU-64 + [128,64] MLP

## 架构

```mermaid
graph TB
    subgraph Observation[观测层]
        O1[时序特征 ~50维]
        O2[regime_risk / anchor_gap / cum_drop_30 / vix_zscore]
        O3[vol_coef / trend_gap / drawdown_streak / below_anchor_ratio]
    end

    subgraph Actor[Actor 网络]
        A1[GRU-64 编码]
        A2[MLP 128→64]
        A3[输出: 5-action delta]
    end

    subgraph ModeRouter[Mode Router 慢速风险锚定]
        M1{infer_mode_from_slow_position}
        M1 -->|pos ≤ 0.25| F[flat_wait<br/>band: 0.00-0.25]
        M1 -->|pos ≤ 0.40| D[defensive_hold<br/>band: 0.10-0.45]
        M1 -->|pos ≤ 0.60| C[crash_rebuild<br/>band: 0.25-0.65]
        M1 -->|pos > 0.60| T[trend_hold<br/>band: 0.45-0.95]
    end

    subgraph BudgetUsage[Budget Usage 映射]
        B1{delta_levels}
        B1 -->|-0.10| U0[usage=0%]
        B1 -->|-0.05| U25[usage=25%]
        B1 -->|0.00| U50[usage=50%]
        B1 -->|+0.05| U75[usage=75%]
        B1 -->|+0.10| U100[usage=100%]
    end

    subgraph PositionCalc[仓位计算]
        P1[floor + usage × band_width]
        P2 -->|cum_drop_30| P3[drawdown_position_cap]
        P3 -->|start_dd=0.10<br/>full_dd=0.30| P4[min_cap=0.40]
    end

    subgraph Execution[执行层]
        E1[|target - current| ≥ threshold?]
        E1 -->|base: 0.04| E2[执行调仓]
        E1 -->|quiet: 0.08| E3[放宽阈值]
        E1 -->|stress: 0.012| E4[紧缩阈值]
        E2 --> E5[trade_lock=5天]
    end

    O1 --> A1
    A3 --> B1
    F & D & C & T --> P1
    U0 & U25 & U50 & U75 & U100 --> P1
    P1 --> P2
    P4 --> E1
```

### 核心机制

1. **双速架构**: 慢速 Mode Router 决定风险区间（band），快速 Actor 决定区间内的使用比例（usage）
2. **Budget Usage 映射**: Actor 输出的 delta 被重新解释为 `usage ratio`（0%-100%），而非直接仓位变化
3. **动态风控**: drawdown_position_cap 在累计回撤 10%-30% 之间线性压缩仓位，最低 cap=0.40（bull 态提升至 0.65，bear 态降低至 0.30）
4. **执行防抖**: trade_lock=5天 + threshold 分级（base/stress/quiet）

## 过程

训练 → validation → checkpoint 选择 → test 回测 → promotion gate 评估。

## 结果

| 指标 | 数值 |
|------|------|
| Return | 274 |
| Sharpe | 0.942 |
| MaxDD | 32.489 |
| AvgPos | 0.269 |
| StressAvgPos | 0.113 |
| 2022 Return | -17.293 |
| 2022 MaxDD | — |
| 2020 Return | 40.877 |
| 2023-24 Return | 82.848 |
| Fixed Split Pass | ❌ |
| Walk Forward Pass | ❌ |

## 结论

表现尚可但不够强。Sharpe=0.942，MaxDD=32.489%。未通过 promotion gate。

## 与基线/相关实验的差异

与 canonical `balanced` 基线的差异：引入风险预算约束，avg_pos 通常更低。


## 实验原始记录

详细配置、动机和迭代历史见：
- `experiments/candidate/risk_budget_usage_v1/notes.md`
- `experiments/candidate/risk_budget_usage_v1/expectations.md`
- `experiments/candidate/risk_budget_usage_v1/profile.yaml`
## 下一步

- 分析未通过 gate 的原因
- 考虑与其他机制组合
