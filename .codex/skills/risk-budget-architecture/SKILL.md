# Risk Budget Usage 架构

## 用途

记录 `risk_budget_usage_v1` 及其变体的两速风险预算架构。该架构是当前 harness 中表现最好的候选模型（Sharpe=0.942，2022 MaxDD=18.7%），核心思想是：

> **慢速路径定区间，快速路径定用量。**

## 架构概览

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

## 关键组件

### 1. Mode Router（慢速风险锚定）

由 rule-based 慢速路径（或 `balanced_rule_residual_mode_v1` actor）决定当前风险模式：

| Mode | Floor | Cap | 含义 |
|------|-------|-----|------|
| flat_wait | 0.00 | 0.25 | 空仓/极低仓位等待 |
| defensive_hold | 0.10 | 0.45 | 防御性持有 |
| crash_rebuild | 0.25 | 0.65 | 崩盘后重建仓位 |
| trend_hold | 0.45 | 0.95 | 趋势确认后持有 |

实现：`src/soxl_quant/policy/two_speed_budget.py::infer_mode_from_slow_position()`

### 2. Budget Usage 映射（快速路径）

Actor 输出 `delta ∈ {-0.10, -0.05, 0, +0.05, +0.10}`，在 `position_mapping: budget_usage` 模式下被映射为 usage ratio：

| Delta | Usage Ratio |
|-------|-------------|
| -0.10 | 0% |
| -0.05 | 25% |
| 0.00 | 50% |
| +0.05 | 75% |
| +0.10 | 100% |

实现：`src/soxl_quant/policy/positioning.py::action_to_budget_usage_position()`

### 3. 仓位计算

```
position = mode_floor + usage_ratio × (mode_cap - mode_floor)
```

然后经过 `drawdown_position_cap` 动态压缩：
- cum_drop_30 ≤ 10%: cap = 1.0
- cum_drop_30 ≥ 30%: cap = min_cap (0.40 normal, 0.65 bull, 0.30 bear)
- 之间线性插值

### 4. 执行层

- **base_threshold**: 0.04 — 正常阈值
- **quiet_threshold**: 0.08 — 低波动期放宽
- **stress_threshold**: 0.012 — 高波动期紧缩
- **trade_lock_days**: 5 — 防抖锁定

## 历史表现

| 指标 | 值 |
|------|-----|
| test_proxy Return | 274% |
| test_proxy Sharpe | 0.942 |
| test_proxy MaxDD | 32.5% |
| test_2022 Return | -17.3% |
| test_2022 MaxDD | **18.7%** |
| test_2022 avg_pos | **0.127** |

**这是唯一通过 test_2022 promotion gate 的候选模型**（MaxDD < 30%, avg_pos < 0.35）。

## 为什么它有效

1. **风险区间隔离**: Mode Router 在 crisis 时强制进入 flat_wait/defensive_hold 低区间
2. **用量而非绝对值**: Actor 只决定「用多少预算」，而非「仓位多少」，避免 Actor 在压力下过度加仓
3. **双层风控**: Mode band + drawdown cap 双重约束
4. **保守执行**: stress_threshold=0.012 在高波动时几乎禁止调仓

## 实验变体

| 变体 | 差异 | 结果 |
|------|------|------|
| `risk_budget_usage_v1` | PPO 原生训练 | Sharpe=0.942 |
| `risk_budget_bc_warmstart_v1` | BC warmstart + PPO fine-tune | 待验证 |
| replay composition | mode_v1 + imitation_v4 路径组合 | Sharpe=1.248 (replay) |

## 扩展迭代

### v2: Mode-Conditional Reward + Dynamic Band Scaling

在 v1 基础上引入两个互补扩展：

#### 1. Mode-Conditional Reward Shaping

在 moderate reward 中根据当前 mode 动态调整奖励/惩罚，强化 mode 语义：

| Mode | Reward 调整 | 目的 |
|------|------------|------|
| flat_wait | `flat_wait_pos_penalty` × pos（非零仓位惩罚） | 强化空仓等待 |
| crash_rebuild | `crash_rebuild_bonus` × Δpos（加仓奖励） | 鼓励底部重建 |
| trend_hold | `trend_hold_alignment_multiplier` × alignment_bonus | 放大趋势对齐奖励 |
| defensive_hold | `defensive_hold_neutral_bonus`（中性稳定奖励） | 防止漂移 |

配置项（`RewardConfig`）：
- `mode_conditional_reward_enabled: bool`
- `flat_wait_pos_penalty: float`
- `crash_rebuild_bonus: float`
- `trend_hold_alignment_multiplier: float`
- `defensive_hold_neutral_bonus: float`

实现：`src/soxl_quant/policy/reward.py::_moderate_reward()`

#### 2. Dynamic Band Scaling

根据 realized_vol_10 动态缩放 mode band 宽度：

```
scale = 1.0 - (realized_vol - baseline) × sensitivity
scale = clamp(scale, min_scale, max_scale)
new_width = original_width × scale
```

- 高波动（vol > baseline）→ scale < 1.0 → 压缩 band → 降低风险暴露
- 低波动（vol < baseline）→ scale > 1.0 → 扩展 band → 增加参与机会

配置项（`ModeRouterConfig`）：
- `use_dynamic_bands: bool`
- `dynamic_band_vol_baseline: float`（默认 0.50）
- `dynamic_band_vol_sensitivity: float`（默认 0.30）
- `dynamic_band_min_scale: float`（默认 0.60）
- `dynamic_band_max_scale: float`（默认 1.20）

实现：`src/soxl_quant/policy/hierarchical_composer.py::_scale_band_by_vol()`

#### v2 实验包

- `experiments/candidate/risk_budget_usage_v2/`
- 目标：proxy Sharpe ≥ 0.95，test_2022 MaxDD ≤ 20%
- Baseline：`risk_budget_usage_v1`
- Changed layer：`reward` + `execution` (mode router)

### v3: Mode-Conditioned Actor Input

在 v2 基础上，将当前 mode 的 one-hot 向量附加到 Actor 的 observation 中：

```
observation (~50 dims) + mode_one_hot (4 dims) → GRU → MLP → action
```

实现要点：
- `ModelConfig.use_mode_conditioned_input: bool`
- `observation_dim_for_profile()` 自动 +4 维
- `SoxlState.interpret()` 在返回 obs 前附加 mode one-hot
- `backtest.py` 在调用 actor 前同样附加 mode one-hot
- Mode 编码顺序：`[trend_hold, defensive_hold, crash_rebuild, flat_wait]`

预期效果：
- 降低学习 mode-specific 策略的样本复杂度
- 改善 mode 过渡期的平滑性
- Actor 直接"知道"自己处于哪个 risk regime

#### v3 实验包

- `experiments/candidate/risk_budget_usage_v3/`
- 目标：proxy Sharpe ≥ 0.95，test_2022 MaxDD ≤ 18%
- Baseline：`risk_budget_usage_v2`
- Changed layer：`network` (observation input)

## 待挖掘方向

- [x] **Band 动态调整** — 已在 v2 实现
- [x] **Mode 条件 reward** — 已在 v2 实现
- [x] **Mode-conditioned input** — 已在 v3 实现
- [ ] **Mode 预测而非推断**: 让 Actor 直接输出 mode + usage，而非只输出 usage
- [ ] **Multi-mode 混合**: 允许多个 mode 的加权混合（soft routing）
- [ ] **Stress 专用 reward**: 在 stress 态下增大仓位惩罚系数
- [ ] **Supervised warmstart**: 用 replay composition 路径做 BC warmstart

## 文件组织

实验包采用**共置（colocation）**原则：每个实验的所有文件放在同一个目录下。

```
experiments/candidate/risk_budget_usage_v1/
├── notes.md          # 动机、假设、结果
├── expectations.md   # 预期与失败模式
├── profile.yaml      # 完整配置
└── card.md           # 运行时卡片（与实验共置）
```

runtime/model/knowledge.md 中的索引表直接链接到 experiments/ 下的 card.md，
不再在 runtime/model/card/ 中保留重复副本。

## 相关文件

- `src/soxl_quant/policy/two_speed_budget.py` — 核心组合逻辑
- `src/soxl_quant/policy/positioning.py` — budget_usage 映射
- `experiments/candidate/risk_budget_usage_v1/profile.yaml` — 完整配置
- `experiments/candidate/risk_budget_usage_v1/notes.md` — 实验记录
- `experiments/candidate/risk_budget_usage_v1/card.md` — 运行时卡片
