# MODEL_SELECTION Knowledge

> 最后更新: 2026-06-13
> 该方向负责 walk-forward retention gate、candidate aggregation、teacher/reference scoring 和下一轮训练目标路由。

## 当前结论

1. **没有现有候选满足当前 retention gate**：`retention_gap_aggregate_v1` 统一检查了 61 个候选/来源，要求完整 `wf_2020` / `wf_2022` / `wf_2023_2024`、最差 WF MaxDD `<20%`、YTD return `>=132.08%` SOXLBacktest reference 且 `>=213.14%` current-best alignment；full gate count 为 `0`。
2. **低回撤路径的问题是收益不足**：SOXLBacktest qlib risk-capped reference 和 QuantBench current ensemble 能接近或通过 WF MaxDD 约束，但 YTD 明显低于 SOXLBacktest/current-best return floors。
3. **高收益路径的问题是 WF 回撤或证据缺口**：现有高 YTD 模型缺完整 WF 证据，或完整 WF 证据下仍突破 20% MaxDD。
4. **SOXLBacktest qlib buy/sell path 只作为 reference baseline**：不能直接作为 retained model；raw teacher WF drawdown 过高，risk-capped variants 又牺牲收益。

## 下一步路由

- 下一条训练/选择分支应走 `train_state_conditioned_multi_objective_with_ytd_participation_and_drawdown_gate`。
- 下一轮 actor screen 必须用 official replay metrics 做 selection source，覆盖三段 WF 和 YTD safe replay end，并拒绝任何 WF MaxDD `>=20%` 或 YTD `<213.14%` 的 checkpoint。
- 主要证据卡片：`runtime/model_selection/card/retention_gap_aggregate_v1/retention_gap_aggregate_v1.ipynb`。

## 2026-06-13 Retention YTD Training Contract

- `retention_ytd_training_objective_handoff_v1` 把当前 retained-model 要求固化成下一轮训练/筛选合同：下一候选 profile 名称为 `state_conditioned_multi_objective_ytd_gate_v1`，仍不允许 promotion 或 actor retention。
- 硬门槛：candidate 必须有 official replay / dashboard replay 证据，覆盖 `wf_2020` / `wf_2022` / `wf_2023_2024` 和 YTD；最差 WF MaxDD `<20%`；YTD return `>=213.143462%`，且至少超过 SOXLBacktest qlib reference floor `132.077978%`。
- 训练/筛选目标：保留 `<=15%` warm-start drawdown margin 作为 checkpoint loss，加入 YTD trend participation floor，但只有在 risk-state coverage 与 drawdown interval behavior 同时改善时才允许提高 reentry/trend floor。
- stop rules：缺 official evidence、WF MaxDD `>=20%`、YTD 低于 current-best alignment、ordinary-vs-official gate flip、或 reentry floor 改善收益但恶化 drawdown 时，直接拒绝训练 handoff / 停止 floor sweep。
- 主要证据卡片：`runtime/model_selection/card/retention_ytd_training_objective_handoff_v1/retention_ytd_training_objective_handoff_v1.ipynb`。
