# EXECUTION Knowledge

> 最后更新: 2026-05-21
> 该方向负责执行层策略：仓位管理、交易频率控制、mode routing 和 overlay。

## 执行策略索引

| Profile | 类别 | Return | Sharpe | MaxDD | AvgPos | 2022 MaxDD | 状态 | 卡片 |
|---------|------|--------|--------|-------|--------|------------|------|------|
| balanced_floor40 | cap | 614 | 0.84 | 65.68 | 0.64 | — | 🟡 Good | [card](cards/balanced_floor40/card.md) |
| balanced_floor40_sym_bear30 | cap | 443 | 0.79 | 61.05 | 0.56 | — | 🟡 Moderate | [card](cards/balanced_floor40_sym_bear30/card.md) |
| balanced_floor40_sym_cap25 | cap | 482 | 0.87 | 55.88 | 0.46 | — | 🟡 Good | [card](cards/balanced_floor40_sym_cap25/card.md) |
| balanced_floor40_sym_crisis30 | cap | 375 | 0.79 | 59.87 | 0.47 | — | 🟡 Moderate | [card](cards/balanced_floor40_sym_crisis30/card.md) |
| balanced_floor40_sym_crisis30_v2 | cap | 357 | 0.84 | 51.62 | 0.40 | — | 🟡 Good | [card](cards/balanced_floor40_sym_crisis30_v2/card.md) |
| balanced_floor40_sym_expectancy | cap | 162 | 0.57 | 65.05 | 0.53 | — | 🟡 Moderate | [card](cards/balanced_floor40_sym_expectancy/card.md) |
| balanced_floor40_sym_expectancy | cap | 162 | 0.57 | 65.05 | 0.53 | — | 🟡 Moderate | [card](cards/balanced_floor40_sym_expectancy/card.md) |
| balanced_floor40_sym | cap | 455 | 0.76 | 74.39 | 0.68 | — | 🟡 Moderate | [card](cards/balanced_floor40_sym/card.md) |
| balanced_floor40_sym_slow_bear | cap | 515 | 0.91 | 46.45 | 0.45 | — | 🟡 Good | [card](cards/balanced_floor40_sym_slow_bear/card.md) |
| balanced_floor40_sym | cap | 444 | 0.79 | 61.55 | 0.55 | — | 🟡 Moderate | [card](cards/balanced_floor40_sym/card.md) |
| balanced_floor40_sym_tconv | cap | 542 | 0.85 | 61.58 | 0.55 | — | 🟡 Good | [card](cards/balanced_floor40_sym_tconv/card.md) |
| balanced_floor40_sym_tight_crash | cap | 397 | 0.80 | 56.53 | 0.48 | — | 🟡 Moderate | [card](cards/balanced_floor40_sym_tight_crash/card.md) |
| balanced_floor40_sym_v2 | cap | 385 | 0.85 | 49.01 | 0.41 | — | 🟡 Good | [card](cards/balanced_floor40_sym_v2/card.md) |
| balanced_floor40_sym_v2 | cap | 447 | 0.78 | 64.77 | 0.60 | — | 🟡 Moderate | [card](cards/balanced_floor40_sym_v2/card.md) |
| balanced_floor40_sym_velocity_rules | cap | 386 | 0.80 | 62.34 | 0.46 | — | 🟡 Good | [card](cards/balanced_floor40_sym_velocity_rules/card.md) |
| balanced_floor40_sym_velocity | cap | 591 | 0.89 | 60.14 | 0.52 | — | 🟡 Good | [card](cards/balanced_floor40_sym_velocity/card.md) |
| balanced_rule_residual_learned_mode_v1 | mode | 507 | 0.87 | 50.01 | 0.47 | — | 🟡 Good | [card](cards/balanced_rule_residual_learned_mode_v1/card.md) |
| balanced_rule_residual_mode_v1 | mode | 886 | 1.09 | 42.32 | 0.45 | — | 🟡 Good | [card](cards/balanced_rule_residual_mode_v1/card.md) |
| balanced_rule_residual_mode_v2 | mode | 694 | 1.01 | 50.40 | 0.45 | — | 🟡 Good | [card](cards/balanced_rule_residual_mode_v2/card.md) |
| selective_release_overlay_v1 | overlay | 1415 | 1.11 | 54.30 | 0.52 | 40.64 | 🟢 Counterfactual | [card](card/selective_release_overlay_v1/selective_release_overlay_review.md) |

## 核心结论

1. **Position cap 是最有效的风控工具**。
2. **Decision frequency = 3 是 sweet spot**。
3. **Trade lock（3-day cooldown）有效**。
4. **Regime-conditional overlay 有方向性**。
5. **Generic overlay retuning 是重复工作**。
6. **Generic `fast_reentry` 不是答案**：它能改善恢复期收益，但在 proxy Sharpe / MaxDD 上被 `dwell_hold_v2` 压制。
7. **当前瓶颈是低换手的 selective release / re-entry classification**：`regime_route_v1` 证明更聪明的放行有价值，但 churn 明显升高。
8. **固定 v22 actor 的 selective release gate 已证明方向正确**：`selective_release_recovery_v1` 把 `test_2023_2024` 从 `57.12% / 0.719` 提到 `82.10% / 0.877`，`test_2022` 完全不变，`test_proxy` 从 `1132.19% / 1.043 / 56.48%` 提到 `1415.10% / 1.105 / 54.30%`。
9. **walk-forward actor 验证未通过 runtime promotion**：`selective_release_walkforward_v1` 显示三种既有 release gate 在 `wf_2023_2024` 有 `6-18` 个 eligible days，但 `changed_days=0`，说明当前执行映射吸收了 boost；不要直接治理化该 overlay。
10. **简单 order-generation 修复仍然不够**：`selective_release_recovery_threshold_bypass_v1` 与 `selective_release_recovery_target_lift_v1` 在 `wf_2023_2024` 都有 `18` 个 eligible days，但仍然 `changed_days=0` / `absorbed_days=18`。
11. **逐日 trace 已定位主吸收层**：`selective_release_walkforward_execution_trace.csv` 显示 threshold-bypass actionable eligible days 的 primary blocker 是 `trade_lock=15/16`，target-lift 是 `trade_lock=12/16`、`min_rebalance=3/16`、`ordered=1/16`。主因不是 release signal、不只是 min_rebalance，也不是 PPO 需要直接 retrain；下一条有效规则应只测试 release-gated trade-lock bypass / shortening。
12. **full trade-lock bypass 不是可推广解**：`selective_release_recovery_trade_lock_bypass_v1` 在 `wf_2023_2024` 把 `changed_days` 拉到 `183`、trades 从 `22` 到 `32`，但 return 从 `-20.21%` 降到 `-21.43%`、MaxDD 从 `38.54%` 升到 `40.28%`；`wf_2022` 保持 unchanged。结论是 trade lock 的确挡住订单，但完全 bypass 只制造 churn。
13. **trade-lock shortening 有诊断价值但仍未过 promotion gate**：`selective_release_recovery_trade_lock_short_v1` 用 `trade_lock_cap=1` 后在 `wf_2023_2024` 把 return 从 `-20.21%` 提到 `-17.87%`、trades 只从 `22` 到 `24`，但 MaxDD 从 `38.54%` 升到 `40.65%`；`wf_2022` 仍 unchanged。下一步必须给 release-gated shortening 加 drawdown / exposure-gap guard，而不是直接推广。
14. **静态 drawdown/exposure-gap entry guard 仍不够**：`selective_release_recovery_trade_lock_guarded_short_v1` 把 target boost、rebalance bypass 和 `trade_lock_cap=1` 都限制在 `current_drawdown<=0.03` 且 exposure gap `>=0.10` 时，但 `wf_2023_2024` 只从 `-20.21%` 到 `-19.63%`，MaxDD 却从 `38.54%` 升到 `42.91%`。不要继续调这个静态阈值；下一步先做 drawdown-path attribution 或引入明确 post-release de-risk / target cap 机制。
15. **drawdown attribution 指向 post-release 风控，而不是 entry guard**：`selective_release_walkforward_drawdown_attribution.csv` 显示 guarded-short 的唯一 execution-ready release order 是 `2024-03-28`，variant MaxDD trough 是 `2024-08-07`，release-to-trough 平均仓位差 `+0.0204`、最大仓位差 `+0.3140`、trough account delta `-0.48%`。后续还有 gate-eligible orders 在 `2024-05-13` 和 `2024-07-09`。下一步应测试 release-state target cap 或 post-release de-risk，而不是继续调入口阈值。
16. **release-state target cap 是首个通过 walk-forward 双指标的 selective-release 变体**：`selective_release_recovery_post_release_cap_v1` 在 `wf_2023_2024` 把 return 从 `-20.21%` 提到 `-11.37%`，MaxDD 从 `38.54%` 降到 `27.66%`，同时 `wf_2022` 完全 unchanged；固定 full-window actor grid 上该变体为 no-op，不破坏原 counterfactual。下一步是 promotion review / runtime governance packaging，而不是继续调 entry guard。
17. **post-release cap 已完成 promotion-review packaging，但不能自动进生产**：`runtime/execution/card/selective_release_post_release_cap_v1/promotion_review.json` 四项检查通过（`wf_2022` unchanged、`wf_2023_2024` return `+8.8381pp`、MaxDD `-10.8815pp`、fixed actor no-op）。后续 profile-bound runtime dry-run 已补齐运行时接线，但 `runtime_auto_select_allowed=false` 仍正确，因为 profile flag 与 base v22 model promotion gate 都未批准。
18. **post-release cap 已完成 profile-bound runtime dry-run，但仍不能 auto-select**：`balanced_rule_residual_v22` 现在绑定 `execution_overlay.name=selective_release_recovery_post_release_cap_v1`，runtime 会自动补 `release_signals_v1`、输出 overlay diagnostics，并在非 dry-run 下单后推进 overlay state。固定 profile dry-run 成功跑到 `2026-05-14` 信号，overlay 字段写出，但 risk gate 仍因 `walk-forward promotion gate not evaluated` 阻断；`promotion_review.json` 已更新为 `runtime_profile_bound=true`、`runtime_auto_select_allowed=false`。

## 待验证方向

- [x] 显式 selective release/re-entry overlay（优先 `dwell_hold_v2` / `regime_route_v1` 风格，不再泛化搜索 `fast_reentry`）— fixed v22 actor counterfactual retained as `selective_release_recovery_v1`
- [x] 对 `selective_release_recovery_v1` 做 walk-forward actor validation，再决定是否进入 runtime overlay / governed execution profile — blocked for promotion because WF actor orders do not change
- [x] 对 `wf_2023_2024` eligible release days 做 target-lift / rebalance-threshold bypass 诊断 — 两者仍被吸收
- [x] 记录 eligible day 逐日 `current_pos`、overlay 前后 target、`min_rebalance`、trade lock、buy cooldown、cap state、lot rounding — blocker narrowed to trade lock
- [x] 测试 release-gated full trade-lock bypass — order generation works but recovery metrics worsen
- [x] 测试比 full bypass 更窄的 trade-lock shortening / buy-cooldown-aware release rule，并继续要求 `wf_2022` unchanged — cap variant improves return but fails MaxDD gate
- [x] 测试 release-gated shortening 的 drawdown / exposure-gap guard — static entry guard still fails MaxDD
- [x] 对 guarded shortening 的唯一 allowed release path 做 drawdown attribution，再决定是否测试 post-release de-risk / target cap — attribution points to post-release de-risk / target cap
- [x] 测试 release-state target cap 或 post-release de-risk，要求 `wf_2022` unchanged 且 `wf_2023_2024` MaxDD 回到 baseline 附近 — post-release cap improves both return and MaxDD
- [x] 对 post-release cap 做 promotion review / runtime governance packaging，不再重复 entry-threshold tuning — packaged as review candidate, runtime auto-select remains blocked
- [x] 如果继续推进该候选，做 profile-bound runtime overlay / dry-run replay 设计；否则选择下一条窄模型优化实验 — runtime path exists, auto-select still blocked by profile flag and model gate
- [ ] 决定是否为 v22/post-release-cap runtime use 做 explicit model-promotion sidecar governance；否则转入下一条窄模型优化实验
- [ ] 动态 decision frequency
- [ ] 仓位上限与 VIX term structure 联动
- [ ] budget-usage actor/runtime mismatch 诊断

## 2026-05-14: Selective Release Overlay Counterfactual

Question: can retained release factors improve recovery if they are gated in execution instead of injected directly into PPO state?

Answer: yes on the fixed `balanced_rule_residual_v22` actor.

| Variant | test_proxy | test_2022 | test_2023_2024 | Decision |
|---|---:|---:|---:|---|
| baseline | `1132.19% / 1.043 / 56.48% / avg_pos 0.5226` | `-30.97% / -0.764 / 40.64% / avg_pos 0.3304` | `57.12% / 0.719 / 34.87% / avg_pos 0.5402` | control |
| `selective_release_recovery_v1` | `1415.10% / 1.105 / 54.30% / avg_pos 0.5230` | unchanged | `82.10% / 0.877 / 36.95% / avg_pos 0.5510` | retain counterfactual |

Mechanism:

- Gate requires `RB_COOLDOWN_PROXY` plus either `TRANS_VIX_DIVERGENCE` or `CICC_RETURN_VOLUME_SYNC`.
- The best retained config uses `boost=0.08`, `max_target=0.65`, `max_regime_risk=0.60`, `cooldown_threshold=0.20`, `vix_threshold=0.05`, `volume_sync_threshold=0.70`.
- It does not fire in `2022`, which is why bear-market exposure and drawdown remain unchanged.

Decision:

- Do not retrain another release-pack PPO branch yet.
- Next valid loop is walk-forward actor validation for the same overlay.
- If it survives, encode it as a governed execution overlay/profile rather than expanding the model feature pack.

## 2026-05-15: Selective Release Walk-Forward Actor Validation

Question: does the retained release overlay survive when applied to saved `balanced_rule_residual_v22` walk-forward actors?

Answer: no, not as an executable runtime overlay yet.

| Variant | wf_2020 | wf_2022 | wf_2023_2024 | Decision |
|---|---:|---:|---:|---|
| baseline | `113.17% / 2.015 / 19.47%` | `-34.11% / -0.926 / 38.27%` | `-20.21% / -0.277 / 38.54%` | control |
| `selective_release_recovery_v1` | unchanged, `0` eligible days | unchanged, `0` eligible days | unchanged, `18` eligible days but `0` changed days | block runtime promotion |

Mechanism:

- The release signals do identify 2024 recovery days, so the factor side is not the primary failure.
- The overlay boost is absorbed by actor target / execution mapping, because eligible days do not translate into changed orders.

Decision:

- Keep `selective_release_overlay_v1` as fixed full-window counterfactual evidence only.
- Do not promote it to a governed runtime profile until an explicit target-lift / rebalance-bypass test changes orders while preserving `wf_2022`.

## 2026-05-20: Selective Release Order-Generation Diagnostic

Question: are the walk-forward release days absorbed only because of the visible rebalance threshold or too-small target lift?

Answer: no. The gate still fires in `wf_2023_2024`, but both minimal execution fixes remain no-ops.

| Variant | wf_2022 | wf_2023_2024 | Decision |
|---|---:|---:|---|
| `selective_release_recovery_threshold_bypass_v1` | unchanged, `0` eligible days | `18` eligible days, `0` changed days, `18` absorbed days | blocked |
| `selective_release_recovery_target_lift_v1` | unchanged, `0` eligible days | `18` eligible days, `0` changed days, `18` absorbed days | blocked |

Mechanism:

- Threshold bypass sets gate-ready rebalance threshold to `0.0`.
- Target lift forces the overlay target to at least `current_pos + 0.08`, capped by the retained `max_target`.
- Both variants reuse the retained `selective_release_recovery_v1` gate; this is an execution mapping diagnostic, not another factor or PPO branch.

Decision:

- Do not promote the overlay and do not retrain from this result.
- The blocker is deeper than the visible threshold or a simple target lift. Instrument per-day execution state on the eligible `wf_2023_2024` dates before testing another rule.

## 2026-05-20: Eligible-Day Execution Trace

Question: when the retained release gate fires on walk-forward recovery days, which execution layer actually absorbs the order?

Answer: primarily trade lock.

| Variant | Trace coverage | Primary blockers | Readout |
|---|---:|---|---|
| `selective_release_recovery_threshold_bypass_v1` | `16 / 18` eligible days | `trade_lock=15`, `ordered=1` | lowering min rebalance to `0.0` is still absorbed because trade lock resets target to current position |
| `selective_release_recovery_target_lift_v1` | `16 / 18` eligible days | `trade_lock=12`, `min_rebalance=3`, `ordered=1` | target lift is usually blocked upstream by trade lock; the non-trade-lock no-order cases are near current/target already |

Artifacts:

- `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_execution_trace.csv`
- `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_execution_trace.json`
- `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_review.md`

Decision:

- Do not broaden the gate or retrain.
- Next loop should test only a release-gated trade-lock bypass or shortening rule, preserving the same retained release gate and `wf_2022` no-eligible defense.

## 2026-05-20: Release-Gated Trade-Lock Bypass

Question: if trade lock is the upstream absorber, does bypassing it under the retained release gate produce a promotable walk-forward change?

Answer: it produces orders, but it is not promotable.

| Variant | wf_2022 | wf_2023_2024 | Decision |
|---|---:|---:|---|
| `selective_release_recovery_trade_lock_bypass_v1` | unchanged, `0` eligible days | `18` eligible days, `183` changed days, trades `32` vs `22`, return `-21.43%` vs `-20.21%`, MaxDD `40.28%` vs `38.54%` | drop |

Mechanism:

- Reuses the retained `selective_release_recovery_v1` gate.
- Adds `rebalance_threshold_override=0.0` and `trade_lock_override=0` only when that gate is ready.
- This confirms the previous trace: trade lock was the main order-generation absorber.

Decision:

- Do not promote the bypass and do not retrain from it.
- The next rule must be narrower: shorten trade lock or bypass only when current exposure is below the release target, and account for buy cooldown.

## 2026-05-20: Release-Gated Trade-Lock Shortening

Question: does capping trade lock at one day under the retained release gate produce a cleaner, promotable order-generation change than full bypass?

Answer: it is cleaner and improves recovery return, but still fails the drawdown gate.

| Variant | wf_2022 | wf_2023_2024 | Decision |
|---|---:|---:|---|
| `selective_release_recovery_trade_lock_short_v1` | unchanged, `0` eligible days | `18` eligible days, `175` changed days, trades `24` vs `22`, return `-17.87%` vs `-20.21%`, Sharpe `-0.229` vs `-0.277`, MaxDD `40.65%` vs `38.54%`, avg_pos `0.4620` vs `0.4764` | diagnostic keep, no promotion |

Mechanism:

- Uses the retained `selective_release_recovery_v1` gate.
- Adds `rebalance_threshold_override=0.0` and `trade_lock_cap=1`.
- `trade_lock_cap` only shortens lock days; it does not raise an unlocked day from `0` to `1`.

Trace:

- `wf_2023_2024` eligible-day trace for the short variant: `16 / 18` rows, primary blockers `trade_lock=12`, `ordered=3`, `buy_cooldown=1`.
- Compared with full bypass (`ordered=11`, trades `32`), shortening reduces churn and improves return, but the remaining orders still widen drawdown.

Decision:

- Do not promote the short variant as a runtime overlay.
- Next valid rule should keep `trade_lock_cap=1` but require a drawdown / exposure-gap guard so release orders are allowed only when they reduce a proven under-exposure problem without raising recovery-window MaxDD.

## 2026-05-20: Guarded Release Trade-Lock Shortening

Question: can a static drawdown / exposure-gap guard make release-gated trade-lock shortening promotable?

Answer: no. It reduces the number of explicit execution-ready eligible rows, but still widens recovery-window drawdown.

| Variant | wf_2022 | wf_2023_2024 | Decision |
|---|---:|---:|---|
| `selective_release_recovery_trade_lock_guarded_short_v1` | unchanged, `0` eligible days | `18` eligible days, `1 / 16` traced rows execution-ready, changed_days `185`, trades `21` vs `22`, return `-19.63%` vs `-20.21%`, Sharpe `-0.233` vs `-0.277`, MaxDD `42.91%` vs `38.54%`, avg_pos `0.4833` vs `0.4764` | drop |

Mechanism:

- Uses the retained `selective_release_recovery_v1` gate.
- Adds `max_current_drawdown=0.03` and `min_release_exposure_gap=0.10`.
- The guard blocks all three release-side effects when it fails: target boost, `rebalance_threshold_override=0.0`, and `trade_lock_cap=1`.
- Decision trace now records `overlay_execution_ready` and `overlay_exposure_gap`.

Trace:

- The guarded variant has only `1 / 16` execution-ready eligible rows in `wf_2023_2024`.
- Despite the sparse guard, path dependency from the allowed early release still increases MaxDD by `+4.37pp`.

Decision:

- Do not promote the guarded short variant.
- Do not spend the next loop on another static guard-threshold tweak.
- Next valid work is path-level drawdown attribution for the allowed release order, then a distinct mechanism such as explicit post-release de-risking or a release target cap if the attribution supports it.

## 2026-05-20: Guarded Release Drawdown Attribution

Question: did the sparse guarded release path actually sit before the drawdown expansion, and what should the next mechanism target?

Answer: yes; the drawdown expansion is path-dependent after release activation, so the next mechanism should manage exposure after release, not retune the entry gate.

| Variant | wf_2023_2024 delta MaxDD | Release orders | Variant peak -> trough | Exposure readout | Decision |
|---|---:|---|---|---|---|
| `selective_release_recovery_trade_lock_bypass_v1` | `+1.74pp` | `2024-03-28` through `2024-10-09` | `2024-06-18 -> 2024-12-10` | avg release-to-trough pos delta `+0.0636` | drop full bypass |
| `selective_release_recovery_trade_lock_short_v1` | `+2.11pp` | `2024-03-28;2024-05-13;2024-05-15` | `2024-07-10 -> 2024-12-19` | avg release-to-trough pos delta `+0.0762` | keep diagnostic only |
| `selective_release_recovery_trade_lock_guarded_short_v1` | `+4.37pp` | execution-ready `2024-03-28`; gate-eligible ordered `2024-05-13;2024-07-09` too | `2024-06-18 -> 2024-08-07` | avg release-to-trough pos delta `+0.0204`, max `+0.3140`, trough account delta `-0.48%` | drop static guard |

Artifacts:

- `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_drawdown_attribution.csv`
- `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_drawdown_attribution.json`
- `runtime/execution/card/selective_release_walkforward_v1/selective_release_walkforward_review.md`

Decision:

- Do not tune `max_current_drawdown` / `min_release_exposure_gap` again as the next step.
- The next valid loop is a release-state target cap or explicit post-release de-risk rule: after release activation, reduce/cap exposure if the path starts drawing down, while preserving the retained gate's `wf_2022` unchanged defense.

## 2026-05-20: Release-State Post-Release Cap

Question: can post-release state management keep the retained release gate's recovery benefit while preventing the drawdown path identified by attribution?

Answer: yes on the saved walk-forward actor grid; it is the first selective-release walk-forward variant that improves return and lowers MaxDD.

| Variant | wf_2022 | wf_2023_2024 | Fixed full-window actor | Decision |
|---|---:|---:|---:|---|
| `selective_release_recovery_post_release_cap_v1` | unchanged, `0` eligible days | return `-11.37%` vs `-20.21%`, Sharpe `-0.133` vs `-0.277`, MaxDD `27.66%` vs `38.54%`, trades `27` vs `22` | no-op on proxy/2020/2022/2023-2024 | promotion-review candidate |

Mechanism:

- Reuses the retained release gate and guarded `trade_lock_cap=1` setup.
- Starts a 90-day release state after an execution-ready release order.
- If release-state drawdown exceeds `4%`, caps target exposure at `0.55`.
- De-risk relaxations are sell-only: `post_release_rebalance_threshold_override=0.0` and `post_release_trade_lock_override=0` only help reduce exposure.

Attribution:

- `wf_2023_2024` first execution-ready release order remains `2024-03-28`.
- Eligible ordered dates are `2024-03-28` and `2024-05-14`.
- Variant peak/trough remains `2024-06-18 -> 2024-08-07`, but delta MaxDD is `-10.88pp` versus baseline.
- Release-to-trough average position delta versus baseline is `-0.0373`, and account value at trough is `+11.58%` versus baseline.

Decision:

- Keep as a promotion-review candidate, not a completed runtime promotion.
- Next work should be promotion governance / runtime packaging checks for the post-release cap and trace semantics.
- Do not spend the next loop on another static entry guard, release threshold, or PPO retrain.

## 2026-05-21: Post-Release Cap Promotion Governance Packaging

Question: can the post-release cap be safely packaged for promotion review without accidentally becoming scheduler-selected production behavior?

Answer: yes. The review candidate is now packaged and explicitly blocked from runtime auto-selection.

Artifacts:

- `src/soxl_quant/policy/execution_overlays.py`
- `runtime/execution/card/selective_release_post_release_cap_v1/promotion_review.md`
- `runtime/execution/card/selective_release_post_release_cap_v1/promotion_review.json`

Checks:

| Check | Result |
|---|---:|
| `wf_2022_unchanged` | pass |
| `wf_2023_2024_return_improved` | pass, delta `+8.8381pp` |
| `wf_2023_2024_maxdd_improved` | pass, delta `-10.8815pp` |
| `fixed_actor_noop` | pass |
| `runtime_profile_bound` | pass after the runtime dry-run loop |
| `runtime_auto_select_allowed` | `false` |

Harness governance:

- The selective-release overlay configs now live in a shared registry instead of duplicated script-local lists.
- `tests/test_profile_registry.py`, `tests/test_execution_overlay_registry.py`, and `tests/test_selective_release_promotion_review.py` are part of `scripts/check_harness.py`.
- The profile registry path now matches the repo canonical `runtime/harness/card/registry/profile.yaml`.

Decision:

- Do not let the scheduler auto-select this overlay yet.
- The next runtime packaging step was the profile-bound dry-run loop below; auto-selection still requires explicit governance approval.

## 2026-05-21: Profile-Bound Runtime Overlay Dry-Run

Question: can the promotion-review candidate be wired into the actual runtime profile and paper runner while keeping auto-selection blocked?

Answer: yes. `balanced_rule_residual_v22` now has `execution_overlay.enabled=true`, `name=selective_release_recovery_post_release_cap_v1`, `state_path=runtime/execution/state/selective_release_post_release_cap_v1.json`, and `auto_select_allowed=false`.

Runtime behavior:

- `runtime_feature_packs()` adds `release_signals_v1` for profile-bound selective-release overlays.
- `SignalIntent` now records `target_before_overlay`, `execution_overlay`, `overlay_gate_ready`, `overlay_execution_ready`, `post_release_state_days`, and `post_release_drawdown`.
- `apply_runtime_execution_overlay()` applies the retained release boost and post-release cap from the shared overlay registry.
- Non-dry-run paper execution advances and saves overlay state only after order submission/audit.
- `promotion_review.json` now records `runtime_profile_bound=true`, while `runtime_auto_select_allowed=false` remains correct because the profile flag is false and the base actor sidecar says `walk-forward promotion gate not evaluated`.

Dry-run evidence:

- Command: `PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --profile balanced_rule_residual_v22 --skip-data-update --dry-run --no-notify`
- Output signal date: `2026-05-14`
- Signal overlay: `selective_release_recovery_post_release_cap_v1`
- Overlay readiness: `overlay_gate_ready=false`, `overlay_execution_ready=false`, `post_release_state_days=0`
- Risk gate: `approved=false`, blocked by `walk-forward promotion gate not evaluated`

Decision:

- The profile-bound runtime overlay / dry-run replay step is complete.
- Do not enable scheduler auto-selection from this evidence alone.
- Next valid step is explicit model-promotion sidecar governance for v22/post-release-cap runtime use, or a separate narrow model-optimization experiment if approval is deferred.
