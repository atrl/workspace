# MODEL Knowledge

_Auto-generated from memory/ integration on 2026-04-24._
_Total entries: 54._

## 2026-04-10 — 2026-04-10 (`dropped`)
Legacy memory note normalized into the repo-wide schema. 500轮 × 10 episodes: **301.1% / Sharpe 0.66 / MaxDD 73.9%** 比200轮(437%/0.85/40.2%)更差 → **过拟合** Git: 8a8fc45 结论: **200轮是最佳 checkpoint** 1. 7处 monkey-patch (gym→gymnasium兼容) 2. 没用 qlib 官

## 2026-04-11 — 2026-04-11 (`dropped`)
Legacy memory note normalized into the repo-wide schema. Reviewed `openai/symphony` as a workflow reference rather than something to mirror literally in this repo Pulled the useful parts into `AGENTS.md`: repo-owned workflow contract proof-

## 2026-04-12 — 2026-04-12 (`dropped`)
Legacy memory note normalized into the repo-wide schema. Found that `configs/soxl/*.yaml` already exposed `position_cap.start_drawdown/full_drawdown/min_cap`, but `src/soxl_quant/rl.py` still hard-coded the cap curve Fixed the cap config to

## 2026-04-13 — 2026-04-13 Generalization Backtest Report (`dropped`)
Legacy memory note normalized into the repo-wide schema. Test whether the SOXL-trained PPO actor (`soxl_balanced_actor.pt`) generalizes to other 3x leveraged ETFs without retraining. | ETF | Underlying | Sector | |-----|-----------|--------

## 2026-04-14 — SOXL 模型优化路线图（基于泛化测试结果） (`reference`)
Legacy memory note normalized into the repo-wide schema. > 测试日期：2026-04-14 > 测试模型：soxl_balanced_actor.pt（PPO，零样本） > 泛化标的：SOXL、TQQQ、UPRO、SPXL、TECL（5只3倍杠杆ETF） | ETF | 策略收益 | B&H 收益 | 超额收益 | 策略回撤 | B&H 回撤 | 交易次数 | |-----|----------|----------|

## 2026-04-15 — MoE + Signal Diversification + Reward Optimization (`superseded`)
Legacy memory note normalized into the repo-wide schema. Date: 2026-04-15 Execute the approved "Approach A – Surgical Upgrade" plan to: 1. Upgrade `RegimeMoENet` with noisy Top-1 sparse routing, load-balancing loss, and a shared expert. 2.

## 2026-04-15 — VIX Ablation Experiment Results (`superseded`)
Legacy memory note normalized into the repo-wide schema. Date: 2026-04-15 Seed: 20260411 | Profile: balanced | Iters: 20 | Architecture: MoE **Baseline**: `balanced` + `--moe` (VIX features fully enabled) **Ablation**: `balanced` + `--moe`

## 2026-04-16 — 2026-04-16: A+B Breakthrough – High Threshold + Trade Lock + 80 Iters (`dropped`)
Legacy memory note normalized into the repo-wide schema. The RL model was losing to the rule-based strategy primarily due to **friction from excessive micro-trades** (~1000+ vs 45). Two levers were identified: **A**: Drastically raise `reba

## 2026-04-16 — 2026-04-16: Backtest-Inspired RL Optimizations (`superseded`)
Legacy memory note normalized into the repo-wide schema. Port 5 rule-based `soxl_backtest` mechanics into the RL reward/risk layer, increase high-vol T-trading on SOXL, and ensure zero-shot generalization beats Buy & Hold on other 3x ETFs.

## 2026-04-17 — 2026-04-17: `balanced_regime_gate` failed as a mainline candidate (`dropped`)
Legacy memory note normalized into the repo-wide schema. New profile: `configs/soxl/balanced_regime_gate.yaml` Intent: keep canonical `balanced` as the style anchor add staged checkpoint scoring penalize validation stress-window average pos

## 2026-04-17 — 2026-04-17: `balanced_regime_gate_v2` also failed (`dropped`)
Legacy memory note normalized into the repo-wide schema. New profile: `configs/soxl/balanced_regime_gate_v2.yaml` Intent: keep the moderate cap routing from `balanced_regime_gate` reduce trend-side encouragement directly: lower `activity_bo

## 2026-04-17 — 2026-04-17: `balanced_floor40_sym_bear30` retrain (`reference`)
Legacy memory note normalized into the repo-wide schema. `qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_floor40_sym_bear30 --skip-data-refresh --seed 20260411` Added `configs/soxl/balanced_floor40_sym_bear30.yaml`. Regis

## 2026-04-17 — 2026-04-17: Post-train backtest review skill (`active`)
Legacy memory note normalized into the repo-wide schema. Added repo-local skill `.codex/skills/post-train-backtest-review/`. The skill standardizes how to compare a new training result against a prior backtest using: `artifacts/models/*_sum

## 2026-04-17 — 2026-04-17: RL beats rule-based SOXL return (698.0%) (`superseded`)
Legacy memory note normalized into the repo-wide schema. The ~130pp gap between RL symmetric A+B (598.5%) and rule-based soxl_backtest (688.6%) was caused by three missing pieces: 1. Accidentally reverted asymmetric thresholds (buy 0.02 / s

## 2026-04-18 — 2026-04-18 checkpoint review and overlay loop (`reference`)
Staged checkpoint review proved the best validation checkpoint appears at iter 40 rather than iter 80, but the selected actor still failed fixed-split 2022 gates. Fixed-actor position-cap sweeps improved 2022 avg_pos only marginally and never passed the full gate. A follow-up single-layer retrain that raised `stress_avg_pos_penalty` from `10` to `25` also failed, with `test_2022 maxDD` still at `46.28%` and `avg_pos` at `0.398`. A reward-only `balanced_reward_guard` retrain also failed, worsening `test_2022` to `47.36% maxDD / 0.412 avg_pos`, so the next loop should stop tuning selector penalties and reward coefficients and instead change the stress set or episode sampling itself.

## 2026-04-18 — 2026-04-18 Experiment Directory Migration (`dropped`)
The flat `configs/soxl/*.yaml` layout still made profile intent too opaque even after adding a registry. The next step was to start colocating profile config with experiment-owned notes and expectations. Added `experiments/README.md`. Added

## 2026-04-18 — 2026-04-18: harness cleanup, indexed memory retrieval, and soft artifact prune (`kept`)
Replaced the broken pytest expectation with a repo-owned check_harness entrypoint, normalized memory into front-matter plus index.json, deleted low-value scripts and superseded actor checkpoints, and upgraded the static dashboard with evidence navigation and persisted state.

## 2026-04-18 — 2026-04-18 phase-balanced training failure (`dropped`)
A training-procedure experiment widened episode coverage with overlapping windows plus phase-balanced oversampling for stress, rebound, and bull-trend regimes. The broader sampling materially changed training (`36` episodes, best checkpoint at `iter 20`), but it did not repair post-2022 excess. `2025` still lagged buy-and-hold by `-10.88%`, `2023-2024` degraded on both return and drawdown, and `2022` broke badly (`47.50%` maxDD, `0.364` avg_pos). Keep the harness support, but drop the branch and do not assume broader bull sampling alone fixes the style gap.

## 2026-04-18 — 2026-04-18 Profile Governance And Retention Reset (`dropped`)
The harness had reached the point where `rl.py`, `cli.py`, and the profile set were too entangled to answer basic questions reliably: which profile is canonical which branches are just ablations which actor checkpoints should still exist wh

## 2026-04-18 — 2026-04-18 stress-release delta feature trial (`dropped`)
Legacy memory note normalized into the repo-wide schema. Profile: `balanced_stress_release_delta` Command: `qlib_venv/bin/python soxl_qlib_strategy_v2.py --profiles balanced_stress_release_delta --skip-data-refresh --seed 20260411` Feature

## 2026-04-18 — 2026-04-18 structural model loop (`dropped`)
A structural retrain with longer observation context, custom action levels, hard regime routing, and regime-stratified stress sampling improved stress exposure versus the failed reward-only branch, but it still failed promotion. `test_2022 avg_pos` passed at `0.338`, yet `test_2022 maxDD` remained too high at `39.86%` and `test_proxy Sharpe` fell to `0.739`. Keep the new harness support for model/sampling profiles, but drop `balanced_structural_v1` as a production candidate and ablate one structural lever at a time if model-side work continues.

## 2026-04-19 — 2026-04-19 balanced residual-over-rule full run (`kept`)
Full `balanced_rule_residual_v1` training completed and selected the `@60` checkpoint over `@80`. The branch materially improved trade-quality versus the old negative-Kelly diagnosis, including positive `micro_<5%`, `small_5_15%`, and `mid_3_10d` buckets, while keeping `2022` drawdown/avg-pos controlled and lifting proxy Sharpe to `0.96`. It is strong enough to keep, but still needs walk-forward evidence before any promotion claim.

## 2026-04-19 — 2026-04-19 dashboard auto latest fix (`kept`)
Fixed the dashboard so training now auto-exports a review CSV, auto-rebuilds the static HTML, defaults to the latest post-train view instead of an older best-Sharpe or browser-saved selection, preserves explicit post-train trade actions, and merges OHLC/indicator fields back into post-train rows. The generated dashboard now resolves `latest_version` to `balanced_rule_residual_v1`.

## 2026-04-19 — Internal RL Facade Shrink (`reference`)
Split artifact persistence and walk-forward workers out of app/train.py and updated repo-internal tests/imports so internal code no longer depends on the rl compatibility facade.

## 2026-04-20 — SOXL Backtest vs SOXLQuant 深度对比与奖励函数优化分析 (`kept`)
全面对比 soxl_backtest（规则驱动）与 src/soxl_quant（RL驱动）的架构、信号、风控和绩效。 深入分析对数奖励 $R_t = \log(W_t/W_{t-1})$ 的理论优劣，指出其已实现在代码中但不足以单独使用。 推荐最佳模型的下一步优化：启用 Kelly-alpha 奖励、regime-conditioned scale、以及将 soxl_backtest 的 panic index 引入 observation。

## 2026-04-20 — 2026-04-20: Seed Sensitivity Clarified (v19) (`kept`)
Legacy memory note normalized into the repo-wide schema. **CLI Backtest**: 635.1% return, Sharpe 0.90, MaxDD 51.0%, trades=336 **test_proxy**: 885.5% return, Sharpe 0.99, MaxDD 52.7% **fixed_split**: PASSED ✓ **Best checkpoint**: iter 20 (s

## 2026-04-20 — 2026-04-20 Tier 1/2/3 harness scaffolding (`dropped`)
Legacy memory note normalized into the repo-wide schema. Implemented the full next-step plan harness laid out in `.cursor/plans/plan_next_src_experiments_5796dd14.plan.md`. No new training runs beyond the seed-ensemble evaluation; all addit

## 2026-04-20 — User Direction Pivot: Stop Trade Compression, Focus on Drawdown + Kelly (`kept`)
Legacy memory note normalized into the repo-wide schema. Date: 2026-04-20 > "交易多不影响， 主要要控制回测和凯利期望 优化的方式应该是模型 奖励 特征结合的结果" Translation: "Many trades don't matter. The main thing is to control drawdown and Kelly expectancy. The optimization ap

## 2026-04-20 — v10: Episode-Level Drawdown Penalty — Code-Level Reward Redesign (`dropped`)
Legacy memory note normalized into the repo-wide schema. Date: 2026-04-20 Profile: `balanced_rule_residual_v10` Baseline: `balanced_rule_residual_v2` (best valid score), `v8/v9` (failed hyperparameter tuning) v8 (strong penalties) and v9 (m

## 2026-04-20 — v14 GRU Ablation Results — Episode Penalty Is Necessary (`reference`)
Legacy memory note normalized into the repo-wide schema. **Date:** 2026-04-20 **Profile:** `balanced_rule_residual_v14` **Hypothesis:** GRU architecture + observation_window alone drives drawdown reduction (penalty is incidental). **Result:

## 2026-04-20 — v15 Weak Penalty Results — Penalty=1.0 Is in a Bad Zone (`reference`)
Legacy memory note normalized into the repo-wide schema. **Date:** 2026-04-20 **Profile:** `balanced_rule_residual_v15` **Hypothesis:** A weaker terminal penalty (1.0) might find a sweet spot between v14 aggression and v12/v13 conservatism.

## 2026-04-20 — v17 Results — Window=10 Reverts to Window=3 Performance (`kept`)
Legacy memory note normalized into the repo-wide schema. **Date:** 2026-04-20 **Profile:** `balanced_rule_residual_v17` **Hypothesis:** More temporal context (10 days) continues to improve over 5 days. **Result:** DIMINISHING RETURNS CONFIR

## 2026-04-20 — v18 Results — v16 Config Is Seed-Sensitive on Return (`kept`)
Legacy memory note normalized into the repo-wide schema. **Date:** 2026-04-20 **Profile:** `balanced_rule_residual_v18` **Hypothesis:** v16's strong result (632.4% return) is reproducible with different seeds. **Result:** PARTIALLY REPRODUC

## 2026-04-20 — v3 Results: Trade Compression Failed — Root Cause Analysis (`dropped`)
Legacy memory note normalized into the repo-wide schema. | Metric | v2 | v3 Target | v3 Actual | Δ | |--------|-----|-----------|-----------|---| | Return | 521.3% | 600-650% | **569.7%** | +48.4pp | | Sharpe | 0.86 | 0.85-0.90 | **0.86** |

## 2026-04-20 — v7 Post-Mortem: [256,128] + Stronger Penalties + Tighter Caps (`kept`)
Legacy memory note normalized into the repo-wide schema. Date: 2026-04-20 Profile: `balanced_rule_residual_v7` Baseline comparison: `balanced_rule_residual_v2` (best drawdown so far), `v5` (best return) | Metric | v2 (baseline) | v5 | v7 |

## 2026-04-20 — v8 Results: Passes Fixed-Split Promotion Gate, But Penalties Degrade Learning (`kept`)
Legacy memory note normalized into the repo-wide schema. Date: 2026-04-20 Profile: `balanced_rule_residual_v8` Baseline: `balanced_rule_residual_v2` (best drawdown), `v7` (previous attempt) | Metric | v2 | v5 | v7 | v8 | |--------|-----|---

## 2026-04-21 — 2026-04-21 release overlay evaluation (`dropped`)
The first explicit `release` execution overlay on top of `balanced_rule_residual_v19` did not produce a materially new behavior. `stress_release_route_v1` was effectively identical to `regime_route_v1` on the key SOXL bear/recovery slices, so this direction should not be treated as a successful next-step mechanism.

## 2026-04-21 — 2026-04-21 v22 VIX term-structure training results (`reference`)
First training run of balanced_rule_residual_v22 with SOXL_USE_VIX_TERM=1 (external VVIX / VIX9D / VIX3M / SKEW features added to the observation). Same seed (12345) and profile otherwise identical to v19 baseline. Feature count 59 → 64 and schema signature flipped as expected. All three fixed-split promotion gates still pass. New features buy stress-window MaxDD improvement (test_2020 −1.9pts, test_2023_2024 −4.9pts) at the cost of a 5.8pt return drag in test_2023_2024. Keep as candidate, do not flip defaults.

## 2026-04-21 — 2026-04-21 v23 narrow transition-state probe (`dropped`)
Trained `balanced_rule_residual_v23` as a narrow 6-feature transition-state clone of v19 under `SOXL_USE_TRANSITION_V23=1`. Fixed-split gate still passed, but the model became broadly more defensive rather than selectively better at stress-release. `test_2022` improved slightly while `test_2020`, `test_2023_2024`, and proxy return dropped sharply. Best checkpoint was iter 0, indicating the new feature pack did not survive training pressure. Branch dropped and archived.

## 2026-04-21 — 2026-04-21 v24 low-frequency coarse residual failure (`dropped`)
Experiment: `balanced_rule_residual_v24` Baseline: `balanced_rule_residual_v19` Mechanism: `decision_frequency: 2` hybrid residual deltas compressed to `[-0.08, 0.0, 0.08]` Intent: test whether medium-strength temporal abstraction can reduc

## 2026-04-22 — 2026-04-22 balanced_rule_residual_mode_v2 result (`dropped`)
Second mode-hierarchy run after `balanced_rule_residual_mode_v1`. This loop had two goals: move mode behavior out of hard-coded helpers and into profile config test whether looser favorable modes could recover upside without losing the hier

## 2026-04-22 — 2026-04-22: Rulebase Imitation Bonus — Full Experiment Series (v1-v7) (`kept`)
Legacy memory note normalized into the repo-wide schema. 针对跑不过rulebase的问题，如果在rulebase的"埋点"上给超额奖励，能提高学习rulebase的能力吗？ Imitation bonus DOES improve the model's ability to learn rulebase timing, as evidenced by v1's 812% proxy return (vs v16 ba

## 2026-04-22 — 2026-04-22: CRITICAL CORRECTION — test_2022 Diagnosis (`reference`)
Legacy memory note normalized into the repo-wide schema. Initial diagnosis claimed the model went 100% position right before the Oct 2022 crash: ``` 2022-10-07: model_pos=1.00, market=-17.69% → single day -17.7% ``` Root cause identified as

## 2026-04-22 — 2026-04-22: Rulebase Imitation Bonus v1 (`dropped`)
Legacy memory note normalized into the repo-wide schema. 针对跑不过rulebase的问题，如果在rulebase的"埋点"上给超额奖励，能提高学习rulebase的能力吗？ Rulebase: 688.6% / 0.824 / 69.88% (45 trades) v16 best model: 632.4% / 0.91 / 50.1% Gap: model underperforms by 56pp on raw

## 2026-04-22 — Kelly Criterion Dynamic Position Control — Implementation (`dropped`)
Legacy memory note normalized into the repo-wide schema. Implemented rolling Kelly criterion features and Kelly tracking reward penalty to address the fixed low-position collapse observed in v2/v4 models. 1. **`src/soxl_quant/data.py`** Add

## 2026-04-22 — Minimal Reward v1 Results — Hypothesis Rejected (`reference`)
Minimal reward (5 terms) + regime budget in training produced 492.3%/0.85/48.8% — identical to v12/v13 baseline. Training degraded from iter 0 (random init was best). The "extra" reward terms (activity_bonus, profit_taking, panic_buy, opportunity_cost) contribute ~140pp of return. Participation signals are necessary, not noise. Budget overlay works on fixed actors, not during training.

## 2026-04-22 — v10 Attention + Simplified Reward + Kelly — Failure Report (`dropped`)
Legacy memory note normalized into the repo-wide schema. **Profile**: `balanced_kelly_attention_v10` **Hypothesis**: A fundamentally re-architected approach would break the "fixed low-position collapse": Network: GRU → WindowSelfAttentionNe

## 2026-04-23 — Risk-budget oracle distillation and composer extraction (`dropped`)
No-promotion oracle distillation used full fixed-replay daily labels for risk_budget_usage_v1. The label set contained action-4 full usage, but the single actor reached only test_proxy 363.8% / Sharpe 1.090 / MaxDD 23.7%, far below fixed replay 2003.6%. Keep the reusable composer extraction; stop trying to compress the two-speed mechanism into one PPO actor.

## 2026-04-23 — Risk-budget usage replay candidate (`reference`)
Fixed replay treated imitation_v4 as a budget usage ratio inside mode_v1 risk bands. It passed all replay acceptance checks: test_2022 MaxDD 25.6%, avg_pos 0.214, test_proxy 2003.6% / Sharpe 1.248 / MaxDD 45.0%. Keep as candidate fixed-replay mechanism; next step must implement runtime/backtest parity before any promotion claim.

## 2026-04-24 — Category-first runtime cleanup (`kept`)
Collapsed the separate evidence surface back into runtime, removed historical codex_exec_plans workpads, and deleted the root multi_asset data copy so iteration starts from category-specific runtime paths and memory indexes.

## 2026-04-24 — Hierarchical online budget runtime generalization (`dropped`)
Implemented a deployable hierarchical online composer with explicit slow mode refresh above the retained imitation_v4 actor. Rule-router variants improved safety modestly, and learned-router variants lifted proxy return to 102-415 percent across the 3x ETF set, but all remained materially below fixed replay and the learned variants spent more 2022 drawdown budget. Keep the composer as a runtime diagnostic, not a production promotion.

## 2026-04-24 — Learned-router conditioned oracle distillation diagnostic (`dropped`)
Explicit slow-mode one-hot conditioning did not improve the learned-router oracle distillation probe. Proxy return fell from 412.9 percent to 387.1 percent and MaxDD rose from 28.7 percent to 30.1 percent, so simple regime identity is not the missing ingredient.

## 2026-04-24 — Learned-router oracle distillation diagnostic (`dropped`)
No-promotion oracle distillation from the deployable learned-router slow5 daily path reached test_proxy 412.9 percent, Sharpe 1.042, and MaxDD 28.7 percent. This is stronger than the prior oracle distill but still far below the target path, so the bottleneck remains single-actor capacity in the learned-router regime.

## 2026-04-24 — Learned-router usage regression probe (`dropped`)
Continuous usage regression on the learned-router teacher path fit the labels very well but produced a much worse policy than the learned-router oracle classifier. Proxy return fell to 222.8 percent and MaxDD rose to 34.7 percent, so discrete label quantization is not the main bottleneck.
