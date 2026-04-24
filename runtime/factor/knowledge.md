# FACTOR Knowledge

_Auto-generated from memory/ integration on 2026-04-24._
_Total entries: 16._

## 2026-04-16 — 2026-04-16: Asymmetric Threshold Experiment (Option 1) (`dropped`)
Legacy memory note normalized into the repo-wide schema. Test "buy easy, sell hard" asymmetric rebalance threshold while keeping the 3-day trade lock and hard regime routing. `rebalance_threshold` now returns **direction-dependent threshold

## 2026-04-17 — 2026-04-17: historical-best `balanced` feature gap (`dropped`)
Legacy memory note normalized into the repo-wide schema. Canonical `balanced` is already competent at: persistent stress defense mature trend capture Its main blind spot is: transition states especially stress-release, bear-market rebounds,

## 2026-04-18 — PPO 诊断报告与优化路线图 (`reference`)
Legacy memory note normalized into the repo-wide schema. > 日期: 2026-04-18 > 背景: RL 无法超越 rule-based  baseline (688.6%)，怀疑 PPO 本身存在结构性问题 | 研究 | 算法比较 | 关键结论 | |------|---------|---------| | **FinRL Contest 2023-2024 Review** (Holzer et al., 20

## 2026-04-18 — Trade compression experiments: execution vs reward vs decision_frequency (`reference`)
Four variants of trade-frequency compression tested on SOXL. decision_frequency=3 achieves the best 2022 MaxDD (38.4%) and good generalization, but sacrifices total return (698% -> 314%). Key insight: execution-layer compression (trade_lock, threshold, decision_frequency) is more effective than reward-layer compression for reducing drawdown without destroying returns.

## 2026-04-18 — 2026-04-18: first transition-feature batch failed as mainline (`dropped`)
Legacy memory note normalized into the repo-wide schema. New temporary branch: `balanced_transition_state` Intent: keep canonical `balanced` reward, cap, and selection rules fixed add transition / persistence features only test whether rebo

## 2026-04-20 — RL vs Rule-Based 差距根因分析与下一步优化路线图 (`active`)
量化分析 RL (balanced_rule_residual_v2, 521.3%) 与 rule-based (688.6%) 之间 167pp 差距的五大根因。提出 panic-index 特征注入、底仓提升至 40-50%、 交易压缩至 <100 笔、regime-conditioned alpha scale 四条优化路径。

## 2026-04-20 — v16 Results — More Temporal Context (window=5) Is a Big Win (`kept`)
Legacy memory note normalized into the repo-wide schema. **Date:** 2026-04-20 **Profile:** `balanced_rule_residual_v16` **Hypothesis:** Increasing observation_window from 3→5 with proven penalty=2.0 improves GRU decision quality. **Result:*

## 2026-04-20 — v5 Results: Flat Penalty 4x Stronger — Trade Compression Still Fails (`dropped`)
Legacy memory note normalized into the repo-wide schema. Date: 2026-04-20 | Metric | v4 | v5 (4x penalty) | Delta | |--------|-----|-----------------|-------| | Return | 730.8% | 780.7% | +50pp | | Sharpe | 0.90 | 0.93 | +0.03 | | MaxDD | 5

## 2026-04-21 — 2026-04-21 core extension cleanup (`dropped`)
Cleaned failed experiment mechanisms out of core runtime paths. Execution experiments now extend through declarative `custom_regimes`, and feature experiments now extend through profile-owned `feature.packs` plus optional experiment-local `extensions.py` files instead of env flags and shared dataclass field growth.

## 2026-04-21 — 2026-04-21 experiment landscape review and research gate hardening (`kept`)
Added a durable repo-wide experiment landscape review and hardened `AGENTS.md` so new experiments now require an explicit baseline, closest success/failure case, external evidence class, experiment card, and comparison-table row. The repo should now bias toward narrower, mechanism-new branches instead of repeating reward, threshold, or bundled-structure loops.

## 2026-04-21 — 2026-04-21 historical integration policy (`active`)
Added a durable historical integration policy to `docs/experiment-landscape-review.md`. The repo should now retain a single core stack (`balanced`, `balanced_rule_residual`, `v19`, and the harness governance path), treat `v22`, checkpoint review, smoke residual checks, and overlay studies as optional modules, and permanently demote reward-only tuning, broad transition bundles, narrow transition retrains without selective execution, generic overlay retuning, and bundled structural retries to evidence-only directions.

## 2026-04-22 — Learned Mode Classifier + Hybrid Experiments — Full Results (`reference`)
Comprehensive evaluation of three architectural approaches to break the 40% drawdown barrier. Learned mode classifier achieves 38.2% MaxDD on fixed actor but training adapts away from it. Regime-conditional budget masking preserves return while modestly improving drawdown. Actor ensemble fails. No configuration achieves 30% proxy MaxDD without large return sacrifice. The frontier appears to be ~35-42% for this leverage level and asset class.

## 2026-04-22 — Phase 0 — Drawdown Budget Action Masking Prototype Results (`reference`)
Prototype of Approach C (drawdown-budget position masking) on fixed actors. Pure mechanical masking works but kills return. Regime-conditional masking (only active when regime_risk > 0.6) preserves return while modestly improving drawdown. mode_v1 already achieves 32% proxy MaxDD, making pure masking redundant. Core bottleneck is transition discrimination, not raw drawdown control.

## 2026-04-23 — Two-speed mode budget replay dropped (`dropped`)
Fixed-signal two-speed replay combined mode_v1 as the slow risk anchor with imitation_v4 as the fast participation path. Proxy and restart participation improved, but test_2022 MaxDD remained 33.9%, above the strict 30% gate. Keep the replay tool and drop the candidate; next work should avoid mild slack/budget retrains and require stronger crisis-state protection or external hedge mechanics.

## 2026-04-24 — 2026-04-24: Regime-Conditional Overlay Discovery — vol_stress (`kept`)
Legacy memory note normalized into the repo-wide schema. After regime-conditional IC analysis revealed that feature predictive power is highly regime-dependent, we tested execution overlays using existing features with regime gating. **Majo

## 2026-04-24 — Transition Signals Experiment Rejected (`dropped`)
Legacy memory note normalized into the repo-wide schema. Date: 2026-04-24 Profile: `balanced_rule_residual_v19_transition` Hypothesis: transition-state features (stress-release detection) would improve RL performance Method: add 7 transitio
