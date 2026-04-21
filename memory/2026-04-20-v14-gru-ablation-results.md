---
date: '2026-04-20'
title: "v14 GRU Ablation Results \u2014 Episode Penalty Is Necessary"
kind: decision
profiles:
- balanced_rule_residual_v14
status: reference
tags:
- balanced_rule_residual_v14
- checkpoint-selection
- decision
- eval-2022
- reference
- walk-forward
artifacts:
- artifacts/models/soxl_balanced_rule_residual_v14_actor.pt
- artifacts/models/soxl_balanced_rule_residual_v14_summary.json
workpad: codex_exec_plans/active/2026-04-10-soxl-agent-loop.md
summary: 'Legacy memory note normalized into the repo-wide schema. **Date:** 2026-04-20
  **Profile:** `balanced_rule_residual_v14` **Hypothesis:** GRU architecture + observation_window
  alone drives drawdown reduction (penalty is incidental). **Result:'
---

# v14 GRU Ablation Results — Episode Penalty Is Necessary

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

**Date:** 2026-04-20
**Profile:** `balanced_rule_residual_v14`
**Hypothesis:** GRU architecture + observation_window alone drives drawdown reduction (penalty is incidental).
**Result:** HYPOTHESIS REJECTED.

## Results

| Profile | Return | Sharpe | MaxDD | test_2022 Return | test_2022 MaxDD |
|---------|--------|--------|-------|------------------|-----------------|
| v2 (MLP baseline) | 521.3% | 0.86 | 50.4% | — | — |
| v12 (GRU + penalty 4.0) | 492.3% | 0.85 | **48.8%** | ~-18% | ~30% |
| v13 (GRU + penalty 2.0) | 492.3% | 0.85 | **48.8%** | ~-18% | ~30% |
| **v14 (GRU only, no penalty)** | **663.2%** | **0.89** | **56.6%** | **-24.1%** | **32.6%** |

## Key Findings

1. **GRU without penalty is MORE aggressive, not less.** MaxDD worsened from 48.8% → 56.6%, exceeding the MLP baseline (50.4%).
2. **The terminal episode penalty is the drawdown brake.** When combined with GRU, the agent learns to use temporal context to anticipate episode-end penalties and reduce risk near boundaries. Without it, the GRU exploits temporal context to chase returns.
3. **GRU amplifies whatever incentive you give it.**
   - No penalty → highest return (663%), worst drawdown (56.6%)
   - MLP, no penalty → middle ground (521%, 50.4%)
   - GRU + penalty → most conservative (492%, 48.8%)
4. **Severe overfitting to validation.** Valid MaxDD was 27.1%, but test MaxDD exploded to 56.6%. The GRU without penalty overfits to the 2020 bull market in validation.
5. **First fixed_split promotion pass!** v14 is the first profile where ALL fixed_split checks passed:
   - test_2022_max_dd: 32.6% < 35% ✓
   - test_2022_avg_pos: 0.26 < 0.35 ✓
   - test_proxy_sharpe: 0.93 > 0.75 ✓
   - But walk_forward is "missing" so overall promotion blocked.

## Lesson

The GRU + terminal penalty combination is a **synergistic pair**: the GRU's temporal modeling enables it to strategically avoid the terminal penalty, which the MLP cannot do as effectively. The penalty is not incidental — it is the primary mechanism for drawdown control.

## Next Directions

1. **Tune penalty strength more finely** (0.5, 1.0, 6.0) — v12/v13 showed 2.0 and 4.0 are identical, but the extremes may differ.
2. **Increase observation_window** (5, 10) — more temporal context may help the GRU plan further ahead.
3. **MLP + terminal penalty ablation** — test if the penalty helps MLP too, or if GRU is uniquely required.
4. **Selection score tuning** — v14's iter 40 selected model had high valid score but terrible test generalization. Adjust return/sharpe/max_dd_penalty weights.
5. **Walk-forward evaluation** for v14 — fixed_split passed for the first time; run walk-forward to see if it also passes.

## Evidence

- Log: `log/balanced_rule_residual_v14.log`
- Summary: `artifacts/models/soxl_balanced_rule_residual_v14_summary.json`
- Actor: `artifacts/models/soxl_balanced_rule_residual_v14_actor.pt`

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
