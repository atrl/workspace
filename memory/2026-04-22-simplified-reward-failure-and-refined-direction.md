---
date: '2026-04-22'
title: Simplified Reward Failure and Refined Direction
kind: review
profiles:
- soxl
status: reference
tags:
- eval-2022
- reference
- review
- soxl
- stress-regime
artifacts: []
workpad: codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
summary: 'simplified_v1 failed: proxy MaxDD 58.5% vs v19 54.8%. Root cause: stripping
  reward removed necessary guardrails (stale_position, trapped_loser, downside_signal,
  risk_budget). User wants: cash respected, conflicting gradients reduced, generalization
  to other 3x ETFs. New approach: remove activity_bonus (forced trading), reduce inactivity_penalty,
  strengthen risk guardrails, keep proven components. Network returns to [128,64]+GRU-64.
  Multi-asset joint training is the next frontier after single-asset reward is fixed.'
---

# Simplified Reward Failure and Refined Direction

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## What happened with simplified_v1

Training completed. Results vs v19 baseline:

| metric | v19 | simplified_v1 | delta |
|--------|-----|---------------|-------|
| proxy return | 735.1% | 776.9% | +5.7% |
| proxy sharpe | 0.9 | 0.91 | +0.01 |
| **proxy maxdd** | **54.8%** | **58.5%** | **+3.7pp** |
| test_2022 return | -22.5% | -25.35% | -2.85pp |
| test_2022 maxdd | 31.1% | 34.01% | +2.91pp |
| test_2023_24 return | 43.6% | 53.68% | +10.08pp |
| test_2023_24 maxdd | 37.3% | 36.11% | -1.19pp |
| trades | ~220 | 297 | +77 |

**Promotion blocked** due to proxy MaxDD 58.5% > gate threshold.

## Root cause analysis

The simplified reward stripped too many protective guardrails:

1. **Removed `stale_position` / `trapped_loser`** → model holds losing positions longer
2. **Removed `downside_signal` penalty** → cannot sense accelerating downside
3. **Removed explicit `risk_budget_penalty` / `high_exposure_penalty`** → risk exposure drifts up
4. **Kept only weak drawdown control** → insufficient episodic risk shaping
5. **Network too large** ([256,128,64]+GRU-128) for 200 training episodes → overfitting
6. **Episode too long** (500 days) → dilutes gradient signal

The return increase came from more trades (297 vs ~220) and higher average exposure, but at the cost of worse drawdown control.

## Key insight from history

`mode_v1` achieved the **best proxy DD (42.3%)** in the entire family using **the same reward as v19**. The difference was entirely structural (persistent mode hierarchy). This proves:

- Reward is not the primary bottleneck for DD control
- State/execution structure matters more than reward complexity
- But reward **incentive alignment** does matter — specifically, the "must trade" pressure

## User feedback interpretation

1. **"Cash should be respected as an asset"** → The `activity_bonus` (1.25) + `inactivity_penalty` (0.80) + `opportunity_cost_penalty` (0.05) create a **triple forced-trading pressure**. Model is penalized for holding cash in all regimes.

2. **"Too many hard-coded reward parameters create conflicting gradients"** → True, but the conflict is not "too many terms" — it is **conflicting directions**: activity_bonus pushes toward trading, while drawdown_penalty pushes away from risk. The model learns to oscillate.

3. **"Training is incomplete with no generalization testing"** → Need multi-asset validation.

## New direction: refined_v1

### Reward philosophy

- **Remove forced-trading incentives** (activity_bonus = 0, inactivity_penalty ↓)
- **Respect cash** (lower opportunity cost, no activity bonus)
- **Strengthen risk guardrails** (drawdown, downside, risk_budget, high_exposure all ↑)
- **Keep proven components** (alpha reward, PBRS, profit_taking, panic_buy, stale_position, trapped_loser)

### Network

Return to proven scale: `[128, 64]` + GRU-64, episode_len=300.

### Multi-asset next step

Previous multi-asset joint training (2026-04-14) showed:
- SOXL in-domain: 371% → 514% (+143pp)
- But MaxDD increased: 49.9% → 58.27%
- All cross-asset DDs were ~half of Buy&Hold

If refined_v1 reward achieves better single-asset DD control, it should be tested in multi-asset joint training next.

## refined_v1 Results

Training completed. Results vs v19:

| metric | v19 | refined_v1 | delta |
|--------|-----|-----------|-------|
| proxy return | 735.1% | 500.6% | -234.5pp |
| proxy sharpe | 0.9 | 0.84 | -0.06 |
| proxy maxdd | 54.8% | 54.6% | -0.2pp |
| test_2020 ret/dd | 109.4/18.1% | 107.4/16.2% | similar / better DD |
| test_2022 ret/dd | -22.5/31.1% | -22.2/31.5% | similar |
| test_2023_24 ret/dd | 43.6/37.3% | 48.8/31.0% | +5.2pp ret, -6.3pp DD |
| trades | ~220 | 325 | +105 |

**Diagnosis**: DD control worked (proxy 54.6%, test_2023_24 31.0%), but return
collapsed 235pp. Model held too much cash, missing intermediate-period upside.
Trades=325 (vs v19 ~220) shows oscillation between cash and position without
conviction.

**Root cause**: Complete removal of activity_bonus + very low opportunity_cost
(0.02) gave no "when to participate" signal.

## Decision

- `simplified_v1`: dropped — radical simplification removes necessary guardrails
- `refined_v1`: candidate but blocked — needs better participation signal
- `refined_v2`: candidate but blocked — adds trend_alignment_bonus (0.20) +
  opportunity_cost (0.04). Results: proxy 556.9%/0.87/53.0% (+56pp ret vs v1,
  -1.6pp DD). Best proxy DD in entire family. Excellent test_2023_24
  (63.3%/29.7%). But still 178pp below v19 return.
- `refined_v3`: candidate but blocked — stronger trend_alignment (0.30) +
  opportunity_cost (0.06) + lower hard_floor (0.15). Results: 637.8%/0.89/54.9%
  (+81pp ret vs v2, +1.9pp DD). Clear trade-off: more trend incentive = more
  return but worse DD in stress and post-stress windows.
- `refined_v4`: dropped — midpoint parameters (trend_alignment=0.25,
  opportunity_cost=0.05, hard_floor=0.18). Result: 480.0%/0.84/54.2%, worse
  than v1. Reward interactions are non-linear — midpoint does not work.
- **Reward-only direction exhausted**. v2 (defensive, 53.0% DD) and v3
  (offensive, 637.8% ret) are the two best points. Neither clears promotion.
- Multi-asset joint training: IN PROGRESS — using refined_v3 reward design
  across SOXL/TQQQ/UPRO/SPXL/TECL. Goal: pooled regime learning improves
  return without sacrificing DD control.
- Next: evaluate multi-asset results. If SOXL in-domain DD stays < 55% and
  return improves vs single-asset refined_v3, keep as candidate.

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
