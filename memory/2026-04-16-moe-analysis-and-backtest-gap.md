---
date: '2026-04-16'
title: '2026-04-16: MoE Regime Analysis & Gap to Rule-Based Backtest'
kind: review
profiles:
- moe
- soxl
status: dropped
tags:
- dropped
- eval-2022
- execution-overlay
- moe
- review
- rule-based
- soxl
artifacts: []
workpad: codex_exec_plans/completed/2026-04/2026-04-10-soxl-agent-loop.md
summary: Legacy memory note normalized into the repo-wide schema. Evaluated the multi-asset
  v2 actor (`soxl_balanced_moe_actor.pt`) on the SOXL test set. Extracted gate softmax
  probabilities for every trading day. Performed an **ablation study** by
---

# 2026-04-16: MoE Regime Analysis & Gap to Rule-Based Backtest

## Context

Legacy memory note normalized into the repo-wide schema.

## Evidence

## 1. MoE Gate Routing Validity Analysis

### Method
- Evaluated the multi-asset v2 actor (`soxl_balanced_moe_actor.pt`) on the SOXL test set.
- Extracted gate softmax probabilities for every trading day.
- Performed an **ablation study** by forcing each expert to make *all* decisions, bypassing the gate.

### Regime vs Expert Usage (natural gate)

| Regime Risk Bucket | Expert 0 | Expert 1 | Expert 2 | Mean Position |
|---|---|---|---|---|
| low (<0.35) | 43.3% | 56.7% | 0.0% | 0.609 |
| mid (0.35-0.55) | 45.6% | 53.3% | 1.1% | 0.558 |
| high (0.55-0.75) | 29.4% | 64.2% | 6.5% | 0.410 |
| extreme (>0.75) | 17.9% | 64.3% | 17.9% | 0.260 |

| Volatility Bucket | Expert 0 | Expert 1 | Expert 2 | Mean Position |
|---|---|---|---|---|
| low (<0.35) | 23.1% | 76.9% | 0.0% | — |
| mid (0.35-0.55) | 43.0% | 57.0% | 0.0% | — |
| high (0.55-0.85) | 42.5% | 57.1% | 0.4% | — |
| extreme (>0.85) | 32.2% | 59.4% | 8.4% | — |

### Expert Ablation Results (fast evaluator, 2020-2026)

| Variant | Return | Sharpe | MaxDD | Trades | Avg Pos |
|---|---|---|---|---|---|
| **Natural gate routing** | **383.3%** | **0.80** | **54.5%** | **914** | **0.49** |
| Force Expert 0 only | 365.3% | 0.79 | 53.6% | 797 | 0.47 |
| Force Expert 1 only | 384.5% | 0.80 | 53.0% | 879 | 0.47 |
| **Force Expert 2 only** | **404.9%** | **0.82** | **55.1%** | **818** | **0.48** |

### Interpretation — Critical Finding
- **Expert specialization exists but the gate is sub-optimal.**
  - Expert 0 tends to be chosen in low/mid regime risk (mean position 0.538 → higher exposure).
  - Expert 2 is increasingly chosen in extreme regime risk (mean position 0.303 → lower exposure).
  - Expert 1 is the "lazy default" chosen ~60% of the time across all regimes.
- **The ablation reveals that Expert 2 alone outperforms the natural gate by +21.6pp return.**
  - This means the gate frequently fails to select Expert 2 in regimes where Expert 2 would have produced better decisions.
  - In other words, the model has learned a useful "bear/crisis" expert, but the gate does not deploy it aggressively enough.
- **Natural routing performance is essentially identical to forcing Expert 1**, confirming that the gate is over-relying on the generic expert.

### Conclusion on MoE Validity
- **The architecture works (experts are differentiated), but the gating policy is the bottleneck.**
- To improve performance, we need to **constrain or supervise the gate** so that it trusts the specialist experts more.

---

## 2. Rule-Based `soxl_backtest` Generalization

### Method
- Used the existing rule-based harness (50% min position, panic-buy / profit-taking logic) on all five 3x ETFs.
- Data fetched from local `data/raw/symbols/{ETF}.csv` with mocked VIX from `data/processed/panels/soxl_soxx_vix_full.csv`.

### Results (2020-01-02 ~ 2026-03-31)

| ETF | Rule-Based | Buy & Hold | Delta | RL Multi-Asset v2 |
|---|---|---|---|---|
| **SOXL** | **688.6%** | 154.3% | ✅ +534.3pp | 414.2% |
| **TQQQ** | 190.2% | 284.3% | ❌ -94.1pp | **285.6%** |
| **UPRO** | 161.3% | 180.5% | ❌ -19.2pp | **235.1%** |
| **SPXL** | 165.7% | 183.2% | ❌ -17.5pp | **217.0%** |
| **TECL** | **340.7%** | 270.1% | ✅ +70.6pp | 239.1% |

### Interpretation
- The **rule-based strategy is a SOXL/TECL specialist**. It crushes B&H on SOXL (+534pp) and wins on TECL (+71pp).
- It **fails on broad-index LETFs** (TQQQ, UPRO, SPXL). The 50% floor + high-volatility dip-buying logic is too conservative for smoother broad indices.
- **RL multi-asset v2 beats the rule-based strategy on TQQQ/UPRO/SPXL**, proving that learned policies can generalize better to broad ETFs.
- However, on **SOXL the gap is massive**: rule-based 688.6% vs RL 414.2% (**-274pp**).

---

## 3. Root Cause: Why RL Trails the Rule-Based Strategy on SOXL by 274pp

### Numbers comparison
| Metric | Rule-Based | RL Multi-Asset v2 | Gap |
|---|---|---|---|
| Return | 688.6% | 414.2% | -274pp |
| Trades | **45** | **1075** | +1030 |
| Avg Pos | 0.57 | ~0.49 | -0.08 |
| MaxDD | 69.9% | 56.1% | -13.8pp |

### Root cause #1: Friction massacre
- Rule-based strategy makes ~45 trades in 6+ years (≈1 trade per 2 months).
- RL makes ~1075 trades (≈1 trade every 1.5 days).
- Even at 0.1% cost, **1,030 extra trades** erode ~20-40% of compounding capital.
- More importantly, **many RL trades are micro-adjustments** (e.g. 0.47 → 0.52) that do not capture meaningful alpha but pay fixed slippage.

### Root cause #2: Gate under-utilizes the best expert
- Ablation shows Expert 2 alone yields 404.9% (fast eval) vs natural gate 383.3%.
- If the gate were perfect, we could capture an immediate +5-6% uplift simply by routing better.
- But even the best single expert (404.9%) is still **~284pp below the rule-based strategy**.

### Root cause #3: Missing "core position + big-swing" logic
- The rule-based strategy always holds a 50% core. It never fully exits. This guarantees it captures 100% of the largest up-legs.
- The RL model, despite floor=0.15, frequently exits entirely or oscillates around 0.3-0.6.
- In the strongest rallies (e.g. 2023-2024, 2025), the rule-based strategy benefits from its core + reloads on dips.
- The RL model often **trades around the trend** rather than **riding it**.

---

## 4. Optimization Roadmap to Beat Rule-Based SOXL

### Option A: Supervise the MoE gate (high confidence, moderate impact)
**Goal**: Force the gate to pick Expert 2 in high-risk regimes and Expert 0 in low-risk regimes.

**Implementation**:
- Add an auxiliary `regime_routing_loss` in `RegimeMoENet` / `SoxlPPOPolicy`:
  - When `regime_risk < 0.35`, target Expert 0.
  - When `regime_risk > 0.65`, target Expert 2.
  - Cross-entropy loss on gate logits, weighted by ~0.1.
- This does not change the experts themselves; it only makes the gate less lazy.

**Expected impact**: +5-10% return, modest drawdown improvement.

### Option B: Drastically reduce trading frequency (high confidence, high impact)
**Goal**: Cut trades from ~1000 to ~100-200, eliminating friction.

**Implementation**:
1. **Raise `rebalance_threshold` baseline** from 0.015 → 0.04 (quiet regime 0.03 → 0.08).
2. **Add a `trade_lock_period`** (e.g. 3-5 days): after any trade, freeze position regardless of action.
3. **Reward shaping**: penalize turnover more heavily when `|last_add_pnl| < 0.05` (no meaningful edge).

**Expected impact**: +50-150% return by cutting friction. This is the single biggest lever.

### Option C: Coarsen action space to mimic rule-based bands (medium confidence, high impact)
**Goal**: Replace 11 fine-grained actions with 3-5 coarse bands.

**Implementation**:
- Change `N_ACTIONS` from 11 to 5:
  - Action 0 → ~0% (or min floor)
  - Action 1 → ~0.3
  - Action 2 → ~0.5 (core position)
  - Action 3 → ~0.75
  - Action 4 → ~1.0
- This naturally prevents micro-adjustments because the smallest possible move is ~20% of the band.

**Expected impact**: Could halve trade count while preserving big-swing decisions. Requires retraining from scratch.

### Option D: Hybrid rule-RL policy (highest confidence, biggest impact)
**Goal**: Combine the best of both worlds: rule-based core position + RL overlay for tactical swings.

**Implementation**:
- Hard floor = 0.50 (like backtest) + cap = 0.90.
- Actor only decides the **tactical overlay** within [0.50, 0.90], i.e. action maps to 0.50 + base * 0.40.
- This guarantees the model always captures the core trend, while RL optimizes when to be 90% vs 50%.

**Expected impact**: This is the closest structural match to the rule-based strategy. It directly addresses the "missing core position" problem and should immediately raise average exposure and reduce whipsaws.

---

## 5. Recommended Next Experiment

**Run Option D (hybrid core + RL overlay) as the highest-EV bet.**

Rationale:
- The rule-based strategy's 688.6% return proves that a **50% core + big-swing overlay** is the dominant alpha source for SOXL.
- Current RL's 414.2% is losing because it lacks that core commitment.
- By hard-coding a 0.50 floor and letting RL only optimize the 0.50-0.90 range, we preserve the backtest's core insight while letting RL improve the tactical overlay (which the backtest does crudely with fixed 10%/30% steps).
- This should also help TQQQ/UPRO/SPXL because it prevents the model from going too flat in broad-index uptrends.

Trade-off:
- 2022 MaxDD will likely increase (closer to the rule-based 69.9%), because a 50% floor prevents rapid de-risking.
- But the return uplift should be large enough to justify it, especially if we keep the drawdown cap (`drawdown_position_cap`) at its current settings.

## Decision

See the summary and status fields in front matter.

## Next Use

Use `scripts/memory_search.py` before opening additional memory files.
