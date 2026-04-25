# Walk-Forward Review: balanced_rule_residual_v19 vs balanced

Date: 2026-04-25

## Executive Summary

**Both `balanced` (canonical) and `balanced_rule_residual_v19` (best fixed-split candidate) fail walk-forward validation on the 2022 bear market.**

This is not a v19-specific failure — it is a **family-level failure** on bear market generalization. The residual rule overlay does not solve the fundamental problem of bear-market overfitting.

## Results

### wf_2020 (Bull Market — COVID Recovery)

| Profile | Valid Sharpe | Test Return | Test Sharpe | Test MaxDD | Avg Pos | Stress Days |
|---------|-------------|-------------|-------------|------------|---------|-------------|
| balanced | 1.67 | +125.8% | **2.30** | 21.9% | 0.577 | 48 |
| v19 | 1.72 | +114.9% | **2.35** | 20.4% | 0.524 | 48 |

**Both models excel in bull markets.** v19 shows slightly better Sharpe (2.35 vs 2.30).

### wf_2022 (Bear Market)

| Profile | Valid Sharpe | Test Return | Test Sharpe | Test MaxDD | Avg Pos | Stress Days |
|---------|-------------|-------------|-------------|------------|---------|-------------|
| balanced | 1.06 | **-25.8%** | **-0.91** | 33.2% | 0.279 | 129 |
| v19 | 1.50 | **-25.2%** | **-0.80** | 35.2% | 0.299 | 129 |

**Both models collapse in bear markets.**

### Promotion Gate Results

| Profile | Gate | test_2022_max_dd | test_2022_avg_pos | test_proxy_sharpe |
|---------|------|-----------------|-------------------|-------------------|
| balanced | ❌ FAIL | 33.2% (pass <35%) | 0.28 (pass <0.35) | -0.91 (fail >0.75) |
| v19 | ❌ FAIL | 35.2% (fail <35%) | 0.30 (pass <0.35) | -0.80 (fail >0.75) |

## Key Findings

1. **Bull-market overfitting is universal** — Both models show excellent valid Sharpe (1.06-1.50) but catastrophic test Sharpe (-0.80 to -0.91) on 2022. The training/valid periods are dominated by bull regimes, and models learn to be always-long.

2. **Position collapse in stress** — Both models drop to ~0.28-0.30 average position during 129 stress days (~half the year), but this defensive positioning is **insufficient** to avoid large drawdowns. The model is scared but still loses 25%.

3. **v19 is not worse than balanced** — In fact, v19's test Sharpe (-0.80) is slightly better than balanced (-0.91), and its test MaxDD (35.2%) is only marginally worse than balanced (33.2%). The residual overlay does not help, but it does not significantly hurt either.

4. **The 35% MaxDD gate is irrelevant** — Both models fail the Sharpe gate by a massive margin (-0.80 vs +0.75 threshold). MaxDD is not the binding constraint.

5. **Stress-day count identical** — Both models experience 129 stress days in 2022, suggesting the stress detection is consistent but the *response* to stress is inadequate.

## Diagnosis

The fundamental issue is that the training data (2010-2021) contains only short bear episodes (2015, 2018, 2020), while the 2022 test period is a **prolonged secular bear market**. The models have never seen sustained multi-month drawdowns during training, so they:

- Overfit to bull-market patterns
- Under-react to prolonged stress
- Maintain too much exposure during the early phase of the bear

## Implications

1. **v19 is NOT promotable** — Fixed-split metrics (proxy Sharpe ~0.97) are misleading. Walk-forward is the true gate.

2. **The residual family needs a new mechanism** — Rule-residual imitation alone does not solve bear-market generalization.

3. **The canonical `balanced` is also broken** — The canonical baseline fails walk-forward too. The entire harness needs bear-market-aware training.

4. **Next experiment priority**:
   - **Data layer**: Include longer bear-market history or synthetic bear episodes
   - **Reward layer**: Add bear-market-specific penalties or opportunity-cost terms
   - **Sampling layer**: Oversample bear-market episodes during training
   - **Feature layer**: Add leading bear-market indicators (credit spreads, yield curve, etc.)

## Decision

- `balanced_rule_residual_v19`: **drop** — walk-forward failure confirmed
- No new residual-family experiments until bear-market generalization is solved
