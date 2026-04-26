# v19_lowcap Training Result Analysis

## Date
2026-04-26

## Goal
Verify whether retraining actor with min_cap=0.20 improves over using old actor with new cap config.

## Method
- Train new actor from scratch with `balanced_rule_residual_v19_lowcap` profile (min_cap=0.20)
- Compare walk-forward results to old actor (trained with min_cap=0.40) running under min_cap=0.20

## Results

### New actor (trained with min_cap=0.20)
| Window | Return | MaxDD | Sharpe | RelOutperf |
|--------|--------|-------|--------|-----------|
| wf_2020 | +130.9% | 24.96% | 1.98 | +71.1pp |
| wf_2022 | -49.1% | **52.79%** | -1.55 | +37.5pp |
| wf_2024 | +10.9% | **39.73%** | 0.39 | +13.5pp |

### Old actor (trained min_cap=0.40, running min_cap=0.20)
| Window | Return | MaxDD | Sharpe | RelOutperf |
|--------|--------|-------|--------|-----------|
| wf_2020 | +106.7% | 22.4% | 1.95 | +53.6pp |
| wf_2022 | -41.9% | **46.7%** | -1.32 | +39.6pp |
| wf_2024 | +29.8% | **32.1%** | 0.71 | +31.5pp |

### Delta (new vs old)
| Window | Return | MaxDD |
|--------|--------|-------|
| wf_2020 | +24.2pp | +2.6pp |
| wf_2022 | -7.2pp | **+6.1pp** |
| wf_2024 | -18.9pp | **+7.6pp** |

## Critical Finding

**Retraining with min_cap=0.20 WORSENS bear-market and recovery performance.**

The new actor learns to be MORE aggressive in bull markets (+24pp in 2020) but this comes at the cost of:
- Worse 2022 MaxDD: 52.8% vs 46.7% (+6.1pp)
- Worse 2024 return: +10.9% vs +29.8% (-19pp)
- Worse 2024 MaxDD: 39.7% vs 32.1% (+7.6pp)

## Root Cause

Training with lower min_cap teaches the actor that it can safely hold higher positions because the cap will protect it. This creates a "moral hazard":
- Actor learns aggressive bull-market behavior
- But when bear market arrives, the actor hasn't learned sufficient defensive behavior
- Result: worse MaxDD in stress periods

## Decision

- **v19_lowcap actor**: **DROP** — worse walk-forward than old actor + new cap
- **v19 profile**: Keep `min_cap=0.20` but use **old actor** (trained with min_cap=0.40)
- Best combination: old actor + new cap config

## Action
1. Update profile_registry to mark v19_lowcap as dropped
2. Update experiment-landscape-review.md with finding
3. Keep v19 canonical profile with min_cap=0.20
