# risk_budget Family Complete Review

Date: 2026-04-25

## Executive Summary

The `risk_budget` family uses a **budget-usage architecture**: a slow mode sets the risk band, and a fast path decides how much of that band to use. This produces **exceptional bear-market defense** (2022 losses of 3-17% vs SOXL B&H -86.6%) but **poor recovery timing** (2023-2024 gains of 40-83% vs SOXL B&H +182%).

**`risk_budget_usage_v1` (native PPO) remains the best candidate** — it passes the promotion gate and achieves the best recovery performance. Subsequent attempts to improve it have either made it more conservative (hurting recovery) or introduced proxy-to-backtest gaps.

## Complete Comparison Table

### 2022 Bear Market (SOXL B&H: -86.6%)

| Profile | Return | MaxDD | vs B&H | Assessment |
|---------|--------|-------|--------|------------|
| v2 | **-3.2%** | **7.4%** | +83.4pp | Best defense, but too conservative |
| BC Warmstart | -5.0% | 10.7% | +81.6pp | Excellent defense |
| Oracle Distill | -5.5% | 11.0% | +81.1pp | Excellent defense |
| Learned Oracle | -11.7% | 13.2% | +74.9pp | Good defense |
| Usage Regression | -15.1% | 17.6% | +71.5pp | Moderate defense |
| **v1 (PPO)** | -17.3% | 18.7% | +69.3pp | Acceptable defense |

### 2023-2024 Recovery (SOXL B&H: +182.4%)

| Profile | Return | Sharpe | vs B&H | Assessment |
|---------|--------|--------|--------|------------|
| **v1 (PPO)** | **+82.8%** | **1.44** | -99.6pp | Best recovery in family |
| Learned Oracle | +78.6% | 1.13 | -103.8pp | Good recovery |
| Usage Regression | +63.0% | 0.98 | -119.4pp | Moderate recovery |
| Oracle Distill | +58.5% | 1.18 | -123.9pp | Weak recovery |
| BC Warmstart | +49.0% | 1.01 | -133.4pp | Weak recovery |
| v2 | +40.1% | 1.23 | -142.3pp | Worst recovery |

### Proxy Period (Full Backtest)

| Profile | Return | Sharpe | MaxDD | Gate | Status |
|---------|--------|--------|-------|------|--------|
| Oracle Distill | 363.8% | **1.09** | 23.7% | — | dropped |
| Learned Oracle | 412.9% | 1.04 | 28.7% | — | dropped |
| **v1 (PPO)** | 274.4% | 0.94 | 32.5% | ✅ | **candidate** |
| BC Warmstart | 250.7% | 0.91 | 27.3% | — | dropped |
| v2 | 158.3% | 0.88 | 20.6% | ❌ | ablation |
| Usage Regression | 222.8% | 0.77 | 34.7% | — | dropped |

## Key Findings

### 1. The Family Is Systematically Conservative

All risk_budget variants maintain very low positions:
- v1 avg_pos: 0.27 (proxy), 0.13 (2022 stress)
- v2 avg_pos: 0.18 (proxy), 0.07 (2022 stress)

This produces excellent drawdown control but misses significant upside in recoveries.

### 2. v1 Is the Sweet Spot

v1 achieves the best balance:
- ✅ Passes promotion gate (Sharpe 0.94 > 0.9)
- ✅ Best recovery in family (+82.8% in 2023-2024)
- ✅ Acceptable bear-market defense (-17.3% vs B&H -86.6%)

Subsequent attempts to improve v1 either:
- Made it more conservative (v2: -3.2% in 2022 but +40.1% in recovery)
- Failed to translate proxy gains to backtest (v3)
- Were dropped for other reasons

### 3. Oracle Distill Has Potential But Weak Recovery

Oracle Distill achieves the highest proxy Sharpe (1.09) and excellent 2022 defense (-5.5%), but its 2023-2024 recovery (+58.5%) is weaker than v1 (+82.8%). The oracle supervision teaches conservative behavior.

### 4. v3 (Mode-Conditioned Input) Failed

v3 had proxy Sharpe 1.07 but:
- Backtest Sharpe only 0.87 (proxy-to-backtest gap)
- wf_2022 MaxDD 59.0% (walk-forward collapse)
- wf_2023_2024 Sharpe 0.22

The mode-conditioned one-hot input leaked information from the future or training distribution.

### 5. BC Warmstart Works But Is Conservative

BC Warmstart achieves good 2022 defense (-5.0%) but weaker recovery (+49.0%). The BC initialization from oracle path teaches conservative behavior that PPO reinforces.

## Comparison with residual Family

| Metric | risk_budget v1 | residual v19 |
|--------|---------------|-------------|
| proxy Sharpe | 0.94 ✅ | 0.79 ✅ |
| proxy MaxDD | 32.5% | 53.9% |
| 2022 return | -17.3% | -25.2% |
| 2022 MaxDD | 18.7% | 32.4% |
| 2023-2024 return | +82.8% | +79.4% |
| avg_pos (proxy) | 0.27 | 0.54 |
| avg_pos (2022) | 0.13 | 0.28 |

**risk_budget v1 has:**
- Better proxy Sharpe (0.94 vs 0.79)
- Better 2022 defense (-17% vs -25%)
- Lower MaxDD across the board
- Similar recovery performance (+83% vs +79%)
- Much lower average position (more conservative)

## Decisions

| Profile | Decision | Reason |
|---------|----------|--------|
| risk_budget_usage_v1 | **keep candidate** | Best balance in family; passes gate |
| risk_budget_usage_v2 | **keep ablation** | Confirms more conservative = worse recovery |
| Oracle Distill | **dropped** | Good proxy Sharpe but weak recovery; not better than v1 |
| BC Warmstart | **dropped** | Good defense but weak recovery; not better than v1 |
| Learned Oracle | **dropped** | Learned oracle worse than direct oracle |
| Usage Regression | **dropped** | Worst proxy Sharpe in family |
| risk_budget_usage_v3 | **dropped** | Proxy-to-backtest gap; walk-forward collapse |

## Next Directions

1. **Improve recovery timing** — The core problem is not bear-market defense (which is excellent) but re-entry speed after bear markets. Consider:
   - Recovery-specific reward bonuses
   - Faster regime transition detection
   - Less conservative baseline position in non-stress periods

2. **Test v1 in walk-forward** — v1 has never been walk-forward validated. Given the corrected understanding of bear-market metrics, walk-forward may reveal it passes where v19 failed.

3. **Do NOT pursue more conservative variants** — v2 already proves this direction hurts recovery.

4. **Do NOT pursue mode-conditioned input** — v3 proved this introduces proxy-to-backtest gaps.
