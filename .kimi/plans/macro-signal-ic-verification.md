# Macro Signal IC Verification Plan

## Goal
Verify whether macro indicators (credit spread, yield curve, DXY, oil) have genuine predictive power on SOXL returns before attempting any re-injection into the RL system.

## Background
`balanced_macro_v1` (macro z-score/delta as policy input) failed promotion gate because:
- test_2022 avg_pos = 0.368 > 0.35 limit (bear market MORE aggressive, not less)
- test_proxy Return = 450% vs baseline 605%

Hypothesis: macro signals have weak/no standalone predictive power, so adding them as policy features introduces noise.

## Plan

### Phase 1: Compute standalone IC for each macro signal (this session)
- Load macro time series from `data/raw/` or `data/processed/`
- Load SOXL daily returns
- Compute Spearman rank correlation (IC) between signal(t) and return(t+1, t+5, t+10, t+21)
- Compute IC by regime (bull/bear) to see if signals only work in specific regimes
- Report: mean IC, IC IR, % positive, t-stat

### Phase 2: Analyze IC results
**If IC is weak (|mean IC| < 0.03 or IR < 0.3):**
- Conclusion: macro signals have no standalone alpha
- Decision: permanently drop macro feature direction
- Action: archive `balanced_macro_v1`, remove `macro_signals` from consideration

**If IC is moderate (0.03 <= |mean IC| < 0.08):**
- Conclusion: macro signals have some predictive power but not strong enough for direct policy input
- Decision: try macro-aware position cap instead of direct policy input
- Action: design `balanced_macro_cap_v1` — use macro signals to modulate position_cap.min_cap

**If IC is strong (|mean IC| >= 0.08):**
- Conclusion: macro signals have genuine alpha
- Decision: investigate why RL failed to learn it
- Action: try different preprocessing (regime encoding, momentum, cross-sectional rank) or architecture (attention to macro subset)

### Phase 3: Implement (if Phase 2 warrants)
- If position cap approach: modify `position_cap` logic in `soxl_qlib_strategy_v2.py`
- If re-preprocessing: modify `_builtin_macro_signals_pack` in `data.py`
- Train, evaluate, compare

## Expected Output
- `scripts/compute_macro_ic.py` — standalone IC computation script
- `runtime/model/card/macro_ic_analysis/` — results and plots
- Updated active_workpad with IC results and Phase 2 decision

## Time Estimate
- Phase 1: 30-60 min
- Phase 2: 15 min
- Phase 3: 2-4 hours (if needed)
