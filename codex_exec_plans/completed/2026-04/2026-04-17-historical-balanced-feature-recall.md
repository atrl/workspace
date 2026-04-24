# Historical Best `balanced` Feature Recall

## Summary

The canonical `balanced` policy is already good at one thing:

- recognizing persistent stress and cutting exposure enough to avoid full buy-and-hold damage

It is materially weaker at a different thing:

- recognizing when a deep-stress state is transitioning into an early rebound or fresh trend restart

That means the next feature loop should not add more generic trend or more generic risk factors. The gap is mainly in **transition-state features** and **regime persistence / release features**.

## Evidence Base

- Strategy slices:
  - `artifacts/models/soxl_balanced_summary.json`
- SOXL detailed daily path:
  - `runtime/generalization/detailed/soxl_detailed_trades.csv`
- Existing feature attribution:
  - `runtime/feature_analysis/summary.json`
  - `runtime/feature_analysis/shap_importance.csv`
  - `runtime/feature_analysis/ic.csv`
- Scenario decomposition:
  - `runtime/generalization/execution_overlay_results.json`

## Where `balanced` Is Good

### 1. Persistent 2022 stress defense

- Fixed-split `test_2022`:
  - return `-25.82%`
  - maxDD `33.04%`
  - avg_pos `0.283`
- Strongest monthly excess came from heavy selloff months:
  - `2022-06`: excess `+41.14%`, avg position `0.299`
  - `2022-04`: excess `+34.77%`, avg position `0.277`
  - `2022-09`: excess `+33.02%`, avg position `0.268`
- One clean defensive window:
  - `2022-04-01..2022-06-30`
  - strategy return `-26.81%`
  - buy-and-hold return `-66.18%`
  - compounded excess `+52.03%`
  - avg position `0.275`

### 2. Long clean trend capture

- `test_2020`:
  - return `96.96%`
  - sharpe `2.16`
  - maxDD `21.12%`
  - avg_pos `0.518`
- `bull_trend_2020_2021` scenario:
  - return `120.99%`
  - sharpe `1.48`
  - avg_pos `0.625`

## Where `balanced` Is Bad

### 1. 2020 crash rebound underinvestment

- Worst monthly excess:
  - `2020-04`: excess `-34.42%`, avg position `0.266`
- Window:
  - `2020-04-01..2020-04-30`
  - strategy return `+10.53%`
  - buy-and-hold return `+38.28%`
  - compounded excess `-34.42%`

### 2. Bear-market rebound re-entry too slow

- `2022-07`: excess `-29.96%`, avg position `0.291`
- `2022-11`: excess `-33.28%`, avg position `0.349`
- These are classic reflex rallies after a long stressed regime. The policy stayed too defensive after the environment had already started to release.

### 3. Early 2023 trend restart recognition too slow

- `2023-01`: excess `-27.16%`, avg position `0.363`
- `2023-05..2024-03` still lagged buy-and-hold despite positive absolute return:
  - strategy return `+89.69%`
  - buy-and-hold return `+229.04%`
  - compounded excess `-57.61%`
  - avg position `0.501`

## Feature Recall From Good vs Bad Windows

The most important existing features are already known from SHAP / IC:

- top SHAP:
  - `relative_strength_21`
  - `vix_zscore`
  - `below_anchor_ratio_63`
  - `anchor_gap_180`
  - `days_since_high_63`
  - `vix_momentum_10`
  - `trend_slope_63`
  - `realized_vol_10`
- top IC cluster:
  - downside / streak / drawdown / cum-drop features

### Good stress-defense signature: `2022-04..2022-06`

Feature state versus the `2020-2026` test window mean:

- much worse long-trend damage:
  - `drawdown_63 -1.645 z`
  - `trend_slope_63 -1.536 z`
  - `anchor_gap_180 -1.438 z`
  - `cum_drop_30 -1.269 z`
- much clearer stress persistence:
  - `regime_risk +1.434 z`
  - `below_anchor_ratio_63 +1.384 z`
  - `days_since_high_63 +1.314 z`
  - `downside_ratio_21 +1.099 z`
  - `vix_zscore +0.900 z`

Interpretation:

- The policy already understands the "persistent damage + persistent stress" state.
- Existing features are sufficient to tell the model to cut exposure when:
  - the asset stays below the anchor
  - downside frequency remains elevated
  - high-vol stress persists for weeks

### Good trend-capture signature: `2020-08..2021-11`

Feature state versus the same baseline:

- strong trend confirmation:
  - `anchor_gap_180 +0.662 z`
  - `trend_slope_63 +0.491 z`
  - `relative_strength_21 +0.230 z`
- calmer regime:
  - `regime_risk -0.595 z`
  - `vix_zscore -0.455 z`
  - `realized_vol_10 -0.424 z`
- healthy structural state:
  - `below_anchor_ratio_63 -0.757 z`
  - `days_since_high_63 -0.391 z`

Interpretation:

- The policy also understands persistent clean-uptrend states.
- Existing anchor/trend features are enough when the trend is already mature and stable.

### Bad rebound signature: `2020-04`

Feature state:

- still looked like deep stress:
  - `drawdown_63 -2.108 z`
  - `trend_slope_63 -1.808 z`
  - `regime_risk +1.674 z`
  - `realized_vol_10 +2.029 z`
  - `vix_zscore +1.129 z`
- but early rebound information had already appeared:
  - `relative_strength_21 +0.692 z`

Interpretation:

- The model saw "stress" correctly, but had weak representation for "stress is ending quickly".
- This is not a missing trend feature problem.
- This is a missing **stress-release / rebound-transition** feature problem.

### Bad rebound signature: `2022-07`

Feature state:

- still structurally broken:
  - `trend_slope_63 -1.946 z`
  - `anchor_gap_180 -1.665 z`
  - `below_anchor_ratio_63 +1.462 z`
  - `days_since_high_63 +1.546 z`
  - `regime_risk +1.041 z`
- but fast downside pressure had already started to cool:
  - `downside_ratio_10 -0.433 z`
  - `vix_momentum_10 -0.525 z`

Interpretation:

- The policy overweights slow structural damage features and underweights fast release features.
- Result: it stays underinvested into violent bear-market rallies.

### Bad restart signature: `2023-01`

Feature state:

- still heavily below the long anchor:
  - `below_anchor_ratio_63 +1.412 z`
  - `anchor_gap_180 -0.577 z`
- but immediate panic state had already normalized:
  - `vix_zscore -0.954 z`
  - `downside_ratio_10 -0.684 z`
  - `regime_risk -0.413 z`

Interpretation:

- The policy lacks a feature for "long trend still damaged, but stress has already cleared and restart probability is rising".
- This again points to missing transition features rather than missing pure trend or pure stress features.

## What This Means For Feature Optimization

The most effective next feature loop should target state transitions, not more factor count.

### High-value feature families to add

1. Stress release speed
- `regime_risk_delta_5`
- `vol_stress_score_delta_5`
- `vix_zscore_delta_5`
- `downside_ratio_10_delta_5`

2. Rebound confirmation after deep damage
- `rebound_from_20d_low`
- `days_since_20d_low`
- `close_vs_ma21_after_drawdown`
- `positive_day_ratio_5_after_deep_drop`

3. Regime persistence / hazard
- `days_above_stress_threshold_21`
- `days_below_anchor_63`
- `stress_then_relief_flag`
- `trend_restart_flag` when short trend turns positive while long anchor damage remains negative

4. Cross-horizon disagreement
- short-term reversal positive while long-term anchor still negative
- short vol falling while long drawdown still deep
- relative strength turning positive before anchor repair

### Low-value directions to avoid

- adding more generic momentum variants that duplicate `anchor_gap_180`, `trend_slope_63`, `relative_strength_21`
- adding more generic risk level features that duplicate `vix_zscore`, `regime_risk`, `below_anchor_ratio_63`
- more reward-only tuning without changing state representation

## Next Experiment Order

1. Add transition features without changing reward weights.
2. Re-run canonical `balanced` style training with the same split and seed.
3. Judge success mainly on:
  - `2020-04`
  - `2022-07`
  - `2022-11`
  - `2023-01`
  - full `test_2022`
4. If those windows improve while mature-stress defense remains intact, only then revisit reward or cap tuning.
