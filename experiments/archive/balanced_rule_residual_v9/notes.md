# v9 Design Notes

## Hypothesis

v8 achieved a major milestone: **first experiment to pass all fixed-split promotion criteria** (test_2022_max_dd=31.31% ≤ 35%, test_2022_avg_pos=0.266 ≤ 0.35, test_proxy_sharpe=0.883 ≥ 0.75). However, v8 suffered from severe learning degradation — valid best score was only 89.54 (vs v2's 104.61), suggesting the 2.7x stronger penalties were overwhelming the alpha signal.

v9 tests whether a **moderate penalty level + lower hard_floor** can maintain v8's excellent 2022 risk control while restoring the model's ability to learn high-alpha behavior.

## Key Changes from v8

1. **drawdown_penalty: 0.45** (was 0.75) — dial back to avoid learning degradation
2. **downside_penalty: 0.20** (was 0.40) — revert to v2 level
3. **risk_budget_penalty: 0.20** (was 0.40) — revert to v2 level
4. **high_exposure_penalty: 0.08** (was 0.15) — revert to v2 level
5. **hard_floor: 0.10** (was 0.20) — allow deeper de-risking during bear markets

## Key Restorations from v8

1. **hidden_sizes: [128, 64]** — proven architecture
2. **alpha_scale: 3.0** — strong sensitivity to negative alpha
3. **position_cap: v2 settings** — looser, proven caps

## Expected Outcome

- Valid best score should improve toward v2's 104.61 (target: >95)
- Test_2022_max_dd should stay below 35% (hard_floor=0.10 should help)
- Test_2022_avg_pos should drop below v8's 0.266 (target: <0.22)
- Overall test MaxDD should improve toward or below v2's 50.4%
- Test Sharpe should stay >0.80

## Risk

If lowering penalties causes 2022 drawdown to spike above 35%, we know v8's high penalties were necessary for that specific gate. If overall MaxDD still exceeds 50%, the v7 post-mortem recommendation applies: move to **code-level reward redesign** (episode-level max drawdown penalty).
