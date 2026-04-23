# Transition Signals Experiment Rejected

Date: 2026-04-24

## Experiment
- Profile: `balanced_rule_residual_v19_transition`
- Hypothesis: transition-state features (stress-release detection) would improve RL performance
- Method: add 7 transition features to v19 baseline, run 20-iteration training

## Results
- Test return: 613.0% (vs v19 baseline 632.4% at 80 iters)
- Test Sharpe: 0.86 (vs 0.91)
- Test MaxDD: 57.6% (vs 50.1%) — **+7.5pp worse**
- Promotion: BLOCKED

## Key insight
Positive standalone IC (+0.21 for TRANS_VIX_DIVERGENCE) does NOT guarantee RL improvement. The features may cause the agent to enter too early during persistent stress.

## Decision
- REJECT transition_signals as core feature pack
- Keep pack in data.py as optional
- Actor checkpoint deleted per retention policy
- Profile registry updated: status=dropped

## Next directions
- Test 1-2 strongest transition features in isolation (not full 7-feature pack)
- Focus on execution overlay or reward tuning for test_2022 improvement
- The core gap (stress-release detection) remains unaddressed
