# Expectations: smoke_rule_residual_v1

## Acceptance Criteria

- Training completes without harness breakage
- Smoke backtest produces non-NaN results
- Rule residual mechanism is functional (position = rule + bounded delta)

## Expected Trade-off

- Smoke profiles are intentionally lightweight (2 iters, 2 episodes)
- Not meant to match production quality metrics
- Purpose is structural validation only

## Failure Modes

- Rule residual overlay breaks position calculation → NaN or crash
- Reward terms incompatible with smoke scale → training divergence
- Profile config missing required fields → harness error

## Comparison-Table Row Key

`smoke_rule_residual_v1`
