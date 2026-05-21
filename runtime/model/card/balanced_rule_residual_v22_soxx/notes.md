# balanced_rule_residual_v22_soxx Retention Audit

Date: 2026-05-21

## Goal

Remove stale active-candidate ambiguity around the direct SOXX actor-feature
branch before opening any new model-optimization loop.

## Baseline And Role

- Canonical baseline: `balanced_rule_residual_v22`.
- Direct comparison: direct actor-feature injection via `feature.packs:
  [vix_term, soxx_signals]`.
- Role: diagnostic cleanup.
- Changed layer: `feature`.
- External reference: repo-owned SOXX underlying dataset plus the SOXX
  fixed-actor cap and walk-forward absorption diagnostics.
- Comparison-table row key: `balanced_rule_residual_v22_soxx`.

## Evidence

Registry regression:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_profile_registry.py::test_blocked_soxx_actor_feature_profile_is_not_active_candidate -q
# red before registry update: status='candidate', keep_actor=True, notes_ref=profile.yaml
# green after registry update: 1 passed
```

The retained training log reports:

```text
balanced_rule_residual_v22_soxx: 1099.3% | Sharpe 1.00 | MaxDD 59.6% | trades 304
promotion: blocked
```

The summary artifact has proxy evidence only:

```text
test_proxy return=1235.02% sharpe=1.038 max_dd=61.09% avg_pos=0.5742
```

The follow-up SOXX fixed-actor cap walk-forward validation also blocked SOXX as
the next promotion mechanism:

```text
wf_2020: SOXX cap return 106.66% vs baseline 113.17%
wf_2022: eligible days 44, cap_changed 0, position_changed 0
wf_2023_2024: eligible days 27, cap_changed 6, position_changed 0
decision: blocked
```

## Decision

- Mark `balanced_rule_residual_v22_soxx` as `dropped` in the profile registry.
- Keep summary, daily CSV, profile, and train log as searchable evidence.
- Do not keep the actor checkpoint.
- Do not repeat direct SOXX actor-feature retraining or SOXX cap threshold
  tuning as the next model optimization loop.

## Follow-up

Future SOXX use needs a distinct mechanism that explains why it will survive
walk-forward execution absorption. The current evidence supports using SOXX as
diagnostic context, not as an active actor feature branch.

## Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_profile_registry.py -q
# 10 passed

PYTHONPATH=. qlib_venv/bin/python scripts/build_runtime_index.py --check
# runtime index OK

PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# passed=194 failures=0

test ! -f runtime/model/card/balanced_rule_residual_v22_soxx/soxl_balanced_rule_residual_v22_soxx_actor.pt && echo actor_removed
# actor_removed
```
