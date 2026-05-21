# Paper Dry-Run Profile Scope v1

## Context

The paper runner intentionally refuses duplicate same-day non-dry execution, so
the `2026-05-20` root paper artifact remains the earlier
`balanced_rule_residual_v19` run until the next fresh post-close date.

However, dry-run preflights for different profiles were all written to the same
`runtime/soxl/paper_trading/<date>/dry_run/` folder after a same-day executed
summary existed. The later budget-usage composer dry-run overwrote the earlier
v22 next-execution preflight, weakening the simulated-paper evidence trail.

## Change

`production/soxl_paper_trading.py` now keeps the shared `dry_run/` folder only
when the existing dry-run summary has the same profile. If a different
profile/composer dry-run would collide, it writes to:

```text
runtime/soxl/paper_trading/<date>/dry_run/<profile>/
```

This preserves idempotent non-dry behavior while allowing multiple same-date
preflight proofs.

## Live Evidence

Default v22 paper dry-run:

```bash
PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py \
  --dry-run --no-notify --skip-data-update
```

Readout:

- auto-selected profile: `balanced_rule_residual_v22`
- output:
  `runtime/soxl/paper_trading/2026-05-20/dry_run/balanced_rule_residual_v22`
- risk gate: approved
- target/current: `0.5` / `0.48408568215209924`

Default scheduler dry-run:

```bash
PYTHONPATH=. qlib_venv/bin/python production/soxl_daily_scheduler.py \
  --dry-run --no-notify --skip-data-update --skip-dashboard
```

Result: scheduler completed and wrote the same scoped v22 dry-run path.

Existing composer dry-run remains available at:

```text
runtime/soxl/paper_trading/2026-05-20/dry_run/summary.json
```

Regression suite:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_paper_trading.py tests/test_daily_scheduler.py \
  tests/test_scheduler_health.py -q
# 40 passed

PYTHONPATH=src:. qlib_venv/bin/python scripts/check_harness.py
# passed=167 failures=0

production/check_scheduler_health.sh
# local scheduler/data/model/broker/dashboard ok
# runtime proof pending for selected v22; Vercel blocked_no_credentials
```

## Decision

Do not force a duplicate non-dry `2026-05-20` execution. The scoped v22 dry-run
is preflight evidence only; runtime proof still requires the next fresh
non-dry paper date to write the selected profile.
