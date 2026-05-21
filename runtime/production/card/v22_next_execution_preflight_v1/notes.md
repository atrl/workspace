# v22 Next Execution Preflight

## Context

- checked_at: `2026-05-21 08:56 CST`
- command: `PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py --broker paper --dry-run --no-notify --skip-data-update`
- output_dir: `runtime/soxl/paper_trading/2026-05-20/dry_run/balanced_rule_residual_v22`

## Result

- data freshness: `last_date=2026-05-20`, `expected=2026-05-20`, `stale_days=0`
- auto-selected profile: `balanced_rule_residual_v22`
- selected actor: `soxl_balanced_rule_residual_v22_actor.pt`
- signal date: `2026-05-20`
- target position: `0.5000`
- current paper position: `0.4841`
- risk gate: `approved`
- position step check: `change=0.016 limit=0.350`
- dry_run: `true`
- executed: `false`
- reconcile: `position aligned`, `action_required=none`

## Boundary

The root executed artifact for `2026-05-20` remains the earlier non-dry-run
`balanced_rule_residual_v19` paper execution. Do not force another non-dry-run
for the same signal date just to manufacture a v22 proof. The next fresh
post-close non-dry paper run should naturally write the first executed v22
summary if auto-selection remains unchanged.

Follow-up: a later composer preflight exposed that the shared `dry_run/` folder
could be overwritten by a different profile. The default v22 preflight now
writes to the profile-scoped dry-run folder above while the composer dry-run
keeps its own summary at the shared dry-run path.
