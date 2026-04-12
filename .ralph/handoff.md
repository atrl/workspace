# Handoff

## Completed

- Wired `position_cap` config end-to-end across training, evaluation, backtest, saved actor bundles, and replay
- Added promotion gates to summary sidecars and runtime blocking
- Added isolated walk-forward workers in `src/soxl_quant/cli.py`
- Added walk-forward artifact namespacing so worker windows no longer overwrite canonical fixed-split artifacts

## In Progress

- Ralph rotation 1 is running with Codex
- Highest-priority repo task is still harness validation, not another reward/cap retune

## Next Steps

- Run the real balanced walk-forward validation with the isolated worker harness:
- `python3 soxl_qlib_strategy_v2.py --profiles balanced --walk-forward --skip-data-refresh --seed 20260411`
- Verify:
- `wf_2023_2024` leaves a readable worker result
- canonical `artifacts/models/soxl_balanced_summary.json` keeps the parent promotion verdict
- no worker summary overwrites the canonical fixed-split summary
- If the run still fails, tighten the harness again instead of switching to unrelated strategy changes

## Notes

- Current repo truth is in:
- `codex_exec_plans/active/2026-04-10-soxl-agent-loop.md`
- `memory/2026-04-12.md`
- Use those files over stale assumptions from the initial Ralph prompt
