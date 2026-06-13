# state_conditioned_checkpoint_replay_v1

- Decision: `checkpoint_selector_no_retention_candidate`.
- Actor retention allowed: `False`.
- WF drawdown gate: `False` using MaxDD `< 20.00%`.
- YTD SOXLBacktest reference gate: `False` using return `>= 132.08%`.
- YTD current-best alignment gate: `False` using return `>= 213.14%`.

## Selected Fold Checkpoints
- wf_2020: `wf_2020:iter0` return `41.16%`, MaxDD `19.25%`, pass=True
- wf_2022: `wf_2022:iter0` return `-20.24%`, MaxDD `21.65%`, pass=False
- wf_2023_2024: `wf_2023_2024:iter8` return `1.51%`, MaxDD `6.97%`, pass=True

## Selected YTD Checkpoint
- `ytd_2026:iter8` return `77.97%`, MaxDD `15.67%`, SOXLBacktest pass=False, current-best pass=False.

## Next Action
- retrain or modify objective; selection-only replay did not satisfy walk-forward drawdown and YTD alignment gates

## Sidecars
- `checkpoint_replay_metrics.csv`
- `fold_checkpoint_selection.csv`
- `ytd_checkpoint_selection.csv`
- `selection.json`
- `summary.json`
