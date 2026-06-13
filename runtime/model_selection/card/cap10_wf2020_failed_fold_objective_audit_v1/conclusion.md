# Cap10 WF2020 Failed Fold Objective Audit

- Decision: `wf2020_failed_fold_requires_bull_reentry_target_revision_no_promotion`.
- Promotion: blocked; no actor is promoted or retained by this diagnostic.
- Failed fold: `wf_2020`; selected `bc_initialization` has relative `-292.75pp` and MaxDD `14.98%`.
- SOXLBacktest `legacy_default` captures the 2020 rebound with rebound return `366.21%` and avg position `0.959`, but full-year MaxDD is `54.28%`.
- Current wf2020 target contract has only `21` rows at `>=70%` target position out of `2470` rows; release slice accuracy is `0.611`.
- Next action: build a fold-local bull/reentry participation target, then rerun three-fold BC/checkpoint metric replay before any full retention training.
- Dashboard note: do not expose this as runnable in v2; dashboard feature snapshots and BUY/SELL summaries should be exported only after a retained actor exists.
