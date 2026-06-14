# release16_rulebase_reversal_target_recipe_v1

- Decision: `rulebase_reversal_failedfold_target_recipe_ready_for_checkpoint_screen_no_promotion`.
- Primary feature set: `rulebase_reversal`; selected threshold `0.5`.
- Precision/recall/raw FPR/effective FPR: `0.437` / `0.684` / `0.342` / `0.000`.
- True/false positives: `569` / `732`; target contract rows: `569`.
- Strict raw-FPR gate: `False`; impact-adjusted recipe gate: `True`; stability gate: `True`; consensus gate: `True`.
- Next action: `run_rulebase_reversal_failedfold_checkpoint_screen_against_soxlbacktest_qlib_buysell_baseline`.

No actor is promoted or retained from this card. The written target contract is train-only and must be screened against the SOXLBacktest qlib buy/sell baseline and the walk-forward retention gates before any model can be kept.
