# Phase7 independent data and workflow pre-fit review

Scope: registered constrained_net_excess_20261003; reviewed before market NN fitting. Data readback only, no portfolio outcomes or forecast fitting/inference.

## Confirmed data evidence

`load_constrained_inputs` hash-verifies the FINISHED parent Recorder52f40778cbe64ca6a5864da356b960d0 and its local/archived summary, source-code map, and eight allowed artifacts before deserialization. The qualified source is bd754e6362b241f194351a1e7aebff6f, as_of2026-09-25. Manifest byte SHA and marker canonical JSON SHA are treated separately.

Actual native readback passed: 8,324 OHLCV rows, 4,162 calendar sessions, 3,956 primary rows,14 existing forecast models. Reconstructed all21 development folds exactly: mature row frames/audits, scalers,27-column masked/standardized features, two returns, failure classification, and transformation audits. Native frame hash b7a59f39feed5aed6e61fec2b4239092c474e453c91d4d133c73f9dcc571a03f; primary hash bf5586573a4c51feb348c8e1a661ccf0f334f3a7d83b9ce9558629bfd16cf02b. Full detail `/tmp/constrained_net_excess_data_readback.json`.

Six focused tests pass: native reconstruction with forecaster fit/predict deliberately forbidden, pre-deserialization byte corruption rejection, parent summary/code identity, native value drift, fold/scaler/matrix mismatch, and forecast-date/strict-prior maturity. An actual-data audit serialization issue (list-valued feature metadata cannot use pandas scalar hash) was fixed before fits; artifact integrity remains exact byte SHA, and the descriptive audit hash explicitly uses JSON.

Data module SHA f140218167b67c42e56be08c269f04754970b2211e00ac415c277743bf631be9; focused test SHA ad3ce6895d7cb7368a5b6697674fe68b8b3bd06fd36710ee19f08dd2bd206451.

## Workflow review

Confirmed: only development folds fit before selection_seal; diagnostic folds prepared after the seal; frozen2023forecast inference only if selected price_vol; all three seed models retained; separate exact own-book ensemble audit; same parent replay/accounting/qualification gates; native reconciliation after writable Recorder exits; actual trade_date>=2026-08-03 for user-window with separate carried and CASH-start books. Existing annual/year/bootstrap/seed/invalid gates remain intact. No allocation models from the old run are loaded.

Two concrete review findings sent to root for correction before source freeze:

1. `fit_folds` numerical_failure originally included seed diagnostics only. `evaluate_ensemble` can independently catch a numerical failure and return an accounting null with audit.failure. This ensemble-only failure must propagate to the replay qualification flag; ordinary economic ensemble rejection remains a valid null, not an integrity failure.
2. Add `src/workflow/anchor_checkpoint_feasibility.py` to this run's source hash map: loader reuses checked_parent_load and _same_evidence from it, and that dependency postdates the Phase5 map.

This review does not certify the optimizer globally, risk generalization or alpha. Full objective/risk-gradient tests belong to independent objective/model reviewers, and source uncertainty/previously reviewed diagnostic history remain unchanged.
