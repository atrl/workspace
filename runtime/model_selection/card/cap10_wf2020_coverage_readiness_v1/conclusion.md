# cap10_wf2020_coverage_readiness_v1

- Decision: `wf2020_missing_target_contract_blocks_actor_retention`.
- Current cap10 joint-contract path has target/BC/checkpoint evidence for `wf_2022` and `wf_2023_2024`, but not `wf_2020`.
- `wf_2020` worker split is train `2010-03-11 -> 2018-12-31`, valid `2019-01-01 -> 2019-12-31`, test `2020-01-01 -> 2020-12-31`.
- Current joint target contract begins at `2020-01-02` and the sparse-release source already records: `sparse-release targets begin in 2020, so a wf_2020 warm-start would train on the test year`.
- Actor retention and promotion remain blocked. Do not retain a cap10 actor from this branch until a non-leaking `wf_2020` target/BC/checkpoint sidecar exists or the branch is explicitly replaced.

## Sidecars
- `runtime/model_selection/card/cap10_wf2020_coverage_readiness_v1/source_sidecar_audit.csv`
- `runtime/model_selection/card/cap10_wf2020_coverage_readiness_v1/wf2020_coverage_audit.csv`
- `runtime/model_selection/card/cap10_wf2020_coverage_readiness_v1/next_run_blockers.csv`
- `runtime/model_selection/card/cap10_wf2020_coverage_readiness_v1/summary.json`
- `runtime/model_selection/card/cap10_wf2020_coverage_readiness_v1/conclusion.md`
