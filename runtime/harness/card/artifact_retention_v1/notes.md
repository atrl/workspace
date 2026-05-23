# Artifact Retention Manifest v1

Date: 2026-05-23

## Goal

Clean historical model ambiguity without deleting checkpoints blindly.

## Changes

- `scripts/prune_artifacts.py` now writes a runtime-owned retention manifest under
  `runtime/harness/card/artifact_retention_v1/retention_manifest.json`.
- Runtime actor paths are resolved from current `runtime/*/card/<profile>/`
  directories and registry rows, including walk-forward suffixes and retained
  checkpoint suffixes.
- Summary JSON files are included in the manifest so actor retention and summary
  retention are visible in one audit surface.

## Manifest Result

```text
keep=131
delete=12
```

Delete candidates are actor checkpoints or generated noise only:

```text
runtime/model/card/balanced_rule_residual_mode_v1/soxl_balanced_rule_residual_mode_v1_actor.pt
runtime/model/card/balanced_rule_residual_mode_v1/soxl_balanced_rule_residual_mode_v1_wf_2020_actor.pt
runtime/model/card/balanced_rule_residual_mode_v1/soxl_balanced_rule_residual_mode_v1_wf_2022_actor.pt
runtime/model/card/balanced_rule_residual_mode_v1/soxl_balanced_rule_residual_mode_v1_wf_2023_2024_actor.pt
runtime/model/card/balanced_rule_residual_v19_lowcap/soxl_balanced_rule_residual_v19_lowcap_actor.pt
runtime/model/card/balanced_rule_residual_v19_small/soxl_balanced_rule_residual_v19_small_actor.pt
runtime/model/card/balanced_rule_residual_v19_small_wf_2020/soxl_balanced_rule_residual_v19_small_wf_2020_actor.pt
runtime/model/card/balanced_rule_residual_v19_small_wf_2022/soxl_balanced_rule_residual_v19_small_wf_2022_actor.pt
runtime/model/card/balanced_rule_residual_v19_small_wf_2023_2024/soxl_balanced_rule_residual_v19_small_wf_2023_2024_actor.pt
runtime/model/card/balanced_rule_residual_v22_release_v1/soxl_balanced_rule_residual_v22_release_v1_actor.pt
runtime/model/card/risk_budget_usage_v1/soxl_risk_budget_usage_v1_actor.pt
src/soxl_quant/__pycache__
```

## Decision

Do not apply deletion automatically in this loop. The user asked to clean
history, but checkpoint deletion is destructive; this card and manifest define
the safe delete set first.

## Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_prune_artifacts.py -q
# 3 passed

PYTHONPATH=. qlib_venv/bin/python scripts/prune_artifacts.py
# [manifest] keep=131 delete=12

MPLCONFIGDIR=/private/tmp/mplconfig PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# [summary] passed=203 failures=0
```
