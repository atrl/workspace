# Checkpoint Retention Harness Support

Date: 2026-05-21

## Context

The v22 checkpoint asset audit showed that the current canonical v22 summary
records only iter `0` and iter `80`, and the repo retains only the final actor.
That blocks a no-retrain checkpoint-selection loop because intermediate actors
cannot be replayed.

## Change

Added profile-controlled checkpoint retention:

```yaml
checkpoint_retention:
  enabled: true
  keep_stage: true
  keep_top_n: 2
```

Defaults remain disabled, so existing profiles and training runs keep their
current artifact behavior unless a new governed experiment explicitly opts in.

Implementation boundaries:

- `src/soxl_quant/config.py` owns `CheckpointRetentionConfig`.
- `src/soxl_quant/app/selection.py` selects retained stage/top-N checkpoints
  from evaluated actors.
- `src/soxl_quant/app/artifacts.py` writes retained actor bundles and records
  summary metadata without mutating the live selected actor.
- `src/soxl_quant/app/train.py` connects selection output to persisted actor
  bundles and summary `retained_checkpoints`.

## Evidence

Focused tests:

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_checkpoint_retention.py -q
# 6 passed
```

The tests cover:

- default disabled config behavior
- top-N selection without stage spam
- stage/top-N union reason labels
- actor bundle save from an override `state_dict` without mutating actor state
- summary `retained_checkpoints` metadata
- retained checkpoint actor bundle path records

## Decision

This is a harness capability, not a model promotion. The next governed model
experiment can opt in to retain stage/top-N actors, making future checkpoint
reviews replayable instead of summary-only.
