# Experiments

`experiments/` stores concise runnable experiment cards. It is not a result-log archive and should not duplicate `runtime/` outputs or `memory/` narratives.

## Structure

- `canonical/<name>/` — long-lived baseline profiles.
- `candidate/<name>/` — active promotion or diagnostic candidates only.
- `archive/<name>/` — inactive historical packages kept for retrieval.
- `index.md` — compact table linking packages to registry status and evidence.

## Package Contract

Active `canonical/` and `candidate/` packages must contain:

- `profile.yaml` — runnable config.
- `notes.md` — short experiment card: motivation, baseline, mechanism delta, evidence links.
- `expectations.md` — acceptance criteria, expected trade-off, failure mode, comparison row key.

Long result writeups, keep/drop decisions, and historical comparisons belong in:

- `memory/`
- `runtime/post_train_review/`
- `runtime/generalization/`
- `docs/experiment-landscape-review.md`
- `docs/research_knowledge/`

## Read Order

1. `experiments/index.md`
2. `configs/soxl/profile_registry.yaml`
3. package `profile.yaml`
4. package `expectations.md`
5. package `notes.md`
6. linked `memory/` or `runtime/` evidence

## Validation

Run `PYTHONPATH=. qlib_venv/bin/python scripts/check_experiment_cards.py` after adding, moving, or retiring packages.
