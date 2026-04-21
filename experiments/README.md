# Experiments

This directory is the preferred home for experiment packages, but not for every
historical run narrative.

The intent is to keep model configuration and experiment-card intent together,
so a reader can answer:

- what profile is this
- why does it exist
- what should it beat or protect against

Structure:

- `canonical/<name>/`
  - long-lived baseline profiles
- `candidate/<name>/`
  - current diagnostic or promotion candidates only
- `archive/<name>/`
  - frozen historical experiment packages that are no longer active candidates

Each experiment package should keep these files together:

- `profile.yaml`
- `notes.md`
- `expectations.md`

Package responsibilities are intentionally narrow:

- `profile.yaml`
  - runnable config
- `expectations.md`
  - acceptance bar, intended deltas, success / failure conditions
- `notes.md`
  - short experiment card:
    - motivation
    - baseline
    - mechanism delta
    - links to result evidence

Do not treat `notes.md` as the full run log once results exist elsewhere.
Long result writeups, keep / drop decisions, and historical comparisons belong in:

- `memory/`
- `runtime/post_train_review/`
- `docs/experiment-landscape-review.md`

Read order inside one experiment package:

1. `profile.yaml`
2. `expectations.md`
3. `notes.md`

The loader currently supports both:

- `experiments/<status>/<name>/profile.yaml`
- legacy `configs/soxl/<name>.yaml`

During migration, the experiments path takes precedence when both exist.

If multiple experiment packages seem to say the same thing, the canonical package wins over candidate packages; candidate packages should only capture the delta from canonical behavior.

## Active vs Archived

The `candidate/` directory should stay narrow. If a profile is no longer an
active promotion or diagnostic branch, move its package to `archive/` instead of
leaving it in `candidate/`.

The registry remains the lifecycle source of truth. A package living under
`archive/` does not imply the profile's registry status must literally be
`archived`; it only means the package is no longer part of the active
candidate set.

## Retrieval Rule

Use this read order when revisiting old work:

1. `docs/experiment-landscape-review.md`
2. `configs/soxl/profile_registry.yaml`
3. the relevant experiment package in `candidate/` or `archive/`
4. `memory/index.json` + `scripts/memory_search.py`
5. `runtime/post_train_review/` for pairwise diffs

If both `experiments/` and `memory/` mention the same run, keep the experiment
package short and let `memory/` hold the result narrative.
