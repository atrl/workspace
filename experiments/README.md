# Experiments

This directory is the preferred home for active profile work.

The intent is to keep model configuration and experiment narrative together, so a reader can answer:

- what profile is this
- why does it exist
- what should it beat or protect against

Structure:

- `canonical/<name>/`
  - long-lived baseline profiles
- `candidate/<name>/`
  - current diagnostic or promotion candidates
- `archive/<name>/`
  - optional future home for retired experiment packages

Each experiment package should keep these files together:

- `profile.yaml`
- `notes.md`
- `expectations.md`

Read order inside one experiment package:

1. `profile.yaml`
2. `expectations.md`
3. `notes.md`

The loader currently supports both:

- `experiments/<status>/<name>/profile.yaml`
- legacy `configs/soxl/<name>.yaml`

During migration, the experiments path takes precedence when both exist.

If multiple experiment packages seem to say the same thing, the canonical package wins over candidate packages; candidate packages should only capture the delta from canonical behavior.
