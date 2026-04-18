# Experiments

This directory is the preferred home for active profile work.

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

The loader currently supports both:

- `experiments/<status>/<name>/profile.yaml`
- legacy `configs/soxl/<name>.yaml`

During migration, the experiments path takes precedence when both exist.
