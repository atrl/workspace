# Archived Experiment Packages

This directory holds frozen experiment packages that are no longer part of the
active `candidate/` set.

Use archive packages when you need:

- the runnable profile config that produced an old result
- the original experiment card / expectation framing
- a short pointer to the evidence trail

Do not use archive packages as the primary result database.

For historical retrieval, prefer this order:

1. `docs/experiment-landscape-review.md`
2. `configs/soxl/profile_registry.yaml`
3. `memory/index.json` + `scripts/memory_search.py`
4. the relevant archive package
5. `evidence/post_train_review/` for pairwise comparisons

Archive packages are intentionally allowed to be less polished than active
candidate packages, but they should still keep:

- `profile.yaml`
- `notes.md`
- `expectations.md` when it existed historically

If an archive note duplicates a full result writeup that already exists in
`memory/`, prefer shortening the note rather than rewriting the result in two
places.
