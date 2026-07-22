# Failure modes

- Actual SOXL history is short relative to the paper; the primary horizon is three years rather than ten.
- Cash has zero return because PIT T-bill/T-bond data is not present in the current qlib store.
- SOXL/SOXX is a concentrated semiconductor mapping, not the paper's broad-market LETF/VETF setting.
- Bootstrap paths reuse a limited return history and cannot create unseen structural regimes.
- The 2023-2026 outer path was already observed before this architecture was designed and is diagnostic only.
