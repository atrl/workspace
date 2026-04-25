---
title: balanced
direction: model
status: candidate
---

# Balanced

Canonical balanced baseline for SOXL.

- Purpose: current reference actor and reference comparison baseline
- Layer ownership: reward + position cap + promotion gate + default execution behavior
- Compare against this before introducing any new candidate profile

## Expectations

# Expectations

- Remains the canonical fixed-split comparison baseline
- Keeps actor retention enabled
- Any successor must beat it with explicit evidence on:
  - `test_2022 max_dd`
  - `test_2022 avg_pos`
  - `test_proxy sharpe`
  - walk-forward sanity
