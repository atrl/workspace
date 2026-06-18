---
name: futu-api
description: Use only for SOXL/SOXX data intake through Futu OpenAPI, not for production trading, scheduling, dashboarding, or broker execution.
---

# Futu API Data Intake

Use this skill only when a SOXL experiment needs Futu-sourced market data or a
read-only probe to refresh `data/raw/` inputs.

## Boundaries

- Keep Futu outputs under `data/raw/external/` or a card-local runtime artifact.
- Do not add production execution, scheduler, broker-order, or dashboard code.
- Treat Futu small-order flow as short-history/live overlay data unless a
  full-history source is explicitly proven.
- Keep tokens and login state outside the repository.

## Default Checks

```bash
PYTHONPATH=. qlib_venv/bin/python - <<'PY'
import futu
print(futu.__version__)
PY
```

If OpenD is unavailable, record `needs data` in the notebook conclusion instead
of changing framework code.
