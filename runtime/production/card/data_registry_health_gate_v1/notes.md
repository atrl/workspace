# Data Registry Health Gate

Date: 2026-05-21

## Goal

Make the production scheduler health check surface `data/datasets.yaml`
freshness directly, instead of relying only on paper trading data freshness.

## Gap

The scheduled data path now refreshes `data/datasets.yaml`, and
`scripts/update_data_registry.py --check` can detect stale stored metadata.
However, `production/check_scheduler_health.sh` did not run that checker, so an
operator could see paper data as fresh while the registry surface was stale.

## Change

- Added a `Data Registry Health` section to
  `production/check_scheduler_health.sh`.
- The section runs:

```bash
qlib_venv/bin/python scripts/update_data_registry.py --check
```

- Added `data registry` to the `Production Goal Gates` summary.
- A failed or unavailable registry check now adds
  `data_registry_freshness` to the overall pending issue list.

## Evidence

```bash
uv run --python qlib_venv/bin/python --with pytest pytest \
  tests/test_scheduler_health.py::test_scheduler_health_reports_stale_data_registry_check -q
# red before implementation: missing Data Registry Health section
# green after implementation: 1 passed

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_scheduler_health.py -q
# 26 passed

production/check_scheduler_health.sh
# Data Registry Health: update_data_registry: OK symbols=6 panels=10 qlib=6
# Production Goal Gates: data registry: ok update_data_registry: OK symbols=6 panels=10 qlib=6
# overall remains pending external_vercel_setup,scheduled_postclose_target,scheduled_paper_proof
```

## Decision

Data registry freshness is now an explicit production health gate. This does
not clear the broader production goal; Vercel credentials/project setup and the
next natural scheduled paper proof windows remain pending.
