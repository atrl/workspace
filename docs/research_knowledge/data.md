# Data Knowledge

## Retained Insight

The harness needs one registry-backed path for raw symbols, external volatility inputs, processed market panels, qlib binary stores, and training caches.

## Failed Pattern

Scattered paths such as `data/generalization`, `data/multi_asset`, root-level profile pickles, and ad-hoc qlib stores make refresh state hard to audit and encourage scripts to duplicate path logic.

## Use With Skills

- `qlib-rl-trading` for registry-backed data preparation and qlib dataset refresh.

## Evidence Path

- `data/datasets.yaml`
- `data/README.md`
- `scripts/update_data_registry.py`
- `src/soxl_quant/core/dataset.py`

## Reusable Method

Update the dataset registry first, then refresh raw CSVs, qlib stores, and training caches through registry-aware helpers. Treat generated qlib stores and caches as rebuildable unless a registry row says otherwise.

## Next Valid Experiment Type

Before new cross-ETF testing, verify each ETF has a raw symbol CSV, a processed market panel with its underlying and volatility fields, and a qlib store recorded in `data/datasets.yaml`.
