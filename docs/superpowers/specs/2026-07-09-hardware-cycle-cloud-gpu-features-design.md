# Hardware Cycle Cloud GPU Features Design

## Goal

Add reproducible hardware-cycle and cloud-GPU pricing features to the SOXL qlib research workflow, then test whether they improve the probability-budget position model without reintroducing overfit template logic.

## Scope

The feature set has two layers:

1. Long-history market proxies from public daily prices, covering the full 2019-01-02 through current SOXL qlib window:
   - AI accelerator chain: NVDA, AMD, AVGO, TSM, ASML.
   - Memory and storage chain: MU, WDC, STX.
   - Cloud capex buyers: AMZN, MSFT, GOOGL, ORCL, META.
   - AI server exposure: SMCI, DELL.
2. Direct cloud GPU rental price indices:
   - AIMultiple monthly cloud GPU index series, including generation-level and GPU-model-level prices.
   - ComputePrices public trend/current endpoints may be fetched for diagnostics, but only dated rows at or before the qlib `as_of` can be admitted into the provider.
   - Vast host market metrics require a host API key, so they are not a default dependency.

PricePerGig product-level storage data is excluded from training because its API terms restrict ML use and long caching. It can remain a manual research reference, not a qlib training source.

## Architecture

`src/data/hardware_cycle.py` will fetch and normalize source data, write a raw snapshot under `data/raw/external/hardware_cycle/`, build `data/qlib/hardware_cycle_v1` with `build_qlib_store_from_frame`, and audit the provider via qlib `D.features`.

`src/factor/hardware_cycle.py` will read the qlib provider and produce lagged open-to-open-safe features. Public-market proxy features use 5/20/60-day returns and 90-day z-scores. Cloud GPU price features use monthly/21-day changes, z-scores, and availability masks. All features are shifted by at least one trading day before joining the SOXL model frame.

`src/workflow/hardware_cycle_probability_budget_open2open.py` will create a new research card rather than rewriting `probability_budget_open2open_v1`. It will reuse the current LGB/XGB consensus probability-budget position model, add a `base_cogalpha_soxx_hardware` feature group, and keep positions plus walk-forward positions recorder-only.

## Data Semantics

The raw data builder must fail closed if any row exceeds the SOXL qlib `as_of`. Cloud GPU monthly values are not backfilled before their first observed snapshot. Earlier rows receive neutral return/z-score values plus an explicit availability flag, so the model can distinguish missing direct cloud-price history from true price stability.

The direct cloud GPU index has shorter and potentially revised history. It is therefore labeled as an external reconstructed index. A model can only be promoted if walk-forward gates pass and the conclusion states this limitation.

## Testing

Focused tests will cover:

- AIMultiple chart extraction from embedded page data.
- Hardware-cycle qlib provider build and `D.features` readback.
- Lagged hardware-cycle factor generation with no same-day leakage.
- New workflow feature-set selection and summary provenance.

Notebook execution will be required for the new research card before committing final artifacts.
