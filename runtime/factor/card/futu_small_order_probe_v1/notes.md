# Futu Small Order Flow Probe v1

Date: 2026-05-23

## Goal

Probe whether Futu OpenAPI can provide small-order flow data for `US.SOXL` and
`US.SOXX` as a candidate factor source.

## External Reference

Futu OpenAPI exposes:

- `get_capital_flow(stock_code, period_type, start, end)`, including
  `sml_in_flow`, `mid_in_flow`, `big_in_flow`, and `super_in_flow`.
- `get_capital_distribution(stock_code)`, including `capital_in_small` and
  `capital_out_small`.

This is a closer fit for SOXL/SOXX than Tushare A-share `moneyflow`, because the
target symbols are US ETFs.

## Probe

Environment:

```bash
qlib_venv/bin/python - <<'PY'
import futu
print(getattr(futu, "__version__", "unknown"))
print(hasattr(futu, "OpenQuoteContext"), hasattr(futu, "PeriodType"))
PY
# 10.02.6218
# True True
```

OpenD state:

```bash
open /Users/rziiiii/futu_opend/Futu_OpenD_10.2.6208_Mac/Futu_OpenD_10.2.6208_Mac/FutuOpenD.app

PYTHONPATH=. qlib_venv/bin/python production/check_futu_opend.py
# OK: OpenD is reachable at 127.0.0.1:11111
```

Quote SDK probe:

```bash
PYTHONPATH=. qlib_venv/bin/python - <<'PY'
from futu import *
quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
ret, data = quote_ctx.get_capital_flow("US.SOXL", period_type=PeriodType.INTRADAY)
print(ret, data)
quote_ctx.close()
PY
# init connect fail ... msg=需要图形验证码
# TimeoutError: timeout
```

Three-minute retry loop:

```text
probe attempts: 18
result: every attempt failed during OpenQuoteContext initialization
message: 需要图形验证码
```

## Successful Retry After Phone Verification

After completing the local Futu phone/UI verification, the same OpenD instance
allowed quote-context access and returned SOXL/SOXX capital-flow rows.

Command shape:

```bash
PYTHONPATH=. qlib_venv/bin/python - <<'PY'
from futu import OpenQuoteContext, PeriodType
# connect to 127.0.0.1:11111 and fetch:
# - get_market_snapshot(["US.SOXL", "US.SOXX"])
# - get_capital_distribution("US.SOXL" / "US.SOXX")
# - get_capital_flow(..., PeriodType.INTRADAY)
# - get_capital_flow(..., PeriodType.DAY, start="2024-01-01", end="2026-05-22")
PY
```

Cached files:

| File | Rows | Start | End |
|------|------|-------|-----|
| `data/raw/external/futu_snapshot_US_SOXL.csv` | 1 | 2026-05-22 | 2026-05-22 |
| `data/raw/external/futu_snapshot_US_SOXX.csv` | 1 | 2026-05-22 | 2026-05-22 |
| `data/raw/external/futu_capital_distribution_US_SOXL.csv` | 1 | 2026-05-22 | 2026-05-22 |
| `data/raw/external/futu_capital_distribution_US_SOXX.csv` | 1 | 2026-05-22 | 2026-05-22 |
| `data/raw/external/futu_capital_flow_intraday_US_SOXL.csv` | 391 | 2026-05-22 | 2026-05-22 |
| `data/raw/external/futu_capital_flow_intraday_US_SOXX.csv` | 391 | 2026-05-22 | 2026-05-22 |
| `data/raw/external/futu_capital_flow_day_US_SOXL.csv` | 251 | 2025-05-23 | 2026-05-22 |
| `data/raw/external/futu_capital_flow_day_US_SOXX.csv` | 251 | 2025-05-23 | 2026-05-22 |

Latest daily readout on `2026-05-22`:

| Symbol | `in_flow` | `sml_in_flow` | `main_in_flow` | Distribution small in/out |
|--------|-----------|---------------|----------------|---------------------------|
| SOXL | +389.4M | +189.8M | +106.1M | 1048.0M / 858.2M |
| SOXX | -76.6M | -93.5M | +14.7M | 307.2M / 400.6M |

The SOXL/SOXX small-flow divergence on the latest day is large enough to justify
running a factor-gate evaluation, but it is not evidence of predictive value by
itself.

## Decision

Futu is now a viable SOXL/SOXX small-order-flow data source for research use:

- OpenD is installed and reachable at `127.0.0.1:11111`.
- `futu-api=10.02.6218` can initialize `OpenQuoteContext` after phone/UI
  verification.
- Daily capital-flow history is available for roughly one year in this probe.
- Intraday capital flow and current-day capital distribution are available for
  the latest closed US session.

Do not train yet. The next step is to build a lagged Futu factor pack and run
the normal IC/backtest/correlation gate against this cached source.

## Verification

```bash
PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py
# update_data_registry: OK symbols=6 panels=10 qlib=6

uv run --python qlib_venv/bin/python --with pytest pytest tests/test_futu_adapter.py -q
# 7 passed

PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py --check
# update_data_registry: OK symbols=6 panels=10 qlib=6

PYTHONPATH=. qlib_venv/bin/python scripts/check_research_knowledge.py
# check_research_knowledge: OK files=7 active_budget=300

MPLCONFIGDIR=/private/tmp/mplconfig PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# [summary] passed=203 failures=0
```

## Next Factor Gate

Build a lagged factor pack around:

- small net flow z-score
- small flow share
- small-vs-big divergence
- stress/recovery-gated small-flow capitulation

Promotion remains blocked until the factor pack passes the repo gates:
`mean_ic > 0.03`, `ic_positive_ratio > 0.55`, `signal_accuracy > 0.55`,
`profit_factor > 1.2`, and max core correlation `<= 0.80`.
