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

## Decision

No Futu small-order data was fetched in this loop. The blocker is external OpenD
UI verification, not SOXL repo data logic:

- OpenD is installed.
- OpenD port `127.0.0.1:11111` is reachable.
- The Python SDK imports successfully.
- The quote context cannot initialize until the Futu OpenD graphical
  verification/login step is completed.

Do not create a small-order factor pack or start training until a successful
quote-context probe writes actual `US.SOXL` / `US.SOXX` capital flow rows.

## Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_futu_adapter.py -q
# 7 passed

PYTHONPATH=. qlib_venv/bin/python scripts/update_data_registry.py --check
# update_data_registry: OK symbols=6 panels=10 qlib=6

PYTHONPATH=. qlib_venv/bin/python scripts/check_research_knowledge.py
# check_research_knowledge: OK files=7 active_budget=300

MPLCONFIGDIR=/private/tmp/mplconfig PYTHONPATH=. qlib_venv/bin/python scripts/check_harness.py
# [summary] passed=203 failures=0
```

## Next Probe After Verification

After completing Futu OpenD graphical verification in the local UI, retry:

```bash
PYTHONPATH=. qlib_venv/bin/python - <<'PY'
from futu import *
quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
for code in ["US.SOXL", "US.SOXX"]:
    ret, data = quote_ctx.get_capital_flow(
        code,
        period_type=PeriodType.DAY,
        start="2026-01-01",
        end="2026-05-22",
    )
    print(code, ret, len(data) if ret == RET_OK else data)
quote_ctx.close()
PY
```

If this returns rows, build a lagged factor pack around:

- small net flow z-score
- small flow share
- small-vs-big divergence
- stress/recovery-gated small-flow capitulation
