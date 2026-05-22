# Futu Simulated Trading Src Adapter And Startup

Date: 2026-05-22

## Goal

- Move the repo-owned Futu simulated-trading broker logic under `src/`.
- Keep production scripts as thin entrypoints.
- Start Futu OpenD and verify whether the Futu broker path is order-ready.

## Change

- `src/soxl_quant/app/futu.py` now owns:
  - `FutuBroker`
  - `check_opend_port`
  - SDK connection timeout handling
  - US security trade context selection via `OpenSecTradeContext(filter_trdmarket=TrdMarket.US)`
- `production/futu_broker.py` is now a compatibility import only.
- Futu production entrypoints import the adapter from `src.soxl_quant.app.futu`.
- `src.soxl_quant` and `src.soxl_quant.app` use lazy `main` imports so importing the Futu adapter does not initialize the full runtime stack.

## Startup Evidence

OpenD app:

```text
/Users/rziiiii/futu_opend/Futu_OpenD_10.2.6208_Mac/Futu_OpenD_10.2.6208_Mac/FutuOpenD.app
```

OpenD port check:

```bash
PYTHONPATH=. qlib_venv/bin/python production/check_futu_opend.py
# OK: OpenD is reachable at 127.0.0.1:11111
```

Futu broker healthcheck:

```bash
PYTHONPATH=. qlib_venv/bin/python production/soxl_paper_trading.py \
  --broker futu --healthcheck \
  --runtime-composer budget_usage \
  --composer-profile v19reg_v22_budget_usage_bull_guard_v1 \
  --slow-profile balanced_rule_residual_v19_reg \
  --fast-profile balanced_rule_residual_v22 \
  --bull-participation-guard \
  --no-notify
```

Result:

```json
{
  "data": {
    "ok": true,
    "last_date": "2026-05-21",
    "expected_last_trading_day": "2026-05-21",
    "stale_days": 0
  },
  "model": {
    "ok": true,
    "name": "v19reg_v22_budget_usage_bull_guard_v1 (slow=balanced_rule_residual_v19_reg, fast=balanced_rule_residual_v22)"
  },
  "broker": {
    "ok": false,
    "msg": "Futu OpenD SDK connection timed out"
  },
  "all_ok": false
}
```

OpenD SDK log emitted `需要图形验证码`, so the blocker is OpenD account/API verification, not missing local data or missing model configuration.

## Verification

```bash
uv run --python qlib_venv/bin/python --with pytest pytest tests/test_futu_adapter.py tests/test_paper_trading.py -q
# 25 passed
```

## Decision

- Futu simulated-trading code ownership is now centered under `src/`.
- OpenD is started and reachable on `127.0.0.1:11111`.
- The broker is not yet order-ready because OpenD requires graphical verification/login completion.
- No Futu order submission was attempted.

## Next Action

Complete the OpenD graphical verification/login/API questionnaire in the Futu OpenD UI, then rerun the Futu healthcheck above. Only after `broker.ok=true` should the repo run a non-dry Futu simulated-trading submission.
