# Futu Sim Order Probe v1

Date: 2026-05-24

## Goal

Verify the Futu simulated-trading account and order channel directly through
OpenD, without using IBKR and without placing a live order.

## Safety Boundary

- Trading environment: `TrdEnv.SIMULATE` only.
- Symbol: `US.SOXL`.
- Test order: `BUY 1` normal limit order.
- Limit price was set below the live snapshot price and the order was
  immediately cancelled after acceptance.
- No full account id or full order id is recorded in this evidence note.

## Evidence

OpenD was reachable:

```bash
PYTHONPATH=. qlib_venv/bin/python production/check_futu_opend.py
# OK: OpenD is reachable at 127.0.0.1:11111
```

Futu SDK context used:

```text
OpenSecTradeContext(filter_trdmarket=TrdMarket.US, host=127.0.0.1, port=11111)
trd_env=SIMULATE
```

Account readout from `accinfo_query(trd_env=SIMULATE, refresh_cache=True)`:

```text
cash=1000000.0
total_assets=1000000.0
currency=N/A
```

Pre-trade simulated position:

```text
position_list_query(code=US.SOXL, trd_env=SIMULATE): []
```

Market snapshot and submitted test order:

```text
snapshot last_price=190.56
test_limit_price=181.03
open_order_count_before=0

place_order:
  ret=OK
  code=US.SOXL
  side=BUY
  order_status=SUBMITTING
  qty=1
  price=181.03
  dealt_qty=0
  dealt_avg_price=0
```

Cancellation and final state:

```text
modify_order(CANCEL): ret=OK
final_order_status=CANCELLED_ALL
final_dealt_qty=0
final_dealt_avg_price=0
position_after=[]
```

Follow-up active-order check:

```bash
PYTHONPATH=. qlib_venv/bin/python - <<'PY'
# Query US.SOXL simulated orders for 2026-05-24 and count active statuses.
PY
# ret 0
# active_order_count 0
```

## Decision

- The Futu simulated account is readable through OpenD.
- The Futu simulated order channel can accept an order and process a cancel.
- The probe left no `US.SOXL` position and no active simulated order.
- The next production proof should come from the scheduled non-dry Futu
  post-close LaunchAgent path, not from IBKR.
