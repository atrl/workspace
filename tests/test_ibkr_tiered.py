import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.ibkr_tiered import TieredFees, settle_tiered_event
from src.backtest.stage1_ibkr import native_tiered_stage1
from src.data.qlib_native import build_qlib_store_from_frame


def tariff():
    card = Path(__file__).resolve().parents[1] / "research/ibkr_tiered_minute_diagnosis_20261013/fee_schedule.json"
    return json.loads(card.read_text())["tariff"]


def test_per_order_minimum_cap_sell_regulatory_fee_and_exact_hold():
    fees = TieredFees(tariff(), slippage_bps=0)
    buy = fees.quote(shares=200, raw_price=50., sell=False, stamp="2023-01-04")
    sell = fees.quote(shares=200, raw_price=50., sell=True, stamp="2023-01-04")
    assert np.isclose(buy["total"], 1.3411145)
    assert np.isclose(sell["total"] - buy["total"], .206)
    capped = fees.quote(shares=10, raw_price=.20, sell=False, stamp="2023-01-04")
    assert capped["brokerage"] == .02 and capped["cap_applied"]
    hold = settle_tiered_event([2., 3.], 100., desired=None, adjusted_prices=[10., 20.],
        raw_prices=[31., 61.], stamp="2023-01-04", fees=fees)
    assert hold["orders"] == [] and hold["cash"] == 100.
    assert fees.monthly_shares == {}


def test_monthly_marginal_tier_and_capped_orders_volume_exclusion():
    fees = TieredFees(tariff(), slippage_bps=0)
    fees.monthly_shares["2023-01"] = 299990
    quoted = fees.quote(shares=1000, raw_price=50., sell=False, stamp="2023-01-04")
    assert np.isclose(quoted["brokerage"], 10 * .0035 + 990 * .002)
    assert fees.monthly_shares["2023-01"] == 299990  # quoting is read-only
    fees.commit(1000, stamp="2023-01-04", cap_applied=quoted["cap_applied"])
    assert fees.monthly_shares["2023-01"] == 300990
    fees.commit(10, stamp="2023-01-04", cap_applied=True)
    assert fees.monthly_shares["2023-01"] == 300990
    assert np.isclose(fees.quote(shares=1000, raw_price=50., sell=False, stamp="2023-02-01")["brokerage"], 3.5)


def test_native_fees_use_raw_shares_with_affine_adjusted_nav_units(tmp_path):
    days = pd.to_datetime(["2023-01-04", "2023-01-05", "2023-01-06"])
    prices = {"SOXL": [100., 103., 102.], "SOXX": [200., 201., 199.]}
    bars = []
    for symbol, values in prices.items():
        for day, price in zip(days, values):
            for stamp in pd.date_range(day + pd.Timedelta(hours=9, minutes=45), day + pd.Timedelta(hours=16), freq="15min"):
                bars.append({"instrument": symbol, "datetime": stamp, "open": price, "high": price,
                    "low": price, "close": price, "volume": 100000., "execution_price": price,
                    "raw_execution_price": 3. * price + 7.})
    provider = tmp_path / "quotes"
    fields = ["open", "high", "low", "close", "volume", "execution_price", "raw_execution_price"]
    build_qlib_store_from_frame(pd.DataFrame(bars), provider_uri=provider, freq="15min", as_of="2023-01-06",
        field_map={name: name for name in fields}, source="deterministic fee-contract test quotes")
    cash, quantities, rows, orders = 100000., np.zeros(2), [], []
    fees = TieredFees(tariff(), slippage_bps=2.)
    frame = pd.DataFrame({"signal_date": pd.to_datetime(["2023-01-03", "2023-01-04"]),
        "trade_date": days[:2], "exit_date": days[1:], "soxl_exit_open": prices["SOXL"][1:],
        "soxx_exit_open": prices["SOXX"][1:]})
    for i, row in frame.iterrows():
        opening = np.array([prices[s][i] for s in ("SOXL", "SOXX")])
        before = float(quantities @ opening + cash)
        event = settle_tiered_event(quantities, cash, desired=np.array([.3, .5, .2]) if i == 0 else None,
            adjusted_prices=opening, raw_prices=3. * opening + 7., stamp=row.trade_date + pd.Timedelta(hours=9, minutes=45), fees=fees)
        quantities, cash = event["quantities"], event["cash"]
        orders += [{**o, "trade_date": row.trade_date, "slot": "open"} for o in event["orders"]]
        exiting = row[["soxl_exit_open", "soxx_exit_open"]].to_numpy(float)
        nav = float(quantities @ exiting + cash)
        weights = np.r_[quantities * exiting, cash] / nav
        rows.append({**row.to_dict(), "strategy_return": nav / before - 1., "native_expected_nav": nav,
            "ending_soxl_quantity": quantities[0], "ending_soxx_quantity": quantities[1], "ending_cash": cash,
            "ending_soxl_weight": weights[0], "ending_soxx_weight": weights[1], "ending_cash_weight": weights[2]})
    book = native_tiered_stage1(frame, {"replay": pd.DataFrame(rows), "orders": pd.DataFrame(orders)},
        provider_uri=provider, tariff=tariff(), slippage_bps=2., initial_cash=100000.)
    assert book["audit"]["status"] == "passed" and len(book["native_orders"]) == 2
    assert book["native_orders"].raw_shares.tolist() == [97, 82]
    assert max(book["audit"]["errors"].values()) < 1e-9
