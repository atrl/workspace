import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.ibkr_tiered import TieredFees
from src.backtest.stage1_multiclock import economic_choice


def test_economic_previews_keep_original_monthly_state_and_hold_pays_no_fee():
    tariff = json.loads((Path(__file__).resolve().parents[1] / "research/ibkr_tiered_minute_diagnosis_20261013/plan.json").read_text())["tariff"]
    fees = TieredFees(tariff, slippage_bps=2); fees.monthly_shares["2023-01"] = 299950
    q, prices = np.array([0.,100.]), np.array([20.,100.])
    choices = [np.array([.2,.3,.5]), np.array([0.,.85,.15])]
    hold, held_fees = economic_choice(q, 10000., prices, prices, np.zeros(2), choices, fees=fees, stamp="2023-01-04")
    assert hold["orders"] == [] and hold["cash"] == 10000.
    bought, used_fees = economic_choice(q, 10000., prices, prices, np.array([.02,0.]), choices, fees=fees, stamp="2023-01-04")
    assert len(bought["orders"]) == 2 and used_fees.monthly_shares["2023-01"] > 299950
    assert fees.monthly_shares == held_fees.monthly_shares == {"2023-01":299950}
    np.testing.assert_array_equal(q,[0.,100.])


def test_illegal_hold_is_not_an_economic_option_even_when_its_forecast_is_best():
    tariff = json.loads((Path(__file__).resolve().parents[1] / "research/ibkr_tiered_minute_diagnosis_20261013/plan.json").read_text())["tariff"]
    result,_ = economic_choice(np.array([500.,0.]),10000.,np.array([20.,100.]),np.array([20.,100.]),np.array([.10,0.]),
        [np.array([.3,.2,.5]),np.array([0.,.5,.5])],fees=TieredFees(tariff,slippage_bps=2),stamp="2023-01-04",allow_hold=False)
    assert any(order["sell"] and order["symbol"] == "SOXL" for order in result["orders"])
    assert result["quantities"][0] == 300.


def test_native_whole_raw_sale_preserves_positive_adjusted_residual(tmp_path):
    from src.backtest.ibkr_tiered import settle_tiered_event
    from src.backtest.stage1_multiclock import native_multiclock
    from src.data.qlib_native import build_qlib_store_from_frame
    tariff = json.loads((Path(__file__).resolve().parents[1] / "research/ibkr_tiered_minute_diagnosis_20261013/plan.json").read_text())["tariff"]
    stamps = pd.to_datetime(["2023-01-04 09:45", "2023-01-04 10:00", "2023-01-05 09:45"])
    values = np.asarray([[100.,200.],[100.,200.],[100.,200.]], dtype=np.float32).astype(float)
    raw_values = values.copy(); raw_values[1,0] = float(np.float32(99.99998))
    bars = []
    for i, stamp in enumerate(stamps):
        for j, symbol in enumerate(("SOXL","SOXX")):
            price=values[i,j]
            bars.append({"datetime":stamp,"instrument":symbol,"open":price,"high":price,"low":price,"close":price,"volume":100000.,"raw_execution_price":raw_values[i,j]})
    provider=tmp_path/'quotes'
    build_qlib_store_from_frame(pd.DataFrame(bars),provider_uri=provider,freq='15min',as_of='2023-01-05',source='native NAV precision regression',field_map={name:name for name in ['open','high','low','close','volume','raw_execution_price']})
    fees=TieredFees(tariff,slippage_bps=2)
    bought=settle_tiered_event([0.,0.],100000.,desired=[.3,0.,.7],adjusted_prices=values[0],raw_prices=raw_values[0],stamp=stamps[0],fees=fees)
    sold=settle_tiered_event(bought['quantities'],bought['cash'],desired=[0.,0.,1.],adjusted_prices=values[1],raw_prices=raw_values[1],stamp=stamps[1],fees=fees)
    assert 0 < sold['quantities'][0] < .001
    nav=np.array([bought['quantities']@values[0]+bought['cash'],sold['quantities']@values[1]+sold['cash']])
    ledger=pd.DataFrame({'nav':nav,'return':nav/np.r_[100000.,nav[:-1]]-1,'cash':[bought['cash'],sold['cash']],
        'soxl_quantity':[bought['quantities'][0],sold['quantities'][0]],'soxx_quantity':0.},index=stamps[:2])
    book=native_multiclock({'ledger':ledger,'orders':pd.DataFrame(bought['orders']+sold['orders'])},provider_uri=provider,plan={'tariff':tariff,'initial_cash':100000},slip=2)
    assert book['audit']['nav_relative_error'] < 1e-12
