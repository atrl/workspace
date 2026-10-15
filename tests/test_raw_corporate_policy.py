from decimal import Decimal
import numpy as np
import pandas as pd
from src.backtest.raw_corporate_policy import IncomeLedger,settle_raw
from src.backtest.native_raw_corporate import RawCashPosition
from src.backtest.ibkr_tiered import TieredFees
import json
from pathlib import Path


def test_receivable_funding_and_native_income_split_payment_once():
    tariff=json.loads(Path('research/frozen_policy_raw_cashbook_20261015/plan.json').read_text())['tariff']
    q,cash,orders=settle_raw([0,0],0.,1000.,desired=[0,1,0],prices=[10,10],stamp='2023-01-04',fees=TieredFees(tariff,slippage_bps=2))
    assert not orders and cash==0 and np.array_equal(q,[0,0])
    position=RawCashPosition(cash=0,position_dict={'SOXX':{'amount':100.,'price':30.}})
    schedule={pd.Timestamp('2024-03-07'):[{'kind':'split','symbol':'SOXX','id':'split','ratio':3}],pd.Timestamp('2024-03-08'):[{'kind':'entitle','symbol':'SOXX','id':'cash','rate':'0.01','available_date':'2024-03-12'}],pd.Timestamp('2024-03-12'):[{'kind':'pay','symbol':'SOXX','id':'cash'}]}
    position.apply_corporate('2024-03-07',schedule);assert position.get_stock_amount('SOXX')==300 and position.calculate_value()==3000
    position.apply_corporate('2024-03-08',schedule);assert position.income.value==3 and position.get_cash()==0 and position.calculate_value()==3003
    position.apply_corporate('2024-03-08',schedule);assert position.income.value==3
    position.apply_corporate('2024-03-12',schedule);assert position.income.value==0 and position.get_cash()==3 and position.calculate_value()==3003
