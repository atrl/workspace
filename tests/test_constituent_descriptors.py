import numpy as np
import pandas as pd
from src.data.constituent_descriptors import reported_basket_descriptors
from src.data.soxx_pit_holdings import PIT_HOLDINGS_COLUMNS


def test_future_report_cannot_change_descriptors_and_no_loss_is_unknown():
    dates=pd.bdate_range('2020-01-02',periods=30)
    prices=pd.DataFrame({'A':np.arange(30)+100.},index=dates)
    row={c:'' for c in PIT_HOLDINGS_COLUMNS};row.update(report_date='2019-09-30',filing_date='2019-12-31',accession='old',ticker='A',weight_pct=100.)
    holdings=pd.DataFrame([row]);expected=reported_basket_descriptors(prices,holdings)
    future={**row,'report_date':'2020-03-31','filing_date':'2020-06-01','accession':'future','ticker':'B'}
    actual=reported_basket_descriptors(prices,pd.concat([holdings,pd.DataFrame([future])]))
    pd.testing.assert_frame_equal(actual,expected)
    assert expected.downside_top5_share.isna().all()
    assert expected.new_low_level20.iloc[:19].isna().all()
    assert expected.new_low_price_contraction5.iloc[-1]==0
