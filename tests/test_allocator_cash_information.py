import numpy as np
import pandas as pd
from src.analysis.allocator_cash_information import cash_labels


def test_native_horizon_maturity_and_tail_preserve_unknown_rows():
    days=pd.bdate_range('2023-01-03',periods=7)
    data=pd.DataFrame({'signal_date':days[:6]-pd.Timedelta(days=1),'trade_date':days[:6],'exit_date':days[1:],'soxx_open':[100,101,102,103,104,105],'soxx_exit_open':[101,102,103,104,105,106],'close_soxx':[101,102,94,104,105,106]})
    result=cash_labels(data,horizons=[1,5],tail_threshold=-.05)
    assert len(result)==6 and result.cash_advantage_h1.notna().sum()==6
    assert result.cash_advantage_h5.notna().sum()==2 and result.tail_h5.isna().sum()==4
    np.testing.assert_allclose(result.cash_advantage_h5.iloc[0],-np.log(105/100))
    assert result.tail_h5.iloc[0]==1
    assert result.label_available_h5.iloc[0]==days[5]+pd.Timedelta(hours=9,minutes=49)
