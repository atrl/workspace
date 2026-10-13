import numpy as np
import pandas as pd
from src.data.skew_information import lagged_skew


def test_two_session_lag_keeps_gap_unknown_instead_of_previous_observation():
    days=pd.bdate_range('2023-01-03',periods=5)
    native=pd.DataFrame({'$skew':[120,np.nan,130,140,150],'$observed':[1,0,1,1,1]},index=days)
    result=lagged_skew(native,days,days,lag=2,eligible_after='2014-02-03')
    assert result.skew_known.tolist()==[0,0,1,0,1]
    assert result.skew_scaled.iloc[2]==2 and result.skew_scaled.iloc[4]==3
    assert result.source_date.iloc[3]==days[1]
    assert result.assumed_available_at.iloc[2]==days[1]+pd.Timedelta(hours=17)
