import pandas as pd
import pytest
from src.data.semis_context_refresh import audit_context


def test_halfday_missing_bar_and_outside_session_date_remain_detectable():
    start=pd.Timestamp('2025-11-28 09:30',tz='America/New_York');end=pd.Timestamp('2025-11-28 13:00',tz='America/New_York')
    schedule=pd.DataFrame({'open':[start],'close':[end]},index=pd.to_datetime(['2025-11-28']))
    bars=pd.DataFrame({'symbol':'QQQ','date':'2025-11-28','timestamp':pd.date_range(start+pd.Timedelta(minutes=5),end,freq='5min'),'price_valid':True})
    assert audit_context(bars,schedule,symbols=['QQQ']).iloc[0].expected_bars==42
    assert audit_context(bars.iloc[:-1],schedule,symbols=['QQQ']).iloc[0].missing_bars==1
    bad=bars.copy();bad.loc[0,'date']='2025-11-29'
    with pytest.raises(ValueError,match='unexpected'):audit_context(bad,schedule,symbols=['QQQ'])
