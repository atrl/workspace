import pandas as pd
import pytest
from src.data.settlement_value_dates import standard_value_dates


def test_service_calendar_and_trade_date_cycle_handle_documented_boundaries():
    cycles=pd.Series([2,1],index=pd.to_datetime(['2023-01-01','2024-05-28']))
    cases=[
        ('2024-05-24','2024-05-29',[True,False,False,False,True,True],['2024-05-24','2024-05-28'],['2024-05-29','2024-05-29']),
        ('2025-01-08','2025-01-10',[True,True,True],['2025-01-08'],['2025-01-09']),
        ('2024-10-11','2024-10-15',[True,False,False,False,True],['2024-10-11'],['2024-10-15']),
        ('2024-03-27','2024-04-01',[True,True,False,False,False,True],['2024-03-27'],['2024-04-01']),
    ]
    for start,end,flags,trades,expected in cases:
        calendar=pd.Series(flags,index=pd.date_range(start,end),dtype='boolean')
        actual=standard_value_dates(trades,settlement_open=calendar,cycle_by_effective_date=cycles)
        assert actual.tolist()==pd.to_datetime(expected).tolist()
    unknown=pd.Series([True,pd.NA,True],index=pd.date_range('2025-01-08',periods=3),dtype='boolean')
    with pytest.raises(ValueError,match='unknown settlement'):standard_value_dates(['2025-01-08'],settlement_open=unknown,cycle_by_effective_date=cycles)
    with pytest.raises(ValueError,match='ends before settlement'):standard_value_dates(['2025-01-10'],settlement_open=unknown,cycle_by_effective_date=cycles)
