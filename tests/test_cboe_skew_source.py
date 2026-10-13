import pandas as pd
import pytest
from src.data.cboe_skew_source import read_skew,build_store


def test_missing_native_session_stays_unknown_and_duplicate_source_rejected(tmp_path):
    raw=b'DATE,SKEW\n01/03/2023,120\n01/05/2023,125\n'
    values=read_skew(raw);calendar=pd.bdate_range('2023-01-03',periods=3)
    native,_=build_store(values,reference_calendar=calendar,provider=tmp_path/'provider',as_of='2023-01-05',source_sha256='fixture')
    assert native.index.equals(calendar)
    assert pd.isna(native.loc[calendar[1],'$skew']) and native.loc[calendar[1],'$observed']==0
    assert native.loc[calendar[0],'$skew']==120 and native.loc[calendar[2],'$skew']==125
    with pytest.raises(ValueError,match='invalid source'):
        read_skew(raw+b'01/05/2023,126\n')
