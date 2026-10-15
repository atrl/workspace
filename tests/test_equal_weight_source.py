import pandas as pd
import pytest
from src.data.equal_weight_source import parse_chart

def test_same_date_adjusted_quote_are_distinct_and_missing_adjustment_fails():
    stamp=int(pd.Timestamp('2020-01-02 09:30',tz='America/New_York').timestamp());calendar=pd.DatetimeIndex(['2020-01-02'])
    result={'meta':{'symbol':'RSP','currency':'USD','exchangeTimezoneName':'America/New_York','dataGranularity':'1d'},'timestamp':[stamp],'indicators':{'quote':[{'open':[99.],'high':[101.],'low':[98.],'close':[100.],'volume':[7.]}],'adjclose':[{'adjclose':[80.]}]},'events':{'dividends':{str(stamp):{'date':stamp,'amount':.2}}}}
    p={'chart':{'result':[result],'error':None}};frame,events,meta=parse_chart(p,'RSP',calendar)
    assert frame.quote_close.iloc[0]==100 and frame.vendor_adj_close.iloc[0]==80 and len(events)==1
    result['indicators']['adjclose']=[]
    with pytest.raises(ValueError,match='adjusted'):parse_chart(p,'RSP',calendar)
