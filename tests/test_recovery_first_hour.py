import pandas as pd
import numpy as np
from src.data.recovery_first_hour import aggregate_first_hour,FIELDS


def test_first_hour_excludes_later_bars_and_preserves_invalid_day():
    stamps=pd.date_range('2020-03-16 09:31',periods=60,freq='min',tz='America/New_York')
    parts=[pd.DataFrame({'symbol':s,'date':'2020-03-16','timestamp':stamps,'open':100.,'high':102.,'low':99.,'close':101.,'volume':10.,'price_valid':True,'is_final':True}) for s in ['SOXL','SOXX']]
    raw=pd.concat(parts,ignore_index=True);before=aggregate_first_hour(raw);late=raw.iloc[[0]].copy();late['timestamp']=pd.Timestamp('2020-03-16 11:00',tz='America/New_York');late['close']=1e9
    pd.testing.assert_frame_equal(before,aggregate_first_hour(pd.concat([raw,late],ignore_index=True)))
    assert before.first_hour_ready.iloc[0]==1 and np.isclose(before.early_ret_soxx.iloc[0],.01) and before.early_volume_soxx.iloc[0]==600
    raw.loc[0,'price_valid']=False;bad=aggregate_first_hour(raw);assert bad.first_hour_ready.iloc[0]==0 and bad[FIELDS].isna().all().all()
