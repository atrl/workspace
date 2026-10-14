import numpy as np
import pandas as pd
from src.data.opening_state_features import opening_state_features,FEATURES,FLAG


def test_opening_state_uses_previous_close_and_preserves_unknown():
    dates=pd.date_range('2024-01-02',periods=4)
    primary=pd.DataFrame(index=dates[:3])
    execution=pd.DataFrame({'signal_date':dates[:3],'trade_date':dates[1:]})
    for a in ['soxl','soxx']:execution['close_'+a]=[100.,110.,120.];execution['known_'+a]=[101.,112.,121.]
    q=pd.DataFrame({'$open930':[100.,105.,115.]*2,'$valid930':[1.,1.,0.]*2},index=pd.MultiIndex.from_product([['SOXL','SOXX'],dates[1:]],names=['instrument','datetime']))
    f,t=opening_state_features(primary,execution,q)
    assert f[FLAG].tolist()==[0.,1.,0.]
    assert f.iloc[[0,2]][FEATURES].isna().all().all()
    assert np.isclose(f.iloc[1].opening_soxx_gap,np.log(105/100))
    altered=execution.copy();altered.loc[1:,'close_soxx']=1e10
    second,_=opening_state_features(primary,altered,q)
    pd.testing.assert_series_equal(f.iloc[1],second.iloc[1])
    assert t.iloc[1].decision_at.hour==9 and t.iloc[1].decision_at.minute==46
