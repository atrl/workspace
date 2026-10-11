import pandas as pd
import numpy as np
from src.data.alfred_initial_release import qualify_paired_information


def _series(dates,values,releases):
    return pd.DataFrame(dict(observation_date=pd.to_datetime(dates),value=values,release_date=pd.to_datetime(releases)))


def test_no_same_day_use_and_latest_unknown_invalidates_feature():
    calendar=pd.bdate_range('2020-01-02',periods=5)
    dates=calendar[:3]
    vxn=_series(dates,[20.,np.nan,25.],dates)
    vix=_series(dates,[10.,12.,15.],dates)
    state,pair,audit=qualify_paired_information(vxn,vix,calendar)
    assert state.known.tolist()==[0.,1.,0.,1.,1.]
    assert np.isnan(state.loc[2,'log_vxn_vix'])
    np.testing.assert_allclose(state.loc[1,'log_vxn_vix'],np.log(2))
    assert audit['price_evaluation_dates_dropped']==0
    assert pair.loc[0,'available_at'].tz_convert('America/New_York').hour==9


def test_premature_market_day_is_quarantined_without_rewriting_release():
    calendar=pd.bdate_range('2018-10-05',periods=4)
    dates=['2018-10-05','2018-10-08']
    releases=['2018-10-05','2018-10-05']
    vxn=_series(dates,[20.,22.19],releases);vix=_series(dates,[15.,15.69],releases)
    state,pair,audit=qualify_paired_information(vxn,vix,calendar)
    assert state.known.tolist()==[0.,1.,0.,0.]
    assert pair.iloc[1].release_date_vxn==pd.Timestamp('2018-10-05')
    assert audit['premature_market_observations']==1
    assert not pair.iloc[1].qualified


def test_release_delay_uses_both_series_same_observation_day():
    calendar=pd.bdate_range('2020-01-02',periods=5)
    vxn=_series(calendar[:1],[20.],calendar[:1])
    vix=_series(calendar[:1],[10.],calendar[2:3])
    state,_,_=qualify_paired_information(vxn,vix,calendar)
    assert state.known.tolist()==[0.,0.,0.,1.,1.]
    assert (state.loc[state.known==1,'observation_date']==calendar[0]).all()
