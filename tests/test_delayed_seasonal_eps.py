import numpy as np
import pandas as pd
from src.data.delayed_seasonal_eps import delayed_seasonal_events,align_delayed_events,FEATURE,FLAG


def test_date_buffer_denominator_and_future_release_are_causal():
    dates=pd.date_range('2015-01-02',periods=14,freq='QS')
    reports=pd.DataFrame([{'quarter':f'FY{2015+i//4}Q{i%4+1}','release_date':str(d.date()),'filing_date':str(d.date()),'api_accepted_raw':str((d+pd.Timedelta(days=1)).date())+'T01:00:00Z','index_accepted_raw':None,'current_eps':str(1+i*i/100)} for i,d in enumerate(dates)])
    calendar=pd.bdate_range('2015-01-01','2019-12-31')
    a=delayed_seasonal_events(reports,[],calendar);assert a[FEATURE].iloc[:12].isna().all()
    values=reports.current_eps.astype(float).to_numpy();diff=values[4:13]-values[:9]
    assert np.isclose(a.iloc[12][FEATURE],diff[-1]/np.std(diff[:-1],ddof=1))
    mutated=reports.copy();mutated.loc[13,'current_eps']='9999';b=delayed_seasonal_events(mutated,[],calendar)
    pd.testing.assert_frame_equal(a.iloc[:13],b.iloc[:13])
    assert (a.available_date>pd.to_datetime(a.latest_literal_date)).all()
    t=a.iloc[12].available_date;aligned=align_delayed_events(a,pd.RangeIndex(2),[t-pd.Timedelta(days=1),t]);assert aligned[FLAG].tolist()==[0.,1.]
    # A split introduced only in the last report rescales numerator and denominator equally.
    split={'event_id':'split','event_type':'stock_split','effective_date':reports.iloc[12].release_date,'known_by_date':reports.iloc[12].release_date,'new_shares_per_old_share':4}
    # Keep effective date strictly between reports to avoid undefined same-day basis.
    split['effective_date']=str((pd.Timestamp(reports.iloc[12].release_date)-pd.Timedelta(days=1)).date())
    changed=reports.iloc[:13].copy();changed.loc[12,'current_eps']=str(float(changed.loc[12,'current_eps'])/4)
    c=delayed_seasonal_events(changed,[split],calendar);assert np.isclose(c.iloc[12][FEATURE],a.iloc[12][FEATURE])
