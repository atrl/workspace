import pandas as pd
from src.data.funding_availability import funding_release_events,align_funding_information


def test_announced_policy_correction_and_publication_day_delay():
    sofr=pd.DataFrame({'observation_date':pd.to_datetime(['2020-03-16']),'release_date':pd.to_datetime(['2020-03-17']),'value':[.26]})
    ioer=pd.DataFrame({'observation_date':pd.to_datetime(['2020-03-16']),'release_date':pd.to_datetime(['2020-03-13']),'value':[1.1]})
    events=funding_release_events(sofr,ioer,ioer.iloc[:0],official_observation_dates=sofr.observation_date,policy_events=[{'series':'IOER','observation_date':'2020-03-16','public_date':'2020-03-15','value':.1}])
    assert abs(events.spread_bp.iloc[0]-16)<1e-9 and events.original_policy_initial_value.iloc[0]==1.1
    state=align_funding_information(events,pd.to_datetime(['2020-03-17','2020-03-18']))
    assert state.known.tolist()==[0.,1.]
    # Stock-market-open bank holiday must not turn underlying Friday into known Monday data.
    sofr=pd.DataFrame({'observation_date':pd.to_datetime(['2026-10-09']),'release_date':pd.to_datetime(['2026-10-13']),'value':[3.87]})
    iorb=pd.DataFrame({'observation_date':pd.to_datetime(['2026-10-09']),'release_date':pd.to_datetime(['2026-10-08']),'value':[3.9]})
    events=funding_release_events(sofr,ioer.iloc[:0],iorb,official_observation_dates=sofr.observation_date,policy_events=[])
    state=align_funding_information(events,pd.to_datetime(['2026-10-12','2026-10-13','2026-10-14']))
    assert state.known.tolist()==[0.,0.,1.]
