import pandas as pd
from src.model.annual_allocator_clock import annual_decision_groups


def test_previous_year_signal_uses_new_year_model_only_after_cutoff():
    dates=pd.to_datetime(['2023-12-28','2023-12-29','2024-01-02'])
    timing=pd.DataFrame({'decision_at':pd.to_datetime(['2023-12-29 09:31','2024-01-02 09:31','2024-01-03 09:31']).tz_localize('America/New_York')},index=dates)
    groups=annual_decision_groups(timing,dates)
    assert list(groups[2023]['signal_dates'])==[dates[0]]
    assert list(groups[2024]['signal_dates'])==[dates[1],dates[2]]
    assert groups[2024]['cutoff']==pd.Timestamp('2024-01-01',tz='America/New_York')
    assert dates[1]<groups[2024]['cutoff'].tz_localize(None)
