import numpy as np
import pandas as pd
from src.data.asia_same_session import align_asian_closes


def test_same_local_date_is_available_before_us_decision_across_dst_and_gap_is_explicit():
    days=pd.to_datetime(['2023-03-09','2023-03-10','2023-03-13','2023-03-14'])
    raw=pd.DataFrame({'twii':[100.,102.,np.nan,104.]},index=days)
    points,audit=align_asian_closes(raw,days,markets={'twii':'Asia/Taipei'},availability_hour=18)
    assert points.known.tolist()==[0,1,0,1]
    assert pd.Timestamp(audit.assumed_available_at.iloc[1]).hour==5
    assert pd.Timestamp(audit.assumed_available_at.iloc[3]).hour==6
    assert points.gap_days.iloc[3]==4
    np.testing.assert_allclose(points.return_value.iloc[3],104/102-1)
