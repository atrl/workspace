import pandas as pd
import pytest
from src.model.annual_recovery import AnnualRecoveryControllers


def test_missing_year_never_borrows_future_model_and_maturity_is_strict():
    audit=pd.DataFrame([{'year':2020,'status':'needs_data','rows':19,'positive_rows':9,'minimum_rows':20,'cutoff':'2020-01-01T00:00:00-05:00','last_label_available_at':'2019-08-16T09:49:00-04:00'},{'year':2021,'status':'fitted','rows':174,'positive_rows':102,'minimum_rows':20,'cutoff':'2021-01-01T00:00:00-05:00','last_label_available_at':'2020-12-23T09:49:00-05:00'}])
    model=object();policy=AnnualRecoveryControllers({2021:model},audit)
    assert policy(pd.Timestamp('2020-12-31'))['frozen_recovery_model'] is None
    assert policy(pd.Timestamp('2021-01-04'))['frozen_recovery_model'] is model
    with pytest.raises(ValueError,match='no declared'):policy(pd.Timestamp('2022-01-03'))
    bad=audit.copy();bad.loc[1,'last_label_available_at']=bad.loc[1,'cutoff']
    with pytest.raises(ValueError,match='crosses cutoff'):AnnualRecoveryControllers({2021:model},bad)
