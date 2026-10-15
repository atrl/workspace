import numpy as np
import pandas as pd
from src.analysis.seed_disagreement_information import mature_before,fit_scaler,transform,full_calendar_bootstrap


def test_strict_maturity_training_only_scaling_and_calendar_unknown_mask():
    dates=pd.Series(pd.to_datetime(['2021-12-31 09:49-05:00','2022-01-01 00:00-05:00','2022-01-03 09:49-05:00']))
    mask=mature_before(dates,'2022-01-01T00:00:00-05:00');assert mask.tolist()==[True,False,False]
    train=np.array([[1.,np.nan],[3.,2.]]);mean,scale=fit_scaler(train)
    assert np.isfinite(transform([[1e9,np.nan]],mean,scale)).all();np.testing.assert_array_equal(mean,[2.,2.])
    values=np.array([1.,np.nan,-.5,2.]);summary,index,replicates=full_calendar_bootstrap(values,mean_block=2,resamples=50,seed=7,alpha=.05)
    assert index.shape==(50,4) and summary['calendar_rows']==4 and summary['scored_rows']==3
    assert (index==1).any()
    expected=np.array([np.nanmean(values[row]) if np.isfinite(values[row]).any() else np.nan for row in index])
    np.testing.assert_allclose(replicates,expected,equal_nan=True)
