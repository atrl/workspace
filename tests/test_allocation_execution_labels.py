import numpy as np
import pandas as pd
import pytest
from src.data.allocation_execution_labels import paired_execution_returns
from src.model.minute_pit_allocator import TARGETS


def test_aligned_returns_preserve_classes_and_mask_both_arms_on_unknown_prices():
    dates=pd.to_datetime(['2022-12-19','2022-12-20','2022-12-21']);trade=dates+pd.Timedelta(days=1);exit=dates+pd.Timedelta(days=2)
    p=pd.DataFrame({c:[.05,.03,.01] for c in TARGETS},index=dates);p['trade_date']=trade;p['exit_date']=exit
    timing=pd.DataFrame({'label_available_at':pd.to_datetime(['2022-12-28 09:31','2022-12-29 09:31','2022-12-30 09:31']).tz_localize('America/New_York')},index=dates)
    execution=pd.DataFrame({'signal_date':dates,'trade_date':trade,'exit_date':exit,'soxl_open':[100,100,np.nan],'soxl_exit_open':[110,90,110],'soxx_open':[100,100,100],'soxx_exit_open':[102,98,101]})
    old,new,audit=paired_execution_returns(p,timing,execution)
    np.testing.assert_allclose(new.soxl_forward_open_return.iloc[:2],[.1,-.1]);np.testing.assert_allclose(old.soxl_forward_open_return.iloc[:2],[.05,.03])
    assert new[list(TARGETS)].iloc[2].isna().all() and old[list(TARGETS)].iloc[2].isna().all()
    pd.testing.assert_frame_equal(old[list(TARGETS[2:])],new[list(TARGETS[2:])]);assert audit.paired_label_known.tolist()==[True,True,False]
    broken=execution.copy();broken.loc[0,'trade_date']=pd.Timestamp('2022-12-21')
    with pytest.raises(ValueError,match='calendar'):paired_execution_returns(p,timing,broken)
