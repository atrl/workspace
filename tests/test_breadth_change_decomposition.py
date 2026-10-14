import numpy as np
import pandas as pd
from src.analysis.breadth_change_decomposition import decompose_breadth_changes


def test_basket_jump_is_not_price_repair_and_missing_old_member_stays_unknown():
    dates=pd.date_range('2020-01-01',periods=3)
    signal=pd.DataFrame([[1.,0.],[1.,0.],[np.nan,1.]],index=dates,columns=['A','B'])
    weights=pd.DataFrame([[.5,.5],[1.,0.],[0.,1.]],index=dates,columns=signal.columns)
    out=decompose_breadth_changes(signal,weights)
    assert out.iloc[1].price_component==0 and out.iloc[1].basket_component==.5
    assert out.iloc[2].current==1 and pd.isna(out.iloc[2].total_change)
    assert out.reconciliation_error.dropna().abs().max()==0
