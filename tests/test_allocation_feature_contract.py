import pandas as pd
import pytest
from src.data.allocation_feature_contract import canonical_return_inputs


def test_alias_contract_preserves_missing_and_rejects_source_disagreement():
    p=pd.DataFrame({f'soxx_ret{h}':[.1,None,.2] for h in (5,20,60)})
    c=p.rename(columns={f'soxx_ret{h}':f'mom{h}' for h in (5,20,60)})
    c['vol20']=[.2,.3,.4]
    view,audit=canonical_return_inputs(p,c)
    assert view.shape==(3,4) and view['primary:soxx_ret5'].isna().sum()==1
    c.loc[0,'mom5']=.4
    with pytest.raises(ValueError,match='values differ'):
        canonical_return_inputs(p,c)
    c.loc[0,'mom5']=.1;c.loc[1,'mom5']=0.
    with pytest.raises(ValueError,match='missing masks differ'):
        canonical_return_inputs(p,c)


def test_disjoint_returns_roundtrip_and_reject_hidden_missingness():
    import numpy as np
    from src.data.allocation_feature_contract import disjoint_momentum_inputs, reconstruct_momentum
    frame=pd.DataFrame({f'primary:{s}_ret{h}':[np.expm1(.01*h),np.expm1(-.01*h)]
                        for s in ('soxl','soxx') for h in (3,5,10,20,60)})
    segmented,_=disjoint_momentum_inputs(frame)
    np.testing.assert_allclose(reconstruct_momentum(segmented),frame,atol=1e-15)
    assert segmented['primary:soxx_log_return_lags_4_5'].iloc[0]==pytest.approx(.02)
    frame.loc[0,'primary:soxx_ret3']=np.nan
    with pytest.raises(ValueError,match='non-nested'):
        disjoint_momentum_inputs(frame)
