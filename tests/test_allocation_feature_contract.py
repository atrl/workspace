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
