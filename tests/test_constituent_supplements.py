import hashlib
import pandas as pd
import pytest
from src.data import constituent_supplements as module


def test_verified_supplement_fills_only_missing_and_rejects_conflicts(tmp_path,monkeypatch):
    receipt=tmp_path/'receipt';receipt.write_text('verified')
    dates=pd.date_range('2020-01-02',periods=2)
    native=pd.DataFrame({'$close':[10.,11.]},index=pd.MultiIndex.from_product([['WOLF'],dates],names=['instrument','datetime']))
    monkeypatch.setattr(module,'load_qlib_feature_frame',lambda *a,**k:native)
    registry={'sources':[{'source_id':'test','required_hashes':{'receipt':hashlib.sha256(receipt.read_bytes()).hexdigest()},'provider_uri':'provider','instrument':'WOLF','field':'$close','start':'2020-01-02','end':'2020-01-03','historical_aliases':['WOLF']}]}
    base=pd.DataFrame({'WOLF':[10.,float('nan')]},index=dates)
    result,audit=module.apply_constituent_supplements(base,root=tmp_path,registry=registry)
    assert result.WOLF.tolist()==[10.,11.] and pd.isna(base.WOLF.iloc[1]) and audit[0]['filled_price_rows']==1
    native.iloc[0,0]=9.
    with pytest.raises(ValueError,match='conflicting'):module.apply_constituent_supplements(base,root=tmp_path,registry=registry)
    receipt.write_text('modified')
    with pytest.raises(ValueError,match='hash differs'):module.apply_constituent_supplements(base,root=tmp_path,registry=registry)
