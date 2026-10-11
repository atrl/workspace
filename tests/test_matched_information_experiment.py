import copy
import hashlib
import json
import pytest
from src.workflow.matched_information_experiment import verify_frozen_inputs


def _plan(tmp_path,monkeypatch):
    import src.data.provider_content_hash as providers
    monkeypatch.setattr(providers,'provider_content_hash',lambda path:('a'*64,{'fixture':'zero-fit contract'}))
    p=dict(horizon_sessions=5,costs_bps=[10.,25.],robust_radius=1.,outer_years=list(range(2016,2027)),
        fallback_years=[],train_start='2011-01-03',fixed_fit_budget=110,fixed_native_cases=12,
        minimum_training_rows=252,minimum_validation_rows=126,hac_lags=5,source_deployment_gate=False,
        gates={'minimum_positive_years':6},scientific_scope={'forward_required':60})
    path=tmp_path/'preregister.json';path.write_text(json.dumps(p))
    p.update(preregister_path=path.name,frozen_files_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest()},
        authorization={'fit':True,'replay':True,'orders':False},
        providers={k:{'provider_uri':k,'content_sha256':'a'*64} for k in ['price','execution','information']})
    return p


def test_fixed_execution_matches_entire_preregister_without_fitting(tmp_path,monkeypatch):
    p=_plan(tmp_path,monkeypatch);verify_frozen_inputs(tmp_path,tmp_path,p)
    for key,value in [('gates',{'minimum_positive_years':1}),('fallback_years',[2016]),
                      ('train_start','2014-01-01'),('source_deployment_gate',True),('scientific_scope',{})]:
        changed=copy.deepcopy(p);changed[key]=value
        with pytest.raises(ValueError,match='preregistered field'):
            verify_frozen_inputs(tmp_path,tmp_path,changed)


def test_missing_provider_or_real_orders_cannot_enter_research_run(tmp_path,monkeypatch):
    p=_plan(tmp_path,monkeypatch);p['providers'].pop('information')
    with pytest.raises(ValueError,match='inventory'):
        verify_frozen_inputs(tmp_path,tmp_path,p)
    p=_plan(tmp_path,monkeypatch);p['authorization']['orders']=True
    with pytest.raises(ValueError,match='orders'):
        verify_frozen_inputs(tmp_path,tmp_path,p)
