import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.workflow.single_information_experiment import verify_information_execution_inputs


def _fixture(tmp_path, monkeypatch):
    import src.data.provider_content_hash as providers
    monkeypatch.setattr(providers, "provider_content_hash", lambda path: ("a" * 64, {}))
    root = Path(__file__).resolve().parents[1]
    module = "src/workflow/single_information_experiment.py"
    plan = dict(horizon_sessions=5, costs_bps=[10.,25.], robust_radius=1., outer_years=list(range(2016,2027)),
        fixed_fit_budget=110, fixed_native_cases=12, minimum_training_rows=252, minimum_validation_rows=126,
        hac_lags=5, source_deployment_gate=False, source_card="qualified", prefit={"card_path":"qualified"},
        route_on_failure="end_fixed_route", report_title="# Fixed source", interpretation_notes=["Source scope only"],
        frozen_code_sha256={module:hashlib.sha256((root/module).read_bytes()).hexdigest()})
    p=tmp_path/"preregister.json";p.write_text(json.dumps(plan))
    plan.update(preregister_path=p.name,frozen_files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()},
                providers={k:{"provider_uri":k,"content_sha256":"a"*64} for k in ["price","execution","information"]},
                authorization={"fit":True,"replay":True,"orders":False})
    return root,plan


def test_fixed_information_context_and_code_identity_are_checked_before_fitting(tmp_path,monkeypatch):
    root,plan=_fixture(tmp_path,monkeypatch)
    verify_information_execution_inputs(tmp_path,tmp_path,plan,source_root=root)
    for key,value in [("report_title","changed"),("route_on_failure","retry_failed_source"),
                      ("source_card","other"),("fixed_fit_budget",120)]:
        changed=copy.deepcopy(plan);changed[key]=value
        with pytest.raises(ValueError,match="preregistered"):
            verify_information_execution_inputs(tmp_path,tmp_path,changed,source_root=root)
    with pytest.raises(ValueError,match="source root"):
        verify_information_execution_inputs(tmp_path,tmp_path,plan,source_root=tmp_path)


def test_modified_frozen_source_cannot_reach_a_model_fit(tmp_path,monkeypatch):
    root,plan=_fixture(tmp_path,monkeypatch)
    relative=next(iter(plan['frozen_code_sha256']))
    plan['frozen_code_sha256'][relative]='0'*64
    p=tmp_path/'preregister.json'
    initial=json.loads(p.read_text());initial['frozen_code_sha256']=plan['frozen_code_sha256']
    p.write_text(json.dumps(initial));plan['frozen_files_sha256'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='frozen code'):
        verify_information_execution_inputs(tmp_path,tmp_path,plan,source_root=root)
