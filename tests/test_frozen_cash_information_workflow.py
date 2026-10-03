from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from src.workflow import frozen_cash_information as workflow
from src.data.integrated_allocation import PRIMARY_FEATURES, COMMON_PRIMARY_FEATURES


@pytest.fixture(scope="module")
def synthetic_bundle():
    """Artificial prices/labels only; the original forecast fitter is a fixture."""
    from src.workflow.conditional_variance import fit_oof_forecasts
    dates = pd.bdate_range("2011-01-03", "2026-09-25", name="datetime").delete(slice(10, 30))
    t = np.arange(len(dates))
    frame = pd.DataFrame({name: np.sin(t * (.01 + j * .002)) + .03 * j
                          for j, name in enumerate(PRIMARY_FEATURES)}, index=dates)
    frame.loc[:, list(PRIMARY_FEATURES[-3:])] = 0.
    days = pd.Series(dates, index=dates)
    def at(values, hour, minute=0):
        return pd.Series(pd.DatetimeIndex(values).tz_localize("America/New_York")
                         + pd.Timedelta(hours=hour, minutes=minute), index=dates)
    frame["signal_date"] = days
    frame["trade_date"], frame["exit_date"] = days.shift(-1), days.shift(-2)
    frame["feature_available_at"] = at(days, 16)
    frame["decision_at"] = at(frame.trade_date, 9, 31)
    frame["return_available_at"] = at(frame.exit_date, 9, 30)
    frame["forecast_available_at"] = at(days.shift(-6), 9, 30)
    frame["auxiliary_available_at"] = at(days.shift(-5), 16)
    frame["soxx_return"] = .004 * np.sin(t * .1)
    frame["soxl_return"] = 3 * frame.soxx_return
    frame.loc[frame.exit_date.isna(), ["soxl_return", "soxx_return"]] = np.nan
    frame["cash_advantage_1d"] = -np.log1p(frame.soxx_return)
    frame["future_vol_5d"] = .003 + .001 * np.cos(t * .04)
    frame.loc[frame.forecast_available_at.isna(), "future_vol_5d"] = np.nan
    frame["underlying_return_5d"] = .002
    frame["underlying_vol_5d"] = .001
    frame.loc[frame.auxiliary_available_at.isna(), ["underlying_return_5d", "underlying_vol_5d"]] = np.nan
    forecasts, audit, models = fit_oof_forecasts(frame, as_of="2026-09-25")
    assert set(models) == set(range(2013, 2027))
    return {"raw":frame, "forecasts":forecasts, "forecast_audit":audit,
            "forecast_models":models, "source_audit":{}, "source_details":{}, "prices":pd.DataFrame()}


@pytest.fixture
def plan():
    return {"source":{"as_of":"2026-09-25"}, "forecast":{"years":list(range(2013, 2027))},
            "source_readback_contract":{"variants":[{"name":name} for name in ("constant","price","price_vol")],
                "training":{"development_signal_years":list(range(2016,2023)),"minimum_rows":252}}}


def _load(monkeypatch, bundle, plan):
    monkeypatch.setattr(workflow,"load_constrained_inputs",lambda root,p: bundle)
    return workflow.load_frozen_information_inputs(Path.cwd(),plan)


def test_native_joint_readback_preserves_unavailable_heads_and_h1_tail(monkeypatch,synthetic_bundle,plan):
    result = _load(monkeypatch,synthetic_bundle,plan)
    rows = result["diagnostic_rows"]
    assert len(rows) == len(synthetic_bundle["raw"])
    assert not rows.loc[rows.index.year <= 2012,"forecast_available"].any()
    assert rows.loc[rows.index.year <= 2012,"prior_mean_prediction"].isna().all()
    assert not rows.iloc[-2:].h1_outcome_available.any()
    assert rows.iloc[-6:-2].score_available.all()  # valid cash outcomes without H5 labels
    assert len(result["head_training_evidence"]) == 14
    annual=result["head_readback_audit"]
    assert annual.loc[annual.available,"prediction_max_absolute_error"].max() == 0.
    assert annual.loc[annual.available,"intercept_mean_absolute_error"].max() < 1e-14
    assert not result["information_source_audit"]["predictive_metrics_computed"]
    for year,evidence in result["head_training_evidence"].items():
        source=synthetic_bundle["raw"].loc[evidence["dates"]]
        cutoff=pd.Timestamp(f"{year}-01-01",tz="America/New_York")
        assert source.forecast_available_at.lt(cutoff).all()
        assert source.return_available_at.lt(cutoff).all()


@pytest.mark.parametrize("fault",["prediction","intercept","scale","feature_order","maturity"])
def test_integrity_faults_block_without_refit(monkeypatch,synthetic_bundle,plan,fault):
    bundle=deepcopy(synthetic_bundle)
    model=bundle["forecast_models"][2026]
    if fault=="prediction":bundle["forecasts"].loc["2026-01-02","cash_advantage_prediction"]+=.001
    elif fault=="intercept":model.named_steps["ridge"].intercept_[0]+=.001
    elif fault=="scale":model.named_steps["standardscaler"].scale_[0]*=2
    elif fault=="feature_order":model.named_steps["standardscaler"].feature_names_in_=np.array(COMMON_PRIMARY_FEATURES[::-1])
    else:bundle["raw"].loc["2014-06-02","forecast_available_at"]=pd.NaT
    with pytest.raises(ValueError,match="blocked:"):_load(monkeypatch,bundle,plan)


def test_invalid_vol_sibling_is_preserved_without_dropping_cash(monkeypatch,synthetic_bundle,plan):
    bundle=deepcopy(synthetic_bundle);model=bundle["forecast_models"][2026];predict=model.predict
    def bad_vol(x):
        result=predict(x);result[:,1]=np.nan;return result
    model.predict=bad_vol
    bundle["forecasts"].loc[bundle["forecasts"].index.year==2026,"future_vol_prediction"]=np.nan
    result=_load(monkeypatch,bundle,plan)
    original=synthetic_bundle["raw"]
    assert result["diagnostic_rows"].loc[original.index.year==2026,"score_available"].sum()==int((original.index.year==2026).sum())-2


def test_invalid_scored_cash_cannot_silently_disappear(monkeypatch,synthetic_bundle,plan):
    bundle=deepcopy(synthetic_bundle);model=bundle["forecast_models"][2026];predict=model.predict
    def bad_cash(x):
        result=predict(x);result[0,0]=np.nan;return result
    model.predict=bad_cash
    date=bundle["forecasts"].index[bundle["forecasts"].index.year==2026][0]
    bundle["forecasts"].loc[date,"cash_advantage_prediction"]=np.nan
    with pytest.raises(ValueError,match="available cash forecasts are invalid"):_load(monkeypatch,bundle,plan)


def _seal_fixture(tmp_path,monkeypatch):
    card=tmp_path/"card";card.mkdir()
    hashes={"a":"source"};monkeypatch.setattr(workflow,"_source_hashes",lambda root:hashes)
    tests={}
    for name in ("test_frozen_cash_information_workflow.py","test_frozen_cash_information.py"):
        p=tmp_path/"tests"/name;p.parent.mkdir(exist_ok=True);p.write_text("test source")
        tests[str(p.relative_to(tmp_path))]=workflow._sha(p)
    identity={"plan_sha256":"plan","registration_sha256":"registration"}
    seal={**identity,"source_sha256":hashes,"test_sha256":tests,"focused_passed":2,
          "static_contracts_passed":True,"data_contracts_passed":True,"analysis_contracts_passed":True,
          "predictive_metrics_started":False}
    return card,identity,seal


@pytest.mark.parametrize("key,value",[("focused_passed",True),("data_contracts_passed","true"),
    ("analysis_contracts_passed",False),("predictive_metrics_started",True)])
def test_bad_initial_seal_blocks_before_loading_or_analysis(tmp_path,monkeypatch,key,value):
    card,identity,seal=_seal_fixture(tmp_path,monkeypatch);seal[key]=value
    workflow._write_json(card/"implementation_seal.json",seal)
    monkeypatch.setattr(workflow,"_read_registered_plan",lambda c:({},identity))
    monkeypatch.setattr(workflow,"load_frozen_information_inputs",lambda *a:pytest.fail("source loader reached"))
    with pytest.raises(ValueError,match="blocked:"):workflow.run(tmp_path,card)


def test_partial_reference_blocks_automatic_repeat(tmp_path):
    workflow._write_json(tmp_path/"recorder_reference.json",{"complete":False})
    with pytest.raises(ValueError,match="partial diagnostic"):
        workflow._load_completed_reference(tmp_path,tmp_path,{}, {},"seal")


@pytest.mark.parametrize("analysis_fails",[False,True])
def test_real_recorder_single_run_cache_and_failure_scope(tmp_path,monkeypatch,analysis_fails):
    import qlib
    from qlib.workflow import R
    from src.analysis import frozen_cash_information as analysis
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE","true")
    qlib.init(provider_uri=str(tmp_path),region="us",kernels=1)
    monkeypatch.setattr(workflow,"CANONICAL_ROOT",tmp_path)
    card,identity,seal=_seal_fixture(tmp_path,monkeypatch)
    for name in ("diagnostic_plan.json","registration.json"): (card/name).write_text("{}")
    workflow._write_json(card/"implementation_seal.json",seal)
    source={"as_of":"2026-09-25","provider_uri":str(tmp_path)}
    plan={"source":source,"artifact_policy":{"policy":"slim_card_recorder_first","workflow_config":"workflow.yaml","recorder_only_artifacts":["inputs/diagnostic_rows.pkl"]},
          "qlib_docs":[],"source_limits":"synthetic fixture","event_diagnostic":{"limitations":"not a portfolio"},
          "uncertainty":{"interpretation":"synthetic fixture"}}
    monkeypatch.setattr(workflow,"_read_registered_plan",lambda c:(plan,identity))
    bundle={"forecasts":pd.DataFrame({"x":[1]}),"diagnostic_rows":pd.DataFrame({"x":[1]}),"source_details":{},
            "information_source_audit":{},"head_readback_audit":pd.DataFrame({"model_year":[2013]}),
            "head_training_evidence":{},"forecast_models":{}}
    loads=[];calls=[]
    monkeypatch.setattr(workflow,"load_frozen_information_inputs",lambda *a:(loads.append(1) or bundle))
    def analyzer(rows,p):
        calls.append(1)
        if analysis_fails:raise RuntimeError("synthetic analysis failure")
        return {**{key:pd.DataFrame({"value":[1.]}) for key in ("point_metrics","calibration","event_metrics","bootstrap_summary")},
                "bootstrap_draws":{},"audit":{"fixture":True}}
    monkeypatch.setattr(analysis,"analyze_information",analyzer)
    if analysis_fails:
        with pytest.raises(RuntimeError,match="synthetic analysis failure"):workflow.run(tmp_path,card)
        reference=json.loads((card/"recorder_reference.json").read_text());assert not reference["complete"]
        with R.uri_context(reference["recorder_uri"]):
            rec=R.get_recorder(experiment_name=workflow.EXPERIMENT,recorder_id=reference["recorder_id"])
            assert rec.client.get_run(rec.id).info.status=="FAILED"
        with pytest.raises(ValueError,match="partial"):workflow.run(tmp_path,card)
    else:
        first=workflow.run(tmp_path,card);second=workflow.run(tmp_path,card)
        assert first==second and len(loads)==len(calls)==1
        assert first["market_model_fits"]==first["new_portfolio_paths"]==0
        reference=json.loads((card/"recorder_reference.json").read_text());assert reference["complete"]
        artifact=Path(first["recorder_local_dir"])/"artifacts/inputs/diagnostic_rows.pkl"
        artifact.write_bytes(b"tampered")
        with pytest.raises(ValueError,match="artifact hash differs"):workflow.run(tmp_path,card)
