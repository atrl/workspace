"""Focused contracts for the fixed-path fee/drawdown audit (0 fit)."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.analysis import fixed_path_fee_drawdown as A
from src.workflow import fixed_path_fee_drawdown_audit as W

ROOT = Path(__file__).resolve().parents[1]


def _base(n=8, rets=None, convention="pre_fee_account"):
    rets = rets if rets is not None else [0.0] * n
    return pd.DataFrame({
        "signal_date": pd.bdate_range("2020-01-01", periods=n),
        "trade_date": pd.bdate_range("2020-01-02", periods=n),
        "exit_date": pd.bdate_range("2020-01-03", periods=n),
        "desired_soxl": 0.2, "desired_soxx": 0.5, "desired_cash": 0.3,
        "recovery_soxl": np.nan, "recovery_soxx": np.nan, "recovery_cash": np.nan,
        "execute": True, "recovery_action": False,
        "soxl_return": rets, "soxx_return": [0.0] * n,
        "early_ret_soxl": 0.0, "early_ret_soxx": 0.0,
        "strategy_return": 0.0, "fees": 0.0, "turnover": 0.0,
        "ending_soxl_weight": 0.2, "ending_soxx_weight": 0.5, "ending_cash_weight": 0.3,
        "funding_convention": convention})


def test_drawdown_retains_initial_one_and_per_row_peak_date():
    daily = pd.DataFrame({"trade_date": pd.bdate_range("2020-01-01", periods=4),
                          "exit_date": pd.bdate_range("2020-01-02", periods=4),
                          "strategy_return": [0.10, -0.5, 0.0, 0.0]})
    st = A.drawdown_stats(daily, emergency_threshold=0.22)
    assert st["sessions"] == 4 and len(st["peak_capital_date"]) == 4
    assert st["peak_capital_date"][0] is None              # initial capital, not a fabricated day
    assert st["peak_capital_date"][1] == str(pd.Timestamp("2020-01-02").date())  # exit_date of the peak-setting session
    assert st["peak_capital_date"][2] == str(pd.Timestamp("2020-01-02").date())  # stays on the recent peak
    assert st["peak_updates"][0]["date"] == str(pd.Timestamp("2020-01-02").date())


def test_drawdown_blocks_empty_nonfinite_and_le_minus_one():
    with pytest.raises(ValueError):
        A.drawdown_stats(pd.DataFrame({"trade_date": [], "strategy_return": []}), emergency_threshold=0.22)
    bad = pd.DataFrame({"trade_date": pd.bdate_range("2020-01-01", periods=2), "strategy_return": [np.nan, 0.0]})
    with pytest.raises(ValueError):
        A.drawdown_stats(bad, emergency_threshold=0.22)
    bad2 = pd.DataFrame({"trade_date": pd.bdate_range("2020-01-01", periods=2), "strategy_return": [-1.5, 0.0]})
    with pytest.raises(ValueError):
        A.drawdown_stats(bad2, emergency_threshold=0.22)


def test_resettle_blocks_invalid_targets_and_convention():
    base = _base()
    base.loc[0, "desired_soxl"] = np.nan  # execute=True with missing target
    with pytest.raises(ValueError, match="missing desired target"):
        A.resettle_preset_path(base, settlement_cost=10.0)
    ok = _base()
    with pytest.raises(ValueError, match="unsupported convention"):
        A.resettle_preset_path(ok, settlement_cost=10.0, convention="magic")
    strflags = _base(); strflags["execute"] = "True"
    with pytest.raises(ValueError, match="non-bool flag"):
        A.resettle_preset_path(strflags, settlement_cost=10.0)


def _settled_base(cost):
    raw = _base(rets=[0.01, -0.02, 0.03, -0.01, 0.02, 0.0, -0.03, 0.04])
    settled = A.resettle_preset_path(raw, settlement_cost=cost)
    base = raw.copy()
    for c in ("strategy_return", "fees", "turnover", "ending_soxl_weight", "ending_soxx_weight", "ending_cash_weight"):
        base[c] = settled[c].to_numpy()
    return base


def test_base_parity_executes_and_tamper_blocks():
    for cost in (10.0, 25.0):
        base = _settled_base(cost)
        daily = A.resettle_preset_path(base, settlement_cost=cost)
        assert A.base_parity(base, daily)["status"] == "passed"
    base = _settled_base(10.0)
    tampered = base.copy(); tampered.loc[0, "strategy_return"] += 0.001
    assert A.base_parity(tampered, A.resettle_preset_path(base, settlement_cost=10.0))["status"] == "failed"


def test_unordered_or_shifted_not_treated_as_same_path():
    base = _settled_base(10.0)
    daily = A.resettle_preset_path(base, settlement_cost=10.0)
    shifted = daily.copy(); shifted["trade_date"] = pd.to_datetime(shifted["trade_date"]) + pd.Timedelta(days=1)
    assert A.base_parity(base, shifted)["status"] == "failed"
    short = daily.iloc[1:]
    assert A.base_parity(base, short)["status"] == "failed"


def test_different_fee_changes_path_not_same():
    base = _base(rets=[-0.1, 0.05, -0.1, 0.05, -0.1, 0.05, -0.1, 0.05])
    d0 = A.resettle_preset_path(base, settlement_cost=0.0)
    d25 = A.resettle_preset_path(base, settlement_cost=25.0)
    assert not np.allclose(d0["strategy_return"], d25["strategy_return"])  # not the same path


def test_cross_window_peak_resets():
    # two independent windows each include the initial 1.0 peak
    for rets in ([-0.5] + [0.0] * 3, [-0.3] + [0.0] * 3):
        st = A.drawdown_stats(pd.DataFrame({"trade_date": pd.bdate_range("2020-01-01", periods=4),
                                            "exit_date": pd.bdate_range("2020-01-02", periods=4),
                                            "strategy_return": rets}), emergency_threshold=0.22)
        assert st["peak_capital_date"][0] is None
        assert st["true_max_drawdown"] <= -abs(rets[0]) + 1e-9


def test_frozen_replay_sha_bound():
    replays, parent = W._load_parent(ROOT, {"source_binding": {}})
    checked = parent["checked"]
    assert set(checked) == set(W.FROZEN_REPLAY_SHA)
    assert set(replays) == set(W.FROZEN_REPLAY_SHA)
    for key, sha in checked.items():
        assert sha == W.FROZEN_REPLAY_SHA[key]
    assert parent["summary"]["native"]["coverage"]["native_passed"] == 20


@pytest.mark.skipif(os.environ.get("FIXED_PATH_RUN_CARD") != "1",
                    reason="integration card run; set FIXED_PATH_RUN_CARD=1")
def test_zero_fit_full_workflow(monkeypatch, tmp_path):
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    def boom(*a, **k):
        raise AssertionError("no fit permitted")
    monkeypatch.setattr(Pipeline, "fit", boom); monkeypatch.setattr(Ridge, "fit", boom)
    source_card = ROOT / "research" / W.EXPERIMENT
    s = W.run_fixed_path_fee_drawdown_audit(root=ROOT, card_dir=tmp_path, source_card_dir=source_card)
    assert s["models_trained"] == 0 and s["cases_count"] == 12
    assert s["native_coverage"]["passed_count"] == 12
    assert set(s["risk_case_coverage"]) == set(s["cases"])
    assert (tmp_path / "summary.json").exists() and not (tmp_path / "preregister_plan.json").exists()
    assert Path(s["recorder_uri"]).is_relative_to(tmp_path)
    assert s["plan_full_sha256"] == W._sha(source_card / "preregister_plan.json")


# ---- deep source-native verification helpers (no native re-run) --------------
def _fake_parent(root, window, cost, base, mutate=None, stages=("prefix", "suffix"), status="passed"):
    card = root / W.PARENT_CARD
    (card / "objects").mkdir(parents=True, exist_ok=True)
    natives, path_index, sha_index, results = {}, {}, {}, {}
    bounds = {"prefix": base.iloc[:3], "suffix": base.iloc[3:] if "prefix" in stages else base}
    for stage in stages:
        rows = bounds[stage]
        frame = pd.DataFrame({"signal_date": rows.signal_date.values, "trade_date": rows.trade_date.values,
                              "exit_date": rows.exit_date.values, "native_return": rows.strategy_return.values,
                              "native_soxl_ending_weight": rows.ending_soxl_weight.values,
                              "native_soxx_ending_weight": rows.ending_soxx_weight.values})
        if mutate:
            frame = mutate(stage, frame)
        name = f"native::{W.LAYER}::{window}::{int(cost)}::{stage}_reconciliation"
        rel = f"objects/{name}"
        pd.to_pickle(frame, card / rel)
        path_index[name] = rel
        sha_index[name] = W._sha(card / rel)
        natives[f"{window}::{int(cost)}::{stage}_reconciliation"] = frame
        results[stage] = {"sessions": len(rows), "tolerance": 1e-7, "status": status}
    results["all_stages_passed"] = True
    summary = {"native": {"results": {f"{W.LAYER}::{window}::{int(cost)}": results}}}
    return {"path_index": path_index, "sha_index": sha_index, "summary": summary, "natives": natives}


def test_native_base_verify_deep_passes_and_catches_tampering(tmp_path):
    base = _base(n=8, rets=[0.0] * 8)
    window = "fixture_window"
    parent = _fake_parent(tmp_path, window, 10.0, base)
    good = W._native_base_verify(tmp_path, base, parent, window=window, cost=10.0)
    assert good["all_stages_passed"] and good["full_calendar_exact"]
    assert good["prefix"]["stage_status_read"] == "passed"

    for label, mutate in (("ret", lambda s, f: f.assign(native_return=f.native_return + 0.01) if s == "prefix" else f),
                          ("w", lambda s, f: f.assign(native_soxl_ending_weight=f.native_soxl_ending_weight + 0.02) if s == "suffix" else f),
                          ("date", lambda s, f: f.assign(trade_date=f.trade_date + pd.Timedelta(days=1)) if s == "suffix" else f)):
        bad = _fake_parent(tmp_path / label, window, 10.0, base, mutate=mutate)
        assert not W._native_base_verify(tmp_path / label, base, bad, window=window, cost=10.0)["all_stages_passed"]

    status_bad = _fake_parent(tmp_path / "st", window, 10.0, base, status="failed")
    assert not W._native_base_verify(tmp_path / "st", base, status_bad, window=window, cost=10.0)["all_stages_passed"]

    (tmp_path / W.PARENT_CARD / parent["path_index"][f"native::{W.LAYER}::{window}::10::prefix_reconciliation"]).write_bytes(b"tampered")
    assert not W._native_base_verify(tmp_path, base, parent, window=window, cost=10.0)["all_stages_passed"]


def test_native_base_verify_outer_has_no_prefix(tmp_path):
    base = _base(n=5, rets=[0.0] * 5)
    window = "outer_fixture"
    parent = _fake_parent(tmp_path, window, 10.0, base, stages=("suffix",))
    v = W._native_base_verify(tmp_path, base, parent, window=window, cost=10.0)
    assert v["all_stages_passed"] and v["prefix"]["status"] == "not_required" and v["suffix"]["sessions"] == 5


def test_actual_vs_hypothetical_uses_shared_source_mask_and_asserts_base():
    source = pd.DataFrame({"trade_date": pd.bdate_range("2020-01-01", periods=4),
                           "emergency_active": [True, False, True, True],
                           "drawdown_before": [0.25, 0.10, 0.30, 0.22]})
    base_settle = pd.DataFrame({"drawdown_before": source["drawdown_before"].to_numpy()})
    same = W._actual_vs_hypothetical(source, base_settle, threshold=0.22, base_settlement=True)
    assert same["actual_dd_before_matches_source"] is True and same["actual_emergency_days"] == 3
    assert same["actual_and_hypothetical"] == 3 and same["actual_and_hypothetical_episodes"]
    with pytest.raises(ValueError):
        W._actual_vs_hypothetical(source, pd.DataFrame({"drawdown_before": [0.30, 0.10, 0.05, 0.01]}),
                                  threshold=0.22, base_settlement=True)
    diff = W._actual_vs_hypothetical(source, pd.DataFrame({"drawdown_before": [0.30, 0.10, 0.05, 0.01]}),
                                     threshold=0.22, base_settlement=False)
    assert diff["actual_dd_before_matches_source"] is False and diff["actual_and_hypothetical"] == 1


def test_fixed0_vs_source_reports_intersection_share_episodes():
    source = pd.DataFrame({"trade_date": pd.bdate_range("2020-01-01", periods=4),
                           "emergency_active": [True, True, False, True],
                           "drawdown_before": [0.25, 0.24, 0.10, 0.22]})
    fixed0 = pd.DataFrame({"drawdown_before": [0.25, 0.10, 0.05, 0.22]})
    out = W._fixed0_vs_source(source, fixed0, threshold=0.22)
    assert out["actual_emergency_days"] == 3 and out["fixed0_emergency_days"] == 2
    assert out["fixed0_and_actual_days"] == 2 and out["actual_only_days"] == 1
    assert out["fixed0_share_of_actual"] == pytest.approx(2 / 3)
    assert out["fixed0_and_actual_episodes"] == [{"start": "2020-01-01", "end": "2020-01-01", "days": 1},
                                                 {"start": "2020-01-06", "end": "2020-01-06", "days": 1}]


def test_source_native_readback_reads_frozen_card_input(tmp_path):
    card = tmp_path / W.EXPERIMENT
    card.mkdir()
    doc = {"recorder_id": W.PARENT_RECORDER, "checks": [{"case": "x", "passed": True, "maximum_ending_weight_error": 1e-9,
                                                         "maximum_return_error": 1e-12}],
           "cash_weight_method": "1-stocks", "not_relying_on_default_zero_missing_schedule_weight_error": True}
    (card / "source_qa.json").write_text(json.dumps(doc))
    sha = W._sha(card / "source_qa.json")
    plan = {"source_binding": {"source_qa_file": "source_qa.json", "source_qa_sha256": sha}}
    out = W._source_native_readback(card, plan)
    assert out["status"] == "recorded" and out["all_passed"] and out["checks_count"] == 1
    plan["source_binding"]["source_qa_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        W._source_native_readback(card, plan)
    (card / "source_qa.json").unlink()
    assert W._source_native_readback(card, plan)["status"] == "needs_data"


def test_native_counterfactual_stages_not_empty_pass(tmp_path):
    # A scheduled case with expected prefix+suffix but an empty stage dict must NOT be
    # reported as passed just because all([]) is True.
    empty = W._counterfactual_stage_verdict({}, status="passed", expected_prefix=752, expected_suffix=1004)
    assert empty["all_stages_passed"] is False and empty["present_all_stages"] is False
    missing_prefix = W._counterfactual_stage_verdict(
        {"suffix": {"status": "passed", "sessions": 1004}}, status="passed", expected_prefix=752, expected_suffix=1004)
    assert missing_prefix["all_stages_passed"] is False
    wrong_count = W._counterfactual_stage_verdict(
        {"prefix": {"status": "passed", "sessions": 1}, "suffix": {"status": "passed", "sessions": 1004}},
        status="passed", expected_prefix=752, expected_suffix=1004)
    assert wrong_count["all_stages_passed"] is False
    good = W._counterfactual_stage_verdict(
        {"prefix": {"status": "passed", "sessions": 752}, "suffix": {"status": "passed", "sessions": 1004}},
        status="passed", expected_prefix=752, expected_suffix=1004)
    assert good["all_stages_passed"] is True and good["present_all_stages"] is True
    no_prefix_case = W._counterfactual_stage_verdict(
        {"suffix": {"status": "passed", "sessions": 934}}, status="passed", expected_prefix=0, expected_suffix=934)
    assert no_prefix_case["all_stages_passed"] is True and "prefix" not in no_prefix_case["stages"]


def test_counterfactual_schedule_carries_resettled_returns_and_ending_weights():
    base = _base(n=6, rets=[0.0, 0.02, -0.01, 0.03, 0.0, -0.02])
    resettled = A.resettle_preset_path(base, settlement_cost=25.0, convention="pre_fee_account")
    sched = W._counterfactual_schedule(base, resettled)
    for key in ("ending_soxl_weight", "ending_soxx_weight", "ending_cash_weight"):
        assert key in sched.columns
        assert np.allclose(sched[key], resettled[key])
    assert np.allclose(sched["strategy_return"], resettled["strategy_return"])


def test_risk_small_table_shapes():
    report = W._risk_report(np.array([0.0, 0.01, -0.02]), np.array([0.0, 0.01, -0.02]))
    cases = {"dev::10::10": {"native": {"risk_status": "confirmed"}}}
    table = W._risk_small_table({"risk_analysis::dev::10::10": report}, cases)
    assert set(table["series"]) == {"native", "research"} and (table["case"] == "dev::10::10").all()


def test_risk_case_coverage_is_the_12_case_set_and_needs_data_not_fabricated():
    cases = {f"{w}::{b}::{s}": {"native": {"risk_status": "confirmed" if s != 0 else "needs_data"}}
             for w in W.WINDOWS for b in (10, 25) for s in (0, 10, 25)}
    assert len(cases) == 12
    report = W._risk_report(np.array([0.0, 0.01]), np.array([0.0, 0.01]))
    risk_objects = {f"risk_analysis::{k}": report for k, v in cases.items()
                    if v["native"]["risk_status"] == "confirmed"}
    table = W._risk_small_table(risk_objects, cases)
    assert set(table["case"]) == set(cases)                     # table covers exactly the 12 cases
    missing = table[table["case"].str.endswith("::0")]
    assert (missing["status"] == "needs_data").all() and missing["value"].isna().all()
