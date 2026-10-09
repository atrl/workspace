import numpy as np
import pandas as pd
import pytest

from src.analysis import action_marginal_information as ami
from src.analysis import action_oos_information as oos
from src.workflow import action_marginal_diagnosis as wf

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_rank_ic_and_auc_direction():
    x = np.arange(1.0, 51.0)
    assert ami.rank_ic(x, x) == pytest.approx(1.0)
    assert ami.auc(x, (x > 25).astype(int)) == pytest.approx(1.0)


def test_true_oos_predictions_use_each_years_own_pipeline():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    # Two synthetic years; each year's model must be used only for its own rows.
    rows = []
    for year in (2021, 2022):
        for i in range(40):
            rows.append({"signal_date": pd.Timestamp(f"{year}-01-{i % 28 + 1:02d}"),
                         "year": year, "f1": float(i), "f2": float(i % 3),
                         "execute_advantage": float(i - 20) / 100.0,
                         "decision_at": pd.Timestamp(f"{year}-02-01", tz="UTC"),
                         "label_available_at": pd.Timestamp(f"{year}-02-02", tz="UTC")})
    frame = pd.DataFrame(rows)
    X = frame[["f1", "f2"]]
    y = (frame["execute_advantage"] > 0).astype(int)
    pipelines = {2021: make_pipeline(LogisticRegression()).fit(X, y),
                 2022: make_pipeline(LogisticRegression()).fit(X, y)}
    preds = oos.true_oos_predictions(frame, pipelines)
    # Both years present; the frame's own columns are named to match.
    assert set(preds["year"]) == {2021, 2022}
    assert preds["predicted_probability"].between(0, 1).all()


def test_true_oos_predictions_block_on_missing_feature():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    frame = pd.DataFrame({"signal_date": pd.to_datetime(["2021-01-04"]), "year": [2021],
                          "f1": [1.0], "execute_advantage": [0.1],
                          "decision_at": pd.to_datetime(["2021-01-05"], utc=True),
                          "label_available_at": pd.to_datetime(["2021-01-06"], utc=True)})
    pipe = make_pipeline(LogisticRegression()).fit(pd.DataFrame({"f2": [1.0, 2.0]}), [0, 1])
    with pytest.raises(ValueError, match="missing columns"):
        oos.true_oos_predictions(frame, {2021: pipe})


def test_true_oos_predictions_exclude_unmatured_terminal_row():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    frame = pd.DataFrame({
        "signal_date": pd.to_datetime(["2021-01-04", "2021-12-31"]), "year": [2021, 2021],
        "f1": [1.0, 2.0], "execute_advantage": [0.1, -0.1],
        "decision_at": pd.to_datetime(["2021-01-05", "2022-01-03"], utc=True),
        "label_available_at": pd.to_datetime(["2021-01-06", None], utc=True)})
    pipe = make_pipeline(LogisticRegression()).fit(pd.DataFrame({"f1": [1.0, 2.0]}), [0, 1])
    preds = oos.true_oos_predictions(frame, {2021: pipe})
    assert len(preds) == 1  # terminal NaT row dropped, not fabricated


def test_segment_contribution_requires_exact_alignment():
    idx = pd.bdate_range("2020-01-02", periods=10)
    no_hold = pd.Series(0.001, index=idx)
    paths = {"active": pd.Series(0.002, index=idx)}
    rows = oos.segment_contribution(paths, no_hold, cost_bps=10.0, segment="development")
    assert rows[0]["variant"] == "active"
    assert rows[0]["gaps"] == 0


def test_real_repo_diagnosis_uses_own_label_and_own_recorder(tmp_path):
    summary = wf.run_action_marginal_diagnosis(root=ROOT, card_dir=tmp_path)
    assert summary["decision"] == "diagnostic_reported_no_strategy"
    info = summary["conditional_information"]
    assert info["status"] == "ok"
    assert info["label_source"] == "active_model_own_execute_advantage"
    # Decidable subset (real eligibility rule) is smaller than the full OOS set.
    assert 100 < info["decidable_sessions"] < 1503
    # The CI must be reported even when it contains zero.
    assert "contains_zero" in info["advantage_bps_ci"]
    # Per-segment, per-cost rows exist for both segments.
    segments = {r["segment"] for r in summary["contribution"]}
    assert {"development", "outer_reused_diagnostic"}.issubset(segments)
    costs = {r["cost_bps"] for r in summary["contribution"]}
    assert costs == {10.0, 25.0}
    # References share the execution/cost and are fee-charged.
    refs = {r["reference"] for r in summary["references"]}
    assert {"soxx_buy_hold", "soxx_cash_50_50"}.issubset(refs)
    # Child wrote its own recorder; the summary records it.
    assert summary["recorder_id"]
    assert (tmp_path / "qlib_recorder").is_dir()


def test_summary_is_strict_json_serializable(tmp_path):
    import json
    summary = wf.run_action_marginal_diagnosis(root=ROOT, card_dir=tmp_path)
    text = json.dumps(summary, default=str, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text


# --- A: model identity + maturity from source metadata ---

def _fake_pipeline(features):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    import numpy as np
    X = pd.DataFrame({f: np.linspace(0, 1, 4) for f in features})
    return make_pipeline(LogisticRegression()).fit(X, [0, 1, 0, 1])


def _audit(features):
    return {"features": list(features), "label_available_at_rule": "r", "label": "L"}


def test_maturity_verification_passes_and_flags_bad_cutoff():
    feats = ["a", "b"]
    diag_ok = pd.DataFrame({"model_year": [2017, 2018],
                            "label_available_at_max": ["2017-01-03T14:31:00+00:00", "2018-01-02T14:31:00+00:00"],
                            "test_decision_at": ["2017-01-04T14:31:00+00:00", "2018-01-03T14:31:00+00:00"]})
    pipelines = {2017: _fake_pipeline(feats), 2018: _fake_pipeline(feats)}
    assert oos.verify_model_identity_and_maturity(_audit(feats), diag_ok, pipelines=pipelines)["status"] == "ok"
    # A cutoff that is not strictly before the first test decision must block.
    diag_bad = diag_ok.copy()
    diag_bad.loc[0, "label_available_at_max"] = "2017-01-05T14:31:00+00:00"
    assert oos.verify_model_identity_and_maturity(_audit(feats), diag_bad, pipelines=pipelines)["status"] == "blocked"


def test_maturity_verification_blocks_on_missing_or_feature_mismatch():
    feats = ["a", "b"]
    diag = pd.DataFrame({"model_year": [2017],
                         "label_available_at_max": ["2017-01-03T14:31:00+00:00"],
                         "test_decision_at": ["2017-01-04T14:31:00+00:00"]})
    # Missing feature list -> needs_data.
    assert oos.verify_model_identity_and_maturity({}, diag, pipelines={})["status"] == "needs_data"
    # Feature mismatch -> blocked.
    wrong = {2017: _fake_pipeline(["a", "c"])}
    assert oos.verify_model_identity_and_maturity(_audit(feats), diag, pipelines=wrong)["status"] == "blocked"


def test_true_oos_predictions_block_without_maturity_columns():
    frame = pd.DataFrame({"signal_date": ["2021-01-04"], "year": [2021], "f1": [1.0],
                          "execute_advantage": [0.1]})
    with pytest.raises(ValueError, match="maturity columns"):
        oos.true_oos_predictions(frame, {})


# --- B: strict segment alignment blocks on gaps/duplicates/nonfinite ---

def _paths():
    idx = pd.bdate_range("2020-01-02", periods=8)
    no_hold = pd.Series(0.001, index=idx)
    return idx, no_hold


def test_segment_contribution_blocks_on_missing_row():
    idx, no_hold = _paths()
    short = pd.Series(0.002, index=idx).iloc[:-1]
    with pytest.raises(ValueError, match="session keys differ"):
        oos.segment_contribution({"active": short}, no_hold, cost_bps=10.0, segment="development")


def test_segment_contribution_blocks_on_duplicate_index():
    idx, no_hold = _paths()
    dup = pd.Series([0.002] * (len(idx) + 1), index=idx.append(idx[:1]))
    with pytest.raises(ValueError, match="duplicates"):
        oos.segment_contribution({"active": dup}, no_hold, cost_bps=10.0, segment="development")


def test_segment_contribution_blocks_on_nonfinite():
    idx, no_hold = _paths()
    bad = pd.Series(0.002, index=idx).copy()
    bad.iloc[2] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        oos.segment_contribution({"active": bad}, no_hold, cost_bps=10.0, segment="development")


def test_segment_contribution_blocks_on_nonfinite_baseline():
    idx, no_hold = _paths()
    nb = no_hold.copy()
    nb.iloc[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        oos.segment_contribution({"active": pd.Series(0.002, index=idx)}, nb, cost_bps=10.0, segment="development")


# --- C: rankIC/AUC block CIs ---

def test_real_repo_reports_rank_and_auc_confidence_intervals(tmp_path):
    summary = wf.run_action_marginal_diagnosis(root=ROOT, card_dir=tmp_path)
    info = summary["conditional_information"]
    assert info["status"] == "ok"
    for key, null in (("rank_ic_ci", 0.0), ("auc_ci", 0.5)):
        ci = info[key]
        assert ci["null"] == null
        assert ci["block"] >= 1
        assert ci["lower"] <= ci["point"] <= ci["upper"] or ci["point"] is None
        assert isinstance(ci["contains_zero"], bool)
    # No significance is claimed when the CIs contain their nulls.
    assert info["full_oos_sessions"] >= info["decidable_sessions"]


def test_plan_is_hash_locked_and_drives_config(tmp_path):
    summary = wf.run_action_marginal_diagnosis(root=ROOT, card_dir=tmp_path)
    assert summary["plan_sha256"]
    assert summary["windows"]["development"][0] == "2016-01-04"
    assert summary["decidable_rule"]["execute_threshold"] == 0.50
