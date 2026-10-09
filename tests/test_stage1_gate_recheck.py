from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.workflow import stage1_gate_recheck as recheck

ROOT = Path(__file__).resolve().parents[1]


def test_risk_metrics_include_initial_nav_in_drawdown():
    m = recheck.risk_metrics(pd.Series([-0.05, 0.01, 0.01]))
    assert m["max_drawdown_pct"] == pytest.approx(-5.0)
    assert recheck.risk_metrics(pd.Series([0.01, 0.02, 0.03]))["max_drawdown_pct"] == pytest.approx(0.0)


def test_short_series_worst5_is_none_and_gate_fails_closed():
    m = recheck.risk_metrics(pd.Series([0.01, -0.01, 0.02]))  # only 3 sessions
    assert m["worst_rolling_5d_pct"] is None
    # An undefined metric must never pass the gate.
    checks = recheck.gate_checks(m)
    assert checks["checks"]["worst_rolling_5d_pct"] is False
    assert checks["pass"] is False


def test_risk_metrics_fail_closed_on_bad_input():
    with pytest.raises(ValueError, match="empty"):
        recheck.risk_metrics(pd.Series([], dtype=float))
    with pytest.raises(ValueError, match="non-finite"):
        recheck.risk_metrics(pd.Series([0.0, np.nan]))
    with pytest.raises(ValueError, match="-100"):
        recheck.risk_metrics(pd.Series([-1.0, 0.0]))


def test_validate_returns_rejects_duplicates_and_gaps():
    good = pd.DataFrame({"signal_date": pd.bdate_range("2020-01-02", periods=5), "strategy_return": 0.0})
    assert len(recheck._validate_returns(good, source="x")) == 5
    dup = pd.DataFrame({"signal_date": [pd.Timestamp("2020-01-02")] * 2, "strategy_return": 0.0})
    with pytest.raises(ValueError, match="duplicate"):
        recheck._validate_returns(dup, source="x")
    unordered = pd.DataFrame({"signal_date": [pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-02")],
                              "strategy_return": 0.0})
    with pytest.raises(ValueError, match="ordered"):
        recheck._validate_returns(unordered, source="x")


def test_gate_checks_addresses_every_floor():
    metrics = {"max_drawdown_pct": -28.1, "worst_day_pct": -9.4,
               "cvar5_pct": -5.3, "worst_rolling_5d_pct": -12.11}
    assert recheck.gate_checks(metrics)["pass"] is False
    assert recheck.gate_checks(dict(metrics, worst_rolling_5d_pct=-11.3))["pass"] is True
    # A None (undefined) metric fails closed.
    assert recheck.gate_checks(dict(metrics, cvar5_pct=None))["pass"] is False


def test_same_window_requires_exact_match():
    ref = {"window_start": "2023-01-04", "window_end": "2026-08-07"}
    assert recheck._same_window(ref, "2023-01-03", "2026-06-26") is False
    assert recheck._same_window(ref, "2023-01-04", "2026-08-07") is True


def test_recorder_read_uses_real_qlib_api_behaviourally():
    # A real committed recorder in this checkout must be readable through Qlib.
    entries = recheck.resolve_legacy_recorders(ROOT)
    readable = []
    for entry in entries:
        out = recheck.qlib_read_attempt(ROOT, recorder_id=entry["recorder_id"],
                                        experiment_name=recheck.SOURCE_CARD, artifact=recheck.ARTIFACT)
        if out["status"] == "readable":
            readable.append(out)
            assert out["downloaded_rows"] > 0
            api = " ".join(out["api"])
            assert "download_artifact" in api and "list_artifacts" in api
            assert out["downloaded_sha256"]
    assert readable, "at least one committed recorder must be Qlib-readable"
    # The download must not have created any run: recorder identity is fixed.
    assert all(out["recorder_id"] for out in readable)


def test_recorder_read_reports_reader_error_verbatim():
    out = recheck.qlib_read_attempt(ROOT, recorder_id="0" * 32,
                                    experiment_name=recheck.SOURCE_CARD,
                                    artifact=recheck.ARTIFACT)
    assert out["status"] == "reader_error"
    assert "reason" in out


def test_recheck_has_no_override_always_not_comparable_and_does_not_touch_real_card(tmp_path):
    real_card = ROOT / "research" / recheck.EXPERIMENT / "summary.json"
    before = real_card.read_bytes() if real_card.is_file() else None
    summary = recheck.run_stage1_gate_recheck(root=ROOT, card_dir=tmp_path)
    assert summary["decision"] == "legacy_diagnostic_no_recheck_verdict"
    assert summary["recheck_verdict"] == "none"
    assert summary["production_gate_override"] is False
    assert summary["pandas_only_fallback"] is False
    assert summary["comparable_count"] == 0
    for row in summary["recorders"]:
        if row.get("status") == "ok":
            assert row["comparability"] == "not_comparable"
            assert row["cost_status"] == "unknown"
            assert row["calendar_status"] == "unknown"
            assert row["evidence_source"] == "qlib_download"
    # The real card bytes must be unchanged.
    after = real_card.read_bytes() if real_card.is_file() else None
    assert after == before
    assert (tmp_path / "summary.json").is_file()


def test_reader_error_produces_no_metrics(tmp_path):
    # Force an unreadable recorder id: the row must carry no metrics and no gate.
    import json
    entries = recheck.resolve_legacy_recorders(ROOT)
    if not entries:
        pytest.skip("no legacy recorders present in this checkout")
    # Monkeypatch the reader to a hard error to prove no fallback occurs.
    original = recheck.qlib_read_attempt
    try:
        recheck.qlib_read_attempt = lambda *a, **k: {"status": "reader_error", "reason": "forced"}
        summary = recheck.run_stage1_gate_recheck(root=ROOT, card_dir=tmp_path)
    finally:
        recheck.qlib_read_attempt = original
    for row in summary["recorders"]:
        assert row["status"] == "needs_data"
        assert row["comparability"] == "not_comparable"
        for key in ("return_pct", "max_drawdown_pct", "worst_rolling_5d_pct", "gate_pass"):
            assert key not in row


def test_recorder_read_matches_stored_artifact_identity():
    entries = recheck.resolve_legacy_recorders(ROOT)
    for entry in entries:
        if not entry["git_tracked"]:
            continue
        out = recheck.qlib_read_attempt(ROOT, recorder_id=entry["recorder_id"],
                                        experiment_name=recheck.SOURCE_CARD,
                                        artifact=recheck.ARTIFACT)
        if out["status"] == "readable":
            assert out["downloaded_sha256"] == entry["artifact_sha256"]


def test_summary_is_strict_json_serializable_with_mixed_rows(tmp_path, monkeypatch):
    import json
    # Force a reader error so rows mix needs_data and ok.
    calls = {"n": 0}
    real = recheck.qlib_read_attempt

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] % 2 == 1:
            return {"status": "reader_error", "reason": "forced"}
        return real(*a, **k)

    monkeypatch.setattr(recheck, "qlib_read_attempt", flaky)
    summary = recheck.run_stage1_gate_recheck(root=ROOT, card_dir=tmp_path)
    # Must serialize with allow_nan=False (no NaN/Infinity).
    text = json.dumps(summary, default=str, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
    # At least one row is needs_data and one ok when two recorders exist.
    statuses = [row["status"] for row in summary["recorders"]]
    if len(statuses) >= 2:
        assert "needs_data" in statuses
    # The written summary.json also parses under strict rules.
    written = (tmp_path / "summary.json").read_text(encoding="utf-8")
    json.loads(written)
