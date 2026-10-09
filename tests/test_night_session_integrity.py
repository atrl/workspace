from pathlib import Path

import pandas as pd

import pytest

from src.analysis import night_session_integrity as integrity

ROOT = Path(__file__).resolve().parents[1]


def test_integrity_correction_marks_invalid_and_records_defects():
    rec = integrity.integrity_correction(ROOT)
    assert rec["status"] == "invalid_for_promotion"
    assert rec["objective_achieved"] is False
    assert rec["promotion_eligible"] is False
    assert any("look_ahead" in d for d in rec["known_defects"])


def test_safe_cutoff_collapses_the_ic():
    unsafe = integrity.recompute_ic(ROOT, cutoff=integrity.UNSAFE_CUTOFF)
    safe = integrity.recompute_ic(ROOT, cutoff=integrity.SAFE_CUTOFF)
    assert unsafe["status"] == "ok" and safe["status"] == "ok"
    assert unsafe["sessions"] == safe["sessions"]
    # The unsafe 09:30 cutoff reads the 09:00-10:00 bar and shows a large IC.
    assert unsafe["ic_soxl"] > 0.15
    # Under a safe 09:00 cutoff the signal collapses toward zero.
    assert abs(safe["ic_soxl"]) < 0.05
    assert abs(safe["ic_excess"]) < 0.05


def test_safe_cutoff_never_exceeds_unsafe_information():
    unsafe = integrity.recompute_ic(ROOT, cutoff=integrity.UNSAFE_CUTOFF)
    safe = integrity.recompute_ic(ROOT, cutoff=integrity.SAFE_CUTOFF)
    assert abs(safe["ic_excess"]) < abs(unsafe["ic_excess"])


def test_bar_span_audit_needs_data_without_snapshot(tmp_path):
    # No snapshot file at all -> needs_data, never an assumption.
    span = integrity.bar_span_audit(tmp_path)
    assert span["status"] == "needs_data"


def test_one_hour_09_bar_is_not_available_at_0930():
    # The 09:00 one-hour bar covers 09:00-10:00 and is only available at 10:00.
    label = pd.Timestamp("2026-10-08 09:00")
    available = integrity.bar_available_at(label, "1h")
    assert available == pd.Timestamp("2026-10-08 10:00")
    decision = pd.Timestamp("2026-10-08 09:30")
    assert integrity.selectable_at(label, decision, "1h") is False
    # The same bar IS available at/after 10:00.
    assert integrity.selectable_at(label, pd.Timestamp("2026-10-08 10:00"), "1h") is True


def test_bar_available_at_rejects_unknown_resolution():
    import pytest as _pytest
    with _pytest.raises(ValueError, match="resolution"):
        integrity.bar_available_at(pd.Timestamp("2026-10-08 09:00"), "2h")


def test_night_session_card_is_marked_invalid_for_promotion():
    import json
    card = ROOT / "research/night_session_signal_20261010/summary.json"
    summary = json.loads(card.read_text())
    assert summary["decision"] == "invalid_for_promotion_look_ahead"
    assert summary["promotion_eligible"] is False
    assert summary["objective_achieved"] is False
    assert summary["integrity"]["status"] == "invalid_for_promotion"
    assert summary["original_metrics_invalid"] is True
    # The original metrics are preserved, just labeled invalid.
    assert summary["integrity"]["superseded_metrics_preserved"]["original_promotion_eligible"] is True


def test_rerun_is_blocked_on_invalid_card(tmp_path):
    import json
    from src.workflow import night_session_signal as wf
    # A card already marked invalid must not be overwritten by a rerun.
    (tmp_path / "summary.json").write_text(json.dumps({"decision": "invalid_for_promotion_look_ahead"}))
    with pytest.raises(ValueError, match="invalid_for_promotion"):
        wf.write_card_files({"x": 1}, card_dir=tmp_path)
    # The original bytes are untouched.
    assert json.loads((tmp_path / "summary.json").read_text())["decision"] == "invalid_for_promotion_look_ahead"


def test_guard_allows_fresh_card(tmp_path):
    from src.workflow import night_session_signal as wf
    # No existing summary: the guard is a no-op.
    wf._guard_invalid_card(tmp_path)


def test_run_entry_is_unconditionally_blocked_any_path(tmp_path):
    import tempfile
    from src.workflow import night_session_signal as wf
    # A completely fresh, empty output directory must NOT be able to run the
    # known-defective workflow and produce a success result.
    for target in (tmp_path / "fresh", Path(tempfile.mkdtemp())):
        with pytest.raises(ValueError, match="blocked_known_integrity_defects"):
            wf.run_night_session_signal(root=ROOT, card_dir=target,
                                        recorder_uri=target / "qlib_recorder")


def test_span_audit_consumes_snapshot_and_reaggregates():
    span = integrity.bar_span_audit(ROOT)
    assert span["status"] == "ok"
    assert span["all_five_fields_identical"] is True
    assert span["snapshot_sha256"] == "4a5f55b37fc3a0298420c32008e6940d6e79c7347bd8da64e546ffe634a97ae1"
    assert span["bound_raw_parquet"] == "5caf8600ca1ad8428032ed8049bd64cca330faed22685d58c05ac1fa3b2b0dd6"
    # Each required day must have been re-aggregated from 12 five-minute bars.
    assert len(span["days"]) == 4
    for row in span["days"]:
        assert row["status"] == "ok"
        assert row["identical"] is True
        assert set(row["fields"]) == {"open", "high", "low", "close", "volume"}


def test_span_audit_needs_data_when_snapshot_incomplete(tmp_path):
    import json
    doc = {"days": [{"day": "2026-10-05", "five_minute_bars": [], "hourly_bar": {}}]}
    (tmp_path / "snap.json").write_text(json.dumps(doc))
    span = integrity.bar_span_audit(tmp_path, snapshot="snap.json")
    assert span["status"] == "needs_data"


def test_span_audit_needs_data_when_one_day_not_exact(tmp_path):
    import json
    import numpy as np

    def stamps(day, n):
        start = pd.Timestamp(day).tz_localize("America/New_York") + pd.Timedelta(hours=9)
        return [(start + pd.Timedelta(minutes=5 * i)).isoformat() for i in range(n)]

    def day(d, n):
        bars = [{"timestamp_et": t, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}
                for t in stamps(d, n)]
        return {"day": d, "five_minute_bars": bars,
                "hourly_bar": {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": float(n)}}

    doc = {"days": [day("2026-10-05", 12), day("2026-10-06", 12),
                    day("2026-10-07", 12), day("2026-10-08", 11)]}
    (tmp_path / "snap.json").write_text(json.dumps(doc))
    span = integrity.bar_span_audit(tmp_path, snapshot="snap.json")
    assert span["status"] == "needs_data"


def test_span_audit_mismatch_when_hourly_differs(tmp_path):
    import json

    def stamps(day):
        start = pd.Timestamp(day).tz_localize("America/New_York") + pd.Timedelta(hours=9)
        return [(start + pd.Timedelta(minutes=5 * i)).isoformat() for i in range(12)]

    def day(d, close):
        bars = [{"timestamp_et": t, "open": 2.0, "high": 2.0, "low": 2.0, "close": 2.0, "volume": 1.0}
                for t in stamps(d)]
        return {"day": d, "five_minute_bars": bars,
                "hourly_bar": {"open": 2.0, "high": 2.0, "low": 2.0, "close": close, "volume": 12.0}}

    import hashlib
    raw = ROOT / "data/raw/external/night_session_20261010_v1/NQ_F.parquet"
    raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    doc = {"local_raw_parquet": {"path": str(raw), "sha256": raw_sha},
           "days": [day("2026-10-05", 2.0), day("2026-10-06", 2.0),
                    day("2026-10-07", 2.0), day("2026-10-08", 9.9)]}
    (tmp_path / "snap.json").write_text(json.dumps(doc))
    span = integrity.bar_span_audit(tmp_path, snapshot="snap.json")
    # All four days present and exact-bar, but one field mismatches -> mismatch.
    assert span["status"] == "mismatch"
    assert span["all_five_fields_identical"] is False


def test_span_audit_needs_data_when_raw_parquet_binding_missing(tmp_path):
    import json
    doc = {"local_raw_parquet": {"path": "nope/missing.parquet", "sha256": "deadbeef"},
           "days": [{"day": "2026-10-05"}]}
    (tmp_path / "snap.json").write_text(json.dumps(doc))
    span = integrity.bar_span_audit(tmp_path, snapshot="snap.json")
    assert span["status"] == "needs_data"
    assert "missing" in span["reason"]
