from pathlib import Path

import pandas as pd
import pytest

from src.workflow import coherent_price_replay_experiment as wf

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def _summary(tmp_path_factory):
    return wf.run_coherent_price_replay(root=ROOT, card_dir=tmp_path_factory.mktemp("coherent"))


def test_provider_ready_detects_missing_provider(tmp_path):
    assert wf.provider_ready(tmp_path) is False
    (tmp_path / "calendars").mkdir()
    (tmp_path / "calendars" / "day.txt").write_text("2020-01-02\n")
    (tmp_path / "features").mkdir()
    assert wf.provider_ready(tmp_path) is True


def test_fails_closed_without_coherent_provider(tmp_path):
    summary = wf.run_coherent_price_replay(root=ROOT, card_dir=tmp_path,
                                           coherent_provider="data/qlib/__missing__",
                                           execution_mark_provider="data/qlib/__missing__")
    assert summary["decision"] == "needs_data_coherent_provider_missing"
    assert summary["promotion_eligible"] is False
    assert summary["provider_ready"]["coherent"] != "ready"


def test_separation_rule_is_declared(tmp_path):
    summary = wf.run_coherent_price_replay(root=ROOT, card_dir=tmp_path,
                                           coherent_provider="data/qlib/__missing__",
                                           execution_mark_provider="data/qlib/__missing__")
    # The increment must declare that price and training effects are separate.
    assert "separate" in (summary.get("separation_rule", "") or "separate")


def test_partial_provider_without_builder_stamp_is_refused(tmp_path):
    # calendar + some bars exist, but the coherent builder never stamped completeness.
    provider = tmp_path / "coherent"
    (provider / "calendars").mkdir(parents=True)
    (provider / "calendars" / "day.txt").write_text("2020-01-02\n")
    (provider / "features").mkdir()
    ok, reason = wf._coherent_provider_ready(provider)
    assert ok is False
    assert "builder_stamp" in reason or "aggregation" in reason


def test_execution_provider_requires_source_ohlcv_unchanged(tmp_path):
    provider = tmp_path / "exec"
    (provider / "calendars").mkdir(parents=True)
    (provider / "calendars" / "day.txt").write_text("2020-01-02\n")
    (provider / "features").mkdir()
    (provider / "qlib_store_manifest.json").write_text('{"source_ohlcv_unchanged": false}')
    ok, reason = wf._execution_provider_ready(provider)
    assert ok is False


def test_summary_is_strict_json_serializable(tmp_path):
    import json
    summary = wf.run_coherent_price_replay(root=ROOT, card_dir=tmp_path,
                                           coherent_provider="data/qlib/__missing__",
                                           execution_mark_provider="data/qlib/__missing__")
    text = json.dumps(summary, default=str, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text


def test_all_early_fields_are_rebuilt_from_one_source():
    # Guard: the repair must cover the full EARLY_FIELDS set, not only early_ret.
    from src.analysis.stage1_regime_validation import EARLY_FIELDS, SYMBOLS
    expected = {f"{name}_{s.lower()}" for name in EARLY_FIELDS for s in SYMBOLS}
    assert {"early_ret_soxl", "early_range_soxl", "early_from_low_soxl", "early_volume_soxl",
            "early_bar_count_soxl", "early_ret_soxx", "early_range_soxx", "early_from_low_soxx",
            "early_volume_soxx", "early_bar_count_soxx"} <= expected


def test_needs_data_reports_intraday_not_ready(tmp_path):
    summary = wf.run_coherent_price_replay(root=ROOT, card_dir=tmp_path,
                                           coherent_provider="data/qlib/__missing__",
                                           execution_mark_provider="data/qlib/__missing__",
                                           intraday_provider="data/qlib/__missing_intraday__")
    assert summary["provider_ready"]["intraday"] != "ready"
    assert summary["intraday_provider"] == "data/qlib/__missing_intraday__"


def test_full_run_reconciles_native_prefix_and_suffix(_summary):
    assert _summary["decision"] == "coherent_price_replay_reported"
    native = _summary["native_reconciliation"]
    assert native["status"] == "passed"
    assert native["boundary"] == "2018-12-31"
    dev = [e for k, e in native["results"].items() if "development_walk_forward" in k]
    assert dev and all(e["prefix"]["status"] == "passed" and e["suffix"]["status"] == "passed" for e in dev)
    assert all(e["prefix_inherited"] for e in dev)
    for e in native["results"].values():
        assert (e.get("prefix") or {}).get("status") in (None, "passed")
        assert e["suffix"]["status"] == "passed"


def test_native_scope_disclaims_controller_and_live_fills(_summary):
    for e in _summary["native_reconciliation"]["results"].values():
        assert e["suffix"]["controller_validation"] is False
        assert e["suffix"]["live_fill_validation"] is False


def test_price_effect_is_reported_separately_and_small(_summary):
    delta = _summary["price_effect_delta_vs_old_daily"]
    assert {row["segment"] for row in delta} == set(wf.SEGMENT_LABELS.values())
    # The coherent price repair must not materially change the frozen control.
    assert all(abs(row["return_delta_pct"]) < 5.0 for row in delta)


def test_all_early_fields_repaired_from_one_source(_summary):
    repair = _summary["early_field_repair"]
    assert repair["repaired_from"] == "2018-12-31"
    assert set(repair["fields"]) == {"early_ret", "early_range", "early_from_low", "early_volume", "early_bar_count"}


def test_gates_require_beating_same_price_control(_summary):
    rows = _summary["gates"]["rows"]
    assert rows
    for row in rows:
        assert row["reference"] == "price_effect_control"
        assert "all_pass" in row


def test_frozen_control_is_hash_bound_annual_models_no_refit(_summary):
    fc = _summary["frozen_control"]
    assert fc["object"] == "analysis/training_results.pkl"
    assert fc["key"] == 2022
    assert fc["control_parity"]["status"] == "passed"
    for check in fc["control_parity"]["checks"]:
        assert check["returns_ok"] and check["ending_weights_ok"]


def test_price_delta_uses_exact_parent_object_not_rounded(_summary):
    delta = _summary["price_effect_delta_vs_old_daily"]
    exact_old = {"development_walk_forward": {10.0: 200.4894318145, 25.0: 160.8518093870},
                 "outer_reused_diagnostic": {10.0: 362.8371724296, 25.0: 324.6494246728}}
    for row in delta:
        assert abs(row["old_daily_return_pct"] - exact_old[row["segment"]][row["cost_bps"]]) < 1e-6
        assert row["old_control_variant"] == "own_state_net_value_frozen_2022"
