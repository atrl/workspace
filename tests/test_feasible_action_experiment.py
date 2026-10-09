"""Verification for the feasible-action training-domain experiment."""
from pathlib import Path

import pandas as pd
import pytest

from src.workflow import feasible_action_experiment as X

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def _summary(tmp_path_factory):
    return X.run_feasible_action_training(root=ROOT, card_dir=tmp_path_factory.mktemp("feasible"))


def test_providers_are_frozen_with_manifest_hashes():
    frozen = X._freeze_providers(ROOT)
    assert set(frozen) == {"intraday_provider", "execution_mark_provider", "coherent_provider"}
    for entry in frozen.values():
        assert entry["manifest_sha256"]
    assert frozen["coherent_provider"]["daily_source_manifest_sha256"]
    assert frozen["intraday_provider"]["trading_calendar_sha256"]


def test_gates_require_full_checks_and_missing_is_needs_data():
    records = [
        {"variant": "candidate_feasible", "segment": "development_walk_forward", "cost_bps": c,
         "return_pct": 100.0, "max_drawdown_pct": -10.0, "worst_day_pct": -5.0, "cvar5_pct": -1.0,
         "worst_rolling_5d_pct": -3.0, "open_trade_days": 50, "total_turnover": 10.0, "terminal_wealth": 2.0}
        for c in (10.0, 25.0)
    ] + [
        {"variant": "no_hold_reference", "segment": "development_walk_forward", "cost_bps": c,
         "return_pct": 90.0, "max_drawdown_pct": -12.0, "worst_day_pct": -6.0, "cvar5_pct": -2.0,
         "worst_rolling_5d_pct": -4.0, "open_trade_days": 80, "total_turnover": 20.0, "terminal_wealth": 2.0}
        for c in (10.0, 25.0)
    ]
    soxx = [{"segment": "development_walk_forward", "gross_return_pct": 200.0}]
    # With no annual replays the annual-wealth-floor gate cannot be computed -> needs_data, not skipped.
    rows = X._gates(records, {}, soxx)["rows"]
    assert rows and rows[0]["status"] == "needs_data"
    assert rows[0]["all_pass"] is False


def test_full_run_control_reproduces_and_native_passes(_summary):
    assert _summary["decision"] == "feasible_action_domain_contrast_reported"
    assert _summary["control_reproduction"] is True
    native = _summary["native_reconciliation"]
    assert native["status"] == "passed"
    for key, entry in native["results"].items():
        assert (entry.get("suffix", {}).get("status") or entry.get("status")) == "passed"
        expected = "pre_fee_account" if key.startswith(("active_hold_baseline", "no_hold_reference")) else "post_fee_desired"
        assert entry["target_convention"] == expected
        assert entry["commission_per_1"] in (0.001, 0.0025)
    assert _summary["promotion_eligible"] is False


def test_model_inventory_includes_control_reproduction_column(_summary):
    inv = _summary["model_inventory"]
    assert inv["control_reproduction"]["reproduces_sealed_control"] is True
    assert inv["control_all_state"]["annual_models"] > 0
    assert inv["candidate_feasible"]["annual_models"] > 0


def test_soxx_benchmark_reports_gross_and_real_net_costs(_summary):
    rows = _summary["soxx_benchmark"]
    assert rows
    for row in rows:
        assert row["gross_return_pct"] > row["net10_return_pct"] > row["net25_return_pct"]


def test_premise_is_marked_not_supported(_summary):
    assert "NOT supported" in _summary["empirical_prior"]["note"]
    assert "not proof" in _summary["empirical_prior"]["framing"]
