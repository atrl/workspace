"""Verification for the NQ overnight 18:00->10:00 post-10:31 diagnostic."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import nq_overnight_signal as A
from src.workflow import nq_overnight_after1031 as W

ROOT = Path(__file__).resolve().parents[1]


def test_sources_frozen_and_receipt_digest_matches_provider():
    frozen = A.freeze_sources(ROOT)
    for entry in frozen.values():
        assert (entry.get("sha256") or entry.get("manifest_sha256"))
    days, digest = A.load_trading_days(ROOT)
    assert digest == frozen["uniform_intraday"]["trading_calendar_sha256"]
    assert set(days.values()) <= {"WHOLE", "MORNING"}


def test_future_bars_do_not_change_the_signal():
    nq = A.read_nq(ROOT)
    days, _ = A.load_trading_days(ROOT)
    base = A.nq_overnight_signal(nq, trading_days=days)
    perturbed = nq.copy()
    mask = perturbed["datetime"].dt.strftime("%H:%M") >= "10:00"
    perturbed.loc[mask, "$close"] = perturbed.loc[mask, "$close"] * 1.5
    other = A.nq_overnight_signal(perturbed, trading_days=days)
    assert np.allclose(base["nq_overnight"].to_numpy(), other["nq_overnight"].to_numpy())


def test_next_open_is_strictly_after_decision():
    calendar, valid = A.build_calendar(ROOT)
    for symbol in A.SYMBOLS:
        exit_at = pd.to_datetime(valid[f"{symbol.lower()}_next_open_at"])
        assert (exit_at > valid["trade_date"] + pd.Timedelta(hours=10, minutes=31)).all()


def test_exact_roundtrip_matches_closed_form():
    for cost in A.COSTS_BPS:
        for r in (-0.03, 0.0, 0.02):
            settled = A._roundtrip_net(np.array([1.0, 0.0, 0.0]), r, 0.0, cost)
            exact = (1.0 + r) * (1.0 - cost / 10_000.0) / (1.0 + cost / 10_000.0) - 1.0
            assert abs(settled - exact) < 1e-10


@pytest.fixture(scope="module")
def _summary(tmp_path_factory):
    return W.run_nq_overnight_after1031(root=ROOT, card_dir=tmp_path_factory.mktemp("nq"))


def test_zero_models_and_direction_fixed(_summary):
    assert _summary["models_trained"] == 0
    assert _summary["parameter_search"] == "none"
    assert _summary["direction"] == 1.0


def test_half_days_from_receipt_and_audited(_summary):
    assert _summary["morning_days_in_range"] > 0
    assert _summary["valid_morning_days"] < _summary["morning_days_in_range"]  # NQ source lacks them
    assert _summary["dropped_needs_data"] > 0


def test_primary_and_economic_point_estimates(_summary):
    primary = _summary["spread_ic_primary"]
    assert primary["status"] in ("ok", "needs_data")
    for row in _summary["economic_long_only"]:
        assert row["legs"].startswith("cash->target")
        assert row["ci_status"] in ("ok", "needs_data")


def test_stop_decision_without_fee_robust_edge(_summary):
    robust = [r for r in _summary["economic_long_only"]
              if r["book"] == A.PRIMARY_BOOK and r["cost_bps"] == 25.0
              and r["ci_status"] == "ok" and r["ci_low_bps"] is not None and r["ci_low_bps"] > 0.0 and r["mean_net_bps"] > 0.0]
    expected = "end_route_no_further_search" if not robust else "conditional_edge_needs_independent_forward"
    assert _summary["decision"] == expected
    assert _summary["promotion_eligible"] is False


def test_summary_is_strict_json(_summary):
    import json
    text = json.dumps(_summary, default=str, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
