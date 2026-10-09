from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import recorder_provenance_audit as audit
from src.workflow import stage1_gate_recheck as recheck

ROOT = Path(__file__).resolve().parents[1]


def test_risk_metrics_match_manual_values():
    r = pd.Series([0.01, -0.02, 0.03, -0.01, 0.005, -0.04, 0.02])
    m = recheck.risk_metrics(r)
    growth = np.cumprod(1 + r.to_numpy())
    assert m["return_pct"] == pytest.approx(100 * (growth[-1] - 1))
    assert m["max_drawdown_pct"] == pytest.approx(
        100 * (growth / np.maximum.accumulate(growth) - 1).min())
    assert m["worst_day_pct"] == pytest.approx(-4.0)


def test_gate_checks_flag_each_floor():
    metrics = {"max_drawdown_pct": -28.1, "worst_day_pct": -9.4,
               "cvar5_pct": -5.3, "worst_rolling_5d_pct": -12.11}
    result = recheck.gate_checks(metrics)
    assert result["pass"] is False
    assert result["checks"]["worst_rolling_5d_pct"] is False
    passing = dict(metrics, worst_rolling_5d_pct=-11.30)
    assert recheck.gate_checks(passing)["pass"] is True


def test_committed_recorders_are_found():
    found = recheck.committed_strategy_returns(ROOT)
    assert len(found) >= 1
    for recorder_id, path in found:
        assert path.is_file()
        assert isinstance(recorder_id, str) and recorder_id


def test_recheck_reports_from_committed_evidence():
    summary = recheck.run_stage1_gate_recheck(root=ROOT)
    assert summary["source_card"] == recheck.SOURCE_CARD
    assert summary["recorders"]
    for row in summary["recorders"]:
        assert "gate_pass" in row
        assert row["end"] >= row["start"]
    # Every reported row is a committed recorder, so the provenance audit must see it.
    table = audit.audit_tree(ROOT)
    assert not table.empty
