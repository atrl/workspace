from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


EXPERIMENT = "soxl_objective_feasibility_and_predictability_audit_v1"


def test_committed_audit_card_is_qlib_native_outer_safe_and_slim() -> None:
    root = Path(__file__).resolve().parents[1]
    card = root / "research" / EXPERIMENT
    summary = json.loads((card / "summary.json").read_text(encoding="utf-8"))
    notebook = json.loads((card / f"{EXPERIMENT}.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert summary["selection_uses_outer_test"] is False
    assert summary["outer_test_use"] == "diagnostic_only_previously_observed"
    assert summary["qlib_dataset"]["api"] == ["DatasetH", "DataHandlerLP", "StaticDataLoader"]
    assert summary["artifact_policy"]["policy"] == "slim_card_recorder_first"
    assert summary["feasibility_protocol"]["oracle_is_nontradable"] is True
    assert summary["feasibility_protocol"]["oracle_dwell_days"] == 3
    assert summary["feasibility_protocol"]["oracle_soxs_max_weight"] == 0.10
    for name in ("predictions.csv", "strategy_returns.csv", "positions.csv", "bootstrap_metrics.csv"):
        assert not (card / name).exists()
    for token in ("qlib.init", "D.calendar", "DatasetH", "DataHandlerLP", "R.get_recorder"):
        assert token in source
    assert "最终中文结论" in source
    assert notebook["cells"][1]["execution_count"] is not None
    assert summary["recorder_id"] in json.dumps(notebook)


def test_audit_reports_feasibility_and_predictability_without_outer_selection() -> None:
    root = Path(__file__).resolve().parents[1]
    card = root / "research" / EXPERIMENT
    summary = json.loads((card / "summary.json").read_text(encoding="utf-8"))
    frontier = pd.read_csv(card / "feasibility_frontier.csv")
    predictability = pd.read_csv(card / "predictability_summary.csv")

    development = frontier.query("segment == 'development' and cost_bps == 10")
    assert not development.query("strategy_kind == 'mechanical_tradable'")["all_primary_gates_pass"].any()
    assert development.query("strategy_kind == 'oracle_nontradable'")["all_primary_gates_pass"].any()
    assert (frontier["selection_uses_outer_test"] == False).all()  # noqa: E712
    assert (predictability["selection_uses_outer_test"] == False).all()  # noqa: E712
    assert set(predictability["feature_set"]) == {"base", "base_plus_soxs"}
    assert {
        "leverage_advantage_1d",
        "leverage_advantage_5d",
        "leverage_advantage_10d",
        "leverage_advantage_20d",
        "cash_advantage_5d",
        "future_vol_5d",
        "tail_event_5d",
    }.issubset(set(predictability["target"]))
    assert not predictability.loc[predictability["target"].str.startswith("leverage_advantage"), "stable_signal"].any()
    assert summary["audit_result"]["incremental_stable_soxs_targets"] == ["cash_advantage_5d"]
    assert summary["audit_result"]["soxs_expands_oracle_pass_count"] is False


def test_oracle_positions_respect_action_and_minimum_dwell_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = json.loads(
        (root / "research" / EXPERIMENT / "summary.json").read_text(encoding="utf-8")
    )
    artifact_root = Path(summary["recorder_local_dir"]) / "artifacts" / EXPERIMENT
    positions = pd.read_csv(artifact_root / "positions.csv")
    oracle = positions.loc[
        positions["strategy_kind"].eq("oracle_nontradable")
        & positions["segment"].eq("development")
        & positions["cost_bps"].eq(10)
    ].copy()

    weights = oracle[["soxl_weight", "soxx_weight", "soxs_weight", "cash_weight"]]
    assert (weights.sum(axis=1).round(12) == 1.0).all()
    assert oracle["soxs_weight"].max() <= 0.10
    assert not ((oracle["soxl_weight"] > 0.0) & (oracle["soxs_weight"] > 0.0)).any()
    for _, strategy in oracle.groupby("strategy", sort=False):
        changed = weights.loc[strategy.index].ne(weights.loc[strategy.index].shift()).any(axis=1)
        run = changed.cumsum()
        lengths = strategy.groupby(run.loc[strategy.index]).size()
        assert (lengths.iloc[:-1] >= 3).all()

    assert (artifact_root / "predictions.csv").exists()
    assert (artifact_root / "strategy_returns.csv").exists()
    assert len(list((artifact_root / "models").glob("*.pkl"))) == 20
