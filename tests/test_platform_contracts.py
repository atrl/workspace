from __future__ import annotations

import pandas as pd

from src.analysis import compare_models, compute_regime_metrics, compute_window_metrics
from src.core import (
    ArtifactManifest,
    DatasetSnapshot,
    build_dashboard_manifest_source_contract,
    build_dashboard_readiness_overview,
)
from src.model import QUANTBENCH_MODEL_NAMES, get_model_spec
from src.objective import OBJECTIVE_NAMES, utility_score


REQUESTED_QUANTBENCH_MODELS = {
    "logistic_clf",
    "ridge_ic",
    "ridge_mse",
    "random_forest_clf",
    "hist_gradient_mse",
    "xgboost_mse",
    "lightgbm_mse",
    "catboost_rank",
    "extra_trees_rank",
    "mlp_mse",
    "sequence_lag_ridge_mse",
    "lstm_mse",
    "gru_mse",
    "alstm_mse",
    "sfm_mse",
    "darnn_mse",
    "hawkes_gru_mse",
    "tcn_mse",
    "mlp_mixer_mse",
    "dlinear_mse",
    "patch_tst_mse",
    "autoformer_mse",
    "fedformer_mse",
    "market_graph_mse",
    "gcn_mse",
    "gat_mse",
    "rgcn_mse",
    "rsr_mse",
    "hypergraph_mse",
    "thgnn_mse",
    "sthan_mse",
    "sthgcn_mse",
}


def test_core_contracts_are_json_stable_and_hashable():
    snapshot = DatasetSnapshot(
        name="soxl_support_daily",
        universe=("SOXL", "SOXX", "VIX"),
        start="2026-01-01",
        end="2026-06-09",
        calendar_path="data/qlib/soxl/calendars/day.txt",
        data_paths=("data/processed/panels/soxl_soxx_vix_full.csv",),
        schema_version="dataset.v1",
    )
    manifest = ArtifactManifest(
        artifact_id="research/example",
        artifact_type="analysis",
        producer="tests",
        schema_version="artifact.v1",
        dataset=snapshot,
        paths={"summary": "analysis_summary.json"},
    )

    payload = manifest.to_dict()
    assert payload["dataset"]["universe"] == ["SOXL", "SOXX", "VIX"]
    assert len(snapshot.schema_hash()) == 64
    assert len(manifest.manifest_hash()) == 64


def test_dashboard_manifest_source_contract_classifies_strong_missing_legacy_and_incomplete_sources():
    strong = build_dashboard_manifest_source_contract(
        {
            "kind": "analysis_manifest",
            "path": "matrix_card/analysis_manifest.json",
            "schema_version": "artifact.v1",
            "manifest_hash": "a" * 64,
            "producer": "src.analysis.exports",
        }
    )
    missing = build_dashboard_manifest_source_contract(None)
    legacy = build_dashboard_manifest_source_contract(
        {
            "kind": "fixed_sidecars",
            "path": "",
            "manifest_hash": "",
            "producer": "src.dashboard.model_matrix",
        }
    )
    incomplete = build_dashboard_manifest_source_contract(
        {
            "kind": "artifact_manifest",
            "path": "artifact_manifest.json",
            "manifest_hash": "",
            "producer": "src.dashboard.model_matrix",
        }
    )

    assert strong["schema_version"] == "core.dashboard_manifest_source_contract.v1"
    assert strong["status"] == "complete"
    assert strong["is_complete"] is True
    assert strong["kind"] == "analysis_manifest"
    assert strong["source_schema_version"] == "artifact.v1"
    assert strong["allowed_kinds"] == ["analysis_manifest", "artifact_manifest"]
    assert strong["readiness_detail"] == "analysis_manifest / complete hash aaaaaaaaaa"

    assert missing["status"] == "missing"
    assert missing["is_complete"] is False
    assert missing["kind"] == ""
    assert missing["readiness_detail"] == "missing / missing hash "

    assert legacy["status"] == "weak_kind"
    assert legacy["is_complete"] is False
    assert legacy["kind"] == "fixed_sidecars"
    assert legacy["error"] == "kind fixed_sidecars not allowed"
    assert legacy["readiness_detail"] == "fixed_sidecars / weak_kind hash "

    assert incomplete["status"] == "incomplete"
    assert incomplete["missing_fields"] == ["manifest_hash"]
    assert incomplete["error"] == "missing manifest_hash"


def test_dashboard_readiness_overview_contract_unifies_manifest_production_notebook_model_gates():
    payload = {
        "manifest_source": {
            "kind": "analysis_manifest",
            "path": "matrix_card/analysis_manifest.json",
            "schema_version": "artifact.v1",
            "manifest_hash": "a" * 64,
            "producer": "src.analysis.exports",
        },
        "production_health": {
            "schema_version": "schedule.production_health.v1",
            "health_status": "healthy",
            "is_healthy": True,
            "missing_required_paths": [],
        },
        "notebook_card_readiness": {
            "schema_version": "dashboard.notebook_card_readiness.v1",
            "summary": {
                "production_published_count": 1,
                "production_template_count": 1,
                "production_manifest_missing_count": 0,
                "production_registry_mismatch_count": 0,
            },
        },
        "requested_model_contract": {
            "schema_version": "model.requested_contract_health.v1",
            "requested_model_count": 32,
            "completed_requested_model_count": 31,
            "missing_from_ledger": ["sthgcn_mse"],
            "incomplete_requested_models": ["sthgcn_mse"],
            "all_requested_models_completed": False,
        },
        "model_run_coverage": {
            "schema_version": "model.run_ledger_summary.v1",
            "expected_model_count": 2,
            "trained_model_count": 2,
            "backtested_model_count": 2,
            "all_models_completed": True,
        },
    }

    overview = build_dashboard_readiness_overview(payload)

    assert overview["schema_version"] == "dashboard.readiness_overview.v1"
    assert overview["status"] == "review"
    assert overview["gate_count"] == 5
    assert overview["pass_count"] == 3
    assert overview["review_count"] == 2
    assert overview["block_count"] == 0
    assert [gate["gate"] for gate in overview["gates"]] == [
        "dashboard_manifest",
        "production_health",
        "notebook_cards",
        "requested_models",
        "model_runs",
    ]
    gates = {gate["gate"]: gate for gate in overview["gates"]}
    assert gates["dashboard_manifest"]["status"] == "pass"
    assert gates["dashboard_manifest"]["detail"] == "analysis_manifest / complete hash aaaaaaaaaa"
    assert gates["production_health"]["detail"] == "healthy / missing 0"
    assert gates["notebook_cards"]["status"] == "review"
    assert gates["notebook_cards"]["detail"] == "published 1 / template 1 / missing 0 / mismatch 0"
    assert gates["requested_models"]["status"] == "review"
    assert gates["requested_models"]["detail"] == "31 / 32 complete / missing 1 / incomplete 1"
    assert gates["model_runs"]["status"] == "pass"
    assert gates["model_runs"]["detail"] == "trained 2 / backtested 2 / expected 2"

    blocked = build_dashboard_readiness_overview({"top_models": [{"model": "ridge_mse"}]})
    assert blocked["status"] == "blocked"
    assert blocked["gate_count"] == 1
    assert blocked["block_count"] == 1
    assert blocked["gates"][0]["gate"] == "dashboard_manifest"
    assert blocked["gates"][0]["detail"] == "missing / missing hash "


def test_quantbench_registry_covers_requested_models_without_rl_reward():
    assert REQUESTED_QUANTBENCH_MODELS.issubset(set(QUANTBENCH_MODEL_NAMES))

    graph_spec = get_model_spec("gcn_mse")
    assert graph_spec.family == "graph"
    assert graph_spec.input_kind == "support_graph"
    assert graph_spec.trade_instrument == "SOXL"

    assert "reward" not in OBJECTIVE_NAMES
    assert {"classification", "mse", "ic", "ranking", "utility"}.issubset(set(OBJECTIVE_NAMES))


def test_utility_score_penalizes_drawdown_and_turnover():
    conservative = utility_score(return_pct=20.0, sharpe=1.2, max_drawdown_pct=8.0, turnover=0.3)
    fragile = utility_score(return_pct=20.0, sharpe=1.2, max_drawdown_pct=30.0, turnover=2.0)

    assert conservative > fragile


def test_analysis_computes_regime_window_and_model_comparison_tables():
    dates = list(pd.date_range("2026-01-01", periods=8, freq="D")) * 2
    frame = pd.DataFrame(
        {
            "date": dates,
            "model": ["ridge_mse"] * 8 + ["gcn_mse"] * 8,
            "nav": [1.0, 1.02, 1.04, 1.01, 1.06, 1.08, 1.07, 1.10] * 2,
            "benchmark_nav": [1.0, 1.01, 1.03, 1.00, 1.04, 1.05, 1.04, 1.06] * 2,
            "position": [0.2, 0.3, 0.4, 0.1, 0.5, 0.6, 0.4, 0.5] * 2,
            "turnover": [0.0, 0.1, 0.1, 0.3, 0.4, 0.1, 0.2, 0.1] * 2,
            "vix": [18, 20, 24, 35, 31, 22, 19, 17] * 2,
            "trend": [0.02, 0.03, 0.01, -0.04, -0.02, 0.01, 0.03, 0.04] * 2,
            "drawdown": [0.0, 0.0, 0.0, -0.08, -0.04, 0.0, -0.01, 0.0] * 2,
        }
    )
    frame.loc[frame["model"] == "gcn_mse", "nav"] *= 1.03

    regime = compute_regime_metrics(frame)
    windows = compute_window_metrics(frame, windows={"first_half": ("2026-01-01", "2026-01-04")})
    comparison = compare_models(frame)

    assert {"model", "regime", "return_pct", "max_drawdown_pct", "stability_score"}.issubset(regime.columns)
    assert windows["window"].tolist() == ["first_half", "first_half"]
    assert comparison.iloc[0]["model"] == "gcn_mse"
    assert comparison.iloc[0]["rank"] == 1
