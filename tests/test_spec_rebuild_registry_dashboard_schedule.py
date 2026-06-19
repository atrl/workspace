from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.core import ArtifactManifest, SCHEMA_VERSION
from src.dashboard import load_dashboard_export, write_dashboard_export
from src.model import MeanReturnModel, ModelRegistry, quantbench_supervised_model_specs
from src.objective import promotion_scores
from src.schedule import RefreshPlan, run_refresh_plan


def test_quantbench_registry_model_persistence_and_promotion_scoring(tmp_path):
    specs = quantbench_supervised_model_specs()
    registry = ModelRegistry.from_specs(specs)

    assert len(specs) >= 8
    assert {"linear", "tree", "sequence", "transformer"}.issubset({spec.family for spec in specs})
    assert {"classification", "mse", "ic", "ranking", "utility"}.issubset({spec.objective for spec in specs})
    assert registry.get("linear_mse").capabilities["supports_feature_importance"] is True
    assert registry.get("crossformer_mse").capabilities["supports_window_sensitivity"] is True

    model = MeanReturnModel().fit(pd.DataFrame({"target": [0.01, 0.03, 0.05]}), target_col="target")
    model_path = tmp_path / "mean_return_model.json"
    model.save(model_path)
    loaded = MeanReturnModel.load(model_path)

    assert loaded.predict(pd.DataFrame({"feature": [1.0, 2.0]})).tolist() == [0.03, 0.03]

    ranked = promotion_scores(
        pd.DataFrame(
            [
                {
                    "model": "linear_mse",
                    "return_pct": 18.0,
                    "max_drawdown_pct": -4.0,
                    "turnover": 0.2,
                    "ic": 0.08,
                    "rank_ic": 0.06,
                },
                {
                    "model": "fast_tree_rank",
                    "return_pct": 25.0,
                    "max_drawdown_pct": -18.0,
                    "turnover": 2.5,
                    "ic": 0.05,
                    "rank_ic": 0.08,
                },
            ]
        ),
        min_score=10.0,
        max_drawdown_floor=-10.0,
    )

    assert ranked["model"].tolist() == ["linear_mse", "fast_tree_rank"]
    assert ranked.loc[0, "promote"] is True
    assert ranked.loc[1, "promote"] is False


def test_dashboard_export_and_schedule_refresh_pipeline_are_manifest_driven(tmp_path):
    manifest = ArtifactManifest(
        kind="analysis_summary",
        path="research/spec_rebuild_foundation_v1/analysis_summary.json",
        source="qlib://soxl",
        date_range=("2026-01-02", "2026-06-09"),
        purpose="dashboard",
        metadata={"experiment": "spec_rebuild_foundation_v1"},
    )
    ranking = pd.DataFrame(
        [
            {"model": "linear_mse", "promotion_score": 14.0, "promote": True},
            {"model": "fast_tree_rank", "promotion_score": 5.0, "promote": False},
        ]
    )
    export_path = tmp_path / "dashboard_manifest.json"
    payload = write_dashboard_export(
        export_path,
        manifests=[manifest],
        model_comparison=ranking,
        generated_by="schedule:data-model-dashboard-refresh",
    )

    assert payload == load_dashboard_export(export_path)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["generated_by"] == "schedule:data-model-dashboard-refresh"
    assert payload["artifacts"][0]["artifact_id"] == manifest.artifact_id
    assert payload["model_comparison"][0]["model"] == "linear_mse"

    events: list[str] = []
    calls = {"data": 0}

    def data_update():
        calls["data"] += 1
        events.append("data")
        if calls["data"] == 1:
            raise RuntimeError("transient data provider error")
        return [manifest]

    def model_refresh():
        events.append("model")
        return ranking

    def dashboard_export(manifests, model_comparison):
        events.append("dashboard")
        return write_dashboard_export(
            tmp_path / "scheduled_dashboard_manifest.json",
            manifests=manifests,
            model_comparison=model_comparison,
            generated_by="scheduled-refresh",
        )

    def notify(summary):
        events.append(f"notify:{summary['status']}")

    plan = RefreshPlan(
        name="data_model_dashboard_refresh",
        lock_path=tmp_path / "locks" / "refresh.lock",
        attempts=2,
        delay_seconds=0.0,
    )
    summary = run_refresh_plan(
        plan,
        data_update=data_update,
        model_refresh=model_refresh,
        dashboard_export=dashboard_export,
        notify=notify,
    )

    scheduled_payload = json.loads((tmp_path / "scheduled_dashboard_manifest.json").read_text(encoding="utf-8"))

    assert events == ["data", "data", "model", "dashboard", "notify:ok"]
    assert summary["status"] == "ok"
    assert summary["attempts"] == 2
    assert summary["artifacts"] == 1
    assert scheduled_payload["generated_by"] == "scheduled-refresh"
    assert not plan.lock_path.exists()
