from __future__ import annotations

import json

from src.analysis import explain_model_artifact, write_model_attribution_card
from src.data import build_supervised_dataset, load_soxl_support_frame
from src.model import fit_model_artifact


def _split_frame():
    frame = load_soxl_support_frame(start="2026-01-01", end="2026-04-30")
    dataset = build_supervised_dataset(frame, horizon=1)
    train = dataset.frame.loc[dataset.frame["date"] <= "2026-03-31"].copy()
    test = dataset.frame.loc[dataset.frame["date"] > "2026-03-31"].copy()
    return dataset, train, test


def test_explain_model_artifact_outputs_coefficients_permutation_and_sensitivity():
    dataset, train, test = _split_frame()
    artifact = fit_model_artifact("ridge_mse", train=train, feature_columns=dataset.feature_columns[:6])

    report = explain_model_artifact(artifact, frame=test, background_frame=train, top_n=4)

    assert report.summary["schema_version"] == "analysis.model_attribution.v1"
    assert report.summary["model"] == "ridge_mse"
    assert report.summary["feature_count"] == 6
    assert report.summary["top_features"]
    assert {"coefficient", "permutation", "sensitivity"}.issubset(set(report.feature_attribution["attribution_kind"]))
    assert {
        "model",
        "feature",
        "attribution_kind",
        "importance",
        "normalized_importance",
        "rank",
        "direction",
    }.issubset(report.feature_attribution.columns)
    assert report.feature_attribution["importance"].notna().all()
    assert report.feature_attribution["rank"].min() == 1


def test_native_graph_artifact_has_generic_sensitivity_attribution():
    dataset, train, test = _split_frame()
    artifact = fit_model_artifact("gcn_mse", train=train, feature_columns=dataset.feature_columns[:6])

    report = explain_model_artifact(artifact, frame=test, background_frame=train, top_n=3)

    assert report.summary["model"] == "gcn_mse"
    assert report.summary["paper_fidelity"] == "native_graph"
    assert report.summary["proxy_model"] is False
    assert "sensitivity" in set(report.feature_attribution["attribution_kind"])
    assert report.summary["attribution_kinds"]["sensitivity"] == 3


def test_write_model_attribution_card_trains_batch_and_writes_dashboard_sidecars(tmp_path):
    dataset, train, test = _split_frame()

    result = write_model_attribution_card(
        tmp_path,
        model_names=("ridge_mse", "gcn_mse"),
        train=train,
        test=test,
        feature_columns=dataset.feature_columns[:6],
        top_n=3,
    )

    summary = json.loads((tmp_path / "analysis_summary.json").read_text(encoding="utf-8"))
    model_summaries = json.loads((tmp_path / "model_attribution_summaries.json").read_text(encoding="utf-8"))

    assert result.summary["schema_version"] == "analysis.model_attribution.batch.v1"
    assert summary["schema_version"] == "analysis.model_attribution.batch.v1"
    assert summary["status"] == "completed"
    assert summary["models"] == ["ridge_mse", "gcn_mse"]
    assert summary["model_count"] == 2
    assert summary["feature_count"] == 6
    assert summary["top_n"] == 3
    assert summary["proxy_models"] == []
    assert summary["attribution_kinds"]["sensitivity"] >= 2
    assert summary["test_start"] == str(test["date"].min().date())
    assert summary["test_end"] == str(test["date"].max().date())
    assert summary["artifacts"] == {
        "feature_attribution": "feature_attribution.csv",
        "model_attribution_summaries": "model_attribution_summaries.json",
        "model_capabilities": "model_capabilities.csv",
        "top_features_by_model": "top_features_by_model.csv",
    }
    assert len(model_summaries) == 2

    feature_attribution = result.feature_attribution
    top_features = result.top_features_by_model
    capabilities = result.model_capabilities

    assert set(feature_attribution["model"]) == {"ridge_mse", "gcn_mse"}
    assert {"permutation", "sensitivity"}.issubset(set(feature_attribution["attribution_kind"]))
    assert set(top_features["model"]) == {"ridge_mse", "gcn_mse"}
    assert top_features.groupby("model").size().max() == 3
    assert {"model", "feature", "normalized_importance"}.issubset(top_features.columns)
    assert capabilities.loc[capabilities["model"].eq("gcn_mse"), "is_proxy"].iloc[0] is False

    assert (tmp_path / "feature_attribution.csv").exists()
    assert (tmp_path / "top_features_by_model.csv").exists()
    assert (tmp_path / "model_capabilities.csv").exists()
