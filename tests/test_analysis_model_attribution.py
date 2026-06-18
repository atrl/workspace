from __future__ import annotations

from src.analysis import explain_model_artifact
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


def test_proxy_graph_artifact_has_generic_sensitivity_attribution():
    dataset, train, test = _split_frame()
    artifact = fit_model_artifact("gcn_mse", train=train, feature_columns=dataset.feature_columns[:6])

    report = explain_model_artifact(artifact, frame=test, background_frame=train, top_n=3)

    assert report.summary["model"] == "gcn_mse"
    assert report.summary["proxy_model"] is True
    assert "sensitivity" in set(report.feature_attribution["attribution_kind"])
    assert report.summary["attribution_kinds"]["sensitivity"] == 3
