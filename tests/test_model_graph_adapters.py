from __future__ import annotations

from src.data import build_supervised_dataset, load_soxl_support_frame
from src.model import fit_model_artifact, load_model_artifact, save_model_artifact
from src.model.training import fit_predict_model, model_adapter_name, model_adapter_status


def _split_frame():
    frame = load_soxl_support_frame(start="2026-01-01", end="2026-06-09")
    dataset = build_supervised_dataset(frame, horizon=1)
    train = dataset.frame.loc[dataset.frame["date"] <= "2026-04-30"].copy()
    test = dataset.frame.loc[dataset.frame["date"] > "2026-04-30"].copy()
    return dataset, train, test


def test_graph_models_use_named_paper_strict_adapters():
    dataset, train, test = _split_frame()

    expected = {
        "market_graph_mse": "market_graph_feature_mse_v1",
        "gcn_mse": "gcn_feature_propagation_mse_v1",
        "gat_mse": "gat_feature_attention_mse_v1",
        "rgcn_mse": "rgcn_relation_feature_mse_v1",
        "rsr_mse": "rsr_relation_strength_mse_v1",
        "hypergraph_mse": "hypergraph_feature_mse_v1",
        "thgnn_mse": "thgnn_temporal_hypergraph_mse_v1",
        "sthan_mse": "sthan_spatio_temporal_attention_mse_v1",
        "sthgcn_mse": "sthgcn_spatio_temporal_hypergraph_mse_v1",
    }
    for model_name, adapter_name in expected.items():
        predictions = fit_predict_model(
            model_name,
            train=train,
            test=test,
            feature_columns=dataset.feature_columns[:8],
        )

        assert predictions["adapter_status"].unique().tolist() == ["paper_strict_adapter"]
        assert predictions["model_adapter"].unique().tolist() == [adapter_name]
        assert predictions["prediction"].notna().all()
        assert model_adapter_status(model_name) == "paper_strict_adapter"
        assert model_adapter_name(model_name) == adapter_name


def test_graph_attention_relation_and_hypergraph_paths_are_distinct():
    dataset, train, test = _split_frame()

    gcn = fit_predict_model("gcn_mse", train=train, test=test, feature_columns=dataset.feature_columns[:8])
    gat = fit_predict_model("gat_mse", train=train, test=test, feature_columns=dataset.feature_columns[:8])
    rgcn = fit_predict_model("rgcn_mse", train=train, test=test, feature_columns=dataset.feature_columns[:8])
    hyper = fit_predict_model("hypergraph_mse", train=train, test=test, feature_columns=dataset.feature_columns[:8])

    assert not gat["prediction"].round(10).equals(gcn["prediction"].round(10))
    assert not rgcn["prediction"].round(10).equals(gcn["prediction"].round(10))
    assert not hyper["prediction"].round(10).equals(gcn["prediction"].round(10))


def test_temporal_graph_paths_are_distinct_from_static_graph_paths():
    dataset, train, test = _split_frame()

    static = fit_predict_model("hypergraph_mse", train=train, test=test, feature_columns=dataset.feature_columns[:8])
    thgnn = fit_predict_model("thgnn_mse", train=train, test=test, feature_columns=dataset.feature_columns[:8])
    sthan = fit_predict_model("sthan_mse", train=train, test=test, feature_columns=dataset.feature_columns[:8])
    sthgcn = fit_predict_model("sthgcn_mse", train=train, test=test, feature_columns=dataset.feature_columns[:8])

    assert not thgnn["prediction"].round(10).equals(static["prediction"].round(10))
    assert not sthan["prediction"].round(10).equals(static["prediction"].round(10))
    assert not sthgcn["prediction"].round(10).equals(static["prediction"].round(10))


def test_gat_artifact_metadata_records_graph_adapter_after_load(tmp_path):
    dataset, train, test = _split_frame()

    artifact = fit_model_artifact("gat_mse", train=train, feature_columns=dataset.feature_columns[:8])
    path = save_model_artifact(tmp_path / "gat_mse.joblib", artifact)
    loaded = load_model_artifact(path)
    predictions = loaded.predict(test)
    metadata = loaded.metadata_payload()

    assert metadata["model_adapter"] == "gat_feature_attention_mse_v1"
    assert loaded.adapter_status == "paper_strict_adapter"
    assert predictions["model_adapter"].unique().tolist() == ["gat_feature_attention_mse_v1"]
    assert predictions["prediction"].notna().all()
