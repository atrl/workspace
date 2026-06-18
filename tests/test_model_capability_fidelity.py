from __future__ import annotations

import json

from src.model import QUANTBENCH_MODEL_NAMES, iter_model_specs
from src.model.capabilities import model_capability_table, model_fidelity_summary
from src.model.matrix import run_quantbench_matrix_smoke


def test_quantbench_specs_have_explicit_fidelity_and_capability_notes():
    specs = tuple(iter_model_specs())

    assert {spec.name for spec in specs} == set(QUANTBENCH_MODEL_NAMES)
    assert {spec.name for spec in specs if spec.paper_fidelity == "paper_oriented"} == set()
    assert {spec.name for spec in specs if not spec.implementation_notes.strip()} == set()


def test_model_capability_table_separates_native_and_proxy_implementations():
    table = model_capability_table()
    summary = model_fidelity_summary()

    assert set(table["model"]) == set(QUANTBENCH_MODEL_NAMES)
    assert {"model", "paper_fidelity", "is_proxy", "implementation_notes", "promotion_blocker"}.issubset(
        table.columns
    )
    assert table.loc[table["model"].eq("xgboost_mse"), "paper_fidelity"].item() == "native_library"
    assert table.loc[table["model"].eq("catboost_rank"), "paper_fidelity"].item() == "native_library"
    assert table.loc[table["model"].eq("dlinear_mse"), "paper_fidelity"].item() == "native_sequence"
    assert table.loc[table["model"].eq("dlinear_mse"), "is_proxy"].item() is False
    assert table.loc[table["model"].eq("patch_tst_mse"), "paper_fidelity"].item() == "native_sequence"
    assert table.loc[table["model"].eq("patch_tst_mse"), "is_proxy"].item() is False
    assert table.loc[table["model"].eq("autoformer_mse"), "paper_fidelity"].item() == "native_sequence"
    assert table.loc[table["model"].eq("autoformer_mse"), "is_proxy"].item() is False
    assert table.loc[table["model"].eq("fedformer_mse"), "paper_fidelity"].item() == "native_sequence"
    assert table.loc[table["model"].eq("fedformer_mse"), "is_proxy"].item() is False
    assert table.loc[table["model"].eq("mlp_mixer_mse"), "paper_fidelity"].item() == "native_sequence"
    assert table.loc[table["model"].eq("mlp_mixer_mse"), "is_proxy"].item() is False
    assert table.loc[table["model"].eq("sfm_mse"), "paper_fidelity"].item() == "native_sequence"
    assert table.loc[table["model"].eq("sfm_mse"), "is_proxy"].item() is False
    assert table.loc[table["model"].eq("hawkes_gru_mse"), "paper_fidelity"].item() == "native_torch"
    assert table.loc[table["model"].eq("hawkes_gru_mse"), "is_proxy"].item() is False
    assert table.loc[table["model"].eq("gcn_mse"), "paper_fidelity"].item() == "native_graph"
    assert table.loc[table["model"].eq("gcn_mse"), "is_proxy"].item() is False

    assert summary["schema_version"] == "model.fidelity.v1"
    assert summary["model_count"] == len(QUANTBENCH_MODEL_NAMES)
    assert summary["fidelity_counts"]["proxy_feature_adapter"] > 0
    assert summary["fidelity_counts"]["native_sequence"] == 6
    assert summary["fidelity_counts"]["native_torch"] == 6
    assert summary["fidelity_counts"]["native_graph"] == 1
    assert summary["fidelity_counts"]["proxy_feature_adapter"] == 11
    assert "patch_tst_mse" in summary["native_models"]
    assert "autoformer_mse" in summary["native_models"]
    assert "fedformer_mse" in summary["native_models"]
    assert "mlp_mixer_mse" in summary["native_models"]
    assert "sfm_mse" in summary["native_models"]
    assert "hawkes_gru_mse" in summary["native_models"]
    assert "gcn_mse" in summary["native_models"]
    assert "gcn_mse" not in summary["proxy_models"]
    assert "sfm_mse" not in summary["proxy_models"]
    assert "hawkes_gru_mse" not in summary["proxy_models"]
    assert "dlinear_mse" in summary["native_models"]
    assert "xgboost_mse" in summary["native_models"]


def test_quantbench_matrix_summary_and_manifest_include_model_fidelity(tmp_path):
    outputs = run_quantbench_matrix_smoke(
        card_dir=tmp_path,
        start="2026-01-01",
        end="2026-06-09",
        train_end="2026-04-30",
        model_names=(
            "ridge_mse",
            "patch_tst_mse",
            "autoformer_mse",
            "fedformer_mse",
            "mlp_mixer_mse",
            "sfm_mse",
            "hawkes_gru_mse",
            "gcn_mse",
            "xgboost_mse",
        ),
    )

    summary = outputs.summary
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text(encoding="utf-8"))

    assert summary["model_fidelity"]["schema_version"] == "model.fidelity.v1"
    assert summary["model_fidelity"]["model_count"] == 9
    assert summary["model_fidelity"]["fidelity_counts"]["native_sequence"] == 5
    assert summary["model_fidelity"]["fidelity_counts"]["native_torch"] == 1
    assert summary["model_fidelity"]["fidelity_counts"]["native_graph"] == 1
    assert summary["model_fidelity"]["fidelity_counts"]["native_library"] == 2
    assert summary["model_fidelity"]["proxy_models"] == []
    assert "patch_tst_mse" in summary["model_fidelity"]["native_models"]
    assert "autoformer_mse" in summary["model_fidelity"]["native_models"]
    assert "fedformer_mse" in summary["model_fidelity"]["native_models"]
    assert "mlp_mixer_mse" in summary["model_fidelity"]["native_models"]
    assert "sfm_mse" in summary["model_fidelity"]["native_models"]
    assert "hawkes_gru_mse" in summary["model_fidelity"]["native_models"]
    assert "gcn_mse" in summary["model_fidelity"]["native_models"]
    assert manifest["metadata"]["model_fidelity"]["fidelity_counts"] == summary["model_fidelity"][
        "fidelity_counts"
    ]
