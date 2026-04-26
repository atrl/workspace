from __future__ import annotations

import json
from pathlib import Path

from src.soxl_quant.app import artifacts as artifact_module
from src.soxl_quant.config import load_profile


def test_persist_summary_bundle_includes_profile_feature_schema(tmp_path: Path | None = None):
    profile = load_profile("balanced")
    output_dir = tmp_path or Path("runtime/model/card")
    output_dir.mkdir(parents=True, exist_ok=True)

    original_summary_artifact_path = artifact_module.summary_artifact_path
    artifact_module.summary_artifact_path = lambda profile_name, artifact_tag="": output_dir / (
        f"{profile_name}_{artifact_tag or 'main'}_summary.json"
    )
    try:
        summary_path = artifact_module.persist_summary_bundle(
            profile=profile,
            seed=12345,
            effective_tag="unit",
            model_path=output_dir / "dummy_actor.pt",
            valid_metrics={"return": 1.0, "sharpe": 0.5, "max_dd": 2.0},
            selection={"best_score": 3.0, "history": [{"iter": 20, "score": 3.0}]},
            slice_result={"test_2022": {"return": 1.0}},
            test_proxy={"return": 1.0, "sharpe": 0.5},
            promotion_report={"accepted": False, "reasons": ["unit-test"]},
        )
    finally:
        artifact_module.summary_artifact_path = original_summary_artifact_path

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["profile"] == "balanced"
    assert payload["feature_schema"]["feature_count"] > 0
    assert payload["feature_schema"]["signature"]


if __name__ == "__main__":
    test_persist_summary_bundle_includes_profile_feature_schema()
    print("test_summary_bundle: OK")
