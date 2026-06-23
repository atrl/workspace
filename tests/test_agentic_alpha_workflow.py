from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


class AgenticAlphaDataset:
    def __init__(self) -> None:
        dates = pd.date_range("2026-01-02", periods=8, freq="D")
        instruments = ["A", "B", "C", "D"]
        rows = []
        for day_index, date in enumerate(dates):
            for inst_index, instrument in enumerate(instruments):
                alpha = inst_index + 1 + (day_index % 3) * 0.2
                # Make daily rank IC strong but not constant so ICIR is meaningful.
                label_1d = alpha * (1.0 + (0.05 if day_index % 2 else -0.03))
                if day_index % 4 == 0 and instrument == "D":
                    label_1d = 0.5
                label_5d = label_1d * 0.9 + inst_index * 0.1
                noise = [0.2, -0.4, 0.1, -0.1][(inst_index + day_index) % 4]
                rows.append(
                    {
                        "instrument": instrument,
                        "datetime": date,
                        "alpha_good": alpha,
                        "alpha_duplicate": alpha * 2.0,
                        "alpha_noise": noise,
                        "LABEL1": label_1d,
                        "LABEL5": label_5d,
                    }
                )
        frame = pd.DataFrame(rows).set_index(["instrument", "datetime"]).sort_index()
        self.frames = {"test": frame}

    def prepare(self, segment: str) -> pd.DataFrame:
        return self.frames[segment].copy()


def test_agentic_alpha_workflow_manifest_keeps_external_workflow_inside_src_boundary(tmp_path):
    from src.workflow import AgenticAlphaWorkflowSpec, build_agentic_alpha_workflow_manifest

    spec = AgenticAlphaWorkflowSpec(
        name="agentic_alpha_sfa_v1",
        route="sfa",
        market="all",
        provider_uri=Path("data/qlib/soxl"),
        recorder_uri=Path("research/agentic_alpha_sfa_v1/mlruns"),
        source_url="https://github.com/JacobDu/agentic-alpha",
    )

    manifest = build_agentic_alpha_workflow_manifest(spec)

    assert manifest["workflow"] == "agentic_alpha_sfa_v1"
    assert manifest["source_url"] == "https://github.com/JacobDu/agentic-alpha"
    assert manifest["stage_order"] == ["research", "retrieve", "generate", "evaluate", "validate", "distill"]
    assert manifest["decision_labels"] == ["Promote", "Iterate", "Drop"]
    assert manifest["src_boundary"]["skill_scripts_imported"] is False
    assert manifest["src_boundary"]["business_logic_surface"] == "src"
    assert manifest["qlib_native_required"] is True
    assert manifest["artifact_policy"]["policy"] == "slim_card_recorder_first"

    output = tmp_path / "workflow_manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    assert json.loads(output.read_text(encoding="utf-8"))["route"] == "sfa"


def test_agentic_alpha_sfa_validation_uses_qlib_dataset_and_distills_decisions(tmp_path):
    from src.factor import validate_agentic_alpha_factors

    report = validate_agentic_alpha_factors(
        AgenticAlphaDataset(),
        segment="test",
        factor_cols=["alpha_good", "alpha_duplicate", "alpha_noise"],
        label_cols={"1d": "LABEL1", "5d": "LABEL5"},
        primary_horizon="1d",
        min_cross_section=3,
        min_abs_rank_icir=0.10,
        max_fdr_p=1.0,
        max_abs_correlation=0.80,
        min_decay_ratio=0.50,
        min_stability_score=0.30,
    )
    paths = report.write_sidecars(tmp_path)
    metrics = report.metrics.set_index("factor")

    assert report.source == "qlib_dataset"
    assert report.workflow == "agentic_alpha_sfa"
    assert set(paths) == {"metrics", "correlation", "summary"}
    assert paths["metrics"].name == "agentic_alpha_factor_validation.csv"
    assert metrics.loc["alpha_good", "decision"] == "Promote"
    assert metrics.loc["alpha_good", "weekly_suitable"] is True
    assert metrics.loc["alpha_duplicate", "redundancy_flag"] == "high_similarity"
    assert metrics.loc["alpha_duplicate", "decision"] == "Drop"
    assert metrics.loc["alpha_noise", "decision"] == "Drop"
    assert report.correlation.loc["alpha_good", "alpha_duplicate"] == 1.0

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["source"] == "qlib_dataset"
    assert summary["workflow"] == "agentic_alpha_sfa"
    assert summary["stage_order"] == ["research", "retrieve", "generate", "evaluate", "validate", "distill"]
    assert summary["decision_counts"]["Promote"] == 1
    assert summary["recorder_first"] is True
