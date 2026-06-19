from __future__ import annotations

import json

import pandas as pd

from src.analysis import load_qlib_recorder_artifacts


def test_qlib_recorder_artifacts_load_into_analysis_report(tmp_path):
    recorder_dir = tmp_path / "recorder"
    recorder_dir.mkdir()
    (recorder_dir / "metrics.json").write_text(
        json.dumps(
            {
                "annualized_return": 0.42,
                "information_ratio": 1.7,
                "max_drawdown": -0.12,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "cum return w cost": 1.0,
                "cum ex return w cost": 0.0,
                "turnover": 0.0,
            },
            {
                "date": "2026-01-05",
                "cum return w cost": 1.1,
                "cum ex return w cost": 0.03,
                "turnover": 0.4,
            },
            {
                "date": "2026-01-06",
                "cum return w cost": 1.05,
                "cum ex return w cost": -0.01,
                "turnover": 0.2,
            },
        ]
    ).to_csv(recorder_dir / "portfolio_report.csv", index=False)
    pd.DataFrame(
        [
            {"metric": "annualized_return", "value": 0.42},
            {"metric": "max_drawdown", "value": -0.12},
            {"metric": "information_ratio", "value": 1.7},
        ]
    ).to_csv(recorder_dir / "risk_analysis.csv", index=False)
    pd.DataFrame(
        [
            {"model": "linear_mse", "return_pct": 10.0, "max_drawdown_pct": -5.0, "turnover": 0.4},
            {"model": "fast_tree_rank", "return_pct": 9.0, "max_drawdown_pct": -3.0, "turnover": 0.3},
        ]
    ).to_csv(recorder_dir / "model_comparison.csv", index=False)

    artifacts = load_qlib_recorder_artifacts(
        recorder_dir,
        experiment="spec_rebuild_foundation_v1",
        recorder_id="recorder-smoke",
    )
    report = artifacts.to_analysis_report(
        windows={"smoke": ("2026-01-02", "2026-01-06")},
        failure_modes=["qlib-recorder-smoke"],
    )

    assert artifacts.summary["experiment"] == "spec_rebuild_foundation_v1"
    assert artifacts.summary["recorder_id"] == "recorder-smoke"
    assert artifacts.summary["qlib_docs"]["recorder"].endswith("/component/recorder.html")
    assert artifacts.portfolio_report["nav"].tolist() == [1.0, 1.1, 1.05]
    assert artifacts.risk_metrics == {
        "annualized_return": 0.42,
        "information_ratio": 1.7,
        "max_drawdown": -0.12,
    }

    assert report.summary["source"] == "qlib_recorder_artifacts"
    assert report.summary["risk_metrics"]["information_ratio"] == 1.7
    assert report.window_metrics.loc[0, "return_pct"] == 5.0
    assert report.model_comparison.iloc[0]["model"] == "fast_tree_rank"

    paths = report.write_sidecars(tmp_path / "sidecars")

    assert "analysis_summary.json" in {path.name for path in paths.values()}
