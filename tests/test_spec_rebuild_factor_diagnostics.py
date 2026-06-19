from __future__ import annotations

import pandas as pd

import src.factor as factor_module


def test_multi_factor_combination_diagnostics_write_controlled_sidecars(tmp_path):
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-02", periods=4, freq="D"),
            "factor_a": [1.0, 2.0, 3.0, 4.0],
            "factor_b": [4.0, 3.0, 2.0, 1.0],
            "forward_return": [0.01, 0.02, 0.03, 0.04],
        }
    )

    assert hasattr(factor_module, "combine_factor_scores")
    assert hasattr(factor_module, "factor_combination_diagnostics")

    scored = factor_module.combine_factor_scores(
        frame,
        weights={"factor_a": 0.75, "factor_b": -0.25},
        output_col="combo_score",
    )
    report = factor_module.factor_combination_diagnostics(
        scored,
        factor_cols=["factor_a", "factor_b"],
        score_col="combo_score",
        forward_return_col="forward_return",
        quantiles=2,
    )
    paths = report.write_sidecars(tmp_path)

    assert scored["combo_score"].is_monotonic_increasing
    assert report.ic_table.to_dict("records") == [
        {"factor": "factor_a", "ic": 1.0},
        {"factor": "factor_b", "ic": -1.0},
        {"factor": "combo_score", "ic": 1.0},
    ]
    assert report.correlation_matrix.loc["factor_a", "factor_b"] == -1.0
    assert report.quantile_returns.to_dict("records") == [
        {"quantile": 1, "mean_forward_return": 0.015, "count": 2},
        {"quantile": 2, "mean_forward_return": 0.035, "count": 2},
    ]
    assert {path.name for path in paths.values()} == {
        "factor_ic.csv",
        "factor_correlation.csv",
        "factor_quantiles.csv",
    }
