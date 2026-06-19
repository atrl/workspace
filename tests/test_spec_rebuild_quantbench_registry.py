from __future__ import annotations

import pandas as pd

import src.model as model_module
from src.model import ModelRegistry, quantbench_supervised_model_specs


def test_quantbench_registry_contains_full_matrix_with_unique_model_contracts():
    specs = quantbench_supervised_model_specs()
    registry = ModelRegistry.from_specs(specs)

    assert len(specs) == 44
    assert len({spec.name for spec in specs}) == 44
    assert registry.families() == [
        "linear",
        "tree",
        "kernel",
        "boosting",
        "sequence",
        "transformer",
        "graph",
        "ensemble",
    ]
    assert set(registry.objectives()) == {"classification", "mse", "ic", "ranking", "utility"}
    assert registry.by_family("graph")[0].capabilities["supports_graph_inputs"] is True
    assert registry.by_objective("utility")


def test_every_quantbench_model_spec_has_a_minimal_fit_predict_smoke_contract():
    frame = pd.DataFrame(
        {
            "return_5d": [0.01, 0.02, -0.01, 0.03],
            "vix": [18.0, 22.0, 28.0, 16.0],
            "volume_zscore": [0.1, -0.2, 1.2, 0.0],
            "drawdown_20d": [-0.02, -0.05, -0.11, -0.01],
            "turnover_proxy": [0.2, 0.4, 0.8, 0.3],
            "sequence_window": [5.0, 10.0, 20.0, 5.0],
            "multi_scale_window": [3.0, 6.0, 12.0, 24.0],
            "graph_neighbor_return": [0.00, 0.01, -0.02, 0.02],
            "target": [0.01, 0.03, -0.02, 0.02],
        }
    )

    specs = quantbench_supervised_model_specs()
    assert hasattr(model_module, "create_model")

    for spec in specs:
        estimator = model_module.create_model(spec)
        fitted = estimator.fit(frame, target_col="target")
        predictions = fitted.predict(frame)

        assert len(predictions) == len(frame)
        assert predictions.notna().all()
