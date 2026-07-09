from __future__ import annotations

import importlib

import pandas as pd
import pytest


def _kelly_module():
    try:
        return importlib.import_module("src.workflow.dynamic_soxx_soxl_cash_kelly_quarter")
    except ModuleNotFoundError:
        pytest.fail("quarter-Kelly SOXX/SOXL/cash workflow module is missing")


def test_quarter_kelly_sizing_uses_only_prior_source_returns():
    module = _kelly_module()
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]),
            "template_target_position": [1.0, 1.0, 1.0, 1.0],
            "template_strategy_return": [0.03, -0.01, 0.02, -0.50],
        }
    )

    sized = module._quarter_kelly_sizing_frame(
        frame,
        lookback=2,
        min_observations=2,
        kelly_fraction=0.25,
        max_kelly_scale=10.0,
        variance_floor=1.0e-12,
    )

    assert sized.loc[0, "kelly_scale"] == 0.0
    assert sized.loc[1, "kelly_scale"] == 0.0
    assert sized.loc[2, "kelly_scale"] == pytest.approx(3.125)
    assert sized.loc[3, "kelly_scale"] == pytest.approx(2.7777777778)
    assert sized.loc[3, "template_strategy_return"] == -0.50
    assert sized.loc[3, "kelly_scale"] > 0.0


def test_quarter_kelly_policy_combines_existing_signal_cap_and_soxx_floor():
    module = _kelly_module()
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]),
            "year": [2026, 2026, 2026, 2026],
            "template_target_position": [1.0, 1.0, 0.5, 1.0],
            "template_strategy_return": [0.03, -0.01, 0.02, -0.50],
            "template_turnover": [0.0, 0.0, 0.5, 0.5],
            "soxx_open_to_open_return": [0.01, -0.01, 0.02, -0.10],
            "soxl_open_to_open_return": [0.03, -0.03, 0.06, -0.30],
        }
    )

    policy = module._simulate_quarter_kelly_policy(
        frame,
        lookback=2,
        min_observations=2,
        kelly_fraction=0.25,
        max_kelly_scale=0.45,
        soxx_floor=0.20,
        cost_per_turnover=0.0,
        policy_id="quarter_kelly_test",
    )

    assert policy["policy_id"] == "quarter_kelly_test"
    assert policy["kelly_fraction"] == 0.25
    assert policy["max_kelly_scale"] == 0.45
    assert policy["selection_uses_test_segment"] is False
    assert policy["kelly_scale"][2] == pytest.approx(0.45)
    assert policy["soxl_weight"][2] == pytest.approx(0.225)
    assert policy["soxx_weight"][2] == pytest.approx(0.10)
    assert policy["cash_weight"][2] == pytest.approx(0.675)
    assert (policy["soxl_weight"] + policy["soxx_weight"] + policy["cash_weight"] <= 1.0 + 1.0e-12).all()


def test_quarter_kelly_runner_is_exported_from_workflow_package():
    module = _kelly_module()
    workflow = importlib.import_module("src.workflow")

    assert workflow.DynamicSoxxSoxlCashKellyQuarterSpec is module.DynamicSoxxSoxlCashKellyQuarterSpec
    assert workflow.run_dynamic_soxx_soxl_cash_kelly_quarter_card is module.run_dynamic_soxx_soxl_cash_kelly_quarter_card
