from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qlib.data.dataset import DatasetH


def _tiny_provider(tmp_path):
    from src.data import build_qlib_store_from_frame

    dates = pd.bdate_range("2024-01-02", periods=18)
    frames = []
    for instrument, offset in [("SOXL", 0.0), ("TQQQ", 0.1)]:
        frames.append(
            pd.DataFrame(
                {
                    "instrument": instrument,
                    "datetime": dates,
                    "ret_5_pctile252": np.linspace(0.1 + offset, 0.9, len(dates)),
                    "high_chop_intensity": np.linspace(0.2, 0.8 - offset, len(dates)),
                    "path_label": np.arange(len(dates)) % 4,
                }
            )
        )
    provider = tmp_path / "qlib" / "path"
    build_qlib_store_from_frame(
        pd.concat(frames, ignore_index=True),
        provider_uri=provider,
        as_of=dates[-1].date().isoformat(),
    )
    return provider


def test_underlying_groups_keep_shared_products_together():
    from src.workflow.multi_3x_high_chop_risk_control import RESEARCH_ETFS, UNDERLYING_GROUPS

    assert UNDERLYING_GROUPS["UPRO"] == UNDERLYING_GROUPS["SPXL"] == "SPY"
    assert UNDERLYING_GROUPS["SOXL"] == "SOXX"
    assert len(RESEARCH_ETFS) == 16
    assert {"SOXL", "TNA", "FAS", "LABU", "ERX", "WANT"}.issubset(RESEARCH_ETFS)


def test_candidate_selection_uses_validation_only_and_has_stable_tie_break():
    from src.workflow.multi_3x_high_chop_risk_control import select_risk_control_candidate

    metrics = pd.DataFrame(
        [
            {"candidate": "rule_baseline", "validation_utility": 0.10, "validation_log_loss": 1.1, "outer_test_return": 9.0},
            {"candidate": "multinomial_logistic", "validation_utility": 0.30, "validation_log_loss": 1.2, "outer_test_return": -9.0},
            {"candidate": "lightgbm_multiclass", "validation_utility": 0.30, "validation_log_loss": 1.3, "outer_test_return": 99.0},
        ]
    )
    selected = select_risk_control_candidate(metrics)

    assert selected["candidate"] == "multinomial_logistic"
    assert selected["selection_uses_outer_test"] is False
    assert selected["selection_metric"] == "validation_utility_desc_then_log_loss_asc"


def test_build_path_dataset_uses_native_qlib_dataset_contract(tmp_path):
    from src.workflow.multi_3x_high_chop_risk_control import build_path_dataset

    provider = _tiny_provider(tmp_path)
    dataset = build_path_dataset(
        provider,
        instruments=["SOXL", "TQQQ"],
        features=["ret_5_pctile252", "high_chop_intensity"],
        train=("2024-01-02", "2024-01-10"),
        valid=("2024-01-11", "2024-01-17"),
        test=("2024-01-18", "2024-01-25"),
    )

    assert isinstance(dataset, DatasetH)
    assert dataset.qlib_dataset_api == ["DatasetH", "DataHandlerLP", "QlibDataLoader"]
    prepared = dataset.prepare("train")
    assert {"$ret_5_pctile252", "$high_chop_intensity", "$path_label"}.issubset(prepared.columns)


def test_promotion_gate_fails_closed_when_risk_or_transfer_evidence_is_missing():
    from src.workflow.multi_3x_high_chop_risk_control import evaluate_promotion_gates

    gates = evaluate_promotion_gates(
        {
            "max_drawdown": -0.12,
            "annualized_volatility": 0.40,
            "cvar_95": -0.05,
            "upside_capture": 0.80,
            "average_risky_weight": 0.45,
        },
        benchmark={"max_drawdown": -0.20, "annualized_volatility": 0.55, "cvar_95": -0.07},
        transfer_fold_pass_rate=0.40,
    )

    assert gates["promotion_decision"] == "not_promoted_risk_control_gates_failed"
    assert gates["gates"]["transfer_fold_pass_rate"]["passed"] is False
    assert gates["all_passed"] is False


def test_calibration_never_reads_outer_test_labels(monkeypatch):
    from src.workflow import multi_3x_high_chop_risk_control as workflow

    valid_y = pd.Series([0, 1, 2, 3] * 4)
    valid_p = np.full((len(valid_y), 4), 0.25)
    temperature = workflow.select_validation_temperature(valid_y, valid_p)

    assert temperature in workflow.CALIBRATION_TEMPERATURES
    with pytest.raises(TypeError):
        workflow.select_validation_temperature(valid_y, valid_p, outer_test_labels=valid_y)
