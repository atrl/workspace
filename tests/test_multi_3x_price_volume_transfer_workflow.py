from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from qlib.data.dataset import DatasetH


def _diagnostic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    parts: list[pd.DataFrame] = []
    dates = pd.bdate_range("2018-01-02", periods=320)
    for group_index, (instrument, group) in enumerate(
        [("AAA", "A"), ("BBB", "B"), ("CCC", "C")]
    ):
        good = rng.uniform(0.0, 1.0, len(dates))
        probability = np.clip(0.03 + 0.60 * good, 0.0, 0.95)
        label = rng.binomial(1, probability)
        bad = good if group_index < 2 else 1.0 - good
        parts.append(
            pd.DataFrame(
                {
                    "instrument": instrument,
                    "underlying_group": group,
                    "datetime": dates,
                    "high_level_state": 1.0,
                    "breakdown_label": label,
                    "good_factor": good,
                    "good_factor_copy": np.clip(good + rng.normal(0.0, 0.005, len(dates)), 0.0, 1.0),
                    "bad_factor": bad,
                    "range_pos_252": good,
                    "low_trend_efficiency_pressure_20": good,
                    "sign_change_density_pressure_20": good,
                    "high_chop_price_pressure": good,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_group_diagnostics_identify_directional_factor_and_quintile_lift():
    from src.workflow.multi_3x_price_volume_transfer import factor_group_diagnostics

    metrics = factor_group_diagnostics(
        _diagnostic_frame(),
        factors=["good_factor", "bad_factor"],
        min_observations=100,
        min_events=10,
    )

    good = metrics.loc[metrics["factor"] == "good_factor"]
    assert len(good) == 3
    assert good["eligible"].all()
    assert good["direction_pass"].all()
    assert good["quintile_lift"].median() > 0.20
    bad_c = metrics.loc[(metrics["factor"] == "bad_factor") & (metrics["underlying_group"] == "C")].iloc[0]
    assert not bool(bad_c["direction_pass"])


def test_factor_selection_applies_transfer_and_redundancy_gates():
    from src.workflow.multi_3x_price_volume_transfer import (
        factor_group_diagnostics,
        select_transferable_factors,
    )

    frame = _diagnostic_frame()
    metrics = factor_group_diagnostics(
        frame,
        factors=["good_factor", "good_factor_copy", "bad_factor"],
        min_observations=100,
        min_events=10,
    )
    correlation = frame[["good_factor", "good_factor_copy", "bad_factor"]].corr(method="spearman")
    result = select_transferable_factors(metrics, correlation=correlation)

    assert len(result["selected_factors"]) == 1
    assert result["selected_factors"][0] in {"good_factor", "good_factor_copy"}
    factor_summary = result["factor_summary"].set_index("factor")
    assert bool(factor_summary.loc["good_factor", "development_gate_passed"])
    assert bool(factor_summary.loc["good_factor_copy", "development_gate_passed"])
    assert factor_summary.loc["bad_factor", "direction_consistency"] < 1.0
    assert "redundant" in set(factor_summary["selection_status"])


def test_development_and_outer_split_keeps_soxl_fully_unseen():
    from src.workflow.multi_3x_price_volume_transfer import split_development_outer

    dates = pd.to_datetime(["2022-12-30", "2023-01-03", "2024-01-03"])
    frame = pd.DataFrame(
        {
            "instrument": ["TQQQ", "SOXL", "SOXL", "TQQQ"],
            "datetime": [dates[0], dates[1], dates[2], dates[2]],
            "underlying_group": ["QQQ", "SOXX", "SOXX", "QQQ"],
        }
    )
    development, outer = split_development_outer(frame, development_end="2022-12-31", outer_start="2023-01-01")

    assert set(development["instrument"]) == {"TQQQ"}
    assert set(outer["instrument"]) == {"SOXL"}
    assert outer["datetime"].min() >= pd.Timestamp("2023-01-01")


def test_outer_composite_metrics_compare_price_and_selected_volume_factors():
    from src.workflow.multi_3x_price_volume_transfer import evaluate_outer_composites

    frame = _diagnostic_frame().query("instrument == 'AAA'").copy()
    frame["datetime"] = pd.bdate_range("2023-01-03", periods=len(frame))
    metrics, scored = evaluate_outer_composites(frame, selected_factors=["good_factor"])

    assert {"price_only", "price_volume"}.issubset(set(metrics["composite"]))
    overall = metrics.loc[metrics["window"] == "overall"].set_index("composite")
    assert overall.loc["price_volume", "auc"] > 0.60
    assert overall.loc["price_volume", "quintile_lift"] > 0.20
    assert {"price_only_score", "price_volume_score"}.issubset(scored.columns)


def test_transfer_gate_fails_closed_without_selected_factors_or_eligible_outer_data():
    from src.workflow.multi_3x_price_volume_transfer import evaluate_transfer_decision

    decision = evaluate_transfer_decision(
        selected_factors=[],
        outer_metrics=pd.DataFrame(),
        eligible_development_groups=3,
    )
    assert decision["decision"] == "not_retained_no_transferable_price_volume_evidence"
    assert not decision["all_passed"]

    with pytest.raises(ValueError, match="eligible development groups"):
        evaluate_transfer_decision(
            selected_factors=["factor"],
            outer_metrics=pd.DataFrame(),
            eligible_development_groups=0,
        )


def test_build_price_volume_dataset_uses_native_qlib_dataset_contract(tmp_path):
    from src.data import build_qlib_store_from_frame
    from src.workflow.multi_3x_price_volume_transfer import build_price_volume_dataset

    dates = pd.bdate_range("2021-01-04", periods=50)
    rows = []
    for instrument in ["AAA", "BBB"]:
        rows.append(
            pd.DataFrame(
                {
                    "instrument": instrument,
                    "datetime": dates,
                    "good_factor": np.linspace(0.0, 1.0, len(dates)),
                    "breakdown_label": np.arange(len(dates)) % 2,
                }
            )
        )
    provider = tmp_path / "qlib" / "price_volume"
    build_qlib_store_from_frame(
        pd.concat(rows, ignore_index=True),
        provider_uri=provider,
        as_of=dates[-1].date().isoformat(),
    )

    dataset = build_price_volume_dataset(
        provider,
        instruments=["AAA", "BBB"],
        features=["good_factor"],
        train=("2021-01-04", "2021-02-05"),
        valid=("2021-02-08", "2021-02-26"),
        test=("2021-03-01", "2021-03-12"),
    )
    assert isinstance(dataset, DatasetH)
    assert dataset.qlib_dataset_api == ["DatasetH", "DataHandlerLP", "QlibDataLoader"]
    assert {"$good_factor", "$breakdown_label"}.issubset(dataset.prepare("train").columns)
