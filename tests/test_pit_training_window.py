import numpy as np
import pandas as pd
import pytest

from src.workflow.pit_allocation_retrain import PRIMARY_LABELS, _mature_training_labels
from src.workflow.pit_training_window import evaluate_base_allocator, filter_training_window


def test_rolling_window_filters_primary_and_auxiliary_rows_without_future_data():
    dates = pd.to_datetime(["2017-12-29", "2018-01-02", "2025-12-31", "2026-01-02"])
    primary = pd.DataFrame({"feature": [1, 2, 3, 4]}, index=dates)
    result = filter_training_window(primary, start="2018-01-01", cutoff="2025-12-31")
    assert result.index.tolist() == dates[1:3].tolist()
    auxiliary = primary.reset_index(names="datetime")
    result_aux = filter_training_window(auxiliary, start="2018-01-01", cutoff="2025-12-31", date_column="datetime")
    assert result_aux.datetime.tolist() == dates[1:3].tolist()


def test_five_return_labels_are_purged_by_actual_exit_date_across_new_year():
    dates = pd.to_datetime(["2025-12-19", "2025-12-22", "2025-12-23", "2025-12-24", "2025-12-26", "2025-12-29", "2025-12-30", "2025-12-31", "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"])
    frame = pd.DataFrame({name: 1.0 for name in PRIMARY_LABELS}, index=dates)
    frame["exit_date"] = pd.Series(dates, index=dates).shift(-2)
    matured, audit = _mature_training_labels(frame, "2025-12-31")
    train = filter_training_window(matured, start="2011-01-03", cutoff="2025-12-31").dropna(subset=PRIMARY_LABELS)
    assert train.index.max() == pd.Timestamp("2025-12-22")
    assert audit["last_training_label_realization"] == "2025-12-31"
    assert matured.loc["2025-12-23":"2025-12-31", PRIMARY_LABELS].isna().all().all()


@pytest.mark.parametrize("cost_bps", [10., 25.])
def test_base_comparison_projects_passive_drift_and_charges_net_cash_fees(cost_bps):
    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    market = pd.DataFrame({
        "trade_date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
        "exit_date": pd.to_datetime(["2026-01-06", "2026-01-07"]),
        "soxl_forward_open_return": [0., 0.], "soxx_forward_open_return": [0., 0.],
    }, index=dates)
    predictions = pd.DataFrame({"soxl_weight": [.35, .35], "soxx_weight": [.52, .52], "cash_weight": [.13, .13]}, index=dates)
    metrics, replay = evaluate_base_allocator(market, predictions, cost_bps=cost_bps)
    assert replay.soxl_weight.le(.30 + 1e-12).all()
    assert replay.cash_weight.ge(.15 - 1e-12).all()
    np.testing.assert_allclose(replay[["ending_soxl_weight", "ending_soxx_weight", "ending_cash_weight"]], [[.3, .55, .15]] * 2, atol=1e-12)
    assert metrics["static_projection_rows"] == 2
    assert metrics["open_trade_days"] == 1
    assert metrics["max_drawdown_pct"] < 0  # Initial fees are part of drawdown.
