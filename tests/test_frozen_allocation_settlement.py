from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.workflow.pit_constituent_auxiliary_training_allocation import CANDIDATES, PitAuxiliaryAllocationNet, PitAuxiliaryAllocationSpec, _smooth_weights
from src.workflow.intraday_recovery_direct_allocation import _frozen_outer_predictions
from src.workflow import stage1_futu_sim as runtime


def _inputs():
    index = pd.to_datetime(["2026-09-23", "2026-09-24", "2026-09-25"])
    predictions = pd.DataFrame({"soxl_weight": .2, "soxx_weight": .5, "cash_weight": .3}, index=index)
    market = pd.DataFrame({
        "soxl_forward_open_return": [0.1, np.nan, np.nan],
        "soxx_forward_open_return": [0.0, np.nan, np.nan],
        "trade_date": pd.to_datetime(["2026-09-24", "2026-09-25", None]),
        "exit_date": pd.to_datetime(["2026-09-25", None, None]),
    }, index=index)
    return predictions, market, {
        "spec": PitAuxiliaryAllocationSpec(root=Path(".")),
        "candidate": next(c for c in CANDIDATES if c.name == "pit_only_aux_005"),
        "execution_provider_uri": Path("unused"),
    }


def test_historical_frozen_extension_passes_marked_returns_to_smoother():
    # The shared smoothing contract is tested directly; its result must include
    # drift on a hold day rather than reuse yesterday's nominal target weight.
    predictions, market, kwargs = _inputs()
    market.loc[:, "soxl_forward_open_return"] = [0.01, .01, .01]
    market.loc[:, "soxx_forward_open_return"] = 0.0
    result, _, audit = runtime._smooth_frozen_allocation(predictions, market=market, **kwargs)
    expected = _smooth_weights(predictions, spec=kwargs["spec"], candidate=kwargs["candidate"],
                               asset_returns=market[["soxl_forward_open_return", "soxx_forward_open_return"]])
    pd.testing.assert_frame_equal(result, expected)
    assert result.iloc[1].soxl_weight > predictions.iloc[1].soxl_weight
    assert audit["status"] == "ready"


def test_frozen_model_historical_loader_preserves_holdings_drift():
    predictions, market, kwargs = _inputs()
    market.loc[:, "soxl_forward_open_return"] = .01
    market.loc[:, "soxx_forward_open_return"] = 0.0
    market["primary"] = 0.0
    common = pd.DataFrame({"common": 0.0}, index=market.index)
    model = PitAuxiliaryAllocationNet(primary_feature_count=1, common_feature_count=1,
                                     hidden_units=16, tail_hidden_units=8)
    state = {
        "primary_features": ["primary"], "common_features": ["common"],
        "primary_scaler": {"median": {"primary": 0.0}, "mean": {"primary": 0.0}, "scale": {"primary": 1.0}},
        "common_scaler": {"median": {"common": 0.0}, "mean": {"common": 0.0}, "scale": {"common": 1.0}},
        "model_state": {key: torch.zeros_like(value).numpy() for key, value in model.state_dict().items()},
        "candidate": asdict(kwargs["candidate"]),
    }
    result = _frozen_outer_predictions(market=market, common=common, state=state, spec=kwargs["spec"])
    assert result.iloc[1].soxl_weight > result.iloc[0].soxl_weight
    assert result.iloc[1].soxx_weight < result.iloc[0].soxx_weight


def test_pending_interval_is_not_filled_with_an_assumed_zero_return():
    predictions, market, kwargs = _inputs()
    result, evidence, audit = runtime._smooth_frozen_allocation(predictions, market=market, **kwargs)
    assert result.iloc[-1].isna().all()
    assert result.iloc[:-1].notna().all().all()
    assert evidence.iloc[-2][["soxl_forward_open_return", "soxx_forward_open_return"]].isna().all()
    assert audit["status"] == "needs_execution_mark"
    assert audit["pending_signal_dates"] == ["2026-09-25"]


def test_current_execution_mark_settles_prior_interval_but_preserves_unobserved_last_return(monkeypatch):
    predictions, market, kwargs = _inputs()
    monkeypatch.setattr(runtime, "_qlib_open_prices", lambda **_: {"SOXL": 100.0, "SOXX": 200.0})
    result, evidence, audit = runtime._smooth_frozen_allocation(
        predictions, market=market, **kwargs, current_prices={"SOXL": 102.0, "SOXX": 200.0},
        execution_date="2026-09-28", current_prices_as_of="2026-09-28T09:31:00-04:00",
    )
    assert result.notna().all().all()
    assert evidence.iloc[-2].soxl_forward_open_return == pytest.approx(.02)
    assert evidence.iloc[-2].exit_date == pd.Timestamp("2026-09-28")
    assert evidence.iloc[-1][["soxl_forward_open_return", "soxx_forward_open_return"]].isna().all()
    assert audit["unobserved_final_return_signal_date"] == "2026-09-25"
    assert audit["unobserved_final_return_consumed"] is False
    assert audit["current_execution_marks"][0]["price_observed_at"] == "2026-09-28T09:31:00-04:00"
    assert result.iloc[-1].soxl_weight == pytest.approx(.2 * 1.02 / 1.004)


@pytest.mark.parametrize("mark", [None, "2026-09-25T09:31:00-04:00", "2026-09-28T08:00:00-04:00", "2026-09-28T09:31:00"])
def test_undated_stale_or_nonregular_quotes_cannot_complete_settlement(mark):
    predictions, market, kwargs = _inputs()
    with pytest.raises(ValueError, match="blocked:"):
        runtime._smooth_frozen_allocation(predictions, market=market, **kwargs,
            current_prices={"SOXL": 102.0, "SOXX": 200.0}, execution_date="2026-09-28", current_prices_as_of=mark)


def test_only_final_unobserved_return_does_not_block_current_target():
    predictions, market, kwargs = _inputs()
    market.loc[predictions.index[-2], ["soxl_forward_open_return", "soxx_forward_open_return"]] = [.02, 0.0]
    result, evidence, audit = runtime._smooth_frozen_allocation(predictions, market=market, **kwargs)
    assert result.notna().all().all()
    assert audit["status"] == "ready"
    assert evidence.iloc[-1][["soxl_forward_open_return", "soxx_forward_open_return"]].isna().all()
