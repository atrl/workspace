from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.workflow.auxiliary_3x_training_data_allocation import (
    CANDIDATES,
    Auxiliary3xTrainingDataSpec,
    _build_dataset,
    _fit_predict,
    _prepare_panel,
    _training_end,
)


def _panel() -> pd.DataFrame:
    dates = pd.date_range("2015-01-02", periods=800, freq="B")
    rows = []
    for instrument in ("SOXL", "SPXL"):
        for offset, date in enumerate(dates):
            rows.append(
                {
                    "datetime": date,
                    "instrument": instrument,
                    "ret_5_z20": np.sin(offset / 20.0),
                    "ret_20_pctile60": (offset % 60) / 60.0,
                    "soxl_forward_open_return": 0.001,
                    "soxx_forward_open_return": 0.0005,
                    "target_soxl_weight": 0.15,
                    "target_soxx_weight": 0.65,
                    "target_cash_weight": 0.20,
                }
            )
    return pd.DataFrame(rows)


def test_candidates_keep_auxiliary_assets_out_of_primary_identity() -> None:
    candidates = {candidate.name: candidate for candidate in CANDIDATES}

    assert candidates["soxl_only_common_features"].weights == {"SOXL": 1.0}
    assert "UPRO" not in {
        asset
        for candidate in CANDIDATES
        for asset in candidate.weights
    }
    assert all(candidate.weights["SOXL"] == 1.0 for candidate in CANDIDATES)


def test_qlib_dataset_preserves_panel_instruments_and_segments() -> None:
    panel = _panel()
    spec = Auxiliary3xTrainingDataSpec(
        root=Path("."),
        development_years=(2016, 2017),
        development_end="2017-12-29",
        outer=("2018-01-01", "2018-01-31"),
        as_of="2018-01-31",
    )

    dataset, audit = _build_dataset(
        panel, ["ret_5_z20", "ret_20_pctile60"], spec
    )
    prepared = _prepare_panel(dataset, "development")

    assert set(prepared["instrument"]) == {"SOXL", "SPXL"}
    assert audit["dataset_class"] == "qlib.data.dataset.DatasetH"
    assert audit["selection_uses_outer_test"] is False


def test_fit_predict_fails_closed_if_auxiliary_asset_enters_test() -> None:
    panel = _panel()
    candidate = next(
        candidate
        for candidate in CANDIDATES
        if candidate.name == "spxl_aux_half_weight"
    )
    spec = Auxiliary3xTrainingDataSpec(root=Path("."), epochs=2)
    train = panel.loc[panel["datetime"].lt("2017-01-01")]
    invalid_test = panel.loc[
        panel["datetime"].between("2017-01-02", "2017-02-01")
        & panel["instrument"].eq("SPXL")
    ]

    with pytest.raises(
        ValueError, match="auxiliary asset entered SOXL test set"
    ):
        _fit_predict(
            train=train,
            test=invalid_test,
            features=["ret_5_z20", "ret_20_pctile60"],
            candidate=candidate,
            spec=spec,
            seed=17,
        )


def test_fit_predict_emits_legal_direct_weights_for_soxl_only_test() -> None:
    panel = _panel()
    candidate = next(
        candidate
        for candidate in CANDIDATES
        if candidate.name == "spxl_aux_half_weight"
    )
    spec = Auxiliary3xTrainingDataSpec(root=Path("."), epochs=3)
    train = panel.loc[panel["datetime"].lt("2017-01-01")]
    test = panel.loc[
        panel["datetime"].between("2017-01-02", "2017-02-01")
        & panel["instrument"].eq("SOXL")
    ]

    prediction, _, diagnostics = _fit_predict(
        train=train,
        test=test,
        features=["ret_5_z20", "ret_20_pctile60"],
        candidate=candidate,
        spec=spec,
        seed=17,
    )

    assert np.allclose(
        prediction[
            ["soxl_weight", "soxx_weight", "cash_weight"]
        ].sum(axis=1),
        1.0,
    )
    assert prediction["soxl_weight"].le(0.30).all()
    assert prediction[["soxl_weight", "soxx_weight", "cash_weight"]].ge(
        0.0
    ).all().all()
    assert diagnostics["auxiliary_train_share"] > 0.0


def test_training_end_uses_calendar_rows_for_purge() -> None:
    panel = _panel()
    calendar = np.sort(pd.to_datetime(panel["datetime"]).unique())
    expected_position = (
        np.searchsorted(calendar, np.datetime64("2017-01-01")) - 21
    )

    actual = _training_end(panel, 2017, purge_rows=20)

    assert actual == pd.Timestamp(calendar[expected_position])
