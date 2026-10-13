import numpy as np
import pandas as pd
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from src.model.capacity_forecast import CapacityForecast
from src.workflow.fed_capacity_information import evaluation_mask


def test_forecast_scalers_and_coefficients_exclude_unmatured_labels():
    dates = pd.bdate_range("2023-01-03", periods=112)
    features = pd.DataFrame({"price": np.sin(np.arange(112)), "capacity": np.cos(np.arange(112) / 20)}, index=dates)
    labels = pd.DataFrame({"return": features.price * .01, "risk": features.capacity - 8}, index=dates)
    meta = pd.DataFrame({"feature_available_at": dates - pd.Timedelta(days=1), "decision_at": dates,
                         "label_available_at": dates + pd.Timedelta(days=30)}, index=dates)
    meta.loc[dates[-2:], "label_available_at"] = pd.Timestamp("2024-01-01")
    fitted = []
    for changed in (False, True):
        x, y = features.copy(), labels.copy()
        if changed:
            x.loc[dates[-2:]] = 1e8; y.loc[dates[-2:]] = -1e8
        data = pd.concat({"feature": x, "label": y, "meta": meta}, axis=1)
        data.index = pd.MultiIndex.from_arrays([dates, ["SOXX"] * len(dates)], names=["datetime", "instrument"])
        dataset = DatasetH(handler=DataHandlerLP.from_df(data), segments={"train": (None, "2023-12-31"), "test": (None, None)})
        fitted.append(CapacityForecast(list(x), list(y), "2024-01-01").fit(dataset))
    assert fitted[0].audit["training_rows"] == 110
    np.testing.assert_array_equal(fitted[0].x_scaler.mean_, fitted[1].x_scaler.mean_)
    np.testing.assert_array_equal(fitted[0].y_scaler.mean_, fitted[1].y_scaler.mean_)
    np.testing.assert_array_equal(fitted[0].estimator.coef_, fitted[1].estimator.coef_)


def test_development_evaluation_excludes_next_year_outcomes():
    truth = pd.DataFrame({"return": [.01, .02, np.nan], "risk": [-8., -7., np.nan]})
    meta = pd.DataFrame({"label_available_at": pd.to_datetime(["2022-12-30 09:49", "2023-01-03 09:49", None])})
    fold = {"name": "development_2022", "end": "2022-12-31"}
    assert evaluation_mask(truth, meta, fold, "2026-10-09").tolist() == [True, False, False]
