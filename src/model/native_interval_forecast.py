"""A fixed Ridge forecast fitted/predicted through native Qlib DatasetH."""
from __future__ import annotations

import pandas as pd
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.base import Model
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class NativeIntervalRidge(Model):
    def __init__(self, *, features: list[str], alpha: float, mature_before: str):
        self.features = features
        self.alpha = alpha
        self.mature_before = pd.Timestamp(mature_before)
        self.pipeline = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        self.training_rows = 0

    def fit(self, dataset, **kwargs):
        raw = dataset.prepare("train", col_set=["feature", "label", "meta"], data_key=DataHandlerLP.DK_R)
        eligible = pd.to_datetime(raw[("meta", "label_available_at")]).le(self.mature_before)
        labels = raw.loc[:, [("label", "soxl"), ("label", "soxx")]]
        eligible &= labels.notna().all(axis=1)
        fit = raw.loc[eligible]
        if len(fit) < 200:
            raise ValueError("blocked: insufficient matured native interval labels")
        self.pipeline.fit(fit["feature"][self.features], fit["label"][["soxl", "soxx"]])
        self.training_rows = len(fit)
        self.last_training_label_at = pd.to_datetime(fit[("meta", "label_available_at")]).max()

    def predict(self, dataset, segment="test"):
        features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_R)
        prediction = self.pipeline.predict(features[self.features])
        return pd.DataFrame(prediction, index=features.index, columns=["pred_soxl", "pred_soxx"])
