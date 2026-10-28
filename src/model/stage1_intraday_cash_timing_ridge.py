"""Walk-forward Qlib ridge receiver for minute cash-deployment timing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from qlib.model.base import Model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class CashTimingRidge(Model):
    def __init__(self, *, features: list[str], label: str, cutoff: str,
                 alpha: float = 10.0):
        self.features = list(features)
        self.label = label
        self.cutoff = pd.Timestamp(cutoff)
        self.alpha = float(alpha)

    def fit(self, dataset, **kwargs):
        data = dataset.prepare("train", col_set=["feature", "label", "meta"])
        maturity = pd.to_datetime(data[("meta", "label_available_at")]).lt(self.cutoff)
        available = pd.to_datetime(data[("meta", "feature_available_at")]).le(
            pd.to_datetime(data[("meta", "decision_at")]))
        if not available.all():
            raise ValueError("blocked: minute feature arrives after decision")
        train = data.loc[maturity].sort_index()
        x = train["feature"][self.features].to_numpy(float)
        y = train[("label", self.label)].to_numpy(float)
        days = pd.to_datetime(train[("meta", "trade_date")]).dt.normalize()
        if (len(train) < 100 or days.nunique() < 100 or
                not np.isfinite(x).all() or not np.isfinite(y).all()):
            raise ValueError("needs_data: finite mature minute labels insufficient")
        count = days.map(days.value_counts()).to_numpy(float)
        weight = 1. / count
        self.scaler = StandardScaler().fit(x)
        scaled = self.scaler.transform(x)
        self.model = Ridge(alpha=self.alpha).fit(scaled, y, sample_weight=weight)
        self.audit = {"training_rows": len(train), "training_days": int(days.nunique()),
            "cutoff": str(self.cutoff), "features": self.features,
            "label": self.label, "alpha": self.alpha,
            "latest_label_available_at": str(pd.to_datetime(train[("meta", "label_available_at")]).max()),
            "coefficients": dict(zip(self.features, map(float, self.model.coef_))),
            "day_weighting": "inverse eligible event rows per trade date"}
        return self

    def predict(self, dataset, segment="test"):
        data = dataset.prepare(segment, col_set="feature")
        x = data[self.features].to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError("needs_data: minute evaluation feature missing")
        return pd.Series(self.model.predict(self.scaler.transform(x)), index=data.index,
                         name="predicted_fee_net_local_value")
