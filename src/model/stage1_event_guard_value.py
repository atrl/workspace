"""Fixed shallow event-time local-value model with day-balanced Qlib rows."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from qlib.model.base import Model


class EventGuardValue(Model):
    def __init__(self, *, features: list[str], cutoff: str, seed: int = 71):
        self.features = list(features)
        self.cutoff = pd.Timestamp(cutoff)
        self.seed = int(seed)

    def fit(self, dataset, **kwargs):
        data = dataset.prepare("train", col_set=["feature", "label", "meta"])
        mature = pd.to_datetime(data[("meta", "label_available_at")]).lt(self.cutoff)
        known = pd.to_datetime(data[("meta", "feature_available_at")]).le(
            pd.to_datetime(data[("meta", "decision_at")]))
        if not known.all():
            raise ValueError("blocked: event feature arrives after decision")
        train = data.loc[mature].sort_index()
        x = train["feature"][self.features].astype(float)
        y = train[("label", "local_advantage")].astype(float)
        days = pd.to_datetime(train[("meta", "trade_date")]).dt.normalize()
        if len(x) < 1000 or days.nunique() < 100 or not np.isfinite(x.to_numpy()).all() or not np.isfinite(y.to_numpy()).all():
            raise ValueError("needs_data: insufficient finite event labels")
        count = days.map(days.value_counts()).to_numpy(float)
        weight = 1. / count
        self.model = LGBMRegressor(n_estimators=60, num_leaves=7,
            min_child_samples=20, learning_rate=.03, reg_lambda=10.,
            max_depth=3, random_state=self.seed, n_jobs=1, verbosity=-1)
        self.model.fit(x, y, sample_weight=weight)
        self.audit = {"training_rows": len(train), "training_days": int(days.nunique()),
            "cutoff": str(self.cutoff),
            "latest_label_available_at": str(pd.to_datetime(train[("meta", "label_available_at")]).max()),
            "features": self.features, "seed": self.seed,
            "day_weighting": "inverse eligible minute rows per trade date",
            "estimator": "LightGBM_60_leaves7_depth3"}
        return self

    def predict(self, dataset, segment="test"):
        data = dataset.prepare(segment, col_set="feature")
        x = data[self.features].astype(float)
        if not np.isfinite(x.to_numpy()).all():
            raise ValueError("needs_data: event evaluation feature missing")
        return pd.Series(self.model.predict(x), index=data.index, name="fee_net_switch_value")
