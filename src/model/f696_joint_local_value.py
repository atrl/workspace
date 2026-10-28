"""Fixed shallow LightGBM value head on a Qlib DatasetH cohort."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from qlib.model.base import Model


class F696JointLocalValue(Model):
    def __init__(self, *, features: list[str], cutoff: str, seed: int = 71):
        self.features = list(features)
        self.cutoff = pd.Timestamp(cutoff)
        self.seed = int(seed)

    def fit(self, dataset, **kwargs):
        data = dataset.prepare("train", col_set=["feature", "label", "meta"])
        mature = pd.to_datetime(data[("meta", "label_available_at")]).lt(self.cutoff)
        known = pd.to_datetime(data[("meta", "feature_available_at")]).lt(
            pd.to_datetime(data[("meta", "decision_at")]))
        if not known.all():
            raise ValueError("blocked: local-value feature arrives after decision")
        train = data.loc[mature].sort_index()
        x = train["feature"][self.features].astype(float)
        y = train[("label", "local_advantage")].astype(float)
        if len(x) < 150 or not np.isfinite(x.to_numpy()).all() or not np.isfinite(y.to_numpy()).all():
            raise ValueError("needs_data: insufficient finite mature Qlib training labels")
        self.model = LGBMRegressor(n_estimators=60, num_leaves=7,
            min_child_samples=20, learning_rate=.03, reg_lambda=10.,
            max_depth=3, random_state=self.seed, n_jobs=1, verbosity=-1)
        self.model.fit(x, y)
        self.training_mean = float(y.mean())
        self.audit = {"training_rows": len(train), "cutoff": str(self.cutoff),
            "latest_mature_label_at": str(pd.to_datetime(data.loc[mature, ("meta", "label_available_at")]).max()),
            "features": self.features, "seed": self.seed, "fixed_estimator": "LightGBM_60_leaves7_depth3"}
        return self

    def predict(self, dataset, segment="test"):
        data = dataset.prepare(segment, col_set="feature")
        x = data[self.features].astype(float)
        if not np.isfinite(x.to_numpy()).all():
            raise ValueError("needs_data: local-value evaluation feature missing")
        return pd.Series(self.model.predict(x), index=data.index, name="local_value")
