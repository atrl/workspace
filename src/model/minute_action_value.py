"""Chronological Qlib Ridge action value with a conditional-mean error bar."""
from __future__ import annotations

import numpy as np
import pandas as pd
from qlib.model.base import Model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class MinuteActionValue(Model):
    def __init__(self, *, features, cutoff, alpha, minimum_observations, hac_days):
        self.features = list(features)
        self.cutoff = pd.Timestamp(cutoff)
        self.alpha = float(alpha)
        self.minimum_observations = int(minimum_observations)
        self.hac_days = int(hac_days)

    def fit(self, dataset, **kwargs):
        data = dataset.prepare("train", col_set=["feature", "label", "meta"])
        maturity = pd.to_datetime(data[("meta", "label_available_at")])
        keep = maturity.notna() & maturity.lt(self.cutoff)
        keep &= np.isfinite(data["feature"][self.features]).all(axis=1)
        keep &= np.isfinite(data[("label", "net_advantage")])
        train = data.loc[keep]
        if len(train) < self.minimum_observations:
            raise ValueError("needs_data: insufficient matured action-value labels")
        self.scaler = StandardScaler().fit(train["feature"][self.features])
        x = self.scaler.transform(train["feature"][self.features]).clip(-8, 8)
        y = train[("label", "net_advantage")].to_numpy(float)
        self.estimator = Ridge(alpha=self.alpha).fit(x, y)
        z = np.column_stack([np.ones(len(x)), x])
        residual = y - self.estimator.predict(x)
        penalty = np.diag([0., *([self.alpha] * x.shape[1])])
        bread = np.linalg.pinv(z.T @ z + penalty)
        score = z * residual[:, None]
        meat = score.T @ score
        dates = pd.DatetimeIndex(train.index.get_level_values("datetime")).normalize()
        # Calendar-day Bartlett HAC; gaps between eligible actions are retained.
        for i in range(len(train)):
            for j in range(i + 1, len(train)):
                gap = (dates[j] - dates[i]).days
                if gap > self.hac_days:
                    break
                weight = 1. - gap / (self.hac_days + 1.)
                pair = np.outer(score[i], score[j])
                meat += weight * (pair + pair.T)
        covariance = bread @ meat @ bread
        eigenvalues, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
        self.covariance = (vectors * np.maximum(eigenvalues, 0.)) @ vectors.T
        self.audit = {"observations": len(train), "latest_label_available_at": str(maturity.loc[keep].max()),
            "fit_cutoff": str(self.cutoff), "features": self.features, "alpha": self.alpha,
            "hac_calendar_days": self.hac_days,
            "uncertainty_role": "conditional-mean estimation heuristic; not calibrated future-return coverage"}
        return self

    def predict_frame(self, features):
        x = self.scaler.transform(features[self.features]).clip(-8, 8)
        z = np.column_stack([np.ones(len(x)), x])
        return pd.DataFrame({"mean": self.estimator.predict(x),
            "mean_error": np.sqrt(np.maximum(0., np.einsum("ij,jk,ik->i", z, self.covariance, z)))}, index=features.index)

    def predict(self, dataset, segment="test"):
        return self.predict_frame(dataset.prepare(segment, col_set="feature"))
