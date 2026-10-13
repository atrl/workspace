"""Fixed small Qlib forecast with training-only scaling and mature labels."""
import numpy as np
import pandas as pd
from qlib.model.base import Model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class CapacityForecast(Model):
    def __init__(self, columns, targets, cutoff, alpha=10.):
        self.columns, self.targets = list(columns), list(targets)
        self.cutoff, self.alpha = pd.Timestamp(cutoff), alpha

    def fit(self, dataset, **kwargs):
        data = dataset.prepare("train", col_set=["feature", "label", "meta"])
        known = pd.to_datetime(data[("meta", "feature_available_at")])
        decision = pd.to_datetime(data[("meta", "decision_at")])
        if not known.le(decision).all():
            raise ValueError("blocked: capacity forecast contains unavailable information")
        maturity = pd.to_datetime(data[("meta", "label_available_at")])
        keep = maturity.notna() & maturity.lt(self.cutoff)
        train = data.loc[keep]
        x, y = train["feature"][self.columns], train["label"][self.targets]
        if len(train) < 100 or not np.isfinite(x).all().all() or not np.isfinite(y).all().all():
            raise ValueError("needs_data: incomplete matured capacity forecast inputs")
        self.x_scaler, self.y_scaler = StandardScaler().fit(x), StandardScaler().fit(y)
        self.estimator = Ridge(alpha=self.alpha).fit(self.x_scaler.transform(x), self.y_scaler.transform(y))
        self.audit = {"training_rows": len(train), "cutoff": str(self.cutoff), "latest_label_available_at": str(maturity.loc[keep].max()),
            "columns": self.columns, "targets": self.targets, "alpha": self.alpha, "daily_labels_overlap": True}
        return self

    def predict(self, dataset, segment="test"):
        data = dataset.prepare(segment, col_set="feature")
        predicted = self.y_scaler.inverse_transform(self.estimator.predict(self.x_scaler.transform(data[self.columns])))
        return pd.DataFrame(predicted, index=data.index, columns=self.targets)
