"""Qlib fixed Ridge forecast with mature-label, training-only missing-value handling."""
import numpy as np
import pandas as pd
from qlib.model.base import Model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class RevenueForecast(Model):
    def __init__(self, columns, targets, cutoff, alpha=10.):
        self.columns, self.targets = list(columns), list(targets)
        self.cutoff, self.alpha = pd.Timestamp(cutoff), alpha

    def fit(self, dataset, **kwargs):
        data = dataset.prepare("train", col_set=["feature", "label", "meta"])
        meta = data["meta"]
        if not pd.to_datetime(meta.feature_available_at).le(pd.to_datetime(meta.decision_at)).all():
            raise ValueError("unavailable training information")
        maturity = pd.to_datetime(meta.label_available_at)
        train = data.loc[maturity.notna() & maturity.lt(self.cutoff)]
        x, y = train["feature"][self.columns], train["label"][self.targets]
        if len(train) < 100 or not np.isfinite(y).all().all() or np.isinf(x).any().any():
            raise ValueError("incomplete matured forecast input")
        self.impute_mean = x.mean().fillna(0.)
        encoded = x.fillna(self.impute_mean)
        self.x_scaler, self.y_scaler = StandardScaler().fit(encoded), StandardScaler().fit(y)
        self.estimator = Ridge(alpha=self.alpha).fit(self.x_scaler.transform(encoded), self.y_scaler.transform(y))
        self.audit = {"training_rows": len(train), "cutoff": str(self.cutoff), "latest_label_available_at": str(maturity.loc[train.index].max()),
            "columns": self.columns, "targets": self.targets, "alpha": self.alpha,
            "train_missing_counts": x.isna().sum().to_dict(), "train_impute_means": self.impute_mean.to_dict(),
            "all_unknown_columns_encoded_constant": x.columns[x.isna().all()].tolist(),
            "raw_unknowns_are_not_zero_surprises": True}
        return self

    def predict(self, dataset, segment="test"):
        x = dataset.prepare(segment, col_set="feature")[self.columns].fillna(self.impute_mean)
        return pd.DataFrame(self.y_scaler.inverse_transform(self.estimator.predict(self.x_scaler.transform(x))), index=x.index, columns=self.targets)
