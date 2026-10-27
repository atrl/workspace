"""Independent three-asset tree allocator for the frozen full-path Qlib frame."""
from __future__ import annotations

import numpy as np
import pandas as pd
from qlib.model.base import Model
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from src.backtest.ibkr_tiered import TieredFees, settle_tiered_event


ACTION_WEIGHTS = np.eye(3, dtype=float)  # SOXL, SOXX, cash


def one_step_net_labels(frame: pd.DataFrame, *, tariff: dict, initial_cash: float) -> np.ndarray:
    """Best 0bp one-step action from identical cash, including round-trip Tiered fees."""
    required = ["soxl_open", "soxx_open", "raw_soxl_open", "raw_soxx_open",
                "soxl_exit_open", "soxx_exit_open", "raw_soxl_exit_open", "raw_soxx_exit_open"]
    if not np.isfinite(frame[required].to_numpy(float)).all():
        raise ValueError("needs_data: incomplete one-step label marks")
    labels = []
    for row in frame.itertuples():
        value = [float(initial_cash)]
        for asset in (0, 1):
            target = ACTION_WEIGHTS[asset]
            fees = TieredFees(tariff, slippage_bps=0)
            buy = settle_tiered_event([0., 0.], initial_cash, desired=target,
                adjusted_prices=[row.soxl_open, row.soxx_open],
                raw_prices=[row.raw_soxl_open, row.raw_soxx_open], stamp=row.trade_date, fees=fees)
            sell = settle_tiered_event(buy["quantities"], buy["cash"], desired=ACTION_WEIGHTS[2],
                adjusted_prices=[row.soxl_exit_open, row.soxx_exit_open],
                raw_prices=[row.raw_soxl_exit_open, row.raw_soxx_exit_open], stamp=row.exit_date, fees=fees)
            value.append(float(sell["cash"]))
        # value is cash, SOXL, SOXX; exact ties in the reordered vector favor SOXL.
        labels.append(int(np.argmax([value[1], value[2], value[0]])))
    return np.asarray(labels, dtype=int)


class IndependentTreeAllocation(Model):
    def __init__(self, *, family: str, columns: list[str], cutoff: str, tariff: dict,
                 initial_cash: float, seed: int, constrained: bool = False,
                 maximum_soxl_weight: float = .30, minimum_cash_weight: float = .15,
                 stop_drawdown: float = .04, stop_multiplier: float = .10):
        self.family, self.columns, self.cutoff = family, list(columns), pd.Timestamp(cutoff)
        self.tariff, self.initial_cash, self.seed = tariff, float(initial_cash), int(seed)
        self.constrained = constrained
        self.maximum_soxl_weight, self.minimum_cash_weight = maximum_soxl_weight, minimum_cash_weight
        self.stop_drawdown, self.stop_multiplier = stop_drawdown, stop_multiplier

    def fit(self, dataset, **kwargs):
        data = dataset.prepare("train", col_set=["feature", "price", "meta"])
        meta = data["meta"]
        mature = pd.to_datetime(meta.label_available_at).lt(self.cutoff)
        if not pd.to_datetime(meta.feature_available_at).lt(pd.to_datetime(meta.decision_at)).all():
            raise ValueError("blocked: training feature arrives after decision")
        train = data.loc[mature].sort_index()
        if len(train) < 300:
            raise ValueError("needs_data: fewer than 300 mature training rows")
        # DatasetH keeps the Qlib-aligned feature and price planes together.
        prices = train["price"].copy()
        for field in ("raw_soxl_exit_open", "raw_soxx_exit_open"):
            if field not in prices:
                raise ValueError("needs_data: raw exit prices absent from Qlib dataset")
        labels = one_step_net_labels(prices, tariff=self.tariff, initial_cash=self.initial_cash)
        if len(np.unique(labels)) < 2:
            raise ValueError("needs_data: single action in training labels")
        if self.family == "xgb":
            from xgboost import XGBClassifier
            classifier = XGBClassifier(n_estimators=70, max_depth=3, learning_rate=.035,
                subsample=.8, colsample_bytree=.8, reg_alpha=.1, reg_lambda=1.2,
                objective="multi:softprob", eval_metric="mlogloss", random_state=self.seed, n_jobs=1)
        elif self.family == "lgbm":
            from lightgbm import LGBMClassifier
            classifier = LGBMClassifier(n_estimators=60, learning_rate=.035, num_leaves=15,
                min_child_samples=35, subsample=.8, colsample_bytree=.8, reg_alpha=.15,
                reg_lambda=.8, random_state=self.seed, verbose=-1, n_jobs=1)
        else:
            raise ValueError(f"unsupported tree family: {self.family}")
        self.pipeline = make_pipeline(SimpleImputer(strategy="median"), classifier)
        self.pipeline.fit(train["feature"][self.columns].astype(float), labels)
        self.audit = {"family": self.family, "cutoff": str(self.cutoff), "training_rows": len(train),
            "latest_label_available_at": str(pd.to_datetime(meta.loc[mature, "label_available_at"]).max()),
            "label_counts": {str(i): int((labels == i).sum()) for i in range(3)},
            "feature_count": len(self.columns), "model_seed": self.seed}
        return self

    def target(self, row, q, cash, known_prices, peak):
        raw = self.pipeline.predict(pd.DataFrame([row[self.columns]], columns=self.columns))[0]
        target = ACTION_WEIGHTS[int(raw)].copy()
        if not self.constrained:
            if int(raw) != 2:
                reserve = getattr(self, "operational_fee_reserve", .001)
                target[:2] *= 1 - reserve
                target[2] = reserve
            return target
        soxl = min(target[0], self.maximum_soxl_weight)
        risky = min(target[0] + target[1], 1 - self.minimum_cash_weight)
        nav = float(np.asarray(q, float) @ np.asarray(known_prices, float) + cash)
        if nav / peak - 1 <= -self.stop_drawdown:
            soxl *= self.stop_multiplier
            risky *= self.stop_multiplier
        return np.array([soxl, risky - soxl, 1 - risky], dtype=float)

    def predict(self, dataset, segment="test"):
        data = dataset.prepare(segment, col_set="feature")
        return pd.Series(self.pipeline.predict(data[self.columns].astype(float)), index=data.index, name="action")
