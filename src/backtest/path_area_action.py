"""Bind frozen path information to each cost account's own opening features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.stage1_domain_heads_raw import replay_stage1_raw
from src.data.path_area_bridge import AREA_COLUMN, CONTROL_COLUMNS
from src.model.path_area_action import build_arm_features, feature_order, original_action_features


class SessionPathModel:
    """A fresh sequential adapter per account; never carries another account's state.

    The legacy opening controller calls predict_proba exactly once per ordered
    trade day, before applying its mandatory guards. The runtime feature row
    comes from that controller's current physical account. Only the preclosed
    prefix columns are supplied by this adapter.
    """

    def __init__(self, pipeline, prefix: pd.DataFrame, *, arm: str, on_prediction_attempt=None):
        dates = pd.DatetimeIndex(prefix.trade_date)
        if (dates.tz is not None or dates.has_duplicates or dates.hasnans
                or not dates.is_monotonic_increasing or not dates.equals(dates.normalize())):
            raise ValueError("blocked: prefix runtime dates differ")
        required = (*CONTROL_COLUMNS, "prefix_mask", AREA_COLUMN)
        if not set(required).issubset(prefix):
            raise ValueError("blocked: runtime prefix fields missing")
        values = prefix.loc[:, [*CONTROL_COLUMNS, AREA_COLUMN]].to_numpy(float)
        mask = prefix.prefix_mask
        if (mask.isna().any() or not mask.isin([0, 1]).all()
                or not np.isfinite(values[mask.eq(1)]).all()
                or not np.isnan(values[mask.eq(0)]).all()):
            raise ValueError("blocked: runtime prefix mask/value contract differs")
        if arm not in ("B", "C"):
            raise ValueError("blocked: unknown action arm")
        self.pipeline, self.prefix, self.arm = pipeline, prefix.copy(), arm
        self.on_prediction_attempt = on_prediction_attempt
        self.calls = 0
        self.rows = []

    def predict_proba(self, current: pd.DataFrame):
        if (len(current) != 1 or tuple(current.columns) != original_action_features()
                or self.calls >= len(self.prefix)):
            raise ValueError("blocked: opening prediction call grid or feature order differs")
        context = self.prefix.iloc[self.calls]
        self.calls += 1
        values = current.iloc[0].to_dict()
        values.update({name: context[name] for name in (*CONTROL_COLUMNS, "prefix_mask", AREA_COLUMN)})
        features = build_arm_features(pd.DataFrame([values]).loc[:, list(feature_order())], self.arm)
        if self.on_prediction_attempt is not None:
            self.on_prediction_attempt()
        probability = np.asarray(self.pipeline.predict_proba(features), dtype=float)
        if (probability.shape != (1, 2) or not np.isfinite(probability).all()
                or (probability < 0).any() or (probability > 1).any()
                or not np.isclose(probability.sum(), 1.0)):
            raise ValueError("blocked: invalid opening action probabilities")
        self.rows.append({"trade_date": context.trade_date,
                          "decision_at": pd.Timestamp(context.trade_date).tz_localize("America/New_York")
                          + pd.Timedelta(hours=9, minutes=46),
                          **features.iloc[0].to_dict(),
                          "execute_probability": float(probability[0, 1])})
        return probability


def replay_path_area_action(frame, *, prefix, pipeline, arm, point_prices, schedule,
                            plan, controller, recovery_spec, action_spec, selection,
                            slippage_bps, on_prediction_attempt=None):
    """Reuse the original guards/recovery/account with a scoped 32-feature adapter."""
    dates = pd.DatetimeIndex(frame.trade_date)
    if not dates.equals(pd.DatetimeIndex(prefix.trade_date)):
        raise ValueError("blocked: prefix and account days differ")
    if not pd.DatetimeIndex(frame.signal_date).equals(pd.DatetimeIndex(prefix.index)):
        raise ValueError("blocked: prefix and account signal rows differ")
    if float(action_spec.execute_probability_threshold) != 0.5:
        raise ValueError("blocked: frozen action threshold differs")
    adapter = SessionPathModel(pipeline, prefix, arm=arm,
                               on_prediction_attempt=on_prediction_attempt)
    scoped_controller = {**controller, "frozen_open_action_value_model": adapter}
    path = replay_stage1_raw(frame, point_prices=point_prices, schedule=schedule,
                            plan=plan, controller=scoped_controller, recovery_spec=recovery_spec,
                            action_spec=action_spec, selection=selection, slippage_bps=slippage_bps)
    if adapter.calls != len(frame) or len(adapter.rows) != len(frame):
        raise ValueError("blocked: action prediction coverage differs")
    features = pd.DataFrame(adapter.rows)
    trace = path["trace"].copy()
    if not dates.equals(pd.DatetimeIndex(trace.trade_date)):
        raise ValueError("blocked: action trace date grid differs")
    for column in features:
        if column != "trade_date":
            trace[column] = features[column].to_numpy(copy=True)
    points = path["points"].set_index("stamp")
    records = []
    for row in frame.itertuples():
        day = pd.Timestamp(row.trade_date)
        stamp = day + pd.Timedelta(hours=9, minutes=47)
        after = points.loc[stamp]
        quotes = point_prices.loc[stamp, ["SOXL", "SOXX"]].to_numpy(float)
        weights = np.r_[after[["q_soxl", "q_soxx"]].to_numpy(float) * quotes,
                        after.cash + after.receivable] / after.nav
        orders = path["orders"]
        day_orders = orders.loc[orders.trade_date.eq(day)] if len(orders) else orders
        records.append({
            **{f"source_target_{asset}_weight": float(getattr(row, asset + "_weight"))
               for asset in ("soxl", "soxx", "cash")},
            **{f"postfill_{asset}_weight": float(weight)
               for asset, weight in zip(("soxl", "soxx", "cash"), weights)},
            "filled_order_count": len(day_orders),
            "total_order_fees_usd": float(day_orders.total.sum()) if len(day_orders) else 0.0,
        })
    additions = pd.DataFrame(records)
    for column in additions:
        trace[column] = additions[column].to_numpy(copy=True)
    for asset in ("soxl", "soxx", "cash"):
        trace[f"projected_target_{asset}_weight"] = trace[f"target_{asset}_weight"]
    path.update(trace=trace, action_features=features,
                action_prediction_calls=adapter.calls, arm=arm)
    return path
