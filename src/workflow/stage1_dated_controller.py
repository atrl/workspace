"""Bind every Stage1 research decision to a predecision action-model vintage."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class DatedModel:
    model: Any | None
    training_end: pd.Timestamp
    artifact_id: str
    explicit_no_model: bool = False

    def __post_init__(self):
        end = pd.Timestamp(self.training_end)
        if pd.isna(end) or end.tzinfo is not None:
            raise ValueError("blocked: dated model needs a naive training-end session")
        object.__setattr__(self, "training_end", end.normalize())
        if not self.artifact_id:
            raise ValueError("blocked: dated model artifact identity is missing")
        if self.model is None and not self.explicit_no_model:
            raise ValueError("blocked: absent model requires an explicit no-model receipt")
        if self.model is not None and self.explicit_no_model:
            raise ValueError("blocked: no-model receipt contains a fitted model")


class DatedStage1Controllers:
    """Map trade dates to signal-year vintages without a frozen-model fallback."""

    def __init__(self, *, calendar: pd.DataFrame,
                 open_models: Mapping[int, DatedModel],
                 recovery_models: Mapping[int, DatedModel]):
        required = {"signal_date", "trade_date"}
        if required - set(calendar) or calendar.empty:
            raise ValueError("blocked: dated controller calendar is incomplete")
        self.by_trade_date: dict[pd.Timestamp, dict[str, Any]] = {}
        years = pd.DatetimeIndex(calendar.signal_date).year
        if set(years) - set(open_models) or set(years) - set(recovery_models):
            raise ValueError("blocked: missing annual opening or recovery model")
        for row in calendar[["signal_date", "trade_date"]].itertuples(index=False):
            signal, trade = pd.Timestamp(row.signal_date), pd.Timestamp(row.trade_date)
            if signal.tzinfo is not None or trade.tzinfo is not None or signal >= trade:
                raise ValueError("blocked: invalid signal/trade clock")
            if trade in self.by_trade_date:
                raise ValueError("blocked: duplicate trade date in annual controller schedule")
            decision = trade + pd.Timedelta(hours=9, minutes=46)
            open_vintage = open_models[signal.year]
            recovery_vintage = recovery_models[signal.year]
            if open_vintage.model is None:
                raise ValueError("blocked: opening action model cannot be absent")
            for role, vintage in (("opening", open_vintage), ("recovery", recovery_vintage)):
                if vintage.training_end >= signal.normalize():
                    raise ValueError(f"blocked: {role} model trained on or after its signal year")
            self.by_trade_date[trade] = {
                "frozen_open_action_value_model": open_vintage.model,
                "frozen_recovery_model": recovery_vintage.model,
                "open_artifact_id": open_vintage.artifact_id,
                "recovery_artifact_id": recovery_vintage.artifact_id,
                "open_training_end": open_vintage.training_end,
                "recovery_training_end": recovery_vintage.training_end,
                "signal_year": signal.year,
                "decision_at": decision,
            }

    def __call__(self, trade_date: pd.Timestamp) -> dict[str, Any]:
        day = pd.Timestamp(trade_date).normalize()
        if day not in self.by_trade_date:
            raise ValueError("blocked: no point-in-time controller for trade date")
        item = self.by_trade_date[day]
        return {"frozen_open_action_value_model": item["frozen_open_action_value_model"],
                "frozen_recovery_model": item["frozen_recovery_model"]}

    def audit(self) -> pd.DataFrame:
        return pd.DataFrame([{"trade_date": day,
            "signal_year": item["signal_year"], "decision_at": item["decision_at"],
            "open_artifact_id": item["open_artifact_id"],
            "recovery_artifact_id": item["recovery_artifact_id"],
            "open_training_end": item["open_training_end"],
            "recovery_training_end": item["recovery_training_end"]}
            for day, item in sorted(self.by_trade_date.items())])
