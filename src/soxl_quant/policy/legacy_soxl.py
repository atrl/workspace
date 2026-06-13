from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..settings import ROOT


@dataclass(frozen=True)
class LegacySoxlConfig:
    use_longbridge: bool = False
    stock_code: str = "SOXL"
    start_date: str = "2020-01-01"
    end_date: str | None = None
    initial_capital: float = 100_000.0
    vix_low: float = 10.0
    vix_high: float = 30.0
    fear_min: float = -100.0
    fear_max: float = 100.0
    sigmoid_steepness: float = 0.5
    volatility_median: float = 1.0
    max_position: float = 0.9
    min_position: float = 0.5
    min_trade_position: float = 0.05
    buy_step_initial: float = 0.3
    buy_step_normal: float = 0.1
    sell_amount: float = 0.1
    position_update_days: int = 20
    low_position_threshold_multiplier: float = 1.0
    buy_score_threshold: float = -50.0
    max_panic_threshold: float = -300.0
    daily_return_threshold: float = -0.15
    cumulative_return_threshold: float = -0.30
    cumulative_days: int = 30
    daily_buy_cooldown: int = 3
    sell_score_threshold: float = 85.0
    sell_profit_target: float = 0.20
    last_buy_profit_target: float = 0.30
    stop_loss_threshold: float = -1.0
    stop_loss_position: float = 0.6
    take_profit_threshold: float = 0.60
    take_profit_position: float = 0.9

    @classmethod
    def from_dates(
        cls,
        *,
        start: str,
        end: str | None = None,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> "LegacySoxlConfig":
        return cls(start_date=start, end_date=end).with_overrides(config_overrides)

    def with_overrides(self, config_overrides: Mapping[str, Any] | None = None) -> "LegacySoxlConfig":
        if not config_overrides:
            return self
        allowed = {item.name for item in fields(self)}
        payload: dict[str, Any] = {}
        for raw_key, value in config_overrides.items():
            key = str(raw_key).lower()
            if key not in allowed:
                raise ValueError(f"unknown legacy SOXL config override: {raw_key!r}")
            payload[key] = value
        return replace(self, **payload)


def _first_existing(paths: list[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    checked = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"missing {label}: checked {checked}")


def _stock_data_path(config: LegacySoxlConfig, root: Path = ROOT) -> Path:
    return _first_existing(
        [
            root / "data" / "symbols" / f"{config.stock_code}.csv",
            root / "data" / "raw" / "symbols" / f"{config.stock_code}.csv",
        ],
        f"{config.stock_code} local stock data",
    )


def _vix_data_path(root: Path = ROOT) -> Path:
    return _first_existing(
        [
            root / "data" / "soxl_soxx_vix_full.csv",
            root / "data" / "processed" / "panels" / "soxl_soxx_vix_full.csv",
        ],
        "SOXL/SOXX/VIX panel",
    )


def build_legacy_soxl_source_frame(
    *,
    start: str,
    end: str | None = None,
    config_overrides: Mapping[str, Any] | None = None,
    root: Path = ROOT,
) -> pd.DataFrame:
    """Load local SOXL/VIX data with the same ordering as legacy YahooDataFetcher."""
    config = LegacySoxlConfig.from_dates(start=start, end=end, config_overrides=config_overrides)
    start_date = pd.to_datetime(config.start_date)
    actual_start = start_date - timedelta(days=30)
    end_date = pd.to_datetime(config.end_date) if config.end_date else pd.Timestamp.now()

    stock = pd.read_csv(_stock_data_path(config, root), parse_dates=["date"])
    stock = stock.rename(columns={"close": "price"})
    stock = stock[(stock["date"] >= actual_start) & (stock["date"] <= end_date)].copy()
    if "price" not in stock.columns:
        raise ValueError("SOXL source data must contain close or price")

    vix = pd.read_csv(_vix_data_path(root), parse_dates=["date"])
    if "vix" not in vix.columns:
        raise ValueError("SOXL/SOXX/VIX panel must contain vix")

    frame = pd.merge(stock[["date", "price"]], vix[["date", "vix"]], on="date", how="left")
    frame["vix"] = frame["vix"].ffill().bfill()
    return frame[frame["date"] >= start_date].reset_index(drop=True)


def _calculate_rsi(prices: pd.Series, period: int = 6) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).ewm(span=period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=period, adjust=False).mean()
    rs = gain / (loss + 1e-5)
    return 100 - (100 / (1 + rs))


def _calculate_historical_volatility(prices: pd.Series, window: int) -> pd.Series:
    returns = np.log(prices / prices.shift(1))
    return returns.rolling(window=window).std() * np.sqrt(252)


def _calculate_base_panic_score(vix: float, config: LegacySoxlConfig) -> float:
    vix_clipped = max(min(float(vix), config.vix_high), config.vix_low)
    vix_mid = (config.vix_high + config.vix_low) / 2.0
    x = -(vix_clipped - vix_mid) * config.sigmoid_steepness
    sigmoid_value = 1.0 / (1.0 + np.exp(-x))
    return float(sigmoid_value * (config.fear_max - config.fear_min) + config.fear_min)


def _calculate_final_panic_score(base_score: float, rsi6: float) -> float:
    if pd.isna(rsi6):
        return float(base_score)
    rsi6_value = float(rsi6)
    if rsi6_value <= 30:
        adjustment = 20 + (30 - rsi6_value) * 2.5
        return float(max(-150, base_score - adjustment))
    if rsi6_value >= 70:
        adjustment = 20 + (rsi6_value - 70)
        return float(min(150, base_score + adjustment))
    return float(base_score)


def _calculate_panic_score(frame: pd.DataFrame, config: LegacySoxlConfig) -> pd.Series:
    base_scores = frame["vix"].apply(lambda value: _calculate_base_panic_score(float(value), config))
    return pd.Series(
        (
            _calculate_final_panic_score(float(base), float(rsi6))
            for base, rsi6 in zip(base_scores, frame["RSI6"], strict=True)
        ),
        index=frame.index,
    )


def build_legacy_soxl_feature_frame(
    *,
    start: str,
    end: str | None = None,
    config_overrides: Mapping[str, Any] | None = None,
    root: Path = ROOT,
) -> pd.DataFrame:
    """Build the non-qlib soxl_backtest feature frame without importing it."""
    config = LegacySoxlConfig.from_dates(start=start, end=end, config_overrides=config_overrides)
    frame = build_legacy_soxl_source_frame(
        start=start,
        end=end,
        config_overrides=config_overrides,
        root=root,
    ).copy()
    frame["RSI6"] = _calculate_rsi(frame["price"], 6)
    frame["MA5"] = frame["price"].rolling(window=5).mean()
    frame["MA20"] = frame["price"].rolling(window=180).mean()
    frame["HV_20"] = _calculate_historical_volatility(frame["price"], window=20).fillna(0)
    frame["HV_30"] = _calculate_historical_volatility(frame["price"], window=30).fillna(0)
    frame["VOL_COEF"] = (frame["HV_20"] / config.volatility_median).clip(0.5, 2.0)
    frame["panic_score"] = _calculate_panic_score(frame, config)
    return frame


@dataclass
class LegacySoxlPolicyState:
    config: LegacySoxlConfig
    position: float = 0.0
    shares: int = 0
    capital: float = 0.0
    position_stack: list[dict[str, float]] = field(default_factory=list)
    avg_price: float = 0.0
    initial_position_built: bool = False

    def __post_init__(self) -> None:
        if self.capital == 0.0:
            self.capital = float(self.config.initial_capital)

    def update_avg_price(self) -> None:
        if not self.position_stack:
            self.avg_price = 0.0
            self.shares = 0
            return
        total_cost = sum(float(pos["price"]) * int(pos["shares"]) for pos in self.position_stack)
        total_shares = sum(int(pos["shares"]) for pos in self.position_stack)
        self.avg_price = total_cost / total_shares if total_shares > 0 else 0.0
        self.shares = int(total_shares)

    def add_position(self, price: float, shares: int) -> None:
        self.position_stack.append({"price": float(price), "shares": int(shares)})
        self.update_avg_price()

    def reduce_position(self, shares_to_sell: int) -> float:
        remaining_shares = int(shares_to_sell)
        sold_value = 0.0
        while remaining_shares > 0 and self.position_stack:
            pos = self.position_stack[-1]
            shares_sold = min(remaining_shares, int(pos["shares"]))
            sold_value += shares_sold * float(pos["price"])
            if shares_sold == int(pos["shares"]):
                self.position_stack.pop()
            else:
                pos["shares"] = int(pos["shares"]) - shares_sold
            remaining_shares -= shares_sold
        self.update_avg_price()
        return sold_value / shares_to_sell if shares_to_sell > 0 else 0.0

    def update_position(self, current_price: float) -> float:
        total_value = self.capital + self.shares * float(current_price)
        self.position = (self.shares * float(current_price)) / total_value if total_value > 0 else 0.0
        self.position = round(self.position, 4)
        return float(total_value)

    def get_last_position(self) -> dict[str, float] | None:
        return self.position_stack[-1] if self.position_stack else None


@dataclass(frozen=True)
class LegacySoxlReplayResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float | int | str]


class LegacySoxlRulePolicy:
    def __init__(self, config: LegacySoxlConfig, state: LegacySoxlPolicyState):
        self.config = config
        self.state = state
        self.last_buy_was_cumulative = False
        self.cumulative_buy_cooldown = 0
        self.last_buy_conditions: list[dict[str, float | int | str]] = []
        self.last_buy_date: pd.Timestamp | None = None
        self.no_buy_reason: str | None = None
        self.last_sell_conditions: list[dict[str, float | str]] = []
        self.no_sell_reason: str | None = None
        self.actual_cumulative_days = 0

    def calculate_max_cumulative_return(
        self,
        frame: pd.DataFrame,
        current_idx: int,
        last_buy_date: pd.Timestamp | None = None,
    ) -> float:
        max_cumulative_return = 0.0
        current_price = float(frame.iloc[current_idx]["price"])
        self.actual_cumulative_days = 0
        if current_idx >= 2:
            max_window = int(self.config.cumulative_days)
            if last_buy_date is not None:
                days_since_last_buy = (pd.Timestamp(frame.iloc[current_idx]["date"]) - last_buy_date).days
                max_window = min(max_window, days_since_last_buy)
            for days in range(2, max_window + 1):
                if current_idx >= days:
                    price_days_ago = float(frame.iloc[current_idx - days]["price"])
                    cumulative_return = (current_price - price_days_ago) / price_days_ago
                    if cumulative_return < max_cumulative_return:
                        max_cumulative_return = cumulative_return
                        self.actual_cumulative_days = days
        return float(max_cumulative_return)

    def should_initial_build_position(self, current_data: pd.Series, frame: pd.DataFrame, current_idx: int) -> bool:
        self.last_buy_conditions = []
        panic_score = float(current_data["panic_score"])
        if panic_score <= self.config.max_panic_threshold:
            self.no_buy_reason = (
                f"panic score {panic_score:.2f} below max threshold "
                f"{self.config.max_panic_threshold:.2f}"
            )
            return False

        base_conditions = (
            panic_score <= self.config.buy_score_threshold + 10
            and panic_score > self.config.max_panic_threshold
        )
        if not base_conditions:
            self.no_buy_reason = (
                f"panic score {panic_score:.2f} does not satisfy initial build threshold"
            )
            return False

        if self.cumulative_buy_cooldown > 0:
            self.no_buy_reason = f"in cumulative buy cooldown for {self.cumulative_buy_cooldown} more days"
            return False

        prev_price = float(frame.iloc[current_idx - 1]["price"])
        daily_return = (float(current_data["price"]) - prev_price) / prev_price
        max_cumulative_return = self.calculate_max_cumulative_return(frame, current_idx, self.last_buy_date)
        vol_coef = float(current_data["VOL_COEF"])

        if self.config.volatility_median != 0:
            daily_threshold = (
                self.config.daily_return_threshold
                * self.config.low_position_threshold_multiplier
                * vol_coef
            )
            cumulative_threshold = (
                self.config.cumulative_return_threshold
                * self.config.low_position_threshold_multiplier
                * vol_coef
            )
        else:
            daily_threshold = self.config.daily_return_threshold * self.config.low_position_threshold_multiplier
            cumulative_threshold = (
                self.config.cumulative_return_threshold * self.config.low_position_threshold_multiplier
            )

        daily_condition = daily_return <= daily_threshold
        cumulative_condition = max_cumulative_return <= cumulative_threshold
        if daily_condition:
            self.last_buy_conditions.append(
                {"type": "daily_drop", "value": float(daily_return), "threshold": float(daily_threshold)}
            )
        if cumulative_condition:
            self.last_buy_conditions.append(
                {
                    "type": "cumulative_drop",
                    "value": float(max_cumulative_return),
                    "threshold": float(cumulative_threshold),
                    "cooldown_days": int(self.actual_cumulative_days),
                }
            )

        if cumulative_condition and not daily_condition:
            self.last_buy_was_cumulative = True
            self.cumulative_buy_cooldown = int(self.actual_cumulative_days)
        else:
            self.last_buy_was_cumulative = False

        return bool(daily_condition or cumulative_condition)

    def should_regular_buy(self, current_data: pd.Series, frame: pd.DataFrame, current_idx: int) -> bool:
        self.last_buy_conditions = []
        if self.state.position >= self.config.max_position:
            self.no_buy_reason = "max position reached"
            return False

        panic_score = float(current_data["panic_score"])
        if panic_score <= self.config.max_panic_threshold:
            self.no_buy_reason = (
                f"panic score {panic_score:.2f} below max threshold "
                f"{self.config.max_panic_threshold:.2f}"
            )
            return False

        if self.cumulative_buy_cooldown > 0:
            self.no_buy_reason = f"in cumulative buy cooldown for {self.cumulative_buy_cooldown} more days"
            return False

        available_position = self.config.max_position - self.state.position
        if available_position < self.config.min_trade_position:
            self.no_buy_reason = "available position below min trade position"
            return False

        if panic_score <= self.config.buy_score_threshold:
            if current_idx > 0:
                prev_price = float(frame.iloc[current_idx - 1]["price"])
                daily_return = (float(current_data["price"]) - prev_price) / prev_price
            else:
                daily_return = 0.0
            max_cumulative_return = self.calculate_max_cumulative_return(frame, current_idx, self.last_buy_date)
            vol_coef = float(current_data["VOL_COEF"])
            self.last_buy_conditions.append(
                {
                    "type": "panic_score",
                    "value": float(panic_score),
                    "threshold": float(self.config.buy_score_threshold),
                }
            )
            if self.config.volatility_median != 0:
                daily_threshold = self.config.daily_return_threshold * vol_coef
                cumulative_threshold = self.config.cumulative_return_threshold * vol_coef
            else:
                daily_threshold = self.config.daily_return_threshold
                cumulative_threshold = self.config.cumulative_return_threshold

            if daily_return <= daily_threshold:
                self.last_buy_conditions.append(
                    {"type": "daily_drop", "value": float(daily_return), "threshold": float(daily_threshold)}
                )
                self.last_buy_was_cumulative = False
                self.cumulative_buy_cooldown = 1
                self.last_buy_date = pd.Timestamp(frame.iloc[current_idx]["date"])
                return True

            if max_cumulative_return <= cumulative_threshold:
                self.last_buy_conditions.append(
                    {
                        "type": "cumulative_drop",
                        "value": float(max_cumulative_return),
                        "threshold": float(cumulative_threshold),
                        "cooldown_days": int(self.actual_cumulative_days),
                    }
                )
                self.last_buy_was_cumulative = True
                self.cumulative_buy_cooldown = int(self.actual_cumulative_days)
                self.last_buy_date = pd.Timestamp(frame.iloc[current_idx]["date"])
                return True

            self.no_buy_reason = "panic score satisfied but drop triggers did not"
        else:
            self.no_buy_reason = "panic score does not satisfy buy threshold"
        return False

    def should_buy(self, current_data: pd.Series, frame: pd.DataFrame, current_idx: int) -> bool:
        self.no_buy_reason = None
        if not self.state.initial_position_built:
            return self.should_initial_build_position(current_data, frame, current_idx)
        return self.should_regular_buy(current_data, frame, current_idx)

    def should_sell(self, current_data: pd.Series) -> str | bool:
        self.last_sell_conditions = []
        self.no_sell_reason = None
        panic_score = float(current_data["panic_score"])
        profit = (
            (float(current_data["price"]) - self.state.avg_price) / self.state.avg_price
            if self.state.avg_price > 0
            else 0.0
        )
        vol_coef = float(current_data["VOL_COEF"])
        max_sell_position = self.state.position - self.config.min_position
        if max_sell_position < self.config.min_trade_position:
            self.no_sell_reason = "sellable position below min trade position"
            return False

        if profit <= self.config.stop_loss_threshold:
            self.last_sell_conditions.append(
                {
                    "type": "stop_loss",
                    "value": float(profit),
                    "threshold": float(self.config.stop_loss_threshold),
                }
            )
            return "stop_loss"

        last_buy = self.state.get_last_position()
        last_buy_profit = (
            (float(current_data["price"]) - float(last_buy["price"])) / float(last_buy["price"])
            if last_buy
            else 0.0
        )

        take_profit_condition = (
            panic_score >= self.config.sell_score_threshold
            and profit * self.state.position >= self.config.take_profit_threshold
            and self.state.position >= self.config.min_position
        )
        if take_profit_condition:
            if self.state.position * self.config.take_profit_position >= self.config.min_trade_position:
                self.last_sell_conditions.extend(
                    [
                        {
                            "type": "panic_score",
                            "value": float(panic_score),
                            "threshold": float(self.config.sell_score_threshold),
                        },
                        {
                            "type": "take_profit",
                            "value": float(profit * self.state.position),
                            "threshold": float(self.config.take_profit_threshold),
                        },
                    ]
                )
                return "take_profit"
            self.no_sell_reason = "take profit position below min trade position"
        elif panic_score < self.config.sell_score_threshold:
            self.no_sell_reason = "panic score below sell threshold"
        elif profit * self.state.position < self.config.take_profit_threshold:
            self.no_sell_reason = "portfolio profit below take profit threshold"

        normal_sell_condition_1 = (
            profit * self.state.position >= self.config.sell_profit_target
            and panic_score >= self.config.sell_score_threshold
            and profit > 0
        )
        last_buy_threshold = (
            self.config.last_buy_profit_target * vol_coef
            if self.config.volatility_median != 0
            else self.config.last_buy_profit_target
        )
        normal_sell_condition_2 = (
            panic_score >= self.config.sell_score_threshold and last_buy_profit >= last_buy_threshold
        )

        if normal_sell_condition_1 or normal_sell_condition_2:
            if self.config.sell_amount >= self.config.min_trade_position:
                if normal_sell_condition_1:
                    self.last_sell_conditions.extend(
                        [
                            {
                                "type": "profit_target",
                                "value": float(profit * self.state.position),
                                "threshold": float(self.config.sell_profit_target),
                            },
                            {
                                "type": "panic_score",
                                "value": float(panic_score),
                                "threshold": float(self.config.sell_score_threshold),
                            },
                        ]
                    )
                if normal_sell_condition_2:
                    self.last_sell_conditions.extend(
                        [
                            {
                                "type": "last_buy_profit",
                                "value": float(last_buy_profit),
                                "threshold": float(last_buy_threshold),
                            },
                            {
                                "type": "panic_score",
                                "value": float(panic_score),
                                "threshold": float(self.config.sell_score_threshold),
                            },
                        ]
                    )
                return "normal_sell"
            self.no_sell_reason = "sell amount below min trade position"
        return False

    def execute_buy(self, current_data: pd.Series, buy_step: float, market_condition: str) -> dict[str, object] | None:
        target_position = min(self.state.position + float(buy_step), self.config.max_position)
        total_value = self.state.capital + self.state.shares * float(current_data["price"])
        buy_amount = total_value * (target_position - self.state.position)
        new_shares = int(buy_amount / float(current_data["price"]))
        if new_shares <= 0:
            return None
        cost = new_shares * float(current_data["price"])
        self.state.add_position(float(current_data["price"]), new_shares)
        self.state.capital -= cost
        new_total_value = self.state.update_position(float(current_data["price"]))
        return {
            "date": pd.Timestamp(current_data["date"]).strftime("%Y-%m-%d"),
            "type": "buy",
            "price": float(current_data["price"]),
            "position": float(self.state.position),
            "score": float(current_data["panic_score"]),
            "total_value": float(new_total_value),
            "shares": int(self.state.shares),
            "avg_price": float(self.state.avg_price),
            "buy_type": "cumulative" if self.last_buy_was_cumulative else "daily",
            "market_condition": market_condition,
            "buy_step": float(buy_step),
            "trigger_conditions": list(self.last_buy_conditions),
        }

    def execute_sell(self, current_data: pd.Series, sell_type: str) -> dict[str, object] | None:
        price = float(current_data["price"])
        total_value = self.state.capital + self.state.shares * price
        if sell_type == "take_profit":
            sell_amount = self.state.position * self.config.take_profit_position
            sell_shares = int((total_value * sell_amount) / price)
            sell_type_str = "take_profit"
        elif sell_type == "stop_loss":
            sell_amount = self.state.position * self.config.stop_loss_position
            sell_shares = int((total_value * sell_amount) / price)
            sell_type_str = "stop_loss"
        else:
            last_buy = self.state.get_last_position()
            last_buy_profit = (price - float(last_buy["price"])) / float(last_buy["price"]) if last_buy else 0.0
            if (
                float(current_data["panic_score"]) >= self.config.sell_score_threshold
                and last_buy_profit >= self.config.last_buy_profit_target
            ):
                last_buy_shares = int(last_buy["shares"]) if last_buy else 0
                remaining_shares = self.state.shares - last_buy_shares
                remaining_value = remaining_shares * price
                new_total_value = self.state.capital + remaining_value + last_buy_shares * price
                new_position = remaining_value / new_total_value if new_total_value > 0 else 0.0
                if new_position >= self.config.min_position:
                    sell_shares = last_buy_shares
                else:
                    target_value = new_total_value * self.config.min_position
                    keep_shares = int(target_value / price)
                    sell_shares = self.state.shares - keep_shares
            else:
                sell_amount = min(self.config.sell_amount, self.state.position - self.config.min_position)
                sell_shares = int((total_value * sell_amount) / price)
            sell_type_str = "sell"

        if sell_shares <= 0:
            return None
        self.state.reduce_position(sell_shares)
        self.state.capital += sell_shares * price
        new_total_value = self.state.update_position(price)
        gain = ((price - self.state.avg_price) / self.state.avg_price) * 100 if self.state.avg_price > 0 else 0.0
        return {
            "date": pd.Timestamp(current_data["date"]).strftime("%Y-%m-%d"),
            "type": sell_type_str,
            "price": price,
            "position": float(self.state.position),
            "score": float(current_data["panic_score"]),
            "gain": float(gain),
            "total_value": float(new_total_value),
            "shares": int(self.state.shares),
            "avg_price": float(self.state.avg_price),
            "trigger_conditions": list(self.last_sell_conditions),
        }


def _max_drawdown_pct(values: list[float]) -> float:
    if not values:
        return 0.0
    account = np.asarray(values, dtype=float)
    peak = np.maximum.accumulate(account)
    return float(np.max((peak - account) / peak) * 100.0)


def _buyhold_return_pct(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    return float((float(frame.iloc[-1]["price"]) / float(frame.iloc[0]["price"]) - 1.0) * 100.0)


def _trade_frame(trades: list[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "date",
        "type",
        "price",
        "position",
        "score",
        "total_value",
        "shares",
        "avg_price",
        "buy_type",
        "market_condition",
        "buy_step",
        "trigger_conditions",
        "gain",
    ]
    return pd.DataFrame(trades, columns=columns)


def replay_legacy_soxl_policy(
    *,
    start: str,
    end: str | None = None,
    config_overrides: Mapping[str, Any] | None = None,
    root: Path = ROOT,
) -> LegacySoxlReplayResult:
    config = LegacySoxlConfig.from_dates(start=start, end=end, config_overrides=config_overrides)
    frame = build_legacy_soxl_feature_frame(
        start=start,
        end=end,
        config_overrides=config_overrides,
        root=root,
    ).reset_index(drop=True)
    state = LegacySoxlPolicyState(config)
    policy = LegacySoxlRulePolicy(config, state)
    trades: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    capital_history: list[float] = []
    position_history: list[float] = []
    days_since_update = 0

    for idx in range(len(frame)):
        current_row = frame.iloc[idx]
        if policy.cumulative_buy_cooldown > 0:
            policy.cumulative_buy_cooldown -= 1

        days_since_update += 1
        if days_since_update >= config.position_update_days:
            if state.shares > 0:
                state.update_position(float(current_row["price"]))
                if not state.initial_position_built and state.position >= config.min_position:
                    state.initial_position_built = True
            days_since_update = 0

        trade_info: dict[str, object] | None = None
        if policy.should_buy(current_row, frame, idx):
            buy_step = config.buy_step_initial if not state.initial_position_built else config.buy_step_normal
            is_overvalued = bool(float(current_row["price"]) > float(current_row["MA20"]))
            if is_overvalued:
                trade_info = policy.execute_buy(current_row, buy_step * 0.4, "overvalued")
            else:
                trade_info = policy.execute_buy(current_row, buy_step, "undervalued")
        else:
            sell_type = policy.should_sell(current_row)
            if sell_type:
                trade_info = policy.execute_sell(current_row, str(sell_type))

        if trade_info:
            trades.append(trade_info)

        total_value = state.capital + state.shares * float(current_row["price"])
        position_history.append(float(state.position))
        capital_history.append(float(total_value))
        daily_rows.append(
            {
                "date": pd.Timestamp(current_row["date"]).strftime("%Y-%m-%d"),
                "price": float(current_row["price"]),
                "strategy_nav": float(total_value),
                "position": float(state.position),
                "panic_score": float(current_row["panic_score"]),
                "HV_20": float(current_row["HV_20"]),
                "HV_30": float(current_row["HV_30"]),
                "VOL_COEF": float(current_row["VOL_COEF"]),
            }
        )

        if not policy.should_buy(current_row, frame, idx) and not policy.should_sell(current_row):
            pass

    daily = pd.DataFrame(daily_rows)
    if not daily.empty:
        initial_price = float(daily.iloc[0]["price"])
        daily["buyhold_nav"] = config.initial_capital * daily["price"] / initial_price
        daily["excess"] = daily["strategy_nav"] - daily["buyhold_nav"]
        daily["excess_pct"] = daily["excess"] / daily["buyhold_nav"] * 100.0
        action_by_date = {str(trade["date"]): str(trade["type"]).upper() for trade in trades}
        daily["action"] = daily["date"].map(action_by_date).fillna("HOLD")

    trade_df = _trade_frame(trades)
    metrics: dict[str, float | int | str] = {
        "requested_start": str(start),
        "start": str(daily.iloc[0]["date"]) if not daily.empty else "",
        "end": str(daily.iloc[-1]["date"]) if not daily.empty else "",
        "rows": int(len(daily)),
        "initial_capital": float(config.initial_capital),
        "final_capital": float(capital_history[-1]) if capital_history else float(config.initial_capital),
        "soxl_buyhold_return_pct": _buyhold_return_pct(frame),
        "rulebase_return_pct": (
            (float(capital_history[-1]) / config.initial_capital - 1.0) * 100.0
            if capital_history
            else 0.0
        ),
        "rulebase_max_drawdown_pct": _max_drawdown_pct(capital_history),
        "average_position_pct": float(np.mean(position_history) * 100.0) if position_history else 0.0,
        "end_position_pct": float(position_history[-1] * 100.0) if position_history else 0.0,
        "trade_count": int(len(trades)),
    }
    return LegacySoxlReplayResult(daily=daily, trades=trade_df, metrics=metrics)


__all__ = [
    "LegacySoxlConfig",
    "LegacySoxlPolicyState",
    "LegacySoxlReplayResult",
    "LegacySoxlRulePolicy",
    "build_legacy_soxl_feature_frame",
    "build_legacy_soxl_source_frame",
    "replay_legacy_soxl_policy",
]
