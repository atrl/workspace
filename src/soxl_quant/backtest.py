from __future__ import annotations

"""Official qlib backtest integration for a trained SOXL actor.

This module is intentionally narrower than runtime: it exists to measure the
policy under qlib's simulator, not to become a production execution engine.
"""

from typing import Dict, List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qlib.backtest import backtest as qlib_backtest
from qlib.backtest.decision import Order, TradeDecisionWO
from qlib.backtest.executor import SimulatorExecutor
from qlib.data import D
from qlib.strategy.base import BaseStrategy
from tianshou.utils.net.discrete import Actor

from .data import FEATURE_COLUMNS
from .rl import PositionLedger, action_to_position, actor_action, make_observation, rebalance_threshold
from .settings import ACCOUNT, LOOKBACK, PositionCapConfig, RESULT_PNG, RESULT_TXT, TEST_SEG


class RLStrategy(BaseStrategy):
    def __init__(
        self,
        actor: Actor,
        feature_df: pd.DataFrame,
        price_series: pd.Series,
        position_cap_config: PositionCapConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.actor = actor
        self.df = feature_df
        self.price_series = price_series
        self.position_cap_config = position_cap_config
        self.last_step = -10
        self.trades = 0
        self.position_history: List[float] = []
        self.ledger = PositionLedger()
        self.peak_account_value = ACCOUNT
        self.price_path = self.price_series.reindex(self.df.index).ffill().bfill().to_numpy(dtype=np.float32)

    def generate_trade_decision(self, execute_result=None):
        """Convert the actor's target weight into qlib orders.

        The same action-to-position logic used by runtime is reused here so
        backtest, replay, and eventual live execution stay aligned.
        """
        step = self.trade_calendar.get_trade_step()
        if step >= self.trade_calendar.get_trade_len() - 2:
            return TradeDecisionWO([], self)
        trade_date = self.trade_calendar.get_step_time(step)[0]
        next_date = self.trade_calendar.get_step_time(step + 1)[0]
        if trade_date not in self.df.index:
            return TradeDecisionWO([], self)
        idx = self.df.index.get_loc(trade_date)
        if idx < LOOKBACK:
            return TradeDecisionWO([], self)

        position = self.trade_position
        cash = position.get_cash()
        shares = position.get_stock_amount("SOXL") if position.check_stock("SOXL") else 0
        price = float(self.price_series.loc[trade_date])
        account_value = cash + shares * price
        self.peak_account_value = max(self.peak_account_value, account_value)
        current_pos = (shares * price) / account_value if account_value > 0 else 0.0
        obs = make_observation(
            self.df,
            idx,
            current_pos,
            account_value,
            self.peak_account_value,
            self.ledger,
            self.price_path,
        )
        regime_risk = float(self.df.iloc[idx]["regime_risk"])
        realized_vol = float(self.df.iloc[idx]["realized_vol_10"])
        trend_strength = abs(float(self.df.iloc[idx]["trend_gap_21"]))
        min_rebalance = rebalance_threshold(regime_risk, realized_vol, trend_strength)

        target_pos = action_to_position(actor_action(self.actor, obs), obs, config=self.position_cap_config)
        if abs(target_pos - current_pos) < min_rebalance or step - self.last_step < 1:
            self.position_history.append(current_pos)
            return TradeDecisionWO([], self)

        target_shares = int(account_value * target_pos / price / 100) * 100
        delta = target_shares - int(shares)
        orders = []
        if delta >= 100:
            orders.append(Order("SOXL", delta, Order.BUY, trade_date, next_date))
        elif delta <= -100:
            orders.append(Order("SOXL", abs(delta), Order.SELL, trade_date, next_date))
        if orders:
            self.last_step = step
            self.trades += 1
            self.ledger.update(current_pos, target_pos, price)
        self.position_history.append(target_pos if orders else current_pos)
        return TradeDecisionWO(orders, self)


def backtest_profile(
    actor: Actor,
    feature_df: pd.DataFrame,
    price_series: pd.Series,
    position_cap_config: PositionCapConfig,
) -> Dict[str, object]:
    """Run the heavyweight qlib backtest used for reported headline metrics."""
    strategy = RLStrategy(
        actor=actor,
        feature_df=feature_df,
        price_series=price_series,
        position_cap_config=position_cap_config,
    )
    portfolio_metric, indicator_metric = qlib_backtest(
        start_time=TEST_SEG[0],
        end_time=TEST_SEG[1],
        strategy=strategy,
        executor=SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True),
        account=ACCOUNT,
        benchmark="SOXL",
        exchange_kwargs=dict(freq="day", deal_price="close", open_cost=0.001, close_cost=0.001, min_cost=5),
    )
    pdf = portfolio_metric.get("1day", portfolio_metric)
    if isinstance(pdf, tuple):
        pdf = pdf[0]
    account = pdf["account"].astype(float)
    daily_ret = account.pct_change().dropna()
    peak = account.cummax()
    return {
        "portfolio": pdf,
        "indicator": indicator_metric,
        "return": (account.iloc[-1] / ACCOUNT - 1) * 100,
        "sharpe": float(daily_ret.mean() / (daily_ret.std() + 1e-8) * np.sqrt(252)),
        "max_dd": float(((peak - account) / peak).max() * 100),
        "trades": strategy.trades,
        "position_history": strategy.position_history,
    }


def save_results(results: Dict[str, Dict[str, object]]) -> None:
    """Persist chart and text summaries for the latest backtest batch."""
    lines = ["SOXL qlib-native PPO backtest", "=" * 72, ""]
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    colors = {"balanced": "#1f77b4", "aggressive": "#d62728"}

    for name, result in results.items():
        pdf = result["portfolio"]
        color = colors.get(name, "#2ca02c")
        axes[0].plot(pdf.index, pdf["account"] / 1000, lw=2.2, color=color, label=name)
        pos_hist = result["position_history"]
        if pos_hist:
            axes[1].plot(pdf.index[: len(pos_hist)], pos_hist, lw=1.5, color=color, alpha=0.9, label=name)
        lines.extend(
            [
                f"[{name}]",
                f"return={result['return']:.2f}%",
                f"sharpe={result['sharpe']:.3f}",
                f"max_dd={result['max_dd']:.2f}%",
                f"trades={result['trades']}",
                "",
            ]
        )
        for slice_name, slice_metrics in result.get("slices", {}).items():
            lines.extend(
                [
                    f"{slice_name}: return={slice_metrics['return']:.2f}% "
                    f"sharpe={slice_metrics['sharpe']:.3f} max_dd={slice_metrics['max_dd']:.2f}%",
                ]
            )
        if result.get("slices"):
            lines.append("")

    axes[0].set_title("SOXL qlib-native PPO | Official qlib.backtest", fontweight="bold")
    axes[0].set_ylabel("Account ($K)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_ylabel("Position")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    vix = D.features(["SOXL"], ["$vix"], start_time=TEST_SEG[0], end_time=TEST_SEG[1]).xs("SOXL", level="instrument")["$vix"]
    axes[2].plot(vix.index, vix.values, color="black", lw=1.3, label="VIX")
    axes[2].axhline(25, color="orange", ls="--", alpha=0.5)
    axes[2].axhline(15, color="green", ls="--", alpha=0.5)
    axes[2].set_ylabel("VIX")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    for name, result in results.items():
        pdf = result["portfolio"]
        dd = pdf["account"] / pdf["account"].cummax() - 1.0
        axes[3].plot(pdf.index, dd.values, lw=1.6, label=name)
    axes[3].set_ylabel("Underwater")
    axes[3].grid(alpha=0.3)
    axes[3].legend()
    axes[3].xaxis.set_major_locator(mdates.YearLocator())
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(RESULT_PNG, dpi=250, bbox_inches="tight")
    RESULT_TXT.write_text("\n".join(lines), encoding="utf-8")
