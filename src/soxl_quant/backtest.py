from __future__ import annotations

"""基于 qlib 官方回测引擎的策略评估模块。

本模块故意保持精简：它的唯一目的是在 qlib 的仿真器下测量策略表现，
而不是变成一个生产级执行引擎。生产执行逻辑在 runtime.py 中。

【与训练模拟器的区别】
  - 训练模拟器（SoxlSimulator）是轻量数学模拟，假设 0.1% 固定成本
  - qlib backtest 更接近真实交易：支持开盘价/收盘价成交、最小佣金、
    涨跌停处理、账户资金实时结算等
  - 两者的 action→position 转换逻辑完全一致，保证训练与评估对齐

【优化提示】
  - 当前 benchmark 设为目标 ETF 本身，可直观对比策略 vs Buy&Hold。
  - 若未来做模型集成（Ensemble），可在此新增 EnsembleRLStrategy。
"""

from typing import Dict, List, Tuple

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
from .policy.environment import TrancheLedger
from .policy.execution import (
    execution_buy_cooldown_days,
    execution_regime,
    execution_trade_lock_days,
    rebalance_threshold,
)
from .policy.modes import ModeState, mode_band, route_mode
from .policy.observation import make_observation
from .policy.policy_builder import actor_action
from .policy.positioning import target_position_for_profile
from .settings import ACCOUNT, LOOKBACK, ExecutionConfig, PositionBudgetConfig, PositionCapConfig, RESULT_PNG, RESULT_TXT, TEST_SEG


class RLStrategy(BaseStrategy):
    """qlib 策略包装器：把训练好的 actor 接入官方回测引擎。"""
    def __init__(
        self,
        actor: Actor,
        feature_df: pd.DataFrame,
        price_series: pd.Series,
        position_cap_config: PositionCapConfig,
        position_budget_config: PositionBudgetConfig | None = None,
        execution_config: ExecutionConfig | None = None,
        mode_router_config=None,
        symbol: str = "SOXL",
        asset_idx: int = 0,
        feature_columns=None,
        **kwargs,
    ):
        self.model_config = kwargs.pop("model_config", None)
        self.overlay_config = kwargs.pop("overlay_config", None)
        super().__init__(**kwargs)
        self.actor = actor          # 训练好的 PPO actor
        self.df = feature_df        # 测试集特征矩阵
        self.price_series = price_series  # 对齐的价格序列
        self.position_cap_config = position_cap_config
        self.position_budget_config = position_budget_config
        self.execution_config = execution_config or ExecutionConfig()
        self.mode_router_config = mode_router_config
        self.symbol = symbol        # 当前交易的 ETF 代码
        self.asset_idx = asset_idx  # 资产 one-hot 索引
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)
        self.last_step = -10        # 冷却期计数，避免连续过度交易
        self.buy_cooldown = 0
        self.trade_lock = 0
        self.trades = 0             # 总成交次数统计
        self.position_history: List[float] = []  # 每日仓位记录，用于画图
        self.trade_signals: List[Tuple[pd.Timestamp, float, float, str]] = []  # (date, price, position, type)
        self.ledger = TrancheLedger()           # 持仓账本，与训练时一致
        self.peak_account_value = ACCOUNT
        self.price_path = self.price_series.reindex(self.df.index).ffill().bfill().to_numpy(dtype=np.float32)
        self.mode_state = ModeState()

    def generate_trade_decision(self, execute_result=None):
        """把 actor 的目标仓位转换为 qlib 的买卖订单。

        这里的 action→position→orders 链路与 runtime/live 执行完全一致，
        确保 backtest、replay、实盘三端对齐。
        
        交易规则：
          1. 用 make_observation 构造 48 维观测向量
          2. actor 输出离散 action（0~10）
          3. action_to_position 映射为连续仓位 [floor, cap]
          4. 若仓位变化 > min_rebalance 且不在冷却期，则生成目标股数订单
          5. 目标股数按 100 股取整（模拟最小交易单位）
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
        shares = position.get_stock_amount(self.symbol) if position.check_stock(self.symbol) else 0
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
            asset_idx=self.asset_idx,
            feature_columns=self.feature_columns,
        )
        mode_router_cfg = self.mode_router_config
        if mode_router_cfg is not None:
            self.mode_state = route_mode(obs, self.mode_state, mode_router_cfg)
        regime_risk = float(self.df.iloc[idx]["regime_risk"])
        realized_vol = float(self.df.iloc[idx]["realized_vol_10"])
        trend_strength = abs(float(self.df.iloc[idx]["trend_gap_21"]))
        cum_drop_30 = float(self.df.iloc[idx]["cum_drop_30"])
        relative_strength = float(self.df.iloc[idx]["relative_strength_21"])
        vix_zscore = float(self.df.iloc[idx]["vix_zscore"])
        downside_ratio_10 = float(self.df.iloc[idx]["downside_ratio_10"])
        regime = execution_regime(
            regime_risk,
            realized_vol,
            trend_strength,
            cum_drop_30=cum_drop_30,
            relative_strength=relative_strength,
            vix_zscore=vix_zscore,
            downside_ratio_10=downside_ratio_10,
            config=self.execution_config,
        )
        model_cfg = self.model_config
        raw_action = actor_action(self.actor, obs)
        target_pos = target_position_for_profile(
            raw_action,
            obs,
            config=self.position_cap_config,
            action_levels=np.asarray(getattr(model_cfg, "action_levels", ()), dtype=np.float32) if model_cfg else None,
            hard_floor=getattr(model_cfg, "hard_floor", None) if model_cfg else None,
            use_hybrid_overlay=bool(getattr(model_cfg, "use_hybrid_overlay", False)) if model_cfg else False,
            delta_levels=np.asarray(getattr(model_cfg, "delta_levels", ()), dtype=np.float32) if model_cfg and getattr(model_cfg, "use_hybrid_overlay", False) else None,
            mode_band=mode_band(self.mode_state.name, mode_router_cfg) if mode_router_cfg and getattr(mode_router_cfg, "enabled", False) else None,
            position_budget_config=self.position_budget_config,
            current_drawdown=max(1.0 - account_value / max(self.peak_account_value, 1e-8), 0.0),
            position_mapping=str(getattr(model_cfg, "position_mapping", "default")) if model_cfg else "default",
        )
        # Apply transition signal overlay if configured
        if self.overlay_config is not None:
            oc = self.overlay_config
            overlay_type = oc.get("type")
            threshold = oc.get("threshold", 0.3)
            boost = oc.get("boost", 0.1)
            regime_thresh = oc.get("regime_threshold", 0.6)
            regime_mode = oc.get("regime_mode", "stress")  # "stress", "bull", or "dual"
            regime_active = False
            if regime_mode == "stress":
                regime_active = regime_risk >= regime_thresh
            elif regime_mode == "bull":
                regime_active = regime_risk <= regime_thresh
            elif regime_mode == "dual":
                regime_active = True  # handle in type-specific logic
            if regime_active:
                signal_val = 0.0
                if overlay_type == "vix_divergence":
                    signal_val = float(self.df.iloc[idx].get("TRANS_VIX_DIVERGENCE", 0.0))
                    if signal_val >= threshold:
                        target_pos = min(target_pos + boost, 1.0)
                elif overlay_type == "spring_compression":
                    signal_val = float(self.df.iloc[idx].get("TRANS_SPRING_COMPRESSION", 0.0))
                    if signal_val >= threshold:
                        target_pos = min(target_pos + boost, 1.0)
                elif overlay_type == "panic_exhaustion":
                    signal_val = float(self.df.iloc[idx].get("TRANS_PANIC_EXHAUSTION", 0.0))
                    if signal_val >= threshold:
                        target_pos = min(target_pos + boost, 1.0)
                elif overlay_type == "vix_zscore_delta":
                    # VIX zscore falling rapidly in stress regime = potential relief
                    signal_val = float(self.df.iloc[idx].get("VIX_ZSCORE_DELTA_5", 0.0))
                    if signal_val <= -threshold:
                        target_pos = min(target_pos + boost, 1.0)
                elif overlay_type == "regime_risk_delta":
                    # Regime risk falling rapidly in stress regime = potential relief
                    signal_val = float(self.df.iloc[idx].get("REGIME_RISK_DELTA_5", 0.0))
                    if signal_val <= -threshold:
                        target_pos = min(target_pos + boost, 1.0)
                elif overlay_type == "combined_vix":
                    # Both VIX divergence AND VIX zscore delta must agree
                    div_val = float(self.df.iloc[idx].get("TRANS_VIX_DIVERGENCE", 0.0))
                    delta_val = float(self.df.iloc[idx].get("VIX_ZSCORE_DELTA_5", 0.0))
                    if div_val >= threshold and delta_val <= -threshold:
                        target_pos = min(target_pos + boost, 1.0)
                elif overlay_type == "fake_breakout":
                    # Negative overlay: reduce position when price bounces too far above MA21 during deep drawdown
                    # IC = -0.1335: high values predict poor forward returns
                    fb_val = float(self.df.iloc[idx].get("CLOSE_VS_MA21_AFTER_DD", 0.0))
                    if fb_val >= threshold:
                        target_pos = max(target_pos - boost, 0.0)
                elif overlay_type == "vol_stress":
                    # Vol of vol spike in stress regime = high conviction entry
                    signal_val = float(self.df.iloc[idx].get("vol_of_vol_21", 0.0))
                    if signal_val >= threshold:
                        target_pos = min(target_pos + boost, 1.0)
                elif overlay_type == "downside_bull":
                    # High short-term downside in bull regime = mean-reversion entry
                    signal_val = float(self.df.iloc[idx].get("downside_ratio_10", 0.0))
                    if signal_val >= threshold:
                        target_pos = min(target_pos + boost, 1.0)
                elif overlay_type == "dual_regime":
                    # Stress: vol_stress, Bull: downside_bull
                    if regime_risk >= regime_thresh:
                        signal_val = float(self.df.iloc[idx].get("vol_of_vol_21", 0.0))
                        if signal_val >= threshold:
                            target_pos = min(target_pos + boost, 1.0)
                    elif regime_risk <= oc.get("bull_regime_threshold", 0.4):
                        signal_val = float(self.df.iloc[idx].get("downside_ratio_10", 0.0))
                        if signal_val >= threshold:
                            target_pos = min(target_pos + boost, 1.0)
                elif overlay_type == "vol_vix_combo":
                    # Both vol_of_vol AND VIX divergence must agree in stress regime
                    vol_val = float(self.df.iloc[idx].get("vol_of_vol_21", 0.0))
                    vix_val = float(self.df.iloc[idx].get("TRANS_VIX_DIVERGENCE", 0.0))
                    vol_thresh = oc.get("vol_threshold", threshold)
                    vix_thresh = oc.get("vix_threshold", 0.05)
                    if vol_val >= vol_thresh and vix_val >= vix_thresh:
                        target_pos = min(target_pos + boost, 1.0)
                elif overlay_type == "rv63_stress":
                    # Realized vol spike in stress regime = high conviction entry
                    signal_val = float(self.df.iloc[idx].get("realized_vol_63", 0.0))
                    if signal_val >= threshold:
                        target_pos = min(target_pos + boost, 1.0)
        direction = "buy" if target_pos > current_pos else ("sell" if target_pos < current_pos else "neutral")
        min_rebalance = rebalance_threshold(
            regime_risk,
            realized_vol,
            trend_strength,
            cum_drop_30=cum_drop_30,
            relative_strength=relative_strength,
            vix_zscore=vix_zscore,
            downside_ratio_10=downside_ratio_10,
            direction=direction,
            mode=self.mode_state.name if mode_router_cfg and getattr(mode_router_cfg, "enabled", False) else None,
            config=self.execution_config,
            mode_router_config=mode_router_cfg,
        )
        if self.trade_lock > 0:
            if abs(target_pos - current_pos) >= min_rebalance:
                target_pos = current_pos
            self.trade_lock -= 1
        if self.buy_cooldown > 0:
            if target_pos > current_pos:
                target_pos = current_pos
            self.buy_cooldown -= 1
        if abs(target_pos - current_pos) < min_rebalance or step - self.last_step < 1:
            self.position_history.append(current_pos)
            return TradeDecisionWO([], self)

        # 目标股数按手（100股）取整，模拟真实交易单位
        target_shares = int(account_value * target_pos / price / 100) * 100
        delta = target_shares - int(shares)
        orders = []
        if delta >= 100:
            orders.append(Order(self.symbol, delta, Order.BUY, trade_date, next_date))
        elif delta <= -100:
            orders.append(Order(self.symbol, abs(delta), Order.SELL, trade_date, next_date))
        if orders:
            self.last_step = step
            self.trades += 1
            self.ledger.update(current_pos, target_pos, price)
            turnover = abs(target_pos - current_pos)
            if target_pos > current_pos + 1e-6:
                self.buy_cooldown = execution_buy_cooldown_days(cum_drop_30, self.execution_config, regime)
            if turnover > self.execution_config.min_trade_turnover:
                self.trade_lock = execution_trade_lock_days(self.execution_config, regime)
            signal_type = "buy" if delta > 0 else "sell"
            self.trade_signals.append((trade_date, price, target_pos if orders else current_pos, signal_type))
        self.position_history.append(target_pos if orders else current_pos)
        return TradeDecisionWO(orders, self)


def backtest_profile(
    actor: Actor,
    feature_df: pd.DataFrame,
    price_series: pd.Series,
    position_cap_config: PositionCapConfig,
    position_budget_config: PositionBudgetConfig | None = None,
    execution_config: ExecutionConfig | None = None,
    mode_router_config=None,
    symbol: str = "SOXL",
    asset_idx: int = 0,
    feature_columns=None,
    test_seg: Tuple[str, str] = TEST_SEG,
    open_cost: float = 0.001,
    close_cost: float = 0.001,
    min_cost: float = 5.0,
    model_config=None,
    overlay_config=None,
) -> Dict[str, object]:
    """运行重量级 qlib 官方回测，产出 headline 指标（return / sharpe / max_dd）。

    参数 test_seg 默认 ("2020-01-01", "2026-03-31")，这是仓库统一使用的
    测试窗口，所有模型对比都应基于该窗口，避免 cherry-picking。
    
    成本假设默认：open_cost=0.001, close_cost=0.001, min_cost=5
    这意味着单边交易成本 0.1%，且有最低 $5 佣金。
    """
    strategy = RLStrategy(
        actor=actor,
        feature_df=feature_df,
        price_series=price_series,
        position_cap_config=position_cap_config,
        position_budget_config=position_budget_config,
        execution_config=execution_config,
        mode_router_config=mode_router_config,
        symbol=symbol,
        asset_idx=asset_idx,
        feature_columns=feature_columns,
        model_config=model_config,
        overlay_config=overlay_config,
    )
    portfolio_metric, indicator_metric = qlib_backtest(
        start_time=test_seg[0],
        end_time=test_seg[1],
        strategy=strategy,
        executor=SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True),
        account=ACCOUNT,
        benchmark=symbol,
        exchange_kwargs=dict(
            freq="day",
            deal_price="close",
            open_cost=float(open_cost),
            close_cost=float(close_cost),
            min_cost=float(min_cost),
        ),
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
        "trade_signals": strategy.trade_signals,
    }


def save_results(results: Dict[str, Dict[str, object]]) -> None:
    """Persist chart and text summaries for the latest backtest batch."""
    if not results:
        return
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
        signals = result.get("trade_signals", [])
        for date, price, pos, stype in signals:
            marker = "^" if stype == "buy" else "v"
            mcolor = "green" if stype == "buy" else "red"
            axes[1].scatter(date, pos, marker=marker, color=mcolor, s=45, zorder=5, alpha=0.8)
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

    # VIX 数据统一存储在 SOXL 的 qlib binary 中，无论结果 key 是什么都从这里读取。
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
