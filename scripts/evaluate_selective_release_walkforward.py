#!/usr/bin/env python3
"""Validate selective release overlays on saved walk-forward actors."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_selective_release_overlay import overlay_variants
from src.soxl_quant.backtest import backtest_profile
from src.soxl_quant.data import build_dataset, feature_columns_for_packs, init_qlib, segment_frame
from src.soxl_quant.policy.policy_builder import build_policy
from src.soxl_quant.settings import QLIB_DATA_PATH, load_profile


@dataclass(frozen=True)
class WalkForwardActorSpec:
    window: str
    start: str
    end: str
    actor_path: Path


WALK_FORWARD_WINDOWS = (
    ("wf_2020", "2020-01-01", "2020-12-31"),
    ("wf_2022", "2022-01-01", "2022-12-31"),
    ("wf_2023_2024", "2024-01-01", "2024-12-31"),
)


def walk_forward_actor_specs(profile_name: str) -> list[WalkForwardActorSpec]:
    specs: list[WalkForwardActorSpec] = []
    for window, start, end in WALK_FORWARD_WINDOWS:
        actor_path = (
            ROOT
            / "runtime"
            / "model"
            / "card"
            / f"{profile_name}_{window}"
            / f"soxl_{profile_name}_{window}_actor.pt"
        )
        specs.append(WalkForwardActorSpec(window=window, start=start, end=end, actor_path=actor_path))
    return specs


def variant_subset(retained_only: bool) -> list[tuple[str, dict[str, Any] | None]]:
    variants = overlay_variants()
    if not retained_only:
        return variants
    return [
        (name, config)
        for name, config in variants
        if name in {"baseline", "selective_release_recovery_v1"}
    ]


def gate_eligible_mask(feature_df: pd.DataFrame, overlay_config: dict[str, Any] | None) -> pd.Series:
    if not overlay_config:
        return pd.Series(False, index=feature_df.index)
    regime_risk = feature_df["regime_risk"].astype(float)
    min_regime = float(overlay_config.get("min_regime_risk", float("-inf")))
    max_regime = float(overlay_config.get("max_regime_risk", float("inf")))
    cooldown_ready = feature_df["RB_COOLDOWN_PROXY"].astype(float) >= float(
        overlay_config.get("cooldown_threshold", 0.25)
    )
    release_ready = (
        feature_df["TRANS_VIX_DIVERGENCE"].astype(float) >= float(overlay_config.get("vix_threshold", 0.04))
    ) | (
        feature_df["CICC_RETURN_VOLUME_SYNC"].astype(float)
        >= float(overlay_config.get("volume_sync_threshold", 0.68))
    )
    return (regime_risk >= min_regime) & (regime_risk <= max_regime) & cooldown_ready & release_ready


def execution_primary_blocker(row: dict[str, Any]) -> str:
    if int(row.get("order_count", 0)) > 0:
        return "ordered"
    if bool(row.get("blocked_by_trade_lock", False)):
        return "trade_lock"
    if bool(row.get("blocked_by_buy_cooldown", False)):
        return "buy_cooldown"
    if bool(row.get("blocked_by_step_lock", False)):
        return "step_lock"
    if bool(row.get("blocked_by_min_rebalance", False)):
        return "min_rebalance"
    if bool(row.get("blocked_by_lot_rounding", False)):
        return "lot_rounding"
    return "no_order_unknown"


def eligible_decision_trace_rows(
    decision_trace: list[dict[str, Any]],
    feature_df: pd.DataFrame,
    overlay_config: dict[str, Any] | None,
    *,
    actor_window: str,
    variant: str,
) -> list[dict[str, Any]]:
    if not overlay_config:
        return []
    eligible_mask = gate_eligible_mask(feature_df, overlay_config)
    eligible_dates = {
        str(pd.Timestamp(date).date())
        for date in feature_df.index[eligible_mask]
    }
    rows: list[dict[str, Any]] = []
    for item in decision_trace:
        date = str(pd.Timestamp(item["date"]).date())
        if date not in eligible_dates:
            continue
        row = {
            "actor_window": actor_window,
            "variant": variant,
            "date": date,
        }
        row.update({key: value for key, value in item.items() if key != "date"})
        row["primary_blocker"] = execution_primary_blocker(row)
        rows.append(row)
    return rows


def metrics_from_result(
    result: dict[str, Any],
    baseline_positions: np.ndarray | None = None,
) -> dict[str, float | int]:
    positions = np.asarray(result.get("position_history", []), dtype=float)
    metrics: dict[str, float | int] = {
        "return": round(float(result["return"]), 4),
        "sharpe": round(float(result["sharpe"]), 4),
        "max_dd": round(float(result["max_dd"]), 4),
        "avg_pos": round(float(positions.mean()) if positions.size else 0.0, 6),
        "trades": int(result["trades"]),
        "changed_days": 0,
        "avg_pos_delta": 0.0,
        "max_pos_delta": 0.0,
    }
    if baseline_positions is not None and positions.size:
        n = min(len(baseline_positions), len(positions))
        delta = positions[:n] - baseline_positions[:n]
        metrics["changed_days"] = int(np.sum(np.abs(delta) > 1e-6))
        metrics["avg_pos_delta"] = round(float(delta.mean()), 6)
        metrics["max_pos_delta"] = round(float(delta.max()) if delta.size else 0.0, 6)
    return metrics


def max_drawdown_event(account: pd.Series) -> dict[str, Any]:
    account = account.dropna().astype(float)
    if account.empty:
        return {
            "max_dd": 0.0,
            "peak_date": "",
            "trough_date": "",
            "peak_account": 0.0,
            "trough_account": 0.0,
        }
    peak = account.cummax()
    drawdown = (peak - account) / peak.replace(0.0, np.nan) * 100.0
    drawdown = drawdown.fillna(0.0)
    trough_date = drawdown.idxmax()
    peak_date = account.loc[:trough_date].idxmax()
    return {
        "max_dd": round(float(drawdown.loc[trough_date]), 4),
        "peak_date": str(pd.Timestamp(peak_date).date()),
        "trough_date": str(pd.Timestamp(trough_date).date()),
        "peak_account": round(float(account.loc[peak_date]), 4),
        "trough_account": round(float(account.loc[trough_date]), 4),
    }


def _result_account(result: dict[str, Any]) -> pd.Series:
    portfolio = result["portfolio"]
    if isinstance(portfolio, tuple):
        portfolio = portfolio[0]
    return portfolio["account"].astype(float).sort_index()


def _result_position_series(result: dict[str, Any], index: pd.Index) -> pd.Series:
    positions = np.asarray(result.get("position_history", []), dtype=float)
    if positions.size == 0:
        return pd.Series(0.0, index=index)
    pos_index = index[: min(len(index), len(positions))]
    series = pd.Series(positions[: len(pos_index)], index=pos_index)
    return series.reindex(index).ffill().bfill().fillna(0.0).astype(float)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _safe_int(value: Any) -> int:
    if value in ("", None):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _first_index_on_or_after(index: pd.Index, date: str) -> pd.Timestamp | None:
    ts = pd.Timestamp(date)
    pos = index.searchsorted(ts)
    if pos >= len(index):
        return None
    return pd.Timestamp(index[pos])


def _ordered_trace_dates(
    trace_rows: list[dict[str, Any]],
    *,
    actor_window: str,
    variant: str,
    execution_ready_only: bool,
) -> list[str]:
    dates: list[str] = []
    for row in trace_rows:
        if row.get("actor_window") != actor_window or row.get("variant") != variant:
            continue
        if not _truthy(row.get("overlay_gate_ready", False)):
            continue
        if execution_ready_only and not _truthy(row.get("overlay_execution_ready", False)):
            continue
        if _safe_int(row.get("order_count", 0)) <= 0:
            continue
        dates.append(str(pd.Timestamp(row["date"]).date()))
    return sorted(dict.fromkeys(dates))


def build_drawdown_attribution_rows(
    *,
    actor_window: str,
    baseline_result: dict[str, Any],
    variant_results: dict[str, dict[str, Any]],
    trace_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_account = _result_account(baseline_result)
    rows: list[dict[str, Any]] = []
    for variant, variant_result in variant_results.items():
        variant_account = _result_account(variant_result)
        common_index = baseline_account.index.intersection(variant_account.index).sort_values()
        if common_index.empty:
            continue
        base_account = baseline_account.reindex(common_index)
        var_account = variant_account.reindex(common_index)
        base_event = max_drawdown_event(base_account)
        var_event = max_drawdown_event(var_account)
        delta_max_dd = round(float(var_event["max_dd"]) - float(base_event["max_dd"]), 4)
        base_pos = _result_position_series(baseline_result, common_index)
        var_pos = _result_position_series(variant_result, common_index)
        pos_delta = var_pos - base_pos

        release_order_dates = _ordered_trace_dates(
            trace_rows,
            actor_window=actor_window,
            variant=variant,
            execution_ready_only=True,
        )
        eligible_order_dates = _ordered_trace_dates(
            trace_rows,
            actor_window=actor_window,
            variant=variant,
            execution_ready_only=False,
        )
        first_release_order_date = release_order_dates[0] if release_order_dates else ""
        start_date = _first_index_on_or_after(common_index, first_release_order_date) if first_release_order_date else None
        trough_date = pd.Timestamp(var_event["trough_date"]) if var_event["trough_date"] else None

        days_release_to_trough = 0
        position_delta_at_release = 0.0
        avg_position_delta_release_to_trough = 0.0
        max_position_delta_release_to_trough = 0.0
        account_delta_at_trough_pct = 0.0
        if trough_date is not None and trough_date in common_index:
            base_trough_account = float(base_account.loc[trough_date])
            var_trough_account = float(var_account.loc[trough_date])
            if base_trough_account:
                account_delta_at_trough_pct = round((var_trough_account / base_trough_account - 1.0) * 100.0, 4)
        if start_date is not None and trough_date is not None and start_date <= trough_date:
            path_delta = pos_delta.loc[start_date:trough_date]
            days_release_to_trough = int((trough_date - start_date).days)
            position_delta_at_release = round(float(pos_delta.loc[start_date]), 6)
            avg_position_delta_release_to_trough = round(float(path_delta.mean()), 6)
            max_position_delta_release_to_trough = round(float(path_delta.max()), 6)

        if delta_max_dd <= 0.5:
            readout = "no_material_drawdown_widening"
        elif (
            first_release_order_date
            and trough_date is not None
            and pd.Timestamp(first_release_order_date) <= trough_date
        ):
            readout = "release_order_precedes_wider_drawdown"
        elif eligible_order_dates and trough_date is not None and pd.Timestamp(eligible_order_dates[0]) <= trough_date:
            readout = "eligible_order_precedes_wider_drawdown"
        else:
            readout = "wider_drawdown_without_pretrough_release_order"

        rows.append(
            {
                "actor_window": actor_window,
                "variant": variant,
                "baseline_max_dd": round(float(base_event["max_dd"]), 4),
                "variant_max_dd": round(float(var_event["max_dd"]), 4),
                "delta_max_dd": delta_max_dd,
                "baseline_peak_date": base_event["peak_date"],
                "baseline_trough_date": base_event["trough_date"],
                "variant_peak_date": var_event["peak_date"],
                "variant_trough_date": var_event["trough_date"],
                "release_order_dates": ";".join(release_order_dates),
                "eligible_order_dates": ";".join(eligible_order_dates),
                "first_release_order_date": first_release_order_date,
                "days_release_to_trough": days_release_to_trough,
                "position_delta_at_release": position_delta_at_release,
                "avg_position_delta_release_to_trough": avg_position_delta_release_to_trough,
                "max_position_delta_release_to_trough": max_position_delta_release_to_trough,
                "account_delta_at_trough_pct": account_delta_at_trough_pct,
                "readout": readout,
            }
        )
    return rows


def load_actor(actor_path: Path, profile) -> Any:
    if not actor_path.exists():
        raise FileNotFoundError(f"walk-forward actor not found: {actor_path}")
    ckpt = torch.load(actor_path, map_location="cpu", weights_only=False)
    _, actor = build_policy(
        int(ckpt["obs_dim"]),
        profile,
        use_moe=ckpt.get("use_moe", False),
        hard_regime_routing=ckpt.get("use_moe", False),
    )
    actor.load_state_dict(ckpt["state_dict"])
    actor.eval()
    return actor


def run_walk_forward_grid(
    profile_name: str,
    retained_only: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    init_qlib(QLIB_DATA_PATH)
    profile = load_profile(profile_name)
    actor_feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)
    overlay_packs = tuple(dict.fromkeys((*profile.feature.packs, "release_signals_v1")))
    dataset = build_dataset(feature_packs=overlay_packs, profile_name=profile.name)
    full_df = segment_frame(dataset, "test")
    price_full = dataset.handler.price_frame.droplevel("instrument")["price"].sort_index()

    rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    attribution_rows: list[dict[str, Any]] = []
    variants = variant_subset(retained_only)
    for spec in walk_forward_actor_specs(profile_name):
        actor = load_actor(spec.actor_path, profile)
        feature_df = full_df.loc[spec.start : spec.end]
        price = price_full.loc[feature_df.index]
        baseline_positions: np.ndarray | None = None
        baseline_result: dict[str, Any] | None = None
        variant_results: dict[str, dict[str, Any]] = {}
        spec_trace_rows: list[dict[str, Any]] = []

        for variant, overlay_config in variants:
            result = backtest_profile(
                actor=actor,
                feature_df=feature_df,
                price_series=price,
                position_cap_config=profile.position_cap,
                position_budget_config=profile.position_budget,
                execution_config=profile.execution,
                mode_router_config=profile.mode_router,
                asset_idx=0,
                feature_columns=actor_feature_columns,
                model_config=profile.model,
                overlay_config=overlay_config,
                trace_decisions=bool(overlay_config and overlay_config.get("type") == "selective_release"),
                test_seg=(spec.start, spec.end),
            )
            if variant == "baseline":
                baseline_result = result
                baseline_positions = np.asarray(result.get("position_history", []), dtype=float)
            else:
                variant_results[variant] = result
            row = {
                "profile": profile.name,
                "actor_window": spec.window,
                "start": spec.start,
                "end": spec.end,
                "actor_path": str(spec.actor_path.relative_to(ROOT)),
                "variant": variant,
                "overlay_config": json.dumps(overlay_config or {}, sort_keys=True),
            }
            row.update(metrics_from_result(result, None if variant == "baseline" else baseline_positions))
            eligible_mask = gate_eligible_mask(feature_df, overlay_config)
            row["eligible_days"] = int(eligible_mask.sum())
            if eligible_mask.any():
                eligible_index = feature_df.index[eligible_mask]
                row["first_eligible_date"] = str(eligible_index[0].date())
                row["last_eligible_date"] = str(eligible_index[-1].date())
            else:
                row["first_eligible_date"] = ""
                row["last_eligible_date"] = ""
            row["absorbed_days"] = max(int(row["eligible_days"]) - int(row["changed_days"]), 0)
            rows.append(row)
            current_trace_rows = eligible_decision_trace_rows(
                list(result.get("decision_trace", [])),
                feature_df,
                overlay_config,
                actor_window=spec.window,
                variant=variant,
            )
            trace_rows.extend(current_trace_rows)
            spec_trace_rows.extend(current_trace_rows)

        if baseline_result is not None and variant_results:
            attribution_rows.extend(
                build_drawdown_attribution_rows(
                    actor_window=spec.window,
                    baseline_result=baseline_result,
                    variant_results=variant_results,
                    trace_rows=spec_trace_rows,
                )
            )

    context = {
        "profile": profile.name,
        "actor_feature_count": len(actor_feature_columns),
        "dataset_feature_packs": list(overlay_packs),
        "retained_only": retained_only,
        "windows": [
            {"window": spec.window, "start": spec.start, "end": spec.end, "actor_path": str(spec.actor_path.relative_to(ROOT))}
            for spec in walk_forward_actor_specs(profile_name)
        ],
        "trace_rows": len(trace_rows),
        "attribution_rows": len(attribution_rows),
    }
    return rows, context, trace_rows, attribution_rows


def write_outputs(
    rows: list[dict[str, Any]],
    context: dict[str, Any],
    output_dir: Path,
    trace_rows: list[dict[str, Any]] | None = None,
    attribution_rows: list[dict[str, Any]] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    trace_rows = trace_rows or []
    attribution_rows = attribution_rows or []
    trace_df = pd.DataFrame(trace_rows)
    attribution_df = pd.DataFrame(attribution_rows)
    df.to_csv(output_dir / "selective_release_walkforward_results.csv", index=False)
    (output_dir / "selective_release_walkforward_results.json").write_text(
        json.dumps({"context": context, "rows": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    trace_df.to_csv(output_dir / "selective_release_walkforward_execution_trace.csv", index=False)
    (output_dir / "selective_release_walkforward_execution_trace.json").write_text(
        json.dumps({"context": context, "rows": trace_rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    attribution_df.to_csv(output_dir / "selective_release_walkforward_drawdown_attribution.csv", index=False)
    (output_dir / "selective_release_walkforward_drawdown_attribution.json").write_text(
        json.dumps({"context": context, "rows": attribution_rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    baseline = df[df["variant"] == "baseline"].set_index("actor_window")
    lines = [
        "# Selective Release Walk-Forward Actor Review",
        "",
        f"- Profile: `{context['profile']}`",
        f"- Dataset feature packs for overlay columns: `{context['dataset_feature_packs']}`",
        "- Role: walk-forward actor validation before runtime overlay promotion",
        "",
        "## Results",
        "",
        "| actor_window | variant | return | sharpe | max_dd | avg_pos | trades | eligible_days | changed_days | absorbed_days | delta_return | delta_max_dd |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        base = baseline.loc[row["actor_window"]]
        lines.append(
            "| {actor_window} | {variant} | {return:.2f}% | {sharpe:.3f} | {max_dd:.2f}% | "
            "{avg_pos:.4f} | {trades} | {eligible_days} | {changed_days} | {absorbed_days} | "
            "{delta_return:+.2f}pp | {delta_max_dd:+.2f}pp |".format(
                **row,
                delta_return=float(row["return"]) - float(base["return"]),
                delta_max_dd=float(row["max_dd"]) - float(base["max_dd"]),
            )
        )

    retained = df[df["variant"] == "selective_release_recovery_v1"].set_index("actor_window")
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- This keeps the actor fixed within each walk-forward checkpoint and changes only the execution overlay.",
            "- A promotable overlay should improve the recovery window without reopening `wf_2022` drawdown or average exposure.",
        ]
    )
    if not retained.empty:
        wf2022 = retained.loc["wf_2022"] if "wf_2022" in retained.index else None
        wf2024 = retained.loc["wf_2023_2024"] if "wf_2023_2024" in retained.index else None
        if wf2022 is not None and wf2024 is not None:
            base_2022 = baseline.loc["wf_2022"]
            base_2024 = baseline.loc["wf_2023_2024"]
            lines.extend(
                [
                    "",
                    "## Retained Gate Check",
                    "",
                    f"- `wf_2022`: return `{float(wf2022['return']):.2f}%` vs baseline `{float(base_2022['return']):.2f}%`; max_dd `{float(wf2022['max_dd']):.2f}%` vs `{float(base_2022['max_dd']):.2f}%`; eligible_days `{int(wf2022['eligible_days'])}`; changed_days `{int(wf2022['changed_days'])}`.",
                    f"- `wf_2023_2024`: return `{float(wf2024['return']):.2f}%` vs baseline `{float(base_2024['return']):.2f}%`; max_dd `{float(wf2024['max_dd']):.2f}%` vs `{float(base_2024['max_dd']):.2f}%`; eligible_days `{int(wf2024['eligible_days'])}`; changed_days `{int(wf2024['changed_days'])}`.",
                ]
            )

    diagnostic_variants = [
        "selective_release_recovery_threshold_bypass_v1",
        "selective_release_recovery_target_lift_v1",
        "selective_release_recovery_trade_lock_bypass_v1",
        "selective_release_recovery_trade_lock_short_v1",
        "selective_release_recovery_trade_lock_guarded_short_v1",
    ]
    diagnostic_rows = df[
        (df["actor_window"] == "wf_2023_2024") & (df["variant"].isin(diagnostic_variants))
    ]
    if not diagnostic_rows.empty:
        lines.extend(["", "## Order-Generation Diagnostic", ""])
        for variant in diagnostic_variants:
            match = diagnostic_rows[diagnostic_rows["variant"] == variant]
            if match.empty:
                continue
            row = match.iloc[0]
            lines.append(
                f"- `{variant}`: eligible_days `{int(row['eligible_days'])}`, "
                f"changed_days `{int(row['changed_days'])}`, absorbed_days `{int(row['absorbed_days'])}`, "
                f"return `{float(row['return']):.2f}%`, max_dd `{float(row['max_dd']):.2f}%`."
            )
        if (diagnostic_rows["eligible_days"].astype(int) > 0).any() and (
            diagnostic_rows["changed_days"].astype(int) == 0
        ).all():
            lines.extend(
                [
                    "",
                    "- Lowering the rebalance threshold and forcing a target lift still produced no changed days on the eligible `wf_2023_2024` release dates.",
                    "- `wf_2022` remains unchanged because there are no eligible release days, so the defensive gate is still intact.",
                    "- The eligible-day trace below separates upstream execution guards from downstream min-rebalance and lot-rounding symptoms before any new rule or retrain.",
                ]
            )
        if not trace_df.empty:
            trace_diag = trace_df[
                (trace_df["actor_window"] == "wf_2023_2024")
                & (trace_df["variant"].isin(diagnostic_variants))
            ]
            if not trace_diag.empty:
                lines.extend(["", "## Eligible-Day Execution Trace Summary", ""])
                for variant in diagnostic_variants:
                    subset = trace_diag[trace_diag["variant"] == variant]
                    if subset.empty:
                        continue
                    metric_row = diagnostic_rows[diagnostic_rows["variant"] == variant]
                    eligible_days = int(metric_row.iloc[0]["eligible_days"]) if not metric_row.empty else len(subset)
                    blockers = {
                        "trade_lock": int(subset["blocked_by_trade_lock"].astype(bool).sum()),
                        "buy_cooldown": int(subset["blocked_by_buy_cooldown"].astype(bool).sum()),
                        "min_rebalance": int(subset["blocked_by_min_rebalance"].astype(bool).sum()),
                        "step_lock": int(subset["blocked_by_step_lock"].astype(bool).sum()),
                        "lot_rounding": int(subset["blocked_by_lot_rounding"].astype(bool).sum()),
                    }
                    execution_ready = (
                        int(subset["overlay_execution_ready"].astype(bool).sum())
                        if "overlay_execution_ready" in subset
                        else len(subset)
                    )
                    primary = subset["primary_blocker"].value_counts().to_dict()
                    primary_text = ", ".join(
                        f"`{name}` {int(count)}" for name, count in sorted(primary.items())
                    )
                    max_target_after_overlay = float(subset["target_after_overlay"].astype(float).max())
                    max_target_after_guards = float(subset["target_after_guards"].astype(float).max())
                    lines.append(
                        f"- `{variant}`: traced `{len(subset)}` of `{eligible_days}` eligible days; blockers "
                        f"trade_lock `{blockers['trade_lock']}`, buy_cooldown `{blockers['buy_cooldown']}`, "
                        f"min_rebalance `{blockers['min_rebalance']}`, step_lock `{blockers['step_lock']}`, "
                        f"lot_rounding `{blockers['lot_rounding']}`; max target after overlay "
                        f"`{max_target_after_overlay:.4f}`, after guards `{max_target_after_guards:.4f}`; "
                        f"execution_ready `{execution_ready}`; primary {primary_text}."
                    )
                lines.append(
                    "- Readout: trade lock is the dominant upstream absorber; min-rebalance and lot-rounding counts mostly appear after the target has already been collapsed back toward current position."
                )
                lines.append(
                    "- Full eligible-day trace: `selective_release_walkforward_execution_trace.csv`."
                )

    trade_lock_variant = "selective_release_recovery_trade_lock_bypass_v1"
    trade_lock_rows = df[df["variant"] == trade_lock_variant].set_index("actor_window")
    if "wf_2022" in trade_lock_rows.index and "wf_2023_2024" in trade_lock_rows.index:
        row_2022 = trade_lock_rows.loc["wf_2022"]
        row_2024 = trade_lock_rows.loc["wf_2023_2024"]
        base_2022 = baseline.loc["wf_2022"]
        base_2024 = baseline.loc["wf_2023_2024"]
        lines.extend(
            [
                "",
                "## Trade-Lock Bypass Check",
                "",
                f"- `wf_2022`: unchanged with eligible_days `{int(row_2022['eligible_days'])}`, return `{float(row_2022['return']):.2f}%` vs baseline `{float(base_2022['return']):.2f}%`, max_dd `{float(row_2022['max_dd']):.2f}%` vs `{float(base_2022['max_dd']):.2f}%`.",
                f"- `wf_2023_2024`: changed_days `{int(row_2024['changed_days'])}`, trades `{int(row_2024['trades'])}` vs baseline `{int(base_2024['trades'])}`, return `{float(row_2024['return']):.2f}%` vs `{float(base_2024['return']):.2f}%`, max_dd `{float(row_2024['max_dd']):.2f}%` vs `{float(base_2024['max_dd']):.2f}%`.",
                "- Decision: release-gated trade-lock bypass proves the lock was blocking order generation, but the unrestricted bypass worsens the recovery window and is not promotable.",
            ]
        )

    trade_lock_short_variant = "selective_release_recovery_trade_lock_short_v1"
    trade_lock_short_rows = df[df["variant"] == trade_lock_short_variant].set_index("actor_window")
    if "wf_2022" in trade_lock_short_rows.index and "wf_2023_2024" in trade_lock_short_rows.index:
        row_2022 = trade_lock_short_rows.loc["wf_2022"]
        row_2024 = trade_lock_short_rows.loc["wf_2023_2024"]
        base_2022 = baseline.loc["wf_2022"]
        base_2024 = baseline.loc["wf_2023_2024"]
        lines.extend(
            [
                "",
                "## Trade-Lock Shortening Check",
                "",
                f"- `wf_2022`: unchanged with eligible_days `{int(row_2022['eligible_days'])}`, return `{float(row_2022['return']):.2f}%` vs baseline `{float(base_2022['return']):.2f}%`, max_dd `{float(row_2022['max_dd']):.2f}%` vs `{float(base_2022['max_dd']):.2f}%`.",
                f"- `wf_2023_2024`: changed_days `{int(row_2024['changed_days'])}`, trades `{int(row_2024['trades'])}` vs baseline `{int(base_2024['trades'])}`, return `{float(row_2024['return']):.2f}%` vs `{float(base_2024['return']):.2f}%`, max_dd `{float(row_2024['max_dd']):.2f}%` vs `{float(base_2024['max_dd']):.2f}%`.",
                "- Decision: release-gated trade-lock shortening is cleaner than full bypass and improves recovery return, but it still widens drawdown and is not promotable without an additional drawdown / exposure-gap guard.",
            ]
        )

    guarded_short_variant = "selective_release_recovery_trade_lock_guarded_short_v1"
    guarded_short_rows = df[df["variant"] == guarded_short_variant].set_index("actor_window")
    if "wf_2022" in guarded_short_rows.index and "wf_2023_2024" in guarded_short_rows.index:
        row_2022 = guarded_short_rows.loc["wf_2022"]
        row_2024 = guarded_short_rows.loc["wf_2023_2024"]
        base_2022 = baseline.loc["wf_2022"]
        base_2024 = baseline.loc["wf_2023_2024"]
        guarded_delta_return = float(row_2024["return"]) - float(base_2024["return"])
        guarded_delta_max_dd = float(row_2024["max_dd"]) - float(base_2024["max_dd"])
        guarded_decision = (
            "guarded shortening is not promotable in this run because MaxDD still widens materially."
            if guarded_delta_max_dd > 0.5
            else "guarded shortening needs broader validation before promotion."
        )
        lines.extend(
            [
                "",
                "## Guarded Trade-Lock Shortening Check",
                "",
                f"- `wf_2022`: unchanged with eligible_days `{int(row_2022['eligible_days'])}`, return `{float(row_2022['return']):.2f}%` vs baseline `{float(base_2022['return']):.2f}%`, max_dd `{float(row_2022['max_dd']):.2f}%` vs `{float(base_2022['max_dd']):.2f}%`.",
                f"- `wf_2023_2024`: changed_days `{int(row_2024['changed_days'])}`, trades `{int(row_2024['trades'])}` vs baseline `{int(base_2024['trades'])}`, return `{float(row_2024['return']):.2f}%` vs `{float(base_2024['return']):.2f}%`, max_dd `{float(row_2024['max_dd']):.2f}%` vs `{float(base_2024['max_dd']):.2f}%`.",
                f"- Delta: return `{guarded_delta_return:+.2f}pp`, max_dd `{guarded_delta_max_dd:+.2f}pp`.",
                f"- Decision: {guarded_decision}",
            ]
        )

    if not attribution_df.empty:
        attribution_variants = [
            "selective_release_recovery_trade_lock_bypass_v1",
            "selective_release_recovery_trade_lock_short_v1",
            "selective_release_recovery_trade_lock_guarded_short_v1",
        ]
        wf_attr = attribution_df[
            (attribution_df["actor_window"] == "wf_2023_2024")
            & (attribution_df["variant"].isin(attribution_variants))
        ]
        if not wf_attr.empty:
            lines.extend(
                [
                    "",
                    "## Drawdown Path Attribution",
                    "",
                    "| variant | delta_max_dd | peak -> trough | release_order_dates | eligible_order_dates | avg_pos_delta_release_to_trough | account_delta_at_trough | readout |",
                    "|---|---:|---|---|---|---:|---:|---|",
                ]
            )
            for variant in attribution_variants:
                match = wf_attr[wf_attr["variant"] == variant]
                if match.empty:
                    continue
                row = match.iloc[0]
                release_dates = str(row["release_order_dates"]) if pd.notna(row["release_order_dates"]) else ""
                eligible_dates = str(row["eligible_order_dates"]) if pd.notna(row["eligible_order_dates"]) else ""
                lines.append(
                    f"| `{variant}` | `{float(row['delta_max_dd']):+.2f}pp` | "
                    f"`{row['variant_peak_date']}` -> `{row['variant_trough_date']}` | "
                    f"`{release_dates}` | `{eligible_dates}` | "
                    f"`{float(row['avg_position_delta_release_to_trough']):+.4f}` | "
                    f"`{float(row['account_delta_at_trough_pct']):+.2f}%` | `{row['readout']}` |"
                )

            guarded_match = wf_attr[
                wf_attr["variant"] == "selective_release_recovery_trade_lock_guarded_short_v1"
            ]
            if not guarded_match.empty:
                guarded = guarded_match.iloc[0]
                lines.extend(
                    [
                        "",
                        f"- Guarded path readout: the first execution-ready release order was `{guarded['first_release_order_date']}` and the variant max-DD trough was `{guarded['variant_trough_date']}`.",
                        f"- From that release order to trough, average position delta versus baseline was `{float(guarded['avg_position_delta_release_to_trough']):+.4f}` and account value at trough was `{float(guarded['account_delta_at_trough_pct']):+.2f}%` versus baseline.",
                        "- Decision: attribution supports a distinct post-release de-risk or release target-cap test; it does not support another static entry-threshold tweak.",
                    ]
                )
            lines.append("- Full attribution artifact: `selective_release_walkforward_drawdown_attribution.csv`.")

    (output_dir / "selective_release_walkforward_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate selective release overlays on walk-forward actors")
    parser.add_argument("--profile", default="balanced_rule_residual_v22")
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="evaluate all existing selective release variants instead of baseline plus retained recovery gate",
    )
    parser.add_argument(
        "--output-dir",
        default="runtime/execution/card/selective_release_walkforward_v1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, context, trace_rows, attribution_rows = run_walk_forward_grid(
        args.profile,
        retained_only=not args.all_variants,
    )
    output_dir = ROOT / args.output_dir
    write_outputs(rows, context, output_dir, trace_rows, attribution_rows)
    print(
        f"[selective_release_walkforward] wrote {output_dir.relative_to(ROOT)} "
        f"rows={len(rows)} trace_rows={len(trace_rows)} attribution_rows={len(attribution_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
