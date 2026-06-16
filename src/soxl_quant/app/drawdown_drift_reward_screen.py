from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DrawdownDriftRewardScreenConfig:
    target_relative_return_pp: float = 30.0
    target_max_dd_pct: float = 15.0
    min_drawdown_failure_capture: float = 0.50
    min_trend_underparticipation_capture: float = 0.50
    max_overreach_share: float = 0.25
    drawdown_capture_weight: float = 45.0
    trend_capture_weight: float = 45.0
    overreach_weight: float = 60.0
    burden_weight: float = 2.0


DEFAULT_RECIPE_GRID: tuple[dict[str, float | int | str], ...] = tuple(
    {
        "recipe": (
            f"dd{drawdown_penalty:g}_thr{drawdown_threshold:g}_pos{position_threshold:g}"
            f"_tf{trend_floor_penalty:g}_floor{trend_floor_position:g}_lb{trend_lookback}"
        ),
        "drawdown_penalty": drawdown_penalty,
        "drawdown_threshold_pct": drawdown_threshold,
        "position_threshold": position_threshold,
        "trend_floor_penalty": trend_floor_penalty,
        "trend_floor_position": trend_floor_position,
        "trend_lookback": trend_lookback,
        "min_buyhold_return_pct": min_buyhold_return,
    }
    for drawdown_penalty in (1.5, 2.0, 3.0)
    for drawdown_threshold in (12.0, 15.0, 18.0)
    for position_threshold in (0.05, 0.10, 0.15)
    for trend_floor_penalty in (1.0, 2.0, 3.0)
    for trend_floor_position in (0.25, 0.35, 0.45)
    for trend_lookback in (21, 63)
    for min_buyhold_return in (5.0,)
)


def _normalise_daily(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"datetime", "strategy_nav", "buyhold_nav", "position"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"daily path missing columns: {missing}")
    out = frame[list(required)].copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    for col in ("strategy_nav", "buyhold_nav", "position"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["datetime", "strategy_nav", "buyhold_nav", "position"])
    out = out.sort_values("datetime", kind="stable").reset_index(drop=True)
    if len(out) < 2:
        raise ValueError("daily path needs at least two rows")
    return out


def _drawdown_pct(nav: pd.Series) -> pd.Series:
    values = pd.to_numeric(nav, errors="coerce").astype(float)
    peak = values.cummax().replace(0.0, np.nan)
    return (1.0 - values / peak).fillna(0.0).clip(lower=0.0) * 100.0


def _window_return_pct(nav: pd.Series, lookback: int) -> pd.Series:
    values = pd.to_numeric(nav, errors="coerce").astype(float)
    periods = max(int(lookback), 1)
    return (values / values.shift(periods).replace(0.0, np.nan) - 1.0) * 100.0


def _gate_rows(gate_gaps: pd.DataFrame, selected_candidate: str) -> pd.DataFrame:
    if gate_gaps.empty:
        return pd.DataFrame(columns=["profile", "window", "failure_type", "gate_pass"])
    if "profile" not in gate_gaps.columns or "window" not in gate_gaps.columns:
        raise ValueError("gate_gaps must include `profile` and `window` columns")
    rows = gate_gaps.copy()
    rows["profile"] = rows["profile"].astype(str)
    rows["window"] = rows["window"].astype(str)
    rows = rows[rows["profile"].eq(str(selected_candidate))]
    if "failure_type" not in rows.columns:
        rows["failure_type"] = np.where(rows.get("gate_pass", False), "pass", "unknown")
    return rows


def _window_from_label(label: str) -> str:
    if "__" in label:
        return label.rsplit("__", 1)[-1]
    parts = label.split("_")
    for idx, part in enumerate(parts):
        if part == "wf" and idx + 1 < len(parts):
            return "_".join(parts[idx : idx + 2])
        if part.startswith("wf"):
            return part
    return label


def _recipe_grid_frame(recipe_grid: pd.DataFrame | None) -> pd.DataFrame:
    if recipe_grid is None:
        return pd.DataFrame(DEFAULT_RECIPE_GRID)
    if recipe_grid.empty:
        return pd.DataFrame(columns=list(DEFAULT_RECIPE_GRID[0].keys()))
    return recipe_grid.copy()


def _float(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _reward_patch(row: Mapping[str, object]) -> str:
    return json.dumps(
        {
            "drawdown_drift_reward_enabled": _float(row, "drawdown_penalty") > 0.0,
            "drawdown_drift_penalty": _float(row, "drawdown_penalty"),
            "drawdown_drift_threshold_pct": _float(row, "drawdown_threshold_pct"),
            "drawdown_drift_position_threshold": _float(row, "position_threshold"),
            "trend_preservation_floor_enabled": _float(row, "trend_floor_penalty") > 0.0,
            "trend_preservation_floor_penalty": _float(row, "trend_floor_penalty"),
            "trend_preservation_floor_position": _float(row, "trend_floor_position"),
            "trend_preservation_lookback": int(_float(row, "trend_lookback", 21.0)),
            "trend_preservation_min_buyhold_return_pct": _float(row, "min_buyhold_return_pct"),
        },
        sort_keys=True,
    )


def _empty_scores() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "recipe",
            "recipe_ready",
            "reward_recipe_loss",
            "drawdown_failure_capture",
            "raw_drawdown_failure_capture",
            "trend_underparticipation_capture",
            "overreach_share",
            "controllable_drawdown_breach_rows",
            "inactive_drawdown_breach_rows",
            "reward_objective_patch_json",
        ]
    )


def screen_drawdown_drift_reward_recipes(
    daily_paths: Mapping[str, pd.DataFrame],
    *,
    gate_gaps: pd.DataFrame,
    recipe_grid: pd.DataFrame | None = None,
    selected_candidate: str,
    config: DrawdownDriftRewardScreenConfig = DrawdownDriftRewardScreenConfig(),
) -> pd.DataFrame:
    """Rank drawdown-drift and trend-preservation reward recipes before training.

    This is a pre-training diagnostic. It scores whether a reward recipe would
    apply pressure to the known failure modes in existing official daily paths;
    it does not imply that the path itself is improved or promotable.
    """

    selected_gate_rows = _gate_rows(gate_gaps, selected_candidate)
    if not daily_paths or selected_gate_rows.empty:
        return _empty_scores()
    failures_by_window = {
        str(row["window"]): str(row.get("failure_type", "unknown"))
        for row in selected_gate_rows.to_dict("records")
    }
    frame_by_label = {str(label): _normalise_daily(path) for label, path in daily_paths.items()}
    grid = _recipe_grid_frame(recipe_grid)
    rows: list[dict[str, object]] = []

    for recipe_row in grid.to_dict("records"):
        recipe_name = str(recipe_row.get("recipe", f"recipe_{len(rows)}"))
        drawdown_penalty = _float(recipe_row, "drawdown_penalty")
        drawdown_threshold_pct = _float(recipe_row, "drawdown_threshold_pct", config.target_max_dd_pct)
        position_threshold = _float(recipe_row, "position_threshold", 0.10)
        trend_floor_penalty = _float(recipe_row, "trend_floor_penalty")
        trend_floor_position = _float(recipe_row, "trend_floor_position", 0.35)
        trend_lookback = max(int(_float(recipe_row, "trend_lookback", 21.0)), 1)
        min_buyhold_return_pct = _float(recipe_row, "min_buyhold_return_pct", 5.0)

        drawdown_breach_rows = 0
        controllable_drawdown_breach_rows = 0
        inactive_drawdown_breach_rows = 0
        drawdown_captured_rows = 0
        trend_opportunity_rows = 0
        trend_captured_rows = 0
        active_rows = 0
        overreach_rows = 0
        pressure_total = 0.0

        for label, daily in frame_by_label.items():
            window = _window_from_label(label)
            failure_type = failures_by_window.get(window, "")
            if not failure_type or failure_type == "pass":
                continue
            drawdown = _drawdown_pct(daily["strategy_nav"])
            position = pd.to_numeric(daily["position"], errors="coerce").fillna(0.0)
            buyhold_return = _window_return_pct(daily["buyhold_nav"], trend_lookback).fillna(0.0)
            strategy_return = _window_return_pct(daily["strategy_nav"], trend_lookback).fillna(0.0)
            relative_return = strategy_return - buyhold_return

            drawdown_breach = drawdown > float(config.target_max_dd_pct)
            controllable_drawdown_breach = drawdown_breach & (position > position_threshold)
            drawdown_active = (
                controllable_drawdown_breach
                & (drawdown > drawdown_threshold_pct)
                & (drawdown_penalty > 0.0)
            )
            trend_opportunity = (
                (buyhold_return >= min_buyhold_return_pct)
                & (relative_return < 0.0)
                & (position < trend_floor_position)
                & (drawdown <= float(config.target_max_dd_pct))
            )
            trend_active = trend_opportunity & (trend_floor_penalty > 0.0)
            row_active = drawdown_active | trend_active

            drawdown_pressure = (
                (drawdown - drawdown_threshold_pct).clip(lower=0.0)
                * (position - position_threshold).clip(lower=0.0)
                * drawdown_penalty
            )
            trend_pressure = (
                (trend_floor_position - position).clip(lower=0.0)
                * trend_floor_penalty
                * (buyhold_return / max(min_buyhold_return_pct, 1e-9)).clip(lower=0.0)
            )

            if failure_type in {"drawdown_control", "joint_tradeoff", "near_miss_drawdown"}:
                drawdown_breach_rows += int(drawdown_breach.sum())
                controllable_drawdown_breach_rows += int(controllable_drawdown_breach.sum())
                inactive_drawdown_breach_rows += int((drawdown_breach & ~controllable_drawdown_breach).sum())
                drawdown_captured_rows += int(drawdown_active.sum())
            if failure_type in {"joint_tradeoff", "underparticipation", "selection_bias", "trend_capture"}:
                trend_opportunity_rows += int(trend_opportunity.sum())
                trend_captured_rows += int(trend_active.sum())

            active_rows += int(row_active.sum())
            overreach_rows += int((trend_active & drawdown_breach).sum())
            pressure_total += float((drawdown_pressure + trend_pressure).sum())

        raw_drawdown_capture = drawdown_captured_rows / drawdown_breach_rows if drawdown_breach_rows else 0.0
        drawdown_capture = (
            drawdown_captured_rows / controllable_drawdown_breach_rows
            if controllable_drawdown_breach_rows
            else raw_drawdown_capture
        )
        trend_capture = trend_captured_rows / trend_opportunity_rows if trend_opportunity_rows else 0.0
        overreach_share = overreach_rows / active_rows if active_rows else 0.0
        pressure_burden = pressure_total / max(active_rows, 1)
        recipe_ready = (
            drawdown_capture >= float(config.min_drawdown_failure_capture)
            and trend_capture >= float(config.min_trend_underparticipation_capture)
            and overreach_share <= float(config.max_overreach_share)
            and drawdown_penalty > 0.0
            and trend_floor_penalty > 0.0
        )
        loss = (
            max(0.0, float(config.min_drawdown_failure_capture) - drawdown_capture)
            * float(config.drawdown_capture_weight)
            + max(0.0, float(config.min_trend_underparticipation_capture) - trend_capture)
            * float(config.trend_capture_weight)
            + max(0.0, overreach_share - float(config.max_overreach_share)) * float(config.overreach_weight)
            + pressure_burden * float(config.burden_weight)
        )

        rows.append(
            {
                "recipe": recipe_name,
                "recipe_ready": bool(recipe_ready),
                "reward_recipe_loss": float(loss),
                "drawdown_failure_capture": float(drawdown_capture),
                "trend_underparticipation_capture": float(trend_capture),
                "overreach_share": float(overreach_share),
                "active_rows": int(active_rows),
                "drawdown_breach_rows": int(drawdown_breach_rows),
                "controllable_drawdown_breach_rows": int(controllable_drawdown_breach_rows),
                "inactive_drawdown_breach_rows": int(inactive_drawdown_breach_rows),
                "drawdown_captured_rows": int(drawdown_captured_rows),
                "raw_drawdown_failure_capture": float(raw_drawdown_capture),
                "trend_opportunity_rows": int(trend_opportunity_rows),
                "trend_captured_rows": int(trend_captured_rows),
                "pressure_burden": float(pressure_burden),
                "drawdown_penalty": float(drawdown_penalty),
                "drawdown_threshold_pct": float(drawdown_threshold_pct),
                "position_threshold": float(position_threshold),
                "trend_floor_penalty": float(trend_floor_penalty),
                "trend_floor_position": float(trend_floor_position),
                "trend_lookback": int(trend_lookback),
                "min_buyhold_return_pct": float(min_buyhold_return_pct),
                "reward_objective_patch_json": _reward_patch(recipe_row),
            }
        )

    if not rows:
        return _empty_scores()
    return pd.DataFrame(rows).sort_values(
        ["recipe_ready", "reward_recipe_loss", "recipe"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def summarize_drawdown_drift_reward_screen(scored: pd.DataFrame) -> dict[str, object]:
    if scored.empty:
        return {
            "decision": "drawdown_drift_reward_recipe_no_candidates",
            "selected_recipe": "",
            "training_allowed_next": False,
            "promotion_allowed": False,
            "next_mechanism": "collect_daily_paths_and_gate_gaps",
        }
    top = scored.iloc[0]
    ready = bool(top["recipe_ready"])
    return {
        "decision": (
            "drawdown_drift_reward_recipe_ready_for_short_training_screen"
            if ready
            else "drawdown_drift_reward_recipe_no_training_signal"
        ),
        "selected_recipe": str(top["recipe"]),
        "selected_reward_recipe_loss": float(top["reward_recipe_loss"]),
        "drawdown_failure_capture": float(top["drawdown_failure_capture"]),
        "trend_underparticipation_capture": float(top["trend_underparticipation_capture"]),
        "overreach_share": float(top["overreach_share"]),
        "training_allowed_next": ready,
        "promotion_allowed": False,
        "reward_objective_patch": json.loads(str(top.get("reward_objective_patch_json", "{}"))),
        "next_mechanism": (
            "short_foldlocal_bc_or_ppo_training_screen"
            if ready
            else "revise_reward_state_or_checkpoint_objective"
        ),
    }


def render_drawdown_drift_reward_screen_review(
    scored: pd.DataFrame,
    summary: Mapping[str, object],
    *,
    top_n: int = 8,
) -> str:
    lines = [
        "# Drawdown-Drift Reward Screen",
        "",
        "## Decision",
        "",
        f"- decision: `{summary.get('decision', '')}`",
        f"- selected_recipe: `{summary.get('selected_recipe', '')}`",
        f"- training_allowed_next: `{summary.get('training_allowed_next', False)}`",
        f"- next_mechanism: `{summary.get('next_mechanism', '')}`",
        "",
        "## Candidate Ranking",
        "",
        "| recipe | ready | loss | dd capture | trend capture | overreach | active rows |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    if scored.empty:
        lines.append("| none | false | 0.00 | 0.00 | 0.00 | 0.00 | 0 |")
    else:
        for row in scored.head(int(top_n)).itertuples(index=False):
            lines.append(
                "| "
                f"{row.recipe} | {bool(row.recipe_ready)} | {float(row.reward_recipe_loss):.2f} | "
                f"{float(row.drawdown_failure_capture):.2f} | "
                f"{float(row.trend_underparticipation_capture):.2f} | "
                f"{float(row.overreach_share):.2f} | {int(row.active_rows)} |"
            )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- This is a reward-objective readiness screen, not a performance backtest.",
            "- A ready recipe may enter a short training screen, but cannot promote an actor or runtime overlay.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "DEFAULT_RECIPE_GRID",
    "DrawdownDriftRewardScreenConfig",
    "render_drawdown_drift_reward_screen_review",
    "screen_drawdown_drift_reward_recipes",
    "summarize_drawdown_drift_reward_screen",
]
