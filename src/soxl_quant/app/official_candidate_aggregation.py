from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OfficialReplayCandidateConfig:
    target_relative_return_pp: float = 30.0
    target_max_dd_pct: float = 15.0
    max_dd_gap_weight: float = 10.0
    robust_relative_margin_pp: float = 2.0
    robust_max_dd_margin_pct: float = 1.0
    near_miss_relative_gap_pp: float = 1.0
    near_miss_max_dd_gap_pct: float = 0.5


def _first_existing(frame: pd.DataFrame, names: tuple[str, ...], label: str) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"candidate frame must include one of {names} for `{label}`")


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _bool_series(frame: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    raw = frame[column]
    if raw.dtype == bool:
        return raw.fillna(default).astype(bool)
    text = raw.astype(str).str.strip().str.lower()
    return text.isin(["1", "true", "yes", "y"])


def normalise_official_replay_candidate_frame(
    frame: pd.DataFrame,
    *,
    source_label: str,
    mechanism: str,
    candidate_col: str = "",
    window_col: str = "",
    relative_col: str = "",
    max_dd_col: str = "",
    avg_pos_col: str = "",
    gate_col: str = "",
    config: OfficialReplayCandidateConfig = OfficialReplayCandidateConfig(),
) -> pd.DataFrame:
    """Convert one official replay candidate table into a common fold-level schema."""

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "source_label",
                "mechanism",
                "candidate",
                "candidate_id",
                "window",
                "relative_return",
                "max_dd",
                "avg_pos",
                "window_gate_pass",
                "relative_gap",
                "max_dd_gap",
                "robust_window_gate",
            ]
        )

    candidate_col = candidate_col or _first_existing(frame, ("variant", "candidate", "profile"), "candidate")
    window_col = window_col or _first_existing(frame, ("fold", "window"), "window")
    relative_col = relative_col or _first_existing(
        frame,
        ("official_warm_relative_return", "relative_return", "relative_return_pp"),
        "relative_return",
    )
    max_dd_col = max_dd_col or _first_existing(
        frame,
        ("official_warm_max_dd", "max_dd", "max_dd_pct"),
        "max_dd",
    )
    avg_pos_col = avg_pos_col or next(
        (name for name in ("official_warm_avg_pos", "avg_pos", "mean_avg_pos") if name in frame.columns),
        "",
    )
    gate_col = gate_col or next(
        (name for name in ("window_gate_pass", "full_gate", "target_gate", "reported_target_gate") if name in frame.columns),
        "",
    )

    relative = _numeric(frame, relative_col)
    max_dd = _numeric(frame, max_dd_col)
    valid_relative = np.isfinite(relative.to_numpy(dtype=float))
    valid_max_dd = np.isfinite(max_dd.to_numpy(dtype=float))
    computed_gate = (
        valid_relative
        & valid_max_dd
        & (relative.to_numpy(dtype=float) >= float(config.target_relative_return_pp))
        & (max_dd.to_numpy(dtype=float) <= float(config.target_max_dd_pct))
    )
    if gate_col:
        window_gate = _bool_series(frame, gate_col).to_numpy(dtype=bool)
    else:
        window_gate = computed_gate

    candidate = frame[candidate_col].astype(str)
    out = pd.DataFrame(
        {
            "source_label": str(source_label),
            "mechanism": str(mechanism),
            "candidate": candidate,
            "candidate_id": [f"{source_label}::{value}" for value in candidate],
            "window": frame[window_col].astype(str),
            "relative_return": relative,
            "max_dd": max_dd,
            "avg_pos": _numeric(frame, avg_pos_col) if avg_pos_col else np.nan,
            "window_gate_pass": window_gate,
        }
    )
    out["relative_gap"] = np.where(
        valid_relative,
        np.maximum(0.0, float(config.target_relative_return_pp) - out["relative_return"].to_numpy(dtype=float)),
        float(config.target_relative_return_pp),
    )
    out["max_dd_gap"] = np.where(
        valid_max_dd,
        np.maximum(0.0, out["max_dd"].to_numpy(dtype=float) - float(config.target_max_dd_pct)),
        float(config.target_max_dd_pct),
    )
    out["robust_window_gate"] = (
        (out["relative_return"] >= float(config.target_relative_return_pp) + float(config.robust_relative_margin_pp))
        & (out["max_dd"] <= float(config.target_max_dd_pct) - float(config.robust_max_dd_margin_pct))
    )
    return out


def _failure_profile(row: pd.Series, config: OfficialReplayCandidateConfig) -> str:
    if bool(row["robust_full_gate"]):
        return "robust_pass"
    if bool(row["full_gate"]):
        return "full_gate_low_margin"
    if bool(row["near_miss"]):
        return "near_miss"
    rel_bad = float(row["max_relative_gap"]) > float(config.near_miss_relative_gap_pp)
    dd_bad = float(row["max_dd_gap"]) > float(config.near_miss_max_dd_gap_pct)
    if rel_bad and dd_bad:
        return "joint_tradeoff"
    if rel_bad:
        return "trend_capture"
    if dd_bad:
        return "drawdown_control"
    return "unknown"


def score_official_replay_candidate_pool(
    candidates: pd.DataFrame,
    *,
    config: OfficialReplayCandidateConfig = OfficialReplayCandidateConfig(),
) -> pd.DataFrame:
    """Score a combined official replay candidate pool with hard and margin gates."""

    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "source_label",
                "mechanism",
                "candidate",
                "window_count",
                "gate_pass_count",
                "full_gate",
                "robust_full_gate",
                "near_miss",
                "total_gate_loss",
                "failure_profile",
            ]
        )

    rows: list[dict[str, object]] = []
    for candidate_id, sub in candidates.groupby("candidate_id", sort=True):
        relative_gap = pd.to_numeric(sub["relative_gap"], errors="coerce").fillna(float(config.target_relative_return_pp))
        max_dd_gap = pd.to_numeric(sub["max_dd_gap"], errors="coerce").fillna(float(config.target_max_dd_pct))
        gate = _bool_series(sub, "window_gate_pass")
        robust_gate = _bool_series(sub, "robust_window_gate")
        max_relative_gap = float(relative_gap.max()) if len(relative_gap) else 0.0
        max_dd_gap_value = float(max_dd_gap.max()) if len(max_dd_gap) else 0.0
        full_gate = bool(gate.all()) if len(gate) else False
        robust_full_gate = bool(robust_gate.all()) if len(robust_gate) else False
        near_miss = bool(
            not full_gate
            and int(gate.sum()) >= max(0, len(gate) - 1)
            and max_relative_gap <= float(config.near_miss_relative_gap_pp)
            and max_dd_gap_value <= float(config.near_miss_max_dd_gap_pct)
        )
        row = {
            "candidate_id": str(candidate_id),
            "source_label": str(sub["source_label"].iloc[0]),
            "mechanism": str(sub["mechanism"].iloc[0]),
            "candidate": str(sub["candidate"].iloc[0]),
            "window_count": int(len(sub)),
            "gate_pass_count": int(gate.sum()),
            "full_gate": full_gate,
            "robust_gate_pass_count": int(robust_gate.sum()),
            "robust_full_gate": robust_full_gate,
            "near_miss": near_miss,
            "min_relative_return": float(pd.to_numeric(sub["relative_return"], errors="coerce").min()),
            "max_max_dd": float(pd.to_numeric(sub["max_dd"], errors="coerce").max()),
            "mean_avg_pos": float(pd.to_numeric(sub["avg_pos"], errors="coerce").mean()),
            "max_relative_gap": max_relative_gap,
            "max_dd_gap": max_dd_gap_value,
            "min_relative_margin": float(
                (pd.to_numeric(sub["relative_return"], errors="coerce") - float(config.target_relative_return_pp)).min()
            ),
            "min_max_dd_margin": float(
                (float(config.target_max_dd_pct) - pd.to_numeric(sub["max_dd"], errors="coerce")).min()
            ),
            "total_gate_loss": float(relative_gap.sum() + max_dd_gap.sum() * float(config.max_dd_gap_weight)),
        }
        row["failure_profile"] = _failure_profile(pd.Series(row), config)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        [
            "robust_full_gate",
            "full_gate",
            "near_miss",
            "total_gate_loss",
            "max_max_dd",
            "min_relative_return",
            "candidate_id",
        ],
        ascending=[False, False, False, True, True, False, True],
    ).reset_index(drop=True)


def score_official_replay_stress_scenarios(
    candidates: pd.DataFrame,
    *,
    stress_scenarios: list[dict[str, object]] | tuple[dict[str, object], ...],
    config: OfficialReplayCandidateConfig = OfficialReplayCandidateConfig(),
) -> pd.DataFrame:
    """Score candidates after simple return haircuts and drawdown add-ons.

    The scenarios intentionally operate on already-generated official replay
    metrics. They are a promotion-blocking robustness check, not a replacement
    for a replay under a changed execution model.
    """

    if candidates.empty or not stress_scenarios:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "scenario",
                "stress_full_gate",
                "stress_gate_pass_count",
                "stress_total_gate_loss",
                "stress_failure_profile",
            ]
        )

    rows: list[dict[str, object]] = []
    relative = pd.to_numeric(candidates["relative_return"], errors="coerce")
    max_dd = pd.to_numeric(candidates["max_dd"], errors="coerce")
    base = candidates.copy()
    base["_relative_return_numeric"] = relative
    base["_max_dd_numeric"] = max_dd
    base["_base_window_gate"] = (
        (base["_relative_return_numeric"] >= float(config.target_relative_return_pp))
        & (base["_max_dd_numeric"] <= float(config.target_max_dd_pct))
    )

    for scenario in stress_scenarios:
        scenario_name = str(scenario.get("scenario", scenario.get("name", "")) or "stress")
        relative_haircut = float(scenario.get("relative_haircut_pp", 0.0) or 0.0)
        max_dd_add = float(scenario.get("max_dd_add_pct", 0.0) or 0.0)
        stressed = base.copy()
        stressed["stress_relative_return"] = stressed["_relative_return_numeric"] - relative_haircut
        stressed["stress_max_dd"] = stressed["_max_dd_numeric"] + max_dd_add
        stressed["stress_relative_gap"] = (
            float(config.target_relative_return_pp) - stressed["stress_relative_return"]
        ).clip(lower=0.0)
        stressed["stress_max_dd_gap"] = (
            stressed["stress_max_dd"] - float(config.target_max_dd_pct)
        ).clip(lower=0.0)
        stressed["stress_window_gate"] = (
            (stressed["stress_relative_return"] >= float(config.target_relative_return_pp))
            & (stressed["stress_max_dd"] <= float(config.target_max_dd_pct))
        )

        for candidate_id, sub in stressed.groupby("candidate_id", sort=True):
            gate = _bool_series(sub, "stress_window_gate")
            stress_full_gate = bool(gate.all()) if len(gate) else False
            gate_pass_count = int(gate.sum())
            relative_gap = pd.to_numeric(sub["stress_relative_gap"], errors="coerce").fillna(
                float(config.target_relative_return_pp)
            )
            max_dd_gap = pd.to_numeric(sub["stress_max_dd_gap"], errors="coerce").fillna(
                float(config.target_max_dd_pct)
            )
            total_loss = float(relative_gap.sum() + max_dd_gap.sum() * float(config.max_dd_gap_weight))
            base_gate = _bool_series(sub, "window_gate_pass") if "window_gate_pass" in sub.columns else _bool_series(sub, "_base_window_gate")
            base_full_gate = bool(base_gate.all()) if len(base_gate) else False
            if stress_full_gate:
                failure_profile = "stress_pass"
            elif base_full_gate:
                failure_profile = "stress_fragile"
            elif gate_pass_count >= max(0, len(gate) - 1):
                failure_profile = "stress_near_miss"
            else:
                rel_bad = float(relative_gap.max()) > float(config.near_miss_relative_gap_pp)
                dd_bad = float(max_dd_gap.max()) > float(config.near_miss_max_dd_gap_pct)
                if rel_bad and dd_bad:
                    failure_profile = "stress_joint_tradeoff"
                elif rel_bad:
                    failure_profile = "stress_trend_capture"
                elif dd_bad:
                    failure_profile = "stress_drawdown_control"
                else:
                    failure_profile = "stress_unknown"
            rows.append(
                {
                    "candidate_id": str(candidate_id),
                    "source_label": str(sub["source_label"].iloc[0]) if "source_label" in sub else "",
                    "mechanism": str(sub["mechanism"].iloc[0]) if "mechanism" in sub else "",
                    "candidate": str(sub["candidate"].iloc[0]) if "candidate" in sub else str(candidate_id),
                    "scenario": scenario_name,
                    "relative_haircut_pp": relative_haircut,
                    "max_dd_add_pct": max_dd_add,
                    "window_count": int(len(sub)),
                    "stress_gate_pass_count": gate_pass_count,
                    "stress_full_gate": stress_full_gate,
                    "stress_min_relative_return": float(pd.to_numeric(sub["stress_relative_return"], errors="coerce").min()),
                    "stress_max_dd": float(pd.to_numeric(sub["stress_max_dd"], errors="coerce").max()),
                    "stress_total_gate_loss": total_loss,
                    "stress_failure_profile": failure_profile,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["scenario", "stress_full_gate", "stress_total_gate_loss", "candidate_id"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)


def render_official_replay_candidate_review(scored: pd.DataFrame) -> str:
    lines = [
        "# Official Replay Candidate Mechanism PK",
        "",
        "| candidate | mechanism | full | robust | near miss | loss | min rel | max dd | failure |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    if scored.empty:
        lines.append("| none | none | false | false | false | 0.00 | 0.00 | 0.00 | none |")
    else:
        for row in scored.itertuples(index=False):
            lines.append(
                "| "
                f"{row.candidate_id} | {row.mechanism} | {bool(row.full_gate)} | "
                f"{bool(row.robust_full_gate)} | {bool(row.near_miss)} | "
                f"{float(row.total_gate_loss):.2f} | {float(row.min_relative_return):.2f}pp | "
                f"{float(row.max_max_dd):.2f}% | {row.failure_profile} |"
            )
    lines.extend(
        [
            "",
            "Near-miss candidates are diagnostic only. They require wider replay or stress testing before promotion.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "OfficialReplayCandidateConfig",
    "normalise_official_replay_candidate_frame",
    "render_official_replay_candidate_review",
    "score_official_replay_candidate_pool",
    "score_official_replay_stress_scenarios",
]
