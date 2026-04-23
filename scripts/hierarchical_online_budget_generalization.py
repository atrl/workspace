#!/usr/bin/env python3
"""Cross-ETF diagnostics for deployable hierarchical online budget composition."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.replay_two_speed_mode_budget import summarize_daily
from src.soxl_quant.data import (
    FEATURE_COLUMNS,
    build_dataset,
    feature_columns_for_packs,
    init_qlib,
    refresh_qlib_binary,
    segment_frame,
)
from src.soxl_quant.policy.environment import TrancheLedger, build_price_path
from src.soxl_quant.policy.execution import (
    execution_buy_cooldown_days,
    execution_regime,
    execution_trade_lock_days,
    rebalance_threshold,
)
from src.soxl_quant.policy.hierarchical_composer import (
    HierarchicalComposerConfig,
    HierarchicalComposerState,
    hierarchical_budget_target_position,
    next_hierarchical_state,
)
from src.soxl_quant.policy.modes import ModeState, mode_band, route_mode
from src.soxl_quant.policy.observation import make_observation
from src.soxl_quant.policy.policy_builder import actor_action, build_policy
from src.soxl_quant.policy.positioning import target_position_for_profile
from src.soxl_quant.settings import ACCOUNT, ETF_UNIVERSE, LOOKBACK, MODEL_DIR, TEST_SEG, TRAIN_SEG, VALID_SEG, load_profile

DATA_DIR = ROOT / "data" / "generalization"
RUNTIME_DIR = ROOT / "runtime" / "generalization"
QLIB_RUNTIME = RUNTIME_DIR / "qlib_stores"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="risk_budget_usage_v1")
    parser.add_argument("--model", default="soxl_balanced_rule_residual_imitation_v4_actor.pt")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "runtime" / "post_train_review" / "hierarchical_online_budget_generalization.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / "runtime" / "post_train_review" / "hierarchical_online_budget_generalization.md",
    )
    parser.add_argument(
        "--daily-dir",
        type=Path,
        default=ROOT / "runtime" / "post_train_review" / "hierarchical_online_budget_daily",
    )
    return parser.parse_args()


def load_actor(model_path: Path, profile_name: str):
    profile = load_profile(profile_name)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    _, actor = build_policy(
        int(ckpt["obs_dim"]),
        profile,
        use_moe=bool(ckpt.get("use_moe", False)),
        hard_regime_routing=bool(ckpt.get("use_moe", False)),
    )
    actor.load_state_dict(ckpt["state_dict"])
    actor.eval()
    return actor, profile


def build_etf_dataset(etf: str, profile) -> tuple[pd.DataFrame, pd.Series]:
    meta = ETF_UNIVERSE[etf]
    csv_path = DATA_DIR / f"{etf.lower()}_with_underlying_vix.csv"
    qlib_dir = QLIB_RUNTIME / etf.lower()
    refresh_qlib_binary(
        csv_path=csv_path,
        qlib_dir=qlib_dir,
        leveraged_etf=etf,
        underlying_etf=str(meta["underlying"]),
        leverage_ratio=float(meta["leverage"]),
    )
    init_qlib(qlib_dir)
    dataset = build_dataset(
        leveraged_etf=etf,
        underlying_etf=str(meta["underlying"]),
        leverage_ratio=float(meta["leverage"]),
        train_seg=TRAIN_SEG,
        valid_seg=VALID_SEG,
        test_seg=TEST_SEG,
        feature_packs=tuple(profile.feature.packs),
        profile_name=profile.name,
    )
    test_df = segment_frame(dataset, "test")
    price = dataset.handler.price_frame.droplevel("instrument")["price"].loc[test_df.index]
    return test_df, price


def _action(prev_pos: float, pos: float, threshold: float = 0.04) -> str:
    delta = pos - prev_pos
    if delta > threshold:
        return "BUY"
    if delta < -threshold:
        return "SELL"
    return "HOLD"


def simulate_variant(
    actor,
    profile,
    df: pd.DataFrame,
    price: pd.Series,
    asset_idx: int,
    feature_columns: list[str] | tuple[str, ...] | None,
    variant: str,
    composer_cfg: HierarchicalComposerConfig | None = None,
) -> pd.DataFrame:
    cash = ACCOUNT
    peak = ACCOUNT
    pos = 0.0
    ledger = TrancheLedger()
    exec_cfg = profile.execution
    buy_cooldown = 0
    trade_lock = 0
    price_path = build_price_path(df)
    values = [cash]
    positions = [0.0]
    actions = ["HOLD"]
    modes = ["flat_wait"]
    raw_targets = [0.0]
    dates = [pd.Timestamp(df.index[LOOKBACK])]
    last_mode = "flat_wait"
    hierarchical_state = HierarchicalComposerState() if composer_cfg is not None else None
    mode_state = ModeState()

    for idx in range(LOOKBACK, len(df) - 1):
        obs = make_observation(
            df,
            idx,
            pos,
            cash,
            peak,
            ledger,
            price_path,
            asset_idx=asset_idx,
            feature_columns=feature_columns or FEATURE_COLUMNS,
        )
        action = actor_action(actor, obs)
        current_drawdown = max(1.0 - cash / max(peak, 1e-8), 0.0)

        if variant == "actor_runtime":
            mode_state = route_mode(obs, mode_state, profile.mode_router)
            target_pos = target_position_for_profile(
                action,
                obs,
                config=profile.position_cap,
                action_levels=np.asarray(profile.model.action_levels, dtype=np.float32),
                hard_floor=profile.model.hard_floor,
                use_hybrid_overlay=profile.model.use_hybrid_overlay,
                delta_levels=np.asarray(profile.model.delta_levels, dtype=np.float32) if profile.model.delta_levels else None,
                mode_band=mode_band(mode_state.name, profile.mode_router) if profile.mode_router.enabled else None,
                position_budget_config=profile.position_budget,
                current_drawdown=current_drawdown,
                position_mapping=profile.model.position_mapping,
            )
            mode_name = mode_state.name
        elif composer_cfg is None:
            raise ValueError("hierarchical variant requires composer config")
        else:
            hierarchical_state = next_hierarchical_state(obs, hierarchical_state, profile.mode_router, composer_cfg)
            mode_name = hierarchical_state.active_mode
            target_pos = hierarchical_budget_target_position(
                action,
                obs,
                active_mode=mode_name,
                profile=profile,
                current_drawdown=current_drawdown,
            )

        row = df.iloc[idx]
        regime_risk = float(np.nan_to_num(row["regime_risk"]))
        realized_vol = float(np.nan_to_num(row["realized_vol_10"]))
        trend_strength = abs(float(np.nan_to_num(row["trend_gap_21"])))
        cum_drop_30 = float(np.nan_to_num(row["cum_drop_30"]))
        relative_strength = float(np.nan_to_num(row["relative_strength_21"]))
        vix_zscore = float(np.nan_to_num(row["vix_zscore"]))
        downside_ratio_10 = float(np.nan_to_num(row["downside_ratio_10"]))
        regime = execution_regime(
            regime_risk,
            realized_vol,
            trend_strength,
            cum_drop_30=cum_drop_30,
            relative_strength=relative_strength,
            vix_zscore=vix_zscore,
            downside_ratio_10=downside_ratio_10,
            config=exec_cfg,
        )
        direction = "buy" if target_pos > pos else ("sell" if target_pos < pos else "neutral")
        min_rebalance = rebalance_threshold(
            regime_risk,
            realized_vol,
            trend_strength,
            cum_drop_30=cum_drop_30,
            relative_strength=relative_strength,
            vix_zscore=vix_zscore,
            downside_ratio_10=downside_ratio_10,
            direction=direction,
            mode=mode_name if composer_cfg is not None else None,
            config=exec_cfg,
            mode_router_config=profile.mode_router if composer_cfg is not None else None,
        )
        if trade_lock > 0:
            if abs(target_pos - pos) >= min_rebalance:
                target_pos = pos
            trade_lock -= 1
        if buy_cooldown > 0:
            if target_pos > pos:
                target_pos = pos
            buy_cooldown -= 1
        if abs(target_pos - pos) < min_rebalance:
            target_pos = pos
        next_ret = float(df.iloc[idx + 1]["return"])
        turnover = abs(target_pos - pos)
        cash *= 1.0 + target_pos * next_ret - turnover * 0.001
        peak = max(peak, cash)
        ledger.update(pos, target_pos, float(price_path[idx]))
        if target_pos > pos + 1e-6:
            buy_cooldown = execution_buy_cooldown_days(cum_drop_30, exec_cfg, regime)
        if turnover > exec_cfg.min_trade_turnover:
            trade_lock = execution_trade_lock_days(exec_cfg, regime)
        action_label = _action(pos, target_pos)
        pos = target_pos
        last_mode = mode_name

        values.append(cash)
        positions.append(pos)
        actions.append(action_label)
        modes.append(last_mode)
        raw_targets.append(target_pos)
        dates.append(pd.Timestamp(df.index[idx + 1]))

    aligned_price = price.reindex(pd.DatetimeIndex(dates)).ffill().bfill().astype(float)
    buyhold_nav = ACCOUNT * aligned_price / float(aligned_price.iloc[0])
    out = pd.DataFrame(
        {
            "datetime": dates,
            "strategy_nav": values,
            "price": aligned_price.to_numpy(dtype=float),
            "buyhold_nav": buyhold_nav.to_numpy(dtype=float),
            "position": positions,
            "action": actions,
            "mode": modes,
            "raw_target_position": raw_targets,
            "variant": variant,
        }
    )
    out["excess"] = out["strategy_nav"] - out["buyhold_nav"]
    out["excess_pct"] = np.where(
        out["buyhold_nav"] > 0,
        (out["strategy_nav"] / out["buyhold_nav"] - 1.0) * 100.0,
        0.0,
    )
    return out


def _write_daily(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _format(metrics: dict[str, float]) -> str:
    return (
        f"{metrics['return']:.1f}% / {metrics['sharpe']:.3f} / "
        f"{metrics['max_dd']:.1f}% / pos {metrics['avg_pos']:.3f}"
    )


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# hierarchical online budget generalization",
        "",
        "This diagnostic keeps the fast `imitation_v4` actor fixed and changes only the deployable runtime policy shape.",
        "",
        "- `actor_runtime`: current single-actor budget-usage runtime with the existing mode router.",
        "- `hier_rule_daily`: online hierarchical composer with rule mode refresh every day.",
        "- `hier_rule_slow5`: same composer, but slow mode refresh every 5 trading days.",
        "- `hier_learned_daily` / `hier_learned_slow5`: same structure using the learned mode classifier.",
        "",
        "| ETF | actor proxy | rule daily proxy | rule slow5 proxy | learned daily proxy | learned slow5 proxy | best |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for etf, result in payload["results"].items():
        variants = result["variants"]
        best_variant = max(
            variants.items(),
            key=lambda item: (item[1]["test_proxy"]["sharpe"], item[1]["test_proxy"]["return"]),
        )[0]
        lines.append(
            f"| {etf} | {_format(variants['actor_runtime']['test_proxy'])} | "
            f"{_format(variants['hier_rule_daily']['test_proxy'])} | "
            f"{_format(variants['hier_rule_slow5']['test_proxy'])} | "
            f"{_format(variants['hier_learned_daily']['test_proxy'])} | "
            f"{_format(variants['hier_learned_slow5']['test_proxy'])} | {best_variant} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(payload["interpretation"]),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.daily_dir.mkdir(parents=True, exist_ok=True)
    profile_name = str(args.profile)
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = MODEL_DIR / model_path
    actor, profile = load_actor(model_path, profile_name)
    feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)

    variant_configs: dict[str, HierarchicalComposerConfig | None] = {
        "actor_runtime": None,
        "hier_rule_daily": HierarchicalComposerConfig(slow_update_every=1, use_learned_router=False),
        "hier_rule_slow5": HierarchicalComposerConfig(slow_update_every=5, use_learned_router=False),
        "hier_learned_daily": HierarchicalComposerConfig(slow_update_every=1, use_learned_router=True),
        "hier_learned_slow5": HierarchicalComposerConfig(slow_update_every=5, use_learned_router=True),
    }

    results: dict[str, Any] = {}
    for asset_idx, etf in enumerate(ETF_UNIVERSE):
        test_df, price = build_etf_dataset(etf, profile)
        per_variant: dict[str, Any] = {}
        for name, cfg in variant_configs.items():
            daily = simulate_variant(
                actor=actor,
                profile=profile,
                df=test_df,
                price=price,
                asset_idx=asset_idx,
                feature_columns=feature_columns,
                variant=name,
                composer_cfg=cfg,
            )
            _write_daily(args.daily_dir / f"{etf.lower()}_{name}.csv", daily)
            per_variant[name] = summarize_daily(daily)
            print(f"{etf} {name}: {_format(per_variant[name]['test_proxy'])}")
        results[etf] = {"variants": per_variant}

    interpretation = (
        "A deployable improvement here must beat `actor_runtime` on proxy participation without giving back the 2022 drawdown compression. "
        "If the hierarchical variants only reshuffle risk while staying low-return, the missing capability is still the fast actor's participation policy, not the router cadence."
    )
    payload = {
        "profile": profile_name,
        "model": str(model_path),
        "results": results,
        "interpretation": interpretation,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(f"[done] wrote {args.output_json}")
    print(f"[done] wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
