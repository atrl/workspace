#!/usr/bin/env python3
from __future__ import annotations

"""No-promotion regression probe from a replay daily path."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tianshou.utils.net.common import Net

from src.soxl_quant.app.evaluate import slice_frame
from src.soxl_quant.app.train import set_global_seed
from src.soxl_quant.config import load_profile
from src.soxl_quant.data import (
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
from src.soxl_quant.policy.modes import ModeState, mode_band, route_mode
from src.soxl_quant.policy.models.attention import WindowSelfAttentionNet
from src.soxl_quant.policy.models.gru import GRUNet
from src.soxl_quant.policy.models.moe import RegimeMoENet
from src.soxl_quant.policy.models.tide import TiDENet
from src.soxl_quant.policy.observation import make_observation
from src.soxl_quant.policy.positioning import apply_position_budget, budget_usage_to_position
from src.soxl_quant.policy.usage_regression import build_budget_usage_regression_dataset_from_frame
from src.soxl_quant.settings import CSV_PATH, LOOKBACK, QLIB_DATA_PATH


class UsageRegressionModel(nn.Module):
    def __init__(self, obs_dim: int, profile, feature_columns: list[str]):
        super().__init__()
        model_cfg = profile.model
        if model_cfg.use_tide:
            backbone = TiDENet(
                obs_dim,
                hidden_size=int(model_cfg.tide_hidden_size),
                feature_dim=len(feature_columns),
                observation_window=max(int(model_cfg.observation_window), 1),
                dropout=float(model_cfg.tide_dropout),
                device="cpu",
            )
        elif getattr(model_cfg, "use_attention", False):
            backbone = WindowSelfAttentionNet(
                obs_dim,
                feature_dim=len(feature_columns),
                observation_window=max(int(model_cfg.observation_window), 1),
                attention_dim=int(getattr(model_cfg, "attention_dim", 64)),
                attention_heads=int(getattr(model_cfg, "attention_heads", 4)),
                mlp_hidden_sizes=list(profile.hidden_sizes),
                device="cpu",
            )
        elif model_cfg.use_gru:
            backbone = GRUNet(
                obs_dim,
                feature_dim=len(feature_columns),
                observation_window=max(int(model_cfg.observation_window), 1),
                gru_hidden_size=int(model_cfg.gru_hidden_size),
                mlp_hidden_sizes=list(profile.hidden_sizes),
                device="cpu",
            )
        elif getattr(model_cfg, "use_regime_router", False):
            backbone = RegimeMoENet(
                obs_dim,
                hidden_size=64,
                n_experts=int(model_cfg.router_experts),
                device="cpu",
                top_k=int(model_cfg.router_top_k),
                noise_std=float(model_cfg.router_noise_std),
            )
        else:
            backbone = Net((obs_dim,), hidden_sizes=list(profile.hidden_sizes), device="cpu")
        self.backbone = backbone
        hidden_dim = int(getattr(backbone, "output_dim", profile.hidden_sizes[-1]))
        self.head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, obs):
        out = self.backbone(obs)
        hidden = out[0] if isinstance(out, tuple) else out
        pred = self.head(hidden)
        return pred.squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="risk_budget_learned_usage_regression_v1")
    parser.add_argument("--replay-daily", type=Path, default=Path("runtime/model/reviews/hierarchical_online_budget_daily/soxl_hier_learned_slow5.csv"))
    parser.add_argument("--seed", type=int, default=20260411)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-json", type=Path, default=Path("runtime/model/reviews/risk_budget_learned_usage_regression_v1.json"))
    parser.add_argument("--output-md", type=Path, default=Path("runtime/model/reviews/risk_budget_learned_usage_regression_v1.md"))
    parser.add_argument("--skip-data-refresh", action="store_true")
    return parser.parse_args()


def train_regressor(model: UsageRegressionModel, obs_array: np.ndarray, target_array: np.ndarray, steps: int, lr: float, batch_size: int):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32)
    target_tensor = torch.as_tensor(target_array, dtype=torch.float32)
    n = int(obs_tensor.shape[0])
    last_loss = 0.0
    last_mae = 0.0
    model.train()
    for _step in range(int(steps)):
        idx = torch.randint(0, n, (min(batch_size, n),))
        batch_obs = obs_tensor[idx]
        batch_targets = target_tensor[idx]
        preds = model(batch_obs)
        loss = loss_fn(preds, batch_targets)
        optim.zero_grad()
        loss.backward()
        optim.step()
        last_loss = float(loss.detach().cpu().item())
        with torch.no_grad():
            last_mae = float(torch.abs(preds - batch_targets).mean().item())
    model.eval()
    return {"loss": last_loss, "mae": last_mae, "steps": int(steps)}


def evaluate_usage_regression_split(model: UsageRegressionModel, df: pd.DataFrame, profile, feature_columns: list[str]) -> dict[str, float]:
    cash = 100_000.0
    peak = cash
    pos = 0.0
    ledger = TrancheLedger()
    exec_cfg = profile.execution
    buy_cooldown = 0
    trade_lock = 0
    price_path = build_price_path(df)
    values = [cash]
    positions = []
    stress_positions = []
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
            asset_idx=0,
            observation_window=profile.model.observation_window,
            feature_columns=feature_columns,
        )
        mode_state = route_mode(obs, mode_state, profile.mode_router)
        with torch.no_grad():
            usage = float(model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).item())
        row = df.iloc[idx]
        current_drawdown = max(1.0 - cash / max(peak, 1e-8), 0.0)
        target_pos = budget_usage_to_position(
            usage,
            obs,
            config=profile.position_cap,
            hard_floor=profile.model.hard_floor,
            mode_band=mode_band(mode_state.name, profile.mode_router),
        )
        target_pos = apply_position_budget(
            target_pos,
            obs,
            current_drawdown=current_drawdown,
            config=profile.position_budget,
        )
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
            mode=mode_state.name,
            config=exec_cfg,
            mode_router_config=profile.mode_router,
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
        pos = target_pos
        positions.append(pos)
        if regime_risk >= 0.65:
            stress_positions.append(pos)
        values.append(cash)
    account = np.array(values)
    daily_ret = np.diff(account) / account[:-1]
    peak_curve = np.maximum.accumulate(account)
    return {
        "return": (account[-1] / account[0] - 1) * 100,
        "sharpe": float(np.mean(daily_ret) / (np.std(daily_ret) + 1e-8) * np.sqrt(252)),
        "max_dd": float(np.max((peak_curve - account) / peak_curve) * 100),
        "avg_pos": float(np.mean(positions)) if positions else 0.0,
        "stress_avg_pos": float(np.mean(stress_positions)) if stress_positions else 0.0,
        "stress_days": float(len(stress_positions)),
    }


def slice_metrics(model: UsageRegressionModel, df, slices, profile, feature_columns: list[str]):
    metrics: dict[str, dict[str, float]] = {}
    for name, seg in slices.items():
        segment = slice_frame(df, seg)
        if len(segment) <= 120:
            continue
        metrics[name] = evaluate_usage_regression_split(model, segment, profile, feature_columns)
    return metrics


def _fmt(metrics: dict[str, float]) -> str:
    return f"{metrics['return']:.1f}% / Sharpe {metrics['sharpe']:.3f} / MaxDD {metrics['max_dd']:.1f}% / avg_pos {metrics.get('avg_pos', 0.0):.3f}"


def write_markdown(path: Path, payload: dict[str, object]) -> None:
    slices = payload["slices"]
    assert isinstance(slices, dict)
    lines = [
        f"# {payload['profile']} Usage Regression Probe",
        "",
        "No-promotion diagnostic trained from a replay daily path with a continuous usage regression head.",
        "",
        "## Dataset",
        "",
        f"- replay_daily: `{payload['replay_daily']}`",
        f"- samples: `{payload['samples']}`",
        f"- target_mean: `{payload['target_mean']:.6f}`",
        f"- target_std: `{payload['target_std']:.6f}`",
        "",
        "## Train Stats",
        "",
        f"- loss: `{payload['train_stats']['loss']:.6f}`",
        f"- mae: `{payload['train_stats']['mae']:.6f}`",
        f"- steps: `{payload['train_stats']['steps']}`",
        "",
        "## Fixed-Split Probe",
        "",
        "| slice | return / sharpe / maxdd / avg_pos |",
        "| --- | --- |",
    ]
    for name in ("test_2020", "test_2022", "test_2023_2024"):
        lines.append(f"| {name} | {_fmt(slices[name])} |")
    lines.append(f"| test_proxy | {_fmt(payload['test_proxy'])} |")
    lines.extend(["", "## Decision", "", str(payload["decision"]), ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_global_seed(args.seed)
    profile = load_profile(args.profile)
    replay_daily = pd.read_csv(args.replay_daily, parse_dates=["datetime"])

    if not args.skip_data_refresh:
        refresh_qlib_binary(CSV_PATH, QLIB_DATA_PATH)
    init_qlib(QLIB_DATA_PATH)
    dataset = build_dataset(feature_packs=tuple(profile.feature.packs), profile_name=profile.name)
    test_df = segment_frame(dataset, "test")
    feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)

    obs_array, target_array = build_budget_usage_regression_dataset_from_frame(
        test_df,
        replay_daily=replay_daily,
        profile=profile,
        sample_size=int(args.sample_size),
        rng_seed=int(args.seed),
    )
    model = UsageRegressionModel(int(obs_array.shape[1]), profile, feature_columns)
    train_stats = train_regressor(model, obs_array, target_array, int(args.steps), float(args.lr), int(args.batch_size))

    slices_spec = {
        "test_2020": ("2020-01-01", "2020-12-31"),
        "test_2022": ("2022-01-01", "2022-12-31"),
        "test_2023_2024": ("2023-01-01", "2024-12-31"),
    }
    slices = slice_metrics(model, test_df, slices_spec, profile, feature_columns)
    test_proxy = evaluate_usage_regression_split(model, test_df, profile, feature_columns)
    payload: dict[str, object] = {
        "profile": profile.name,
        "replay_daily": str(args.replay_daily),
        "seed": int(args.seed),
        "samples": int(len(obs_array)),
        "target_mean": float(np.mean(target_array)),
        "target_std": float(np.std(target_array)),
        "train_stats": train_stats,
        "slices": slices,
        "test_proxy": test_proxy,
        "decision": "no-promotion diagnostic only; continuous usage regression from held-out replay daily path",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(args.output_md, payload)
    print(f"[usage-regression:{profile.name}] samples={len(obs_array)} stats={train_stats}")
    print(f"[usage-regression:{profile.name}] test_proxy={_fmt(test_proxy)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
