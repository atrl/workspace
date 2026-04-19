from __future__ import annotations

import json
from typing import Dict

import torch

from ..data import FEATURE_COLUMNS, feature_schema_signature
from ..settings import EXTRA_OBS_DIM, MODEL_DIR, N_ASSETS, model_artifact_path, summary_artifact_path


def serialize_position_cap(profile) -> Dict[str, float]:
    return {
        "start_drawdown": profile.position_cap.start_drawdown,
        "full_drawdown": profile.position_cap.full_drawdown,
        "min_cap": profile.position_cap.min_cap,
        "curve_power": profile.position_cap.curve_power,
        "bull_min_cap": profile.position_cap.bull_min_cap,
        "bull_vol_threshold": profile.position_cap.bull_vol_threshold,
        "bull_anchor_threshold": profile.position_cap.bull_anchor_threshold,
        "bear_min_cap": profile.position_cap.bear_min_cap,
        "bear_regime_threshold": profile.position_cap.bear_regime_threshold,
    }


def serialize_selection_score(profile) -> Dict[str, float | bool]:
    return {
        "return_weight": profile.selection_score.return_weight,
        "sharpe_weight": profile.selection_score.sharpe_weight,
        "max_dd_penalty": profile.selection_score.max_dd_penalty,
        "avg_pos_penalty": profile.selection_score.avg_pos_penalty,
        "stress_avg_pos_penalty": profile.selection_score.stress_avg_pos_penalty,
        "stress_threshold": profile.selection_score.stress_threshold,
        "staged_training": profile.selection_score.staged_training,
        "stage_iters": profile.selection_score.stage_iters,
    }


def serialize_model_config(profile) -> Dict[str, object]:
    return {
        "observation_window": int(profile.model.observation_window),
        "action_levels": [float(v) for v in profile.model.action_levels],
        "use_hybrid_overlay": bool(profile.model.use_hybrid_overlay),
        "delta_levels": [float(v) for v in profile.model.delta_levels],
        "hard_floor": None if profile.model.hard_floor is None else float(profile.model.hard_floor),
        "use_regime_router": bool(profile.model.use_regime_router),
        "hard_regime_routing": bool(profile.model.hard_regime_routing),
        "router_experts": int(profile.model.router_experts),
        "router_top_k": int(profile.model.router_top_k),
        "router_noise_std": float(profile.model.router_noise_std),
    }


def serialize_sampling_config(profile) -> Dict[str, object]:
    return {
        "regime_stratified": bool(profile.sampling.regime_stratified),
        "phase_balanced": bool(profile.sampling.phase_balanced),
        "mild_risk_threshold": float(profile.sampling.mild_risk_threshold),
        "severe_risk_threshold": float(profile.sampling.severe_risk_threshold),
        "mild_drawdown_threshold": float(profile.sampling.mild_drawdown_threshold),
        "severe_drawdown_threshold": float(profile.sampling.severe_drawdown_threshold),
        "mild_repeat": int(profile.sampling.mild_repeat),
        "severe_repeat": int(profile.sampling.severe_repeat),
        "persistent_stress_days": float(profile.sampling.persistent_stress_days),
        "trend_repeat": int(profile.sampling.trend_repeat),
        "rebound_repeat": int(profile.sampling.rebound_repeat),
        "episode_stride": int(profile.sampling.episode_stride),
        "bull_anchor_threshold": float(profile.sampling.bull_anchor_threshold),
        "bull_trend_threshold": float(profile.sampling.bull_trend_threshold),
        "bull_vol_threshold": float(profile.sampling.bull_vol_threshold),
        "bull_risk_threshold": float(profile.sampling.bull_risk_threshold),
        "rebound_anchor_ceiling": float(profile.sampling.rebound_anchor_ceiling),
        "rebound_trend_floor": float(profile.sampling.rebound_trend_floor),
        "rebound_risk_ceiling": float(profile.sampling.rebound_risk_ceiling),
        "rebound_risk_release": float(profile.sampling.rebound_risk_release),
    }


def feature_schema_payload() -> Dict[str, object]:
    return {
        "signature": feature_schema_signature(),
        "feature_count": len(FEATURE_COLUMNS),
        "extra_obs_dim": EXTRA_OBS_DIM,
        "n_assets": N_ASSETS,
    }


def artifact_tag_suffix(artifact_tag: str, use_moe: bool, ablate_vix: bool) -> str:
    moe_tag = "moe_novix" if (use_moe and ablate_vix) else ("moe" if use_moe else "")
    return f"{artifact_tag}_{moe_tag}" if (artifact_tag and moe_tag) else (artifact_tag or moe_tag)


def persist_actor_bundle(profile, actor, obs_dim: int, seed: int, effective_tag: str):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = model_artifact_path(profile.name, effective_tag)
    torch.save(
        {
            "profile": profile.name,
            "seed": seed,
            "hidden_sizes": list(profile.hidden_sizes),
            "obs_dim": obs_dim,
            "n_actions": int(getattr(actor, "output_dim", 0) or 11),
            "feature_schema": feature_schema_payload(),
            "position_cap": serialize_position_cap(profile),
            "selection_score": serialize_selection_score(profile),
            "model_config": serialize_model_config(profile),
            "sampling_config": serialize_sampling_config(profile),
            "state_dict": actor.state_dict(),
        },
        model_path,
    )
    return model_path


def persist_summary_bundle(
    profile,
    seed: int,
    effective_tag: str,
    model_path,
    valid_metrics,
    selection,
    slice_result,
    test_proxy,
    promotion_report,
):
    summary_path = summary_artifact_path(profile.name, effective_tag)
    summary_path.write_text(
        json.dumps(
            {
                "profile": profile.name,
                "seed": seed,
                "feature_schema": feature_schema_payload(),
                "position_cap": serialize_position_cap(profile),
                "selection_score": serialize_selection_score(profile),
                "model_config": serialize_model_config(profile),
                "sampling_config": serialize_sampling_config(profile),
                "model_path": str(model_path),
                "valid": valid_metrics,
                "best_score": selection["best_score"],
                "selection_history": selection["history"],
                "slices": slice_result,
                "test_proxy": test_proxy,
                "promotion": promotion_report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary_path


__all__ = [
    "artifact_tag_suffix",
    "feature_schema_payload",
    "persist_actor_bundle",
    "persist_summary_bundle",
    "serialize_model_config",
    "serialize_position_cap",
    "serialize_sampling_config",
    "serialize_selection_score",
]
