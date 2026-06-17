from __future__ import annotations

import json
from typing import Dict

import torch

from ..data import feature_columns_for_packs, feature_schema_signature
from ..settings import EXTRA_OBS_DIM, MODEL_DIR, N_ASSETS, model_artifact_path, summary_artifact_path
from .model_retention_gate import strict_retention_gate_status


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


def serialize_position_budget(profile) -> Dict[str, object]:
    return {
        "enabled": bool(profile.position_budget.enabled),
        "budget": float(profile.position_budget.budget),
        "regime_threshold": float(profile.position_budget.regime_threshold),
        "thresholds": [[float(v) for v in row] for row in profile.position_budget.thresholds],
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


def serialize_checkpoint_retention(profile) -> Dict[str, object]:
    cfg = profile.checkpoint_retention
    return {
        "enabled": bool(cfg.enabled),
        "keep_stage": bool(cfg.keep_stage),
        "keep_top_n": int(cfg.keep_top_n),
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
        "use_temporal_conv": bool(getattr(profile.model, "use_temporal_conv", False)),
        "temporal_conv_channels": int(getattr(profile.model, "temporal_conv_channels", 0)),
        "temporal_conv_kernel": int(getattr(profile.model, "temporal_conv_kernel", 0)),
        "use_gru": bool(profile.model.use_gru),
        "gru_hidden_size": int(profile.model.gru_hidden_size),
        "use_attention": bool(getattr(profile.model, "use_attention", False)),
        "attention_dim": int(getattr(profile.model, "attention_dim", 0)),
        "attention_heads": int(getattr(profile.model, "attention_heads", 0)),
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


def serialize_warm_start(profile) -> Dict[str, object]:
    ws = profile.warm_start
    return {
        "enabled": bool(ws.enabled),
        "steps": int(ws.steps),
        "lr": float(ws.lr),
        "batch_size": int(ws.batch_size),
        "sample_size": int(ws.sample_size),
    }


def feature_schema_payload(profile) -> Dict[str, object]:
    feature_columns = feature_columns_for_packs(profile.feature.packs, profile.name)
    return {
        "signature": feature_schema_signature(profile.feature.packs, profile.name),
        "feature_count": len(feature_columns),
        "extra_obs_dim": EXTRA_OBS_DIM,
        "n_assets": N_ASSETS,
    }


def artifact_tag_suffix(artifact_tag: str, use_moe: bool, ablate_vix: bool) -> str:
    moe_tag = "moe_novix" if (use_moe and ablate_vix) else ("moe" if use_moe else "")
    return f"{artifact_tag}_{moe_tag}" if (artifact_tag and moe_tag) else (artifact_tag or moe_tag)


def persist_actor_bundle(profile, actor, obs_dim: int, seed: int, effective_tag: str, state_dict_override=None):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = model_artifact_path(profile.name, effective_tag)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = state_dict_override if state_dict_override is not None else actor.state_dict()
    torch.save(
        {
            "profile": profile.name,
            "seed": seed,
            "hidden_sizes": list(profile.hidden_sizes),
            "obs_dim": obs_dim,
            "n_actions": int(getattr(actor, "output_dim", 0) or 11),
            "feature_schema": feature_schema_payload(profile),
            "position_cap": serialize_position_cap(profile),
            "position_budget": serialize_position_budget(profile),
            "selection_score": serialize_selection_score(profile),
            "checkpoint_retention": serialize_checkpoint_retention(profile),
            "model_config": serialize_model_config(profile),
            "sampling_config": serialize_sampling_config(profile),
            "warm_start": serialize_warm_start(profile),
            "state_dict": state_dict,
        },
        model_path,
    )
    return model_path


def _checkpoint_tag(effective_tag: str, iter_value: int) -> str:
    suffix = f"checkpoint_iter{int(iter_value)}"
    return f"{effective_tag}_{suffix}" if effective_tag else suffix


def checkpoint_retention_gate_payload(
    profile,
    retained_checkpoints: list[dict],
    retention_gate_evidence: dict | None = None,
) -> dict[str, object]:
    if not retained_checkpoints:
        return {
            "profile": profile.name,
            "status": "not_requested",
            "reason": "no retained checkpoint candidates",
        }
    if not bool(getattr(profile.checkpoint_retention, "enabled", False)):
        return {
            "profile": profile.name,
            "status": "disabled",
            "reason": "checkpoint retention disabled",
        }
    gate_ok, gate_reason = strict_retention_gate_status(
        retention_gate_evidence,
        profile_name=profile.name,
    )
    return {
        "profile": profile.name,
        "status": "passed" if gate_ok else "blocked",
        "reason": gate_reason,
    }


def final_model_retention_payload(
    *,
    profile,
    model_path,
    promotion_report: Dict[str, object],
    retention_gate_evidence: dict | None = None,
) -> dict[str, object]:
    """State whether the main actor is a final retained model.

    Training always writes a replayable actor for audit/backtest reproduction.
    This payload is the stricter human-facing retention decision: a model is not
    final-retained until fixed-split promotion, walk-forward promotion, and the
    SOXLBacktest/strict retention gate all pass.
    """

    walk_forward_evaluated = bool(promotion_report.get("walk_forward_evaluated", False))
    promotion_eligible = bool(promotion_report.get("eligible", False))
    payload: dict[str, object] = {
        "profile": profile.name,
        "model_path": str(model_path),
        "status": "blocked",
        "final_model_retained": False,
        "reason": "",
        "promotion_eligible": promotion_eligible,
        "walk_forward_evaluated": walk_forward_evaluated,
        "required_gates": ["fixed_split", "walk_forward", "strict_retention"],
    }
    if not walk_forward_evaluated:
        payload["reason"] = "walk-forward promotion gate not evaluated"
        return payload
    if not promotion_eligible:
        fixed_split = promotion_report.get("fixed_split")
        walk_forward = promotion_report.get("walk_forward")
        if isinstance(fixed_split, dict) and not bool(fixed_split.get("eligible", False)):
            payload["reason"] = "fixed-split promotion gate failed"
        elif isinstance(walk_forward, dict) and not bool(walk_forward.get("eligible", False)):
            payload["reason"] = "walk-forward promotion gate failed"
        else:
            payload["reason"] = "model promotion gate failed"
        return payload
    gate_ok, gate_reason = strict_retention_gate_status(
        retention_gate_evidence,
        profile_name=profile.name,
    )
    payload["reason"] = gate_reason
    if gate_ok:
        payload["status"] = "passed"
        payload["final_model_retained"] = True
    return payload


def persist_retained_checkpoint_bundles(
    profile,
    actor,
    obs_dim: int,
    seed: int,
    effective_tag: str,
    retained_checkpoints: list[dict],
    retention_gate_evidence: dict | None = None,
) -> list[dict]:
    records: list[dict] = []
    retention_gate_record = checkpoint_retention_gate_payload(
        profile,
        retained_checkpoints,
        retention_gate_evidence,
    )
    if retained_checkpoints and bool(getattr(profile.checkpoint_retention, "enabled", False)):
        gate_ok, _gate_reason = strict_retention_gate_status(
            retention_gate_evidence,
            profile_name=profile.name,
        )
        if not gate_ok:
            return []
    for checkpoint in retained_checkpoints:
        iter_value = int(checkpoint["iter"])
        path = persist_actor_bundle(
            profile=profile,
            actor=actor,
            obs_dim=obs_dim,
            seed=seed,
            effective_tag=_checkpoint_tag(effective_tag, iter_value),
            state_dict_override=checkpoint["state_dict"],
        )
        records.append(
            {
                "iter": iter_value,
                "score": float(checkpoint["score"]),
                "path": str(path),
                "reasons": list(checkpoint.get("reasons", [])),
                "metrics": dict(checkpoint.get("metrics", {})),
                **(
                    {"retention_gate": retention_gate_record}
                    if bool(getattr(profile.checkpoint_retention, "enabled", False))
                    else {}
                ),
            }
        )
    return records


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
    retained_checkpoints=None,
    checkpoint_retention_decision=None,
    final_model_retention_decision=None,
):
    retained_checkpoints = retained_checkpoints or []
    summary_path = summary_artifact_path(profile.name, effective_tag)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "profile": profile.name,
                "seed": seed,
                "feature_schema": feature_schema_payload(profile),
                "position_cap": serialize_position_cap(profile),
                "position_budget": serialize_position_budget(profile),
                "selection_score": serialize_selection_score(profile),
                "checkpoint_retention": serialize_checkpoint_retention(profile),
                "model_config": serialize_model_config(profile),
                "sampling_config": serialize_sampling_config(profile),
                "warm_start": serialize_warm_start(profile),
                "model_path": str(model_path),
                "retained_checkpoints": retained_checkpoints,
                "checkpoint_retention_decision": checkpoint_retention_decision,
                "final_model_retention_decision": final_model_retention_decision,
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


def persist_final_model_retention_decision(summary_path: Path, decision: dict[str, object]) -> None:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["final_model_retention_decision"] = decision
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_actor_bundle(profile_name: str):
    """Load a single actor bundle by profile name.

    Thin wrapper around `soxl_quant.runtime.load_actor_bundle` that lives in
    the `app` layer so scripts can import it without pulling in the full
    runtime/paper-execution surface.
    """
    from ..runtime import load_actor_bundle as _runtime_load

    return _runtime_load(profile_name)


def load_actor_bundles(profile_names: list[str]) -> list:
    """Load multiple actor bundles for ensemble evaluation.

    Returns a list of (actor, bundle_path, bundle) tuples.
    """
    return [load_actor_bundle(name) for name in profile_names]


def build_actor_ensemble(profile_names: list[str], method: str = "mean_logits"):
    del profile_names, method
    raise RuntimeError(
        "Legacy actor ensembles have been retired. Use the simplified supervised "
        "Alpha158/cadence model-selection artifacts instead."
    )


__all__ = [
    "artifact_tag_suffix",
    "build_actor_ensemble",
    "checkpoint_retention_gate_payload",
    "feature_schema_payload",
    "final_model_retention_payload",
    "load_actor_bundle",
    "load_actor_bundles",
    "persist_actor_bundle",
    "persist_final_model_retention_decision",
    "persist_retained_checkpoint_bundles",
    "persist_summary_bundle",
    "serialize_model_config",
    "serialize_checkpoint_retention",
    "serialize_position_cap",
    "serialize_sampling_config",
    "serialize_selection_score",
    "serialize_warm_start",
]
