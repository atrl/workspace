from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from src.soxl_quant.app import artifacts as artifact_module
from src.soxl_quant.app.artifacts import (
    final_model_retention_payload,
    persist_final_model_retention_decision,
    persist_retained_checkpoint_bundles,
)
from src.soxl_quant.app.checkpoint_retention import select_retained_checkpoints
from src.soxl_quant.config import CheckpointRetentionConfig, load_profile


class TinyActor(torch.nn.Module):
    output_dim = 1

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))


def test_checkpoint_retention_config_defaults_disabled():
    profile = load_profile("balanced")

    assert profile.checkpoint_retention == CheckpointRetentionConfig()
    assert profile.checkpoint_retention.enabled is False
    assert profile.checkpoint_retention.keep_stage is False
    assert profile.checkpoint_retention.keep_top_n == 0


def test_select_retained_checkpoints_keeps_top_n_without_stage_spam():
    profile = replace(
        load_profile("balanced"),
        checkpoint_retention=CheckpointRetentionConfig(enabled=True, keep_stage=False, keep_top_n=2),
    )
    candidates = [
        {"iter": 0, "score": 1.0, "state_dict": {"w": 0}},
        {"iter": 20, "score": 3.0, "state_dict": {"w": 20}},
        {"iter": 40, "score": 2.0, "state_dict": {"w": 40}},
    ]

    retained = select_retained_checkpoints(candidates, profile)

    assert [entry["iter"] for entry in retained] == [20, 40]
    assert retained[0]["reasons"] == ["top_n"]
    assert retained[1]["reasons"] == ["top_n"]


def test_select_retained_checkpoints_stage_union_marks_top_n_reason():
    profile = replace(
        load_profile("balanced"),
        checkpoint_retention=CheckpointRetentionConfig(enabled=True, keep_stage=True, keep_top_n=1),
    )
    candidates = [
        {"iter": 0, "score": 1.0, "state_dict": {"w": 0}},
        {"iter": 20, "score": 3.0, "state_dict": {"w": 20}},
        {"iter": 40, "score": 2.0, "state_dict": {"w": 40}},
    ]

    retained = select_retained_checkpoints(candidates, profile)

    assert [entry["iter"] for entry in retained] == [0, 20, 40]
    assert retained[1]["reasons"] == ["stage", "top_n"]


def test_persist_actor_bundle_can_save_retained_state_without_mutating_actor():
    profile = load_profile("balanced")
    actor = TinyActor()
    retained_state = {"weight": torch.tensor([2.0])}

    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "retained_actor.pt"
        original_model_artifact_path = artifact_module.model_artifact_path
        artifact_module.model_artifact_path = lambda profile_name, artifact_tag="": output_path
        try:
            saved_path = artifact_module.persist_actor_bundle(
                profile,
                actor,
                obs_dim=3,
                seed=123,
                effective_tag="checkpoint_iter20",
                state_dict_override=retained_state,
            )
        finally:
            artifact_module.model_artifact_path = original_model_artifact_path

        bundle = torch.load(saved_path, map_location="cpu", weights_only=False)

    assert float(bundle["state_dict"]["weight"][0]) == 2.0
    assert float(actor.state_dict()["weight"][0]) == 1.0


def test_persist_summary_bundle_records_retained_checkpoint_paths():
    profile = load_profile("balanced")

    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        original_summary_artifact_path = artifact_module.summary_artifact_path
        artifact_module.summary_artifact_path = lambda profile_name, artifact_tag="": output_dir / "summary.json"
        try:
            summary_path = artifact_module.persist_summary_bundle(
                profile=profile,
                seed=12345,
                effective_tag="unit",
                model_path=output_dir / "dummy_actor.pt",
                valid_metrics={"return": 1.0, "sharpe": 0.5, "max_dd": 2.0},
                selection={"best_score": 3.0, "history": [{"iter": 20, "score": 3.0}]},
                slice_result={"test_2022": {"return": 1.0}},
                test_proxy={"return": 1.0, "sharpe": 0.5},
                promotion_report={"accepted": False, "reasons": ["unit-test"]},
                retained_checkpoints=[
                    {
                        "iter": 20,
                        "score": 3.0,
                        "path": "runtime/model/card/balanced/soxl_balanced_checkpoint_iter20_actor.pt",
                        "reasons": ["top_n"],
                    }
                ],
            )
        finally:
            artifact_module.summary_artifact_path = original_summary_artifact_path

        payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert payload["retained_checkpoints"] == [
        {
            "iter": 20,
            "score": 3.0,
            "path": "runtime/model/card/balanced/soxl_balanced_checkpoint_iter20_actor.pt",
            "reasons": ["top_n"],
        }
    ]


def test_persist_retained_checkpoint_bundles_writes_actor_files_and_summary_records():
    profile = load_profile("balanced")
    actor = TinyActor()
    retained = [
        {
            "iter": 20,
            "score": 3.0,
            "metrics": {"return": 1.0},
            "state_dict": {"weight": torch.tensor([2.0])},
            "reasons": ["top_n"],
        }
    ]

    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        original_model_artifact_path = artifact_module.model_artifact_path
        artifact_module.model_artifact_path = lambda profile_name, artifact_tag="": output_dir / (
            f"{profile_name}_{artifact_tag}_actor.pt"
        )
        try:
            records = persist_retained_checkpoint_bundles(
                profile=profile,
                actor=actor,
                obs_dim=3,
                seed=123,
                effective_tag="unit",
                retained_checkpoints=retained,
            )
        finally:
            artifact_module.model_artifact_path = original_model_artifact_path

        saved_path = Path(records[0]["path"])
        bundle = torch.load(saved_path, map_location="cpu", weights_only=False)

    assert saved_path.name == "balanced_unit_checkpoint_iter20_actor.pt"
    assert float(bundle["state_dict"]["weight"][0]) == 2.0
    assert records == [
        {
            "iter": 20,
            "score": 3.0,
            "path": str(saved_path),
            "reasons": ["top_n"],
            "metrics": {"return": 1.0},
        }
    ]


def test_persist_retained_checkpoint_bundles_blocks_enabled_retention_without_strict_gate():
    profile = replace(
        load_profile("balanced"),
        checkpoint_retention=CheckpointRetentionConfig(enabled=True, keep_stage=True, keep_top_n=1),
    )
    actor = TinyActor()
    retained = [
        {
            "iter": 20,
            "score": 3.0,
            "metrics": {"return": 1.0},
            "state_dict": {"weight": torch.tensor([2.0])},
            "reasons": ["top_n"],
        }
    ]

    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        original_model_artifact_path = artifact_module.model_artifact_path
        artifact_module.model_artifact_path = lambda profile_name, artifact_tag="": output_dir / (
            f"{profile_name}_{artifact_tag}_actor.pt"
        )
        try:
            records = persist_retained_checkpoint_bundles(
                profile=profile,
                actor=actor,
                obs_dim=3,
                seed=123,
                effective_tag="unit",
                retained_checkpoints=retained,
            )
        finally:
            artifact_module.model_artifact_path = original_model_artifact_path

    assert records == []
    assert not list(output_dir.glob("*.pt"))


def _strict_gate_pass_evidence(profile: str = "balanced") -> dict[str, object]:
    return {
        "profile": profile,
        "status": "candidate",
        "source_path": f"runtime/model/card/{profile}/{profile}_soxl_daily.csv",
        "ytd_return_pct": 220.0,
        "ytd_max_drawdown_pct": 12.0,
        "current_best_return_pct": 220.0,
        "required_best_return_pct": 209.0,
        "required_soxlbacktest_return_pct": 146.0,
        "wf_windows_present": "wf_2020,wf_2022,wf_2023_2024",
        "worst_wf_max_drawdown_pct": 19.0,
        "missing_wf_window_count": 0,
        "wf_drawdown_gap_pct": 0.0,
        "best_return_gap_pct": 0.0,
        "soxlbacktest_return_gap_pct": 0.0,
        "gate_pass": True,
        "retention_decision": "retain_promotable",
        "failures": [],
    }


def test_persist_retained_checkpoint_bundles_allows_explicit_strict_gate_pass():
    profile = replace(
        load_profile("balanced"),
        checkpoint_retention=CheckpointRetentionConfig(enabled=True, keep_stage=True, keep_top_n=1),
    )
    actor = TinyActor()
    retained = [
        {
            "iter": 20,
            "score": 3.0,
            "metrics": {"return": 1.0},
            "state_dict": {"weight": torch.tensor([2.0])},
            "reasons": ["top_n"],
        }
    ]
    retention_gate_evidence = _strict_gate_pass_evidence()

    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        original_model_artifact_path = artifact_module.model_artifact_path
        artifact_module.model_artifact_path = lambda profile_name, artifact_tag="": output_dir / (
            f"{profile_name}_{artifact_tag}_actor.pt"
        )
        try:
            records = persist_retained_checkpoint_bundles(
                profile=profile,
                actor=actor,
                obs_dim=3,
                seed=123,
                effective_tag="unit",
                retained_checkpoints=retained,
                retention_gate_evidence=retention_gate_evidence,
            )
        finally:
            artifact_module.model_artifact_path = original_model_artifact_path

        saved_path = Path(records[0]["path"])

    assert saved_path.name == "balanced_unit_checkpoint_iter20_actor.pt"
    assert records[0]["retention_gate"] == {
        "profile": "balanced",
        "status": "passed",
        "reason": "strict retention gate passed",
    }


def test_final_model_retention_blocks_without_walk_forward_evaluation():
    profile = load_profile("balanced")

    decision = final_model_retention_payload(
        profile=profile,
        model_path=Path("runtime/model/card/balanced/soxl_balanced_actor.pt"),
        promotion_report={
            "eligible": False,
            "fixed_split": {"eligible": True, "checks": {}},
            "walk_forward": {"eligible": False, "status": "missing", "checks": {}},
            "walk_forward_evaluated": False,
        },
    )

    assert decision == {
        "profile": "balanced",
        "model_path": "runtime/model/card/balanced/soxl_balanced_actor.pt",
        "status": "blocked",
        "final_model_retained": False,
        "reason": "walk-forward promotion gate not evaluated",
        "promotion_eligible": False,
        "walk_forward_evaluated": False,
        "required_gates": ["fixed_split", "walk_forward", "strict_retention"],
    }


def test_final_model_retention_distinguishes_failed_walk_forward_from_missing():
    profile = load_profile("balanced")

    decision = final_model_retention_payload(
        profile=profile,
        model_path=Path("actor.pt"),
        promotion_report={
            "eligible": False,
            "fixed_split": {"eligible": True, "checks": {}},
            "walk_forward": {
                "eligible": False,
                "checks": {"wf_2023_2024_return": {"passed": False, "actual": -1.0, "limit": 0.0}},
            },
            "walk_forward_evaluated": True,
        },
    )

    assert decision["status"] == "blocked"
    assert decision["final_model_retained"] is False
    assert decision["reason"] == "walk-forward promotion gate failed"
    assert decision["walk_forward_evaluated"] is True


def test_final_model_retention_requires_strict_gate_after_promotion_pass():
    profile = load_profile("balanced")

    decision = final_model_retention_payload(
        profile=profile,
        model_path=Path("actor.pt"),
        promotion_report={
            "eligible": True,
            "fixed_split": {"eligible": True, "checks": {}},
            "walk_forward": {"eligible": True, "checks": {}},
            "walk_forward_evaluated": True,
        },
    )

    assert decision["status"] == "blocked"
    assert decision["final_model_retained"] is False
    assert decision["reason"] == "strict retention gate evidence missing"


def test_final_model_retention_passes_only_with_strict_gate_evidence():
    profile = load_profile("balanced")

    decision = final_model_retention_payload(
        profile=profile,
        model_path=Path("actor.pt"),
        promotion_report={
            "eligible": True,
            "fixed_split": {"eligible": True, "checks": {}},
            "walk_forward": {"eligible": True, "checks": {}},
            "walk_forward_evaluated": True,
        },
        retention_gate_evidence=_strict_gate_pass_evidence(),
    )

    assert decision == {
        "profile": "balanced",
        "model_path": "actor.pt",
        "status": "passed",
        "final_model_retained": True,
        "reason": "strict retention gate passed",
        "promotion_eligible": True,
        "walk_forward_evaluated": True,
        "required_gates": ["fixed_split", "walk_forward", "strict_retention"],
    }


def test_final_model_retention_blocks_weak_gate_pass_without_walkforward_fields():
    profile = load_profile("balanced")

    decision = final_model_retention_payload(
        profile=profile,
        model_path=Path("actor.pt"),
        promotion_report={
            "eligible": True,
            "fixed_split": {"eligible": True, "checks": {}},
            "walk_forward": {"eligible": True, "checks": {}},
            "walk_forward_evaluated": True,
        },
        retention_gate_evidence={"profile": "balanced", "gate_pass": True},
    )

    assert decision["status"] == "blocked"
    assert decision["final_model_retained"] is False
    assert "strict retention gate missing required fields" in decision["reason"]


def test_final_model_retention_blocks_soxlbacktest_reference_as_actor():
    profile = replace(load_profile("balanced"), name="soxl_backtest_policy_reimplementation_v1")
    evidence = _strict_gate_pass_evidence("soxl_backtest_policy_reimplementation_v1")
    evidence["status"] = "reference"
    evidence["retention_decision"] = "keep_reference_baseline"

    decision = final_model_retention_payload(
        profile=profile,
        model_path=Path("reference_actor.pt"),
        promotion_report={
            "eligible": True,
            "fixed_split": {"eligible": True, "checks": {}},
            "walk_forward": {"eligible": True, "checks": {}},
            "walk_forward_evaluated": True,
        },
        retention_gate_evidence=evidence,
    )

    assert decision["status"] == "blocked"
    assert decision["final_model_retained"] is False
    assert decision["reason"] == "SOXLBacktest reference baseline cannot be retained as actor"


def test_persist_final_model_retention_decision_updates_summary():
    with TemporaryDirectory() as tmpdir:
        summary = Path(tmpdir) / "summary.json"
        summary.write_text('{"profile": "balanced", "promotion": {"eligible": false}}', encoding="utf-8")
        decision = {
            "profile": "balanced",
            "status": "blocked",
            "final_model_retained": False,
            "reason": "walk-forward promotion gate failed",
        }

        persist_final_model_retention_decision(summary, decision)

        payload = json.loads(summary.read_text(encoding="utf-8"))

    assert payload["final_model_retention_decision"] == decision
