from __future__ import annotations

from pathlib import Path

from src.soxl_quant.config import experiment_dir_for_profile, list_profile_names, load_profile, resolve_profile_path
from src.soxl_quant.registry import REGISTRY_PATH, load_profile_registry, validate_profile_registry


def test_registry_covers_all_profiles():
    registry = load_profile_registry()
    assert set(list_profile_names()).issubset(set(registry))


def test_registry_uses_repo_canonical_card_path():
    assert REGISTRY_PATH.as_posix().endswith("runtime/harness/card/registry/profile.yaml")
    assert REGISTRY_PATH.exists()


def test_registry_has_single_canonical_per_family():
    issues = validate_profile_registry()
    assert not [issue for issue in issues if issue.startswith("multiple canonical profiles")]


def test_dropped_profiles_do_not_keep_actors():
    registry = load_profile_registry()
    offenders = [name for name, entry in registry.items() if entry.status == "dropped" and entry.keep_actor]
    assert offenders == []


def test_blocked_reward_diagnostics_do_not_keep_dashboard_actors():
    registry = load_profile_registry()
    blocked_reward_diagnostics = [
        "drawdown_rebound_reward_train_v1",
        "mode_conditioned_rebound_reward_train_v2",
        "state_conditioned_drawdown_rebound_reward_train_v1",
    ]

    assert all(registry[name].status == "ablation" for name in blocked_reward_diagnostics)
    assert all(registry[name].keep_actor is False for name in blocked_reward_diagnostics)


def test_blocked_soxx_actor_feature_profile_is_not_active_candidate():
    registry = load_profile_registry()
    entry = registry["balanced_rule_residual_v22_soxx"]

    assert entry.status != "candidate"
    assert entry.keep_actor is False
    assert entry.notes_ref.endswith("runtime/model/card/balanced_rule_residual_v22_soxx/notes.md")


def test_legacy_profiles_resolve_to_runtime_card_dirs():
    assert resolve_profile_path("balanced").as_posix().endswith("runtime/model/card/balanced/profile.yaml")
    assert resolve_profile_path("smoke").as_posix().endswith("runtime/harness/card/smoke/profile.yaml")


def test_experiment_dir_lookup_is_optional_for_legacy_profiles():
    assert experiment_dir_for_profile("balanced") is None
    assert experiment_dir_for_profile("smoke") is None


def test_profile_listing_keeps_migrated_and_legacy_profiles():
    names = set(list_profile_names())
    assert "balanced" in names
    assert "smoke" in names
    assert "aggressive" in names
    assert "registry" not in names


def test_candidate_directory_stays_narrow():
    candidate_root = Path("experiments/candidate")
    if not candidate_root.exists():
        return
    registry = load_profile_registry()
    candidate_dirs = sorted(path.name for path in candidate_root.iterdir() if path.is_dir())
    assert candidate_dirs != []
    assert all(registry[name].status == "candidate" for name in candidate_dirs)


def test_archive_directory_contains_only_non_active_packages():
    registry = load_profile_registry()
    archive_root = Path("experiments/archive")
    if not archive_root.exists():
        return
    archive_dirs = sorted(path.name for path in archive_root.iterdir() if path.is_dir())
    assert all(registry[name].status != "candidate" for name in archive_dirs)


def test_state_conditioned_multi_objective_candidate_is_pre_promotion_only():
    name = "state_conditioned_multi_objective_ytd_gate_v1"
    registry = load_profile_registry()
    entry = registry[name]
    profile = load_profile(name)

    assert entry.status == "candidate"
    assert entry.baseline == "mode_conditioned_rebound_reward_train_v2"
    assert entry.keep_actor is False
    assert entry.keep_summary is True
    assert entry.notes_ref.endswith(
        "runtime/model_selection/card/state_conditioned_multi_objective_ytd_gate_v1/"
        "state_conditioned_multi_objective_ytd_gate_v1.ipynb"
    )
    assert profile.reward.state_conditioned_reward_enabled is True
    assert profile.reward.drawdown_drift_reward_enabled is True
    assert profile.reward.trend_preservation_floor_enabled is True
    assert profile.feature.packs == ("vix_term", "transition_signals", "soxx_spread_oscillation")
    assert profile.promotion_gate.max_test_2022_drawdown <= 20.0
    assert profile.promotion_gate.min_wf_relative_outperformance >= 30.0
    assert profile.selection_score.use_pareto is True
    assert profile.checkpoint_retention.enabled is True
