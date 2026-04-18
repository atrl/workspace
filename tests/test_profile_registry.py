from __future__ import annotations

from src.soxl_quant.config import experiment_dir_for_profile, list_profile_names, resolve_profile_path
from src.soxl_quant.registry import load_profile_registry, validate_profile_registry


def test_registry_covers_all_profiles():
    registry = load_profile_registry()
    assert set(registry) == set(list_profile_names())


def test_registry_has_single_canonical_per_family():
    issues = validate_profile_registry()
    assert not [issue for issue in issues if issue.startswith("multiple canonical profiles")]


def test_dropped_profiles_do_not_keep_actors():
    registry = load_profile_registry()
    offenders = [name for name, entry in registry.items() if entry.status == "dropped" and entry.keep_actor]
    assert offenders == []


def test_migrated_profiles_resolve_to_experiment_dirs():
    assert resolve_profile_path("balanced").as_posix().endswith("experiments/canonical/balanced/profile.yaml")
    assert resolve_profile_path("balanced_checkpoint_review").as_posix().endswith(
        "experiments/candidate/balanced_checkpoint_review/profile.yaml"
    )
    assert resolve_profile_path("smoke").as_posix().endswith("experiments/canonical/smoke/profile.yaml")


def test_experiment_dir_lookup_for_migrated_profiles():
    assert experiment_dir_for_profile("balanced").as_posix().endswith("experiments/canonical/balanced")
    assert experiment_dir_for_profile("smoke").as_posix().endswith("experiments/canonical/smoke")


def test_profile_listing_keeps_migrated_and_legacy_profiles():
    names = set(list_profile_names())
    assert "balanced" in names
    assert "balanced_checkpoint_review" in names
    assert "smoke" in names
    assert "aggressive" in names
