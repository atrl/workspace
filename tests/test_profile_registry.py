from __future__ import annotations

from pathlib import Path

from src.soxl_quant.config import experiment_dir_for_profile, list_profile_names, resolve_profile_path
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
