from __future__ import annotations

from src.soxl_quant.config import list_profile_names
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
