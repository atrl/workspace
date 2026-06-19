from __future__ import annotations

from pathlib import Path


def test_rebuild_uses_spec_active_surfaces():
    root = Path(__file__).resolve().parents[1]

    active_paths = ["data", "src", "research", "production", "tests"]
    for rel_path in active_paths:
        assert (root / rel_path).exists(), f"{rel_path} should be active for the qlib rebuild"

    retired_paths = ["dashboard.html", "dashboard_data", "web", "scripts", "codex_exec_plans"]
    for rel_path in retired_paths:
        assert not (root / rel_path).exists(), f"{rel_path} should not be a top-level active surface"


def test_remaining_skills_are_minimal_and_qlib_or_data_focused():
    root = Path(__file__).resolve().parents[1]
    skills = sorted(path.parent.name for path in (root / ".codex" / "skills").glob("*/SKILL.md"))

    assert skills == ["futu-api", "soxl-qlib-experiment"]


def test_legacy_runtime_keeps_only_cleanup_and_current_experiment_cards():
    root = Path(__file__).resolve().parents[1]
    runtime_cards = sorted(path.relative_to(root).as_posix() for path in (root / "runtime").glob("*/*/*"))

    assert runtime_cards == [
        "runtime/harness/card/project_cleanup_architecture_v1",
        "runtime/model_selection/card/soxlbacktest_qlib_reference_v1",
    ]
