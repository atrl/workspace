from __future__ import annotations

from pathlib import Path


def test_cleanup_removes_retired_active_surfaces():
    root = Path(__file__).resolve().parents[1]

    retired_paths = [
        "production",
        "dashboard.html",
        "dashboard_data",
        "web",
        "src",
        "scripts",
        "codex_exec_plans",
    ]
    for rel_path in retired_paths:
        assert not (root / rel_path).exists(), f"{rel_path} should be retired from the clean architecture"


def test_remaining_skills_are_minimal_and_qlib_or_data_focused():
    root = Path(__file__).resolve().parents[1]
    skills = sorted(path.parent.name for path in (root / ".codex" / "skills").glob("*/SKILL.md"))

    assert skills == ["futu-api", "soxl-qlib-experiment"]


def test_runtime_keeps_only_cleanup_and_current_experiment_cards():
    root = Path(__file__).resolve().parents[1]
    runtime_cards = sorted(path.relative_to(root).as_posix() for path in (root / "runtime").glob("*/*/*"))

    assert runtime_cards == [
        "runtime/harness/card/project_cleanup_architecture_v1",
        "runtime/model_selection/card/soxlbacktest_qlib_reference_v1",
    ]
