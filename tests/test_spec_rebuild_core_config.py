from __future__ import annotations

import pytest

import src.core as core


def test_run_config_validates_active_surface_and_manifest_payload():
    assert hasattr(core, "RunConfig")
    config = core.RunConfig(
        name="spec_rebuild_foundation_v1",
        surface="research",
        qlib_uri="qlib://soxl/day",
        as_of="2026-06-09",
        output_dir="research/spec_rebuild_foundation_v1",
        artifacts=["summary.json", "analysis_summary.json"],
    )

    payload = config.to_manifest()

    assert payload["schema_version"] == "1.0"
    assert payload["name"] == "spec_rebuild_foundation_v1"
    assert payload["surface"] == "research"
    assert payload["qlib_uri"] == "qlib://soxl/day"
    assert payload["artifacts"] == ["analysis_summary.json", "summary.json"]
    assert payload["config_id"] == config.config_id


def test_run_config_rejects_skill_and_retired_output_surfaces():
    with pytest.raises(ValueError, match="active surface"):
        core.RunConfig(
            name="bad",
            surface="skill",
            qlib_uri="qlib://soxl/day",
            as_of="2026-06-09",
            output_dir=".codex/skills/soxl-qlib-experiment",
            artifacts=["SKILL.md"],
        )

    with pytest.raises(ValueError, match="retired"):
        core.RunConfig(
            name="bad-dashboard",
            surface="research",
            qlib_uri="qlib://soxl/day",
            as_of="2026-06-09",
            output_dir="dashboard_data",
            artifacts=["latest.json"],
        )
