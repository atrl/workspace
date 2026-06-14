from __future__ import annotations

import json
from pathlib import Path

from scripts.refresh_dashboard_model_replays import (
    ReplayCandidate,
    csv_last_date,
    discover_replay_candidates,
    refresh_model_replays,
)


def _write_registry(root: Path, rows: list[dict]) -> None:
    registry = root / "runtime" / "harness" / "card" / "registry"
    registry.mkdir(parents=True)
    registry_rows = "\n".join(
        "\n".join(
            [
                f"- name: {row['name']}",
                f"  status: {row.get('status', 'candidate')}",
                f"  keep_actor: {str(row.get('keep_actor', True)).lower()}",
                f"  keep_summary: {str(row.get('keep_summary', True)).lower()}",
            ]
        )
        for row in rows
    )
    (registry / "profile.yaml").write_text(f"profiles:\n{registry_rows}\n", encoding="utf-8")


def _write_retention_gate(root: Path, rows: list[dict]) -> None:
    path = root / "runtime" / "model_selection" / "card" / "model_retention_walkforward_gate_v1"
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")


def _profile(root: Path, name: str, *, actor: bool = True, profile_yaml: bool = True) -> None:
    card = root / "runtime" / "model" / "card" / name
    card.mkdir(parents=True)
    if actor:
        (card / f"soxl_{name}_actor.pt").write_bytes(b"actor")
    if profile_yaml:
        (card / "profile.yaml").write_text(f"name: {name}\n", encoding="utf-8")


def _daily_csv(path: Path, end: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "datetime,strategy_nav,price,buyhold_nav,position,excess,excess_pct,action",
                "2026-05-19,1000000,100,1000000,0,0,0,HOLD",
                f"{end},1000000,100,1000000,0,0,0,HOLD",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_discover_replay_candidates_requires_registry_actor_and_profile(tmp_path: Path):
    _write_registry(
        tmp_path,
        [
            {"name": "fresh", "status": "canonical", "keep_actor": True},
            {"name": "dropped", "status": "dropped", "keep_actor": True},
            {"name": "no_actor", "status": "candidate", "keep_actor": True},
            {"name": "no_profile", "status": "candidate", "keep_actor": True},
            {"name": "summary_only", "status": "candidate", "keep_actor": False},
        ],
    )
    _profile(tmp_path, "fresh")
    _profile(tmp_path, "dropped")
    _profile(tmp_path, "no_actor", actor=False)
    _profile(tmp_path, "no_profile", profile_yaml=False)
    _profile(tmp_path, "summary_only")
    _write_retention_gate(
        tmp_path,
        [
            {"profile": "fresh", "gate_pass": True, "retention_decision": "retain_promotable", "failures": []},
            {"profile": "no_actor", "gate_pass": True, "retention_decision": "retain_promotable", "failures": []},
            {"profile": "no_profile", "gate_pass": True, "retention_decision": "retain_promotable", "failures": []},
        ],
    )

    candidates = discover_replay_candidates(tmp_path)

    assert [candidate.name for candidate in candidates] == ["fresh", "no_actor", "no_profile"]
    assert candidates[0].output_path == (
        tmp_path / "runtime" / "model" / "card" / "fresh" / "fresh_soxl_daily.csv"
    )
    assert candidates[0].skip_reason == ""
    assert candidates[1].skip_reason == "missing_actor_artifact"
    assert candidates[2].skip_reason == "missing_profile_yaml"


def test_discover_replay_candidates_requires_strict_retention_gate(tmp_path: Path):
    _write_registry(
        tmp_path,
        [
            {"name": "passed", "status": "canonical", "keep_actor": True},
            {"name": "failed", "status": "candidate", "keep_actor": True},
            {"name": "missing_gate_row", "status": "candidate", "keep_actor": True},
        ],
    )
    _profile(tmp_path, "passed")
    _profile(tmp_path, "failed")
    _profile(tmp_path, "missing_gate_row")
    _write_retention_gate(
        tmp_path,
        [
            {"profile": "passed", "gate_pass": True, "retention_decision": "retain_promotable", "failures": []},
            {
                "profile": "failed",
                "gate_pass": False,
                "retention_decision": "retain_as_evidence_only",
                "failures": ["wf_max_drawdown_ge_20:wf_2022", "ytd_return_below_best_alignment"],
            },
            {
                "profile": "soxl_backtest_policy_reimplementation_v1",
                "gate_pass": True,
                "retention_decision": "keep_reference_baseline",
                "failures": [],
            },
        ],
    )

    candidates = discover_replay_candidates(tmp_path)

    by_name = {candidate.name: candidate for candidate in candidates}
    assert by_name["passed"].skip_reason == ""
    assert by_name["failed"].skip_reason == (
        "strict_retention_gate_failed:wf_max_drawdown_ge_20:wf_2022,ytd_return_below_best_alignment"
    )
    assert by_name["missing_gate_row"].skip_reason == "missing_strict_retention_gate_row"


def test_discover_replay_candidates_fails_closed_without_retention_gate_summary(tmp_path: Path):
    _write_registry(tmp_path, [{"name": "retained", "status": "canonical", "keep_actor": True}])
    _profile(tmp_path, "retained")

    candidates = discover_replay_candidates(tmp_path)

    assert [candidate.name for candidate in candidates] == ["retained"]
    assert candidates[0].skip_reason == "missing_strict_retention_gate"


def test_refresh_model_replays_exports_only_stale_profiles(tmp_path: Path):
    current = ReplayCandidate(
        name="current",
        status="canonical",
        actor_path=tmp_path / "runtime/model/card/current/soxl_current_actor.pt",
        profile_path=tmp_path / "runtime/model/card/current/profile.yaml",
        output_path=tmp_path / "runtime/model/card/current/current_soxl_daily.csv",
    )
    stale = ReplayCandidate(
        name="stale",
        status="candidate",
        actor_path=tmp_path / "runtime/model/card/stale/soxl_stale_actor.pt",
        profile_path=tmp_path / "runtime/model/card/stale/profile.yaml",
        output_path=tmp_path / "runtime/model/card/stale/stale_soxl_daily.csv",
    )
    _daily_csv(current.output_path, "2026-05-21")
    _daily_csv(stale.output_path, "2026-05-20")
    calls: list[str] = []

    def fake_export(candidate: ReplayCandidate) -> None:
        calls.append(candidate.name)
        _daily_csv(candidate.output_path, "2026-05-21")

    manifest = refresh_model_replays(
        tmp_path,
        [current, stale],
        latest_data_date="2026-05-22",
        safe_backtest_end="2026-05-21",
        exporter=fake_export,
    )

    assert calls == ["stale"]
    assert manifest["latest_data_date"] == "2026-05-22"
    assert manifest["safe_backtest_end"] == "2026-05-21"
    assert manifest["counts"] == {"current": 1, "refreshed": 1, "failed": 0, "skipped": 0}
    assert csv_last_date(stale.output_path) == "2026-05-21"

    persisted = json.loads((tmp_path / "dashboard_data" / "model_replay_freshness.json").read_text())
    assert persisted["profiles"][0]["result"] == "current"
    assert persisted["profiles"][1]["result"] == "refreshed"


def test_refresh_model_replays_reports_skipped_keep_actor_profiles(tmp_path: Path):
    missing_actor = ReplayCandidate(
        name="missing_actor",
        status="canonical",
        actor_path=tmp_path / "runtime/model/card/missing_actor/soxl_missing_actor_actor.pt",
        profile_path=tmp_path / "runtime/model/card/missing_actor/profile.yaml",
        output_path=tmp_path / "runtime/model/card/missing_actor/missing_actor_soxl_daily.csv",
        skip_reason="missing_actor_artifact",
    )

    manifest = refresh_model_replays(
        tmp_path,
        [missing_actor],
        latest_data_date="2026-05-22",
        safe_backtest_end="2026-05-21",
        exporter=lambda _candidate: (_ for _ in ()).throw(AssertionError("skipped profiles must not export")),
    )

    assert manifest["counts"] == {"current": 0, "refreshed": 0, "failed": 0, "skipped": 1}
    assert manifest["profiles"] == [
        {
            "profile": "missing_actor",
            "status": "canonical",
            "actor": "runtime/model/card/missing_actor/soxl_missing_actor_actor.pt",
            "profile_yaml": "runtime/model/card/missing_actor/profile.yaml",
            "output": "runtime/model/card/missing_actor/missing_actor_soxl_daily.csv",
            "before_end": "",
            "after_end": "",
            "result": "skipped",
            "reason": "missing_actor_artifact",
        }
    ]
