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

    candidates = discover_replay_candidates(tmp_path)

    assert [candidate.name for candidate in candidates] == ["fresh"]
    assert candidates[0].output_path == (
        tmp_path / "runtime" / "model" / "card" / "fresh" / "fresh_soxl_daily.csv"
    )


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
