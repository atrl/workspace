from __future__ import annotations

from pathlib import Path

import scripts.check_source_boundaries as source_boundaries


ROOT = Path(__file__).resolve().parents[1]
RETIRED_WORKSPACE = "/Users/rziiiii/.openclaw/workspace"


def test_scripts_do_not_hardcode_retired_workspace_path():
    offenders: list[str] = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        if path.name == "check_source_boundaries.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if RETIRED_WORKSPACE in text:
            offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []


def test_source_boundary_checker_rejects_retired_workspace_path(tmp_path, monkeypatch, capsys):
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "stale.py").write_text(f'ROOT = "{RETIRED_WORKSPACE}"\n', encoding="utf-8")
    universe_dir = tmp_path / "runtime" / "data" / "universes"
    universe_dir.mkdir(parents=True)
    (universe_dir / "3x_etf_core.yaml").write_text("symbols: []\n", encoding="utf-8")

    monkeypatch.setattr(source_boundaries, "ROOT", tmp_path)

    assert source_boundaries.main() == 1
    assert "retired workspace path" in capsys.readouterr().out
