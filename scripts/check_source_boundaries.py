#!/usr/bin/env python3
"""Validate source-of-truth boundaries between memory, experiments, runtime, production, and data."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
OLD_ACTIVE_WORKPAD = "codex_exec_plans/active/2026-04-10-soxl-agent-loop.md"
RETIRED_EVIDENCE_PREFIXES = (
    "evidence/",
    "evidence/post_train_review",
    "evidence/generalization",
    "evidence/diagnosis",
    "evidence/feature_analysis",
    "evidence/index.json",
)
OLD_DATA_PREFIXES = (
    "data/generalization",
    "data/multi_asset",
    "qlib_soxl_binary",
    "data/symbols",
    "data/external",
)


def iter_text_files(*roots: str) -> list[Path]:
    paths: list[Path] = []
    for root_name in roots:
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".md", ".yaml", ".yml", ".json", ".py", ".txt"}:
                paths.append(path)
    return paths


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def main() -> int:
    failures: list[str] = []

    if (ROOT / "evidence").exists():
        failures.append("top-level evidence/ directory should not exist; use runtime/<category>/")
    if (ROOT / "multi_asset").exists():
        failures.append("top-level multi_asset/ directory should not exist; use data/qlib and data/cache")
    if (ROOT / "docs" / "factor_research").exists():
        failures.append("docs/factor_research/ should not exist; generated factor outputs belong in runtime/factor/analysis")

    for path in iter_text_files("codex_exec_plans"):
        text = read(path)
        if OLD_ACTIVE_WORKPAD in text:
            failures.append(f"{path.relative_to(ROOT)} references old active workpad")

    for path in iter_text_files("experiments"):
        text = read(path)
        if "experiments/candidate/" in text and "/archive/" in str(path):
            failures.append(f"{path.relative_to(ROOT)} archive card references candidate path")
        for prefix in RETIRED_EVIDENCE_PREFIXES:
            if prefix in text:
                failures.append(f"{path.relative_to(ROOT)} references retired evidence path {prefix}")

    runtime = ROOT / "runtime"
    if runtime.exists():
        for child in runtime.iterdir():
            if child.name not in {"README.md", "index.json", "profile_registry.yaml", "soxl", "model", "data", "factor", "network", "execution", "multi_asset", "harness", "reward", "universes"}:
                failures.append(f"runtime contains non-runtime surface: {child.relative_to(ROOT)}")

    for path in iter_text_files("production"):
        text = read(path)
        for prefix in RETIRED_EVIDENCE_PREFIXES:
            if prefix in text:
                failures.append(f"{path.relative_to(ROOT)} references retired evidence path {prefix}")

    for path in iter_text_files("docs"):
        text = read(path)
        for prefix in RETIRED_EVIDENCE_PREFIXES:
            if prefix in text:
                failures.append(f"{path.relative_to(ROOT)} references retired evidence path {prefix}")

    for path in iter_text_files("scripts", "src", "production"):
        if path.name == "check_source_boundaries.py":
            continue
        text = read(path)
        for prefix in OLD_DATA_PREFIXES:
            if prefix in text:
                failures.append(f"{path.relative_to(ROOT)} references old data path {prefix}")

    for path in (ROOT / "data" / "qlib").glob("*/qlib.yaml"):
        text = read(path)
        for prefix in OLD_DATA_PREFIXES:
            if prefix in text:
                failures.append(f"{path.relative_to(ROOT)} embeds old provider path {prefix}")

    universe_path = ROOT / "runtime" / "universes" / "3x_etf_core.yaml"
    if not universe_path.exists():
        failures.append("missing runtime/universes/3x_etf_core.yaml")

    if failures:
        print("check_source_boundaries: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("check_source_boundaries: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
