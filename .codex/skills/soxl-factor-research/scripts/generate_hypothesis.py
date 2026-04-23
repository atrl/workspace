"""Generate factor hypothesis cards from rulebase and external sources.

Usage:
    qlib_venv/bin/python .codex/skills/soxl-factor-research/scripts/generate_hypothesis.py \
      --output docs/factor_research/hypotheses.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Import the rulebase hypotheses
sys.path.insert(0, str(ROOT / ".codex" / "skills" / "soxl-factor-mining" / "scripts"))
from extract_rulebase_signals import RULEBASE_HYPOTHESES  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate hypothesis cards")
    parser.add_argument("--output", default=str(ROOT / "docs" / "factor_research" / "hypotheses.yaml"))
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": "1.0",
        "count": len(RULEBASE_HYPOTHESES),
        "sources": ["rulebase"],
        "hypotheses": RULEBASE_HYPOTHESES,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(payload, f, allow_unicode=True, sort_keys=False)

    print(f"Wrote {len(RULEBASE_HYPOTHESES)} hypotheses to {output_path}")


if __name__ == "__main__":
    main()
