#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soxl_quant.app.metric_replay_handoff import write_metric_replay_handoff_audit  # noqa: E402


SOURCE_CARD = ROOT / "runtime/model_selection/card/cap10_wf2020_bull_participation_diagnostic_v1"
REPLAY_CARD = ROOT / "runtime/model_selection/card/cap10_three_fold_checkpoint_metric_replay_v2"
OUTPUT_DIR = ROOT / "runtime/model_selection/card/cap10_metric_replay_handoff_audit_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit whether cap10 metric replay closes the mixed-metric handoff.")
    parser.add_argument("--source-card", type=Path, default=SOURCE_CARD)
    parser.add_argument("--replay-card", type=Path, default=REPLAY_CARD)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    audit = write_metric_replay_handoff_audit(
        source_card_dir=args.source_card,
        replay_card_dir=args.replay_card,
        output_dir=args.output_dir,
    )
    print(
        "metric_replay_handoff "
        f"decision={audit.summary['decision']} "
        f"completed={audit.summary['metric_replay_handoff_completed']} "
        f"failed_folds={','.join(audit.summary['failed_folds']) or 'none'} "
        f"output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
