"""Run the sealed one-step shadow with the documented US quote-state adapter.

This wrapper preserves the old research bundle's sealed scheduler and policy
bytes. Only quote collection is replaced; the existing shadow still observes
the actual SIM ledger and does not simulate a candidate-owned portfolio or
submit orders. Futu's documented US regular-session state is AFTERNOON:
https://openapi.futunn.com/futu-api-doc/qa/quote.html (Q14).
"""
from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path

from src.schedule.stage1_economic_shadow import run_shadow_schedule
from src.schedule.us_open_quotes import read_open_quotes


def run_stage1_shadow_open(*, root, config_path, now=None):
    quote_reader = read_open_quotes if now is None else partial(read_open_quotes, now=now)
    return run_shadow_schedule(root=root, config_path=config_path, now=now, quote_reader=quote_reader)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_stage1_shadow_open(root=args.root, config_path=args.config),
                     ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
