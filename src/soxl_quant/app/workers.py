from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from ..compat import patch_gym_compat
from ..data import build_dataset, init_qlib, segment_frame
from ..settings import QLIB_DATA_PATH, ROOT, load_profile
from .evaluate import compact_result, describe_metrics, slice_frame
from .train import _build_multi_asset_sources, run_profile, set_global_seed

WalkForwardWindow = Tuple[str, Tuple[str, str], Tuple[str, str], Tuple[str, str]]

WALK_FORWARD_WINDOWS: Tuple[WalkForwardWindow, ...] = (
    ("wf_2020", ("2010-03-11", "2018-12-31"), ("2019-01-01", "2019-12-31"), ("2020-01-01", "2020-12-31")),
    ("wf_2022", ("2010-03-11", "2020-12-31"), ("2021-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31")),
    ("wf_2023_2024", ("2010-03-11", "2022-12-31"), ("2023-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31")),
)

WORKER_RESULT_DIR = ROOT / "runtime" / "soxl" / "wf_workers"


def run_walk_forward_isolated(
    profile,
    smoke: bool,
    seed: int,
    skip_data_refresh: bool,
    multi_asset: bool = False,
    use_moe: bool = False,
    ablate_vix: bool = False,
) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    WORKER_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for name, train_seg, valid_seg, test_seg in WALK_FORWARD_WINDOWS:
        print(
            f"\n[walk-forward:{profile.name}:{name}:isolated] "
            f"train={train_seg[0]}->{train_seg[1]} valid={valid_seg[0]}->{valid_seg[1]} test={test_seg[0]}->{test_seg[1]}"
        )
        worker_output = WORKER_RESULT_DIR / f"{profile.name}-{name}-seed{seed}.json"
        if worker_output.exists():
            worker_output.unlink()
        cmd = [
            sys.executable,
            str(ROOT / "soxl_qlib_strategy_v2.py"),
            "--profiles",
            profile.name,
            "--seed",
            str(seed),
            "--worker-run-profile",
            profile.name,
            "--worker-window-name",
            name,
            "--worker-train-start",
            train_seg[0],
            "--worker-train-end",
            train_seg[1],
            "--worker-valid-start",
            valid_seg[0],
            "--worker-valid-end",
            valid_seg[1],
            "--worker-test-start",
            test_seg[0],
            "--worker-test-end",
            test_seg[1],
            "--worker-output",
            str(worker_output),
        ]
        if smoke:
            cmd.append("--smoke")
        if skip_data_refresh:
            cmd.append("--skip-data-refresh")
        if multi_asset:
            cmd.append("--multi-asset")
        if use_moe:
            cmd.append("--moe")
        if ablate_vix:
            cmd.append("--ablate-vix")
        proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
        if not worker_output.exists():
            raise RuntimeError(f"walk-forward worker failed before writing result: {name} exit={proc.returncode}")
        worker_payload = json.loads(worker_output.read_text(encoding="utf-8"))
        results[name] = worker_payload
        print(describe_metrics(f"walk-forward:{profile.name}:{name}", worker_payload))
        if smoke:
            break
    return results


def run_worker_profile(args: argparse.Namespace) -> None:
    patch_gym_compat()
    set_global_seed(args.seed)
    init_qlib(QLIB_DATA_PATH)
    dataset = build_dataset(ablate_vix=args.ablate_vix)
    full_df = pd.concat(
        [
            segment_frame(dataset, "train"),
            segment_frame(dataset, "valid"),
            segment_frame(dataset, "test"),
        ]
    ).sort_index()
    price_frame = dataset.handler.price_frame.droplevel("instrument")["price"].sort_index()
    profile = load_profile(args.worker_run_profile)
    if args.smoke or args.worker_run_profile == "smoke":
        profile = load_profile("smoke")
    multi_asset_sources = None
    if args.multi_asset:
        multi_asset_sources = _build_multi_asset_sources(ROOT / "data" / "multi_asset", verbose=False)

    result = run_profile(
        profile=profile,
        train_df=slice_frame(full_df, (args.worker_train_start, args.worker_train_end)),
        valid_df=slice_frame(full_df, (args.worker_valid_start, args.worker_valid_end)),
        test_df=slice_frame(full_df, (args.worker_test_start, args.worker_test_end)),
        price_series=price_frame.loc[args.worker_test_start : args.worker_test_end],
        smoke=args.smoke,
        seed=args.seed,
        artifact_tag=args.worker_window_name,
        multi_asset_sources=multi_asset_sources,
        use_moe=args.moe,
        ablate_vix=args.ablate_vix,
    )
    output_path = Path(args.worker_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(compact_result(result), indent=2), encoding="utf-8")
    os._exit(0)


__all__ = [
    "WALK_FORWARD_WINDOWS",
    "WORKER_RESULT_DIR",
    "run_walk_forward_isolated",
    "run_worker_profile",
]
