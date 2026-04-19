from __future__ import annotations

import os
import warnings
from pathlib import Path

from qlib.data import D

from ..compat import patch_gym_compat
from ..data import FEATURE_COLUMNS, build_dataset, init_qlib, refresh_qlib_binary, segment_frame
from ..settings import CSV_PATH, QLIB_DATA_PATH, TEST_SEG
from .sources import build_multi_asset_sources
from .train import set_global_seed

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")


def bootstrap_runtime(args):
    patch_gym_compat()
    set_global_seed(args.seed)

    print("=" * 72)
    print("SOXL Strategy v2 - qlib native package")
    print("=" * 72)

    if not args.skip_data_refresh:
        csv_df = refresh_qlib_binary(CSV_PATH, QLIB_DATA_PATH)
        print(f"[data] qlib binary refreshed: {len(csv_df)} rows, end={csv_df.index[-1].date()}")
    else:
        print("[data] skip binary refresh")

    init_qlib(QLIB_DATA_PATH)
    probe = D.features(["SOXL", "SOXX"], ["$close"], start_time="2026-03-30", end_time="2026-04-02")
    print(f"[qlib] instruments ok: {sorted(probe.index.get_level_values('instrument').unique().tolist())}")

    dataset = build_dataset(ablate_vix=args.ablate_vix)
    train_df = segment_frame(dataset, "train")
    valid_df = segment_frame(dataset, "valid")
    test_df = segment_frame(dataset, "test")
    price_series = dataset.handler.price_frame.droplevel("instrument")["price"].loc[TEST_SEG[0] : TEST_SEG[1]]

    print(f"[dataset] train={len(train_df)} valid={len(valid_df)} test={len(test_df)} features={len(FEATURE_COLUMNS)}")
    if args.ablate_vix:
        print("[ablation] VIX features zeroed out, regime_risk -> regime_risk_no_vix")

    multi_asset_sources = None
    if args.multi_asset:
        multi_asset_sources = build_multi_asset_sources(Path(QLIB_DATA_PATH).parent / "multi_asset", verbose=True)

    return dataset, train_df, valid_df, test_df, price_series, multi_asset_sources


__all__ = ["bootstrap_runtime"]
