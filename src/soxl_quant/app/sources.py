from __future__ import annotations

from pathlib import Path

from ..core.dataset import get_market_panel_path, get_qlib_dir, get_training_cache_path
from ..data import build_dataset, init_qlib, refresh_qlib_binary, segment_frame
from ..settings import ETF_UNIVERSE, QLIB_DATA_PATH


def build_multi_asset_sources(multi_asset_dir: Path, profile=None, verbose: bool = True):
    multi_asset_dir.mkdir(parents=True, exist_ok=True)
    multi_asset_sources = []
    for asset_idx, (etf, meta) in enumerate(ETF_UNIVERSE.items()):
        underlying = meta["underlying"]
        leverage = meta["leverage"]
        csv_path = get_market_panel_path(etf)
        if not csv_path.exists():
            if verbose:
                print(f"[warn] missing {csv_path}, skipping {etf}")
            continue
        qlib_dir = get_qlib_dir(etf)
        refresh_qlib_binary(
            csv_path=csv_path,
            qlib_dir=qlib_dir,
            leveraged_etf=etf,
            underlying_etf=underlying,
            leverage_ratio=leverage,
        )
        init_qlib(qlib_dir)
        dataset = build_dataset(
            leveraged_etf=etf,
            underlying_etf=underlying,
            leverage_ratio=leverage,
            feature_packs=tuple(profile.feature.packs) if profile else (),
            profile_name=profile.name if profile else None,
        )
        etf_train_df = segment_frame(dataset, "train")
        pkl_path = get_training_cache_path(f"{etf}_train")
        pkl_path.parent.mkdir(parents=True, exist_ok=True)
        etf_train_df.to_pickle(pkl_path)
        multi_asset_sources.append((pkl_path, etf_train_df, asset_idx))
        if verbose:
            print(f"[multi-asset] {etf} train={len(etf_train_df)} rows -> {pkl_path.name}")
    if not multi_asset_sources:
        raise RuntimeError("no multi-asset sources available")
    if verbose:
        print(f"[multi-asset] total pooled episodes will be generated from {len(multi_asset_sources)} ETFs")
    init_qlib(QLIB_DATA_PATH)
    return multi_asset_sources


__all__ = ["build_multi_asset_sources"]
