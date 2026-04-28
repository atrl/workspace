from __future__ import annotations

import re
from pathlib import Path

from .config import (
    DEFAULT_ACTION_LEVELS,
    ExecutionConfig,
    ModelConfig,
    PositionBudgetConfig,
    PositionCapConfig,
    PromotionGateConfig,
    RewardConfig,
    SamplingConfig,
    SelectionScoreConfig,
    StrategyConfig,
    load_profile,
)
from .core.universe import load_etf_universe

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DATA_DIR = DATA_DIR / "cache"
QLIB_STORE_DIR = DATA_DIR / "qlib"
CSV_PATH = PROCESSED_DATA_DIR / "panels" / "soxl_soxx_vix_full.csv"
QLIB_DATA_PATH = QLIB_STORE_DIR / "soxl"
RESULT_PNG = ROOT / "soxl_qlib_strategy_v2.png"
RESULT_TXT = ROOT / "soxl_qlib_strategy_v2_results.txt"
PROFILE_DIR = ROOT / "runtime"
MODEL_DIR = ROOT / "runtime/model/card"
RUNTIME_DIR = ROOT / "runtime/soxl"
ACCOUNT = 1_000_000
LOOKBACK = 60
N_ACTIONS = 11
EXTRA_OBS_DIM = 15
N_ASSETS = 5
TRAIN_SEG = ("2010-03-11", "2018-12-31")
VALID_SEG = ("2019-01-01", "2019-12-31")
TEST_SEG = ("2020-01-01", "2026-12-31")

ETF_UNIVERSE = load_etf_universe()


def artifact_stem(profile_name: str, artifact_tag: str = "") -> str:
    stem = f"soxl_{profile_name}"
    cleaned_tag = re.sub(r"[^a-zA-Z0-9._-]+", "_", artifact_tag.strip())
    if cleaned_tag:
        stem = f"{stem}_{cleaned_tag}"
    return stem


def model_artifact_path(profile_name: str, artifact_tag: str = "") -> Path:
    return MODEL_DIR / profile_name / f"{artifact_stem(profile_name, artifact_tag)}_actor.pt"


def summary_artifact_path(profile_name: str, artifact_tag: str = "") -> Path:
    return MODEL_DIR / profile_name / f"{artifact_stem(profile_name, artifact_tag)}_summary.json"
