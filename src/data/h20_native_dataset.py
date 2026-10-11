"""Reusable real H20 native DatasetH builder (src/data/h20_native_dataset.py; 25 features, 0 fit).

Reuses the parent benchmark-terminal recipe: ``prepare_horizon_inputs`` for the
qualified 25-feature frame and native calendar, then rebuilds H20 labels from the parent
EXECUTION provider opens, assembles a real ``DatasetH`` with the 25 features and the two
H20 labels, attaches the horizon scope, and returns the prepared shape. Terminal unknown
labels remain in the dataset. No model is fit here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.convex_allocation import FEATURE_COLUMNS
from src.data.generic_horizon import (attach_horizon_scope, build_horizon_labels_from_qlib,
                                      strict_positive_int)
from src.data.integrated_allocation import build_native_dataset, frame_sha256
from src.data.oos_horizon_inputs import prepare_horizon_inputs

LABEL_COLUMNS_H20 = ("soxl_return_h20", "soxx_return_h20")
METADATA_COLUMNS_H20 = ("signal_date", "feature_available_at", "decision_at", "trade_date", "exit_date",
                        "return_available_at", "exit_date_h20", "label_h20_available_at")


def build_h20_native_dataset(root, parent_plan, *, horizon_sessions: int = 20) -> dict:
    """Build the real H20 DatasetH from the parent plan (read-only, no fit)."""
    horizon_sessions = strict_positive_int(horizon_sessions)
    root = Path(root)
    bundle = prepare_horizon_inputs(root, parent_plan)
    frame = bundle["frame"].copy()
    calendar = pd.DatetimeIndex(bundle["calendar"])
    source = parent_plan["source"]
    execution_uri = root / source["execution_provider_uri"]
    as_of = source["as_of"]
    original_hash = frame_sha256(frame)
    labels, label_audit = build_horizon_labels_from_qlib(
        frame, provider_uri=execution_uri, start_time=str(calendar[0].date()), end_time=as_of,
        symbols=("SOXL", "SOXX"), horizon_sessions=horizon_sessions, availability_hour=9, availability_minute=31)
    missing = [c for c in (*LABEL_COLUMNS_H20, *METADATA_COLUMNS_H20) if c not in labels.columns]
    if missing:
        raise ValueError(f"blocked: H20 frame lacks {missing}")
    dataset = build_native_dataset(labels, feature_columns=FEATURE_COLUMNS, label_columns=LABEL_COLUMNS_H20,
                                   metadata_columns=METADATA_COLUMNS_H20, as_of=as_of)
    attach_horizon_scope(dataset, horizon_sessions)
    dataset.calendar = calendar
    prepared = dataset.prepare("all", col_set=["feature", "label", "metadata"])
    features = list(prepared["feature"].columns)
    label_columns = list(prepared["label"].columns)
    terminal_unknown = int(prepared["label"].isna().any(axis=1).sum())
    covariance = bundle["covariance"]
    if getattr(covariance, "shape", None) is not None and len(covariance) != len(labels):
        raise ValueError("blocked: bundle covariance is not aligned to the full frame index")
    return {
        "dataset": dataset, "frame": labels, "calendar": calendar, "prepared": prepared, "covariance": covariance,
        "audit": {
            "stage": "native_h20_dataset",
            "feature_rows": int(prepared.shape[0]),
            "feature_columns": features,
            "label_columns": label_columns,
            "n_features": len(features),
            "n_labels": len(label_columns),
            "calendar_rows": int(len(calendar)),
            "calendar_first": str(calendar[0].date()), "calendar_last": str(calendar[-1].date()),
            "train_start": parent_plan.get("train_start"),
            "source_original_frame_sha256": original_hash,
            "source_transformed_frame_sha256": frame_sha256(labels),
            "qualified_source_audit": bundle["audit"].get("qualified_source"),
            "label_provider": "execution",
            "terminal_unknown": terminal_unknown,
            "horizon_sessions": horizon_sessions,
            "attachment": "attach_horizon_scope + bound native calendar",
            "fit_performed": False,
        },
    }


def load_parent_plan(root, name="benchmark_terminal_value_20261009") -> dict:
    return json.loads((Path(root) / "research" / name / "plan.json").read_text())
