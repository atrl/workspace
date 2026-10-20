"""Pure P3 action-model preflight and one-arm fitting primitives.

Callers own the authorized Qlib Recorder/Dataset reads and must pass aligned,
already prepared frames. This module has no source-loading, Recorder, evaluation,
or cross-validation path. One ``fit_one_arm`` call performs exactly one fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.path_area_bridge import AREA_COLUMN, CONTROL_COLUMNS

Arm = Literal["B", "C"]
TRAINING_CUTOFF = pd.Timestamp("2023-01-01 00:00:00", tz="America/New_York")
PRE_2022_CUTOFF = pd.Timestamp("2022-01-01 00:00:00", tz="America/New_York")
MINIMUM_FIT_ROWS = 200
MINIMUM_PRE_2022_ROWS = 200
PROBABILITY_THRESHOLD = 0.50


def original_action_features() -> tuple[str, ...]:
    """Return the canonical 22-feature order from the frozen Stage1 module."""
    from src.workflow.stage1_open_action_value import ACTION_FEATURES

    features = tuple(ACTION_FEATURES)
    if len(features) != 22 or len(set(features)) != 22:
        raise ValueError("blocked: canonical Stage1 action feature contract differs")
    return features


def feature_order() -> tuple[str, ...]:
    """P3 order: original 22, eight controls, mask, and final area slot."""
    result = (*original_action_features(), *CONTROL_COLUMNS, "prefix_mask", AREA_COLUMN)
    if len(result) != 32 or len(set(result)) != 32:
        raise ValueError("blocked: P3 action feature contract differs")
    return result


@dataclass(frozen=True)
class TrainingPreflight:
    """Aligned, filtered pre-2023 fit rows plus auditable support counts."""

    features: pd.DataFrame
    target: pd.Series
    row_count: int
    class_counts: dict[int, int]
    input_rows: int
    eligible_rows: int
    mature_rows: int
    finite_original_rows: int
    pre_2022_rows: int
    pre_2022_class_counts: dict[int, int]
    observed_area_rows: int
    cutoff: pd.Timestamp
    pre_2022_cutoff: pd.Timestamp
    failures: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class FittedArm:
    """One fitted arm and compact metadata for the caller's sealed review."""

    arm: Arm
    pipeline: Pipeline
    feature_order: tuple[str, ...]
    row_count: int
    class_counts: dict[int, int]
    pre_2022_rows: int
    cutoff: pd.Timestamp
    pre_2022_cutoff: pd.Timestamp
    threshold: float
    parameters: dict[str, object]


def _ny_maturity(values: pd.Series) -> pd.DatetimeIndex:
    parsed = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if parsed.tz is None:
        raise ValueError("blocked: label_available_at must carry its source timezone")
    return parsed.tz_convert("America/New_York")


def _trade_dates(values: pd.Series) -> pd.DatetimeIndex:
    parsed = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if parsed.tz is not None:
        parsed = parsed.tz_convert("America/New_York").tz_localize(None)
    return parsed


def preflight_training_rows(
    features: pd.DataFrame,
    labels: pd.Series,
    metadata: pd.DataFrame,
) -> TrainingPreflight:
    """Validate exact row alignment and select only eligible, mature train rows.

    ``features`` must already be mapped from native ``prepare_groups(dataset,
    "all")`` to the decisions' actual ``trade_date`` order. ``labels`` is the
    local EXECUTE-minus-HOLD Delta (ties map to HOLD); ``metadata`` supplies
    ``eligible``, ``label_status``, ``trade_date``, and timezone-aware
    ``label_available_at``.
    The caller remains responsible for verifying Dataset labels against its
    separately admitted label frame before calling this pure function.
    """
    order = feature_order()
    if not isinstance(features, pd.DataFrame) or not features.columns.is_unique:
        raise ValueError("blocked: features must be a uniquely-columned DataFrame")
    if tuple(features.columns) != order:
        raise ValueError("blocked: P3 feature columns/order differ from frozen contract")
    if not isinstance(labels, pd.Series) or not isinstance(metadata, pd.DataFrame):
        raise TypeError("features, labels, and metadata must be prepared pandas frames")
    if not features.index.equals(labels.index) or not features.index.equals(metadata.index):
        raise ValueError("blocked: features, labels, and metadata index grids differ")
    required = {"eligible", "label_status", "trade_date", "label_available_at"}
    if not required.issubset(metadata.columns):
        raise ValueError("blocked: preflight metadata fields are incomplete")
    if features.index.has_duplicates:
        raise ValueError("blocked: duplicate training row index")

    eligible = metadata["eligible"]
    if eligible.isna().any() or not eligible.isin([True, False]).all():
        raise ValueError("blocked: eligibility must be complete boolean evidence")
    dates = _trade_dates(metadata["trade_date"])
    if (dates.hasnans or not dates.equals(dates.normalize()) or dates.has_duplicates
            or not dates.is_monotonic_increasing):
        raise ValueError("blocked: trade_date values must be normalized, unique, and ordered")
    if (dates >= TRAINING_CUTOFF.tz_localize(None)).any():
        raise ValueError("blocked: P3 preflight input includes 2023+ trade dates")
    maturity = _ny_maturity(metadata["label_available_at"])

    original = features.loc[:, list(original_action_features())].to_numpy(dtype=np.float64)
    finite_original = pd.Series(np.isfinite(original).all(axis=1), index=features.index)
    delta = pd.to_numeric(labels, errors="raise").to_numpy(dtype=np.float64)
    finite_label = pd.Series(np.isfinite(delta), index=features.index)
    label_mature = metadata["label_status"].eq("mature")
    maturity_before = pd.Series(
        label_mature.to_numpy() & (~maturity.isna()) & (maturity < TRAINING_CUTOFF),
        index=features.index)
    eligible = eligible.astype(bool)
    mask = pd.to_numeric(features["prefix_mask"], errors="raise")
    if mask.isna().any() or not mask.isin([0, 1]).all():
        raise ValueError("blocked: prefix_mask must contain only 0/1")
    path_values = features.loc[:, [*CONTROL_COLUMNS, AREA_COLUMN]].to_numpy(dtype=np.float64)
    complete_prefix = mask.eq(1).to_numpy()
    if not np.isfinite(path_values[complete_prefix]).all():
        raise ValueError("blocked: complete prefix has missing/nonfinite path inputs")
    if not np.isnan(path_values[~complete_prefix]).all():
        raise ValueError("blocked: missing prefix must retain NaN path inputs")

    eligible_rows = int(eligible.sum())
    finite_original_rows = int(finite_original.sum())
    mature_rows = int((eligible & maturity_before).sum())
    fit_mask = eligible & finite_original & finite_label & maturity_before
    fit_features = features.loc[fit_mask, list(order)].astype(np.float64).copy()
    fit_target = pd.Series((delta[fit_mask.to_numpy()] > 0).astype("int8"),
                           index=features.index[fit_mask], name="execute_positive_delta")
    classes = {0: int((fit_target == 0).sum()), 1: int((fit_target == 1).sum())}
    observed_area_rows = int((mask.loc[fit_mask].eq(1)
                              & np.isfinite(fit_features[AREA_COLUMN])).sum())
    pre_2022_mask = fit_mask & pd.Series(
        (dates < PRE_2022_CUTOFF.tz_localize(None)) & (~maturity.isna())
        & (maturity < PRE_2022_CUTOFF),
        index=features.index)
    pre_2022_target = pd.Series((delta[pre_2022_mask.to_numpy()] > 0).astype("int8"),
                                index=features.index[pre_2022_mask])
    pre_classes = {0: int((pre_2022_target == 0).sum()), 1: int((pre_2022_target == 1).sum())}

    failures: list[str] = []
    if len(fit_features) < MINIMUM_FIT_ROWS:
        failures.append("fewer_than_200_fit_rows")
    if int(pre_2022_mask.sum()) < MINIMUM_PRE_2022_ROWS:
        failures.append("fewer_than_200_strict_mature_pre_2022_rows")
    if not all(classes.values()):
        failures.append("fit_rows_do_not_contain_both_classes")
    if observed_area_rows == 0:
        failures.append("no_observed_signed_area_in_fit_rows")
    return TrainingPreflight(
        features=fit_features,
        target=fit_target,
        row_count=len(fit_features),
        class_counts=classes,
        input_rows=len(features),
        eligible_rows=eligible_rows,
        mature_rows=mature_rows,
        finite_original_rows=finite_original_rows,
        pre_2022_rows=int(pre_2022_mask.sum()),
        pre_2022_class_counts=pre_classes,
        observed_area_rows=observed_area_rows,
        cutoff=TRAINING_CUTOFF,
        pre_2022_cutoff=PRE_2022_CUTOFF,
        failures=tuple(failures),
    )


def build_arm_features(features: pd.DataFrame, arm: Arm) -> pd.DataFrame:
    """Build matched B/C matrices; B receives a fixed neutral area of zero."""
    if arm not in ("B", "C"):
        raise ValueError("arm must be B or C")
    order = feature_order()
    if tuple(features.columns) != order:
        raise ValueError("blocked: arm feature columns/order differ from frozen contract")
    result = features.loc[:, list(order)].astype(np.float64).copy()
    if arm == "B":
        result[AREA_COLUMN] = 0.0
    return result


def fit_one_arm(preflight: TrainingPreflight, arm: Arm) -> FittedArm:
    """Fit one fixed P3 logistic arm; caller tracks the two-fit aggregate cap."""
    if not preflight.ready:
        raise ValueError("blocked: preflight gates failed: " + ",".join(preflight.failures))
    if arm not in ("B", "C"):
        raise ValueError("arm must be B or C")
    features = build_arm_features(preflight.features, arm)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=0.10, class_weight="balanced",
             max_iter=2000, random_state=42)),
    ])
    pipeline.fit(features, preflight.target)
    imputed = pipeline.named_steps["imputer"].transform(features)
    if (features.shape[1] != 32 or imputed.ndim != 2 or imputed.shape[1] != 32
            or pipeline.named_steps["scaler"].n_features_in_ != 32
            or pipeline.named_steps["classifier"].n_features_in_ != 32):
        raise ValueError("blocked: fitted action pipeline did not retain all 32 features")
    parameters: dict[str, object] = {
        "imputer": {"strategy": "mean"},
        "scaler": "StandardScaler",
        "classifier": {"family": "LogisticRegression", "C": 0.10,
                        "class_weight": "balanced", "max_iter": 2000,
                        "random_state": 42},
        "target": "local_execute_delta > 0; ties are HOLD",
        "threshold": PROBABILITY_THRESHOLD,
        "area_semantics": "B=0; C=observed signed area",
    }
    return FittedArm(arm=arm, pipeline=pipeline, feature_order=feature_order(),
        row_count=preflight.row_count, class_counts=dict(preflight.class_counts),
        pre_2022_rows=preflight.pre_2022_rows, cutoff=preflight.cutoff,
        pre_2022_cutoff=preflight.pre_2022_cutoff,
        threshold=PROBABILITY_THRESHOLD, parameters=parameters)
