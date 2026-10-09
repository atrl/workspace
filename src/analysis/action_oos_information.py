"""True out-of-sample marginal-information test for the Stage1 action classifier.

The action layer is a per-year sklearn Pipeline (impute -> scale -> logistic). To
measure its conditional information honestly we must fit nothing new: for each
year Y we use the *saved* pipeline for Y, which was fit only on data strictly
before Y, predict on year Y's rows, and compare that probability with the true
baseline-domain ``execute_advantage`` label of the SAME rows.

Cross-account returns (active account minus an independently reset no_hold
account) are NOT the classifier's training target and must never be substituted
for ``execute_advantage``. When a strictly local label is not available the
result is ``needs_data``.

This module is pure computation: no recorder access, no fitting, no tuning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANNUALIZATION_SESSIONS = 252
SCOPE = "active_model_own_oos_label"


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


FEATURE_CANDIDATES = (
    "portfolio_drawdown", "current_beta_cap", "raw_effective_beta", "risk_state",
    "raw_soxl_weight", "raw_soxx_weight", "raw_cash_weight", "tail_probability",
    "alpha_probability", "portfolio_ret_1d", "portfolio_ret_3d", "portfolio_ret_5d",
    "pretrade_soxl_weight", "pretrade_soxx_weight", "pretrade_cash_weight",
    "soxl_weight_delta", "soxx_weight_delta", "effective_beta_delta", "open_turnover",
    "target_soxl_weight", "target_soxx_weight", "target_cash_weight",
)


def pipeline_feature_names(pipeline) -> list[str]:
    names = getattr(pipeline, "feature_names_in_", None)
    if names is None:
        return []
    return [str(name) for name in names]


def verify_model_identity_and_maturity(dataset_audit: dict, diagnostics: pd.DataFrame, *,
                                       pipelines: dict) -> dict:
    """Verify each model year's training cutoff and identity are real metadata.

    Requires the frozen ``dataset_audit.json`` (feature list, training cutoffs,
    label rule) and ``training_diagnostics.csv`` (per-year cutoff and label
    availability). For every model year it asserts
    ``label_available_at_max < test_decision_at`` and that the pipeline's input
    features match the audited feature list. Missing metadata is ``needs_data``;
    no date is trusted by default.
    """
    if not isinstance(dataset_audit, dict) or not isinstance(diagnostics, pd.DataFrame):
        return {"status": "needs_data", "reason": "dataset audit or diagnostics missing"}
    features = dataset_audit.get("features")
    if not isinstance(features, list) or not features:
        return {"status": "needs_data", "reason": "dataset audit has no feature list"}
    required = {"model_year", "label_available_at_max", "test_decision_at"}
    if not required.issubset(diagnostics.columns):
        return {"status": "needs_data", "reason": "training diagnostics missing cutoff columns"}
    checks = []
    for row in diagnostics.itertuples(index=False):
        raw_year = row.model_year
        try:
            year = int(raw_year)
        except (TypeError, ValueError):
            year = None
        avail = pd.Timestamp(row.label_available_at_max)
        test = pd.Timestamp(row.test_decision_at)
        ok = avail < test
        # Only integer model years can be matched to a saved pipeline feature set.
        pipeline = pipelines.get(year) if (year is not None and isinstance(pipelines, dict)) else None
        # Identity is only meaningful for an integer model year that maps to a
        # saved pipeline; a frozen outer alias has its own single model.
        identity_ok = (pipeline is None) if year is None else (
            pipeline is not None and pipeline_feature_names(pipeline) == features)
        checks.append({"model_year": raw_year, "label_available_at_max": str(avail),
                       "test_decision_at": str(test), "maturity_ok": bool(ok),
                       "feature_identity_ok": bool(identity_ok),
                       "identity_required": year is not None})
    all_ok = all(c["maturity_ok"] and (c["feature_identity_ok"] or not c["identity_required"])
                 for c in checks)
    return {"status": "ok" if all_ok else "blocked",
            "label_rule": dataset_audit.get("label_available_at_rule"),
            "label_name": dataset_audit.get("label"),
            "checks": checks,
            "reason": None if all_ok else "a model year failed maturity or feature identity"}


def true_oos_predictions(frame: pd.DataFrame, pipelines: dict, *,
                         label_column: str = "execute_advantage",
                         year_column: str = "year") -> pd.DataFrame:
    """Predict each year with that year's saved pipeline; keep the realized label.

    Only rows whose label is strictly matured (``label_available_at`` present and
    after the decision) are scored. Missing maturity columns BLOCK the run: a
    default of "matured" would treat an unknown date as trustworthy.
    """
    _require(label_column in frame.columns, f"frame is missing label column {label_column}")
    _require(year_column in frame.columns, f"frame is missing {year_column}")
    _require("label_available_at" in frame.columns and "decision_at" in frame.columns,
             "frame is missing maturity columns label_available_at/decision_at")
    work = frame.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"])
    avail = pd.to_datetime(work["label_available_at"], utc=True, errors="coerce")
    decision = pd.to_datetime(work["decision_at"], utc=True, errors="coerce")
    matured = avail.notna() & (avail > decision)
    work = work[matured].copy()
    _require(not work.empty, "no matured rows to score")
    records = []
    for year, group in work.groupby(year_column, sort=True):
        pipeline = pipelines.get(int(year))
        if pipeline is None:
            continue
        names = pipeline_feature_names(pipeline)
        _require(names, f"pipeline for {year} exposes no feature_names_in_")
        missing = [c for c in names if c not in group.columns]
        _require(not missing, f"pipeline for {year} needs missing columns {missing}")
        probability = pipeline.predict_proba(group.loc[:, names])[:, 1]
        block = pd.DataFrame({
            "signal_date": group["signal_date"].to_numpy(),
            "year": int(year),
            "predicted_probability": probability,
            "realized_label": group[label_column].to_numpy(float),
        })
        records.append(block)
    _require(records, "no year had a saved pipeline for its own rows")
    return pd.concat(records, ignore_index=True).sort_values("signal_date").reset_index(drop=True)


def decidable_mask(frame: pd.DataFrame, *, threshold: float,
                   maximum_skippable_turnover: float,
                   mandatory_derisk_beta_delta: float,
                   probability_column: str = "predicted_probability") -> pd.Series:
    """Rows where the model actually decides (not forced by the route rules).

    Mirrors ``decide_open_execution``: execution is forced when the change is a
    mandatory de-risk, a large transition, or the probability is non-finite. Only
    the remaining rows are the classifier's decision-eligible subset. The
    probability column name is explicit so a caller cannot silently pass the
    wrong series.
    """
    _require(probability_column in frame.columns, f"frame is missing {probability_column}")
    _require({"effective_beta_delta", "open_turnover"}.issubset(frame.columns),
             "frame is missing the route-rule inputs")
    prob = pd.to_numeric(frame[probability_column], errors="coerce")
    mandatory = pd.to_numeric(frame["effective_beta_delta"], errors="coerce") < -mandatory_derisk_beta_delta
    large = pd.to_numeric(frame["open_turnover"], errors="coerce") > maximum_skippable_turnover
    forced = mandatory | large | ~np.isfinite(prob)
    return (~forced) & np.isfinite(prob)


def conditional_information(predictions: pd.DataFrame, *, decidable: pd.Series | None = None,
                            block: int = 20, resamples: int = 2000, seed: int = 20261009) -> dict:
    """Rank IC / AUC / calibration of OOS probability vs the matured own label.

    Uncertainty is estimated on the FULL trading-day index: move a 20-session
    block over the whole indexed series (so the sparse eligibility is preserved
    inside each block) and recompute rankIC/AUC from the decidable rows present
    in the resample. The point estimates use only the decidable subset. Both the
    advantage mean and the rankIC/AUC point estimates carry a block CI; no
    significance is claimed beyond what those intervals show.
    """
    from src.analysis.action_marginal_information import auc, block_bootstrap_ci, calibration_table, rank_ic

    _require({"predicted_probability", "realized_label"}.issubset(predictions.columns),
             "predictions frame is missing probability or label")
    full = predictions.dropna(subset=["predicted_probability", "realized_label"]).sort_values("signal_date")
    mask = decidable.reindex(full.index).fillna(False) if decidable is not None else pd.Series(True, index=full.index)
    dec = full[mask.to_numpy(bool)]
    if dec.empty:
        return {"status": "needs_data", "reason": "no aligned decidable probability/label rows", "scope": SCOPE}
    pred = dec["predicted_probability"].to_numpy(float)
    realized = dec["realized_label"].to_numpy(float)
    positive = realized > 0
    # Block CI for rankIC/AUC over the full indexed series (eligibility preserved).
    rank_ci = _metric_block_ci(full, mask, block=block, resamples=resamples, seed=seed, metric=rank_ic)
    auc_ci = _metric_block_ci(full, mask, block=block, resamples=resamples, seed=seed + 1, metric=auc, binary=True)
    return {
        "status": "ok",
        "label_source": "active_model_own_execute_advantage",
        "scope": SCOPE,
        "decidable_rule": "not mandatory_derisk and not large_transition and finite probability",
        "decidable_sessions": int(len(dec)),
        "full_oos_sessions": int(len(full)),
        "positive_advantage_rate": float(positive.mean()),
        "mean_advantage": float(realized.mean()),
        "mean_predicted_probability": float(pred.mean()),
        "rank_ic": rank_ic(pred, realized),
        "auc": auc(pred, positive),
        "rank_ic_ci": rank_ci,
        "auc_ci": auc_ci,
        "advantage_bps_ci": block_bootstrap_ci(realized * 1e4),
        "quintile_mean_advantage_bps": _quintiles(pred, realized),
        "calibration": calibration_table(pred, realized, bins=5).to_dict(orient="records"),
        "interpretation": (
            "AUC near 0.5 or rankIC/AUC/advantage CIs that contain their null means no conditional "
            "information was found for the classifier's own matured label; it does not prove that "
            "no information exists."),
    }


def _metric_block_ci(frame: pd.DataFrame, mask: pd.Series, *, block: int, resamples: int,
                     seed: int, metric, binary: bool = False) -> dict:
    """Moving-block CI for a rank/AUC metric over the full indexed series.

    Blocks are drawn from the full index; within each drawn block only the
    decidable rows contribute to the metric, so a sparse decision set is not
    compressed into a handful of events.
    """
    n = len(frame)
    _require(n > block, "series must exceed the block length")
    prob = frame["predicted_probability"].to_numpy(float)
    label = frame["realized_label"].to_numpy(float)
    flags = mask.to_numpy(bool)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    values = []
    for _ in range(resamples):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        sel = idx[flags[idx]]
        if sel.size < 30:
            continue
        p, y = prob[sel], label[sel]
        value = metric(p, (y > 0) if binary else y)
        if value is not None and np.isfinite(value):
            values.append(value)
    if len(values) < 50:
        return {"point": None, "lower": None, "upper": None, "block": block,
                "resamples": len(values), "contains_zero": None, "reason": "insufficient resamples"}
    values = np.asarray(values)
    null = 0.0 if not binary else 0.5
    lower, upper = float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
    return {"point": float(np.median(values)), "lower": lower, "upper": upper,
            "block": block, "resamples": int(len(values)), "null": null,
            "contains_zero": bool(lower <= null <= upper)}


def _quintiles(pred: np.ndarray, realized: np.ndarray, *, bins: int = 5) -> list[float | None]:
    frame = pd.DataFrame({"p": pred, "y": realized}).dropna()
    if len(frame) < bins:
        return []
    bucket = pd.qcut(frame["p"], bins, labels=False, duplicates="drop")
    return [float(frame.loc[bucket == i, "y"].mean() * 1e4) for i in range(bins)]


def segment_contribution(paths: dict[str, pd.Series], no_hold: pd.Series, *,
                         cost_bps: float, segment: str) -> list[dict]:
    """Per-variant net contribution within ONE segment, each from its own init.

    The no_hold index and every variant must share EXACTLY the same unique,
    ordered session keys with finite values. A missing session, a duplicate, or a
    non-finite value BLOCKS the segment rather than being dropped.
    """
    from src.analysis.action_marginal_information import period_metrics

    base = no_hold
    _require(not base.index.has_duplicates, f"{segment}: no_hold index has duplicates")
    _require(base.index.is_monotonic_increasing, f"{segment}: no_hold index is not ordered")
    _require(np.isfinite(base.to_numpy(float)).all(), f"{segment}: no_hold has non-finite values")
    rows = []
    for name, path in paths.items():
        if path.index.has_duplicates:
            raise ValueError(f"blocked: {segment}/{name} index has duplicates")
        if not path.index.equals(base.index):
            missing = base.index.difference(path.index)
            extra = path.index.difference(base.index)
            raise ValueError(
                f"blocked: {segment}/{name} session keys differ from no_hold "
                f"(missing={len(missing)}, extra={len(extra)})")
        values = path.to_numpy(float)
        _require(np.isfinite(values).all(), f"{segment}/{name} has non-finite returns")
        metrics = period_metrics(values)
        base_metrics = period_metrics(base.to_numpy(float))
        rows.append({
            "segment": segment, "cost_bps": float(cost_bps), "variant": name,
            "sessions": int(len(path)), "gaps": 0,
            "return_pct": metrics["return_pct"],
            "no_hold_return_pct": base_metrics["return_pct"],
            "net_contribution_pct": metrics["return_pct"] - base_metrics["return_pct"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
            "cvar5_pct": metrics["cvar5_pct"],
            "sharpe": metrics["sharpe"],
        })
    return rows
