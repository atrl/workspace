"""Read-only audits of retained SOXL training coverage and target feature drift."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.workflow.direct_continuous_soxx_allocation import DirectContinuousAllocationSpec, _load_frame
from src.workflow.pit_allocation_retrain import _mature_training_labels, _training_auxiliary, validate_frozen_bundle
from src.workflow.pit_constituent_auxiliary_training_allocation import (
    COMMON_FEATURES, PitAuxiliaryAllocationSpec, _soxx_common_training_frame,
)


BASE_CARD = "pit_constituent_auxiliary_training_allocation_retrained_20260928"
ACTION_CARD = "stage1_open_action_value_net_fees_20260928"
LABELS = ["soxl_forward_open_return", "soxx_forward_open_return", "tail_event_5d", "tail_cash_target_5d", "relative_alpha_5d"]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_dir(card: Path) -> tuple[dict, Path]:
    summary = _read(card / "summary.json")
    matches = list(card.glob(f"qlib_recorder/*/{summary['recorder_id']}"))
    if len(matches) != 1:
        raise ValueError(f"blocked: retained recorder is not unique under {card}")
    return summary, matches[0] / "artifacts" / summary["experiment"]


def overlap_clusters(dates, *, calendar: pd.DatetimeIndex, horizon: int = 5) -> pd.Series:
    """Merge overlapping forward-label supports across all instruments together.

    Dates separated by at most horizon-1 sessions have overlapping labels. These
    operational clusters are not independent market episodes or statistical N_eff.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().unique().sort_values()
    positions = calendar.get_indexer(dates)
    if np.any(positions < 0):
        raise ValueError("label date is absent from the shared qlib calendar")
    clusters, group, previous = [], 0, None
    for position in positions:
        if previous is None or position - previous >= horizon:
            group += 1
        clusters.append(f"label_overlap_{group:03d}")
        previous = position
    return pd.Series(clusters, index=dates, name="overlap_cluster")


def auxiliary_coverage(auxiliary: pd.DataFrame, *, calendar: pd.DatetimeIndex,
                       epochs: int, batch_size: int) -> dict[str, pd.DataFrame]:
    frame = auxiliary.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.normalize()
    frame["year"] = frame["datetime"].dt.year
    frame["row_draw_probability"] = 1.0 / len(frame)
    frame["nominal_loss_weight_share"] = frame.training_weight / frame.training_weight.sum()
    frame["crash"] = frame.synthetic_2x_crash_label.eq(1.0)
    assignments = overlap_clusters(frame.loc[frame.crash, "datetime"], calendar=calendar)
    frame["overlap_cluster"] = frame.datetime.map(assignments)
    crash = frame.loc[frame.crash].copy()
    draws = epochs * min(batch_size, len(frame))
    yearly = frame.groupby("year").agg(
        rows=("instrument", "size"), unique_dates=("datetime", "nunique"), instruments=("instrument", "nunique"),
        crash_rows=("crash", "sum"), row_draw_probability=("row_draw_probability", "sum"),
        nominal_loss_weight_share=("nominal_loss_weight_share", "sum"),
    )
    yearly["crash_dates"] = crash.groupby("year").datetime.nunique().reindex(yearly.index, fill_value=0)
    yearly["operational_overlap_clusters"] = crash.groupby("year").overlap_cluster.nunique().reindex(yearly.index, fill_value=0)
    yearly["expected_row_draws_over_training"] = yearly.row_draw_probability * draws
    episodes = crash.groupby("overlap_cluster").agg(
        first_signal_date=("datetime", "min"), last_signal_date=("datetime", "max"),
        crash_rows=("instrument", "size"), unique_dates=("datetime", "nunique"), instruments=("instrument", "nunique"),
        row_draw_probability=("row_draw_probability", "sum"), nominal_loss_weight_share=("nominal_loss_weight_share", "sum"),
    ).reset_index()
    concentrations = []
    for dimension in ("year", "datetime", "instrument"):
        mass = frame.groupby(dimension).nominal_loss_weight_share.sum().sort_values(ascending=False)
        concentrations.append({"grouping": dimension, "group_count": len(mass),
                               "largest_group_nominal_loss_share": float(mass.iloc[0]),
                               "top5_group_nominal_loss_share": float(mass.head(5).sum()),
                               "inverse_weight_concentration": float(1.0 / np.square(mass).sum()),
                               "interpretation": "nominal loss-weight concentration; not gradient influence or IID sample size"})
    if not episodes.empty:
        mass = episodes.nominal_loss_weight_share / episodes.nominal_loss_weight_share.sum()
        concentrations.append({"grouping": "overlap_cluster_conditional_on_crash_labels", "group_count": len(mass),
                               "largest_group_nominal_loss_share": float(mass.max()),
                               "top5_group_nominal_loss_share": float(mass.nlargest(5).sum()),
                               "inverse_weight_concentration": float(1.0 / np.square(mass).sum()),
                               "interpretation": "conditional crash-label nominal weights; not gradient influence or IID sample size"})
    return {"auxiliary_year_coverage": yearly.reset_index(), "auxiliary_crash_overlap_clusters": episodes,
            "auxiliary_sampling_concentration": pd.DataFrame(concentrations)}


def feature_drift(train: pd.DataFrame, target: pd.DataFrame, *, features: list[str],
                  scaler: dict, feature_group: str) -> pd.DataFrame:
    """Use training-only raw-feature quantiles; retained scaler statistics stay fixed."""
    rows = []
    for feature in features:
        reference = pd.to_numeric(train[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        observed = pd.to_numeric(target[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if reference.notna().sum() < 2:
            raise ValueError(f"blocked: insufficient training reference for {feature}")
        q01, q99 = reference.quantile([.01, .99]).tolist()
        mean, scale, median = (float(scaler[name][feature]) for name in ("mean", "scale", "median"))
        if not np.isfinite([mean, scale, median]).all() or scale <= 0:
            raise ValueError(f"blocked: invalid retained scaler for {feature}")
        observed_z = (observed.fillna(median) - mean) / scale
        imputed_train = reference.fillna(reference.median())
        inside = observed.ge(q01) & observed.le(q99)
        rows.append({
            "feature_group": feature_group, "feature": feature, "reference_basis": "reconstructed training-only raw feature distribution",
            "training_rows": int(reference.notna().sum()), "train_q01": q01, "train_q99": q99,
            "target_rows": len(observed), "target_missing_rows": int(observed.isna().sum()),
            "target_outside_q01_q99_rows": int((observed.notna() & ~inside).sum()),
            "target_outside_q01_q99_fraction": float((~inside.loc[observed.notna()]).mean()) if observed.notna().any() else None,
            "target_latest_value": float(observed.iloc[-1]) if pd.notna(observed.iloc[-1]) else None,
            "target_latest_outside_q01_q99": bool(not inside.iloc[-1]) if pd.notna(observed.iloc[-1]) else None,
            "retained_scaler_mean": mean, "retained_scaler_scale": scale, "retained_scaler_median": median,
            "target_scaler_clip_rows": int(observed_z.abs().gt(8.0).sum()),
            "target_latest_unclipped_z": float(observed_z.iloc[-1]),
            "reconstructed_scaler_mean_abs_difference": abs(float(imputed_train.mean()) - mean),
            "reconstructed_scaler_scale_abs_difference": abs(float(imputed_train.std(ddof=1)) - scale),
            "retained_scaler_matches_reconstructed_training": bool(
                np.isclose(imputed_train.mean(), mean, rtol=1e-6, atol=1e-9)
                and np.isclose(imputed_train.std(ddof=1), scale, rtol=1e-6, atol=1e-9)
                and np.isclose(reference.median(), median, rtol=1e-6, atol=1e-9)),
        })
    return pd.DataFrame(rows)


def _action_coverage(artifact: Path, target_start: str, target_end: str) -> tuple[pd.DataFrame, dict]:
    training = pd.read_csv(artifact / "action_value_training_frame.csv")
    dataset = _read(artifact / "dataset_audit.json")
    maturity = pd.to_datetime(training.label_available_at, utc=True)
    bounds = dataset["segments"]["train_outer"]
    in_segment = pd.to_datetime(training.signal_date).between(pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]))
    training = training.loc[in_segment & maturity.notna() & maturity.lt(pd.Timestamp(dataset["training_cutoffs"]["train_outer"]))
                            & training.execute_advantage.notna()].copy()
    replay = pd.read_csv(artifact / "portfolio_report.csv")
    training["state_source"] = "baseline_counterfactual_training"
    dev = replay.loc[replay.segment.eq("development_walk_forward")].copy()
    dev["state_source"] = "own_on_policy_development_validation"
    target = replay.loc[replay.trade_date.between(target_start, target_end)].copy()
    target["state_source"] = "own_on_policy_target_window_diagnostic"
    tables = []
    for source in (training, dev, target):
        counts = source.groupby(["state_source", "risk_state"]).agg(rows=("signal_date", "size"),
                   mean_portfolio_drawdown=("portfolio_drawdown", "mean"), mean_beta_cap=("current_beta_cap", "mean")).reset_index()
        counts["fraction_within_source"] = counts.rows / max(1, len(source))
        tables.append(counts)
    return pd.concat(tables, ignore_index=True), {
        "actual_open_model_training_rows": len(training), "action_feature_count": len(dataset["features"]),
        "training_low_target_beta_le_0_3_rows": int((3 * training.target_soxl_weight + training.target_soxx_weight <= .3 + 1e-12).sum()),
        "training_positive_execute_advantage_rows": int(training.execute_advantage.gt(0).sum()),
        "risk_clock_in_action_features": "risk_clock" in dataset["features"],
        "training_state_source": "baseline counterfactual branches; not own-on-policy training",
        "validation_state_source": "own-on-policy equity, peak, history and marked holdings",
        "target_window_state_source": "historical validation replay; not actual broker holdings",
        "target_on_policy_validation_rows": len(target),
        "target_first_trade_date": target.trade_date.min() if len(target) else None,
        "target_last_trade_date": target.trade_date.max() if len(target) else None,
    }


def analog_training_coverage(events: pd.DataFrame, *, primary: pd.DataFrame,
                             auxiliary: pd.DataFrame, base_returns: pd.DataFrame) -> pd.DataFrame:
    """Annotate supplied matches; never retrieve or select analogs here."""
    rows = []
    if events.empty:
        return pd.DataFrame()
    selected = events.loc[events.representative & events.cohort.eq("training_2011_2022")
                          & events.representative_rank.le(3)]
    for event in selected.to_dict("records"):
        start, end = pd.Timestamp(event["support_start"]), pd.Timestamp(event["support_end"])
        aux = auxiliary.loc[pd.to_datetime(auxiliary.datetime).between(start, end)]
        evaluation_start = event["outcome_start"] if event["retrieval_kind"].startswith("B") else event["feature_start"]
        evaluation_end = event["outcome_end"] if event["retrieval_kind"].startswith("B") else event["feature_end"]
        evaluated = base_returns.loc[
            base_returns.segment.eq("development_walk_forward") & base_returns.cost_bps.eq(10)
            & pd.to_datetime(base_returns.trade_date).ge(pd.Timestamp(evaluation_start))
            & pd.to_datetime(base_returns.exit_date).le(pd.Timestamp(evaluation_end))
        ]
        model_return = float(((1 + evaluated.strategy_return).prod() - 1) * 100) if len(evaluated) else None
        soxx_return = (float(((1 + evaluated.soxx_return).prod() - 1) * 100)
                       if len(evaluated) and "soxx_return" in evaluated else None)
        rows.append({"event_id": event["event_id"], "retrieval_kind": event["retrieval_kind"],
                     "window_spec": event["window_spec"], "support_start": start, "support_end": end,
                     "match_distance": event.get("distance"),
                     "match_below_p5_reference": event.get("below_p5_reference"),
                     "match_status": ("within_preregistered_distance_threshold" if event.get("below_p5_reference") is True
                                      else "nearest_only_or_threshold_not_reported"),
                     "primary_mature_training_rows": int(primary.index.to_series().between(start, end).sum()),
                     "auxiliary_fit_rows": len(aux), "auxiliary_unique_dates": aux.datetime.nunique(),
                     "wf_base_model_rows": len(evaluated),
                     "wf_base_model_mean_cash": float(evaluated.cash_weight.mean()) if len(evaluated) else None,
                     "wf_base_model_return_pct": model_return,
                     "matched_soxx_open_to_open_price_return_pct": soxx_return,
                     "wf_base_minus_soxx_price_return_pp": model_return - soxx_return if model_return is not None and soxx_return is not None else None,
                     "wf_first_trade_date": evaluated.trade_date.min() if len(evaluated) else None,
                     "wf_last_exit_date": evaluated.exit_date.max() if len(evaluated) else None,
                     "evaluation_status": "retained_annual_walk_forward_base_only" if len(evaluated) else "unavailable_no_retained_walk_forward_rows",
                     "return_timing": "contained trade_date/exit_date open-to-open intervals; not full Stage1 or analog close-to-close return",
                     "cost_basis": "base retained 10bps convention; SOXX comparison is price-only before trading costs"})
    return pd.DataFrame(rows)


def audit_soxl_training_coverage(
    *, root: Path | str, artifact_root: Path | str | None = None,
    target_start: str = "2026-07-27", target_end: str = "2026-09-25",
    analog_events: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return only small audit tables; caller owns the unified notebook/Recorder."""
    from qlib.data import D

    root, artifacts = Path(root).resolve(), Path(artifact_root or root).resolve()
    base_card, action_card = artifacts / "research" / BASE_CARD, artifacts / "research" / ACTION_CARD
    bundle = validate_frozen_bundle(base_card)
    base_summary, base_artifact = _artifact_dir(base_card)
    action_summary, action_artifact = _artifact_dir(action_card)
    if action_summary["source_stage1"]["allocation_source_recorder_id"] != base_summary["recorder_id"]:
        raise ValueError("blocked: action and base retained source identities differ")
    provenance = _read(base_artifact / "provenance.json")
    selected = _read(base_artifact / "selected_model.json")
    with (base_artifact / "models" / f"{selected['candidate']}.pkl").open("rb") as handle:
        state = pickle.load(handle)["outer_refit"]
    cutoff = provenance["training_spec"]["development_end"]
    spec_values = {key: value for key, value in provenance["training_spec"].items() if key != "root"}
    for key in ("development_years", "outer", "costs_bps"):
        spec_values[key] = tuple(spec_values[key])
    spec = PitAuxiliaryAllocationSpec(root=root, **spec_values)
    source_hashes_before = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in provenance["source_sha256"]}
    market, source_audit, _ = _load_frame(DirectContinuousAllocationSpec(
        root=root, train_start=spec.train_start, development_years=spec.development_years,
        development_end=spec.development_end, outer=spec.outer, purge_rows=spec.purge_rows,
    ), include_incomplete_latest=True)
    market, label_audit = _mature_training_labels(market, cutoff)
    primary = market.loc[provenance["training_spec"]["train_start"]:cutoff].dropna(subset=LABELS)
    common = _soxx_common_training_frame(spec, end=target_end).reindex(market.index)
    target = market.loc[target_start:target_end]
    if target.empty or target.index.max().date().isoformat() != target_end:
        raise ValueError("blocked: target feature window is unavailable in the fixed qlib snapshot")
    primary_drift = feature_drift(primary, target, features=state["primary_features"], scaler=state["primary_scaler"], feature_group="primary_derived_features")
    common_drift = feature_drift(common.reindex(primary.index), common.reindex(target.index), features=state["common_features"],
                                 scaler=state["common_scaler"], feature_group="common_OHLCV_derived_features")
    auxiliary, _ = _training_auxiliary(spec)
    auxiliary = auxiliary.loc[pd.to_datetime(auxiliary.datetime).ge(provenance["training_spec"]["train_start"])]
    auxiliary = auxiliary.dropna(subset=[*COMMON_FEATURES, "synthetic_2x_crash_label", "synthetic_2x_upside_label", "training_weight"])
    auxiliary = auxiliary.loc[pd.to_numeric(auxiliary["$pit_available"]).gt(.5)].copy()
    # The auxiliary reader initializes its Qlib provider; use that native calendar.
    calendar = pd.DatetimeIndex(D.calendar(start_time="2011-01-03", end_time=cutoff, freq="day"))
    training_spec = provenance["training_spec"]
    tables = auxiliary_coverage(auxiliary, calendar=calendar, epochs=training_spec["epochs"], batch_size=training_spec["auxiliary_batch_size"])
    tables["feature_drift"] = pd.concat([primary_drift, common_drift], ignore_index=True)
    tables["action_state_coverage"], action_audit = _action_coverage(action_artifact, target_start, target_end)
    diagnostic = pd.read_csv(base_artifact / "training_diagnostics.csv")
    outer = diagnostic.loc[diagnostic.fold_year.eq("outer_refit")].iloc[0]
    tables["retained_fit_counts"] = diagnostic.loc[:, ["fold_year", "train_end", "primary_train_rows", "auxiliary_train_rows", "auxiliary_crash_rows"]].copy()
    tables["retained_action_fit_counts"] = pd.read_csv(action_artifact / "training_diagnostics.csv")
    tables["primary_year_label_coverage"] = primary.assign(year=primary.index.year).groupby("year").agg(
        rows=("tail_event_5d", "size"), tail_positive_rows=("tail_event_5d", "sum"),
        positive_relative_alpha_rows=("relative_alpha_5d", lambda x: int(x.gt(0).sum()))).reset_index()
    primary_tail_dates = primary.index[primary.tail_event_5d.eq(1)]
    primary_clusters = overlap_clusters(primary_tail_dates, calendar=calendar)
    primary_episodes = primary_clusters.rename_axis("signal_date").reset_index()
    tables["primary_tail_overlap_clusters"] = primary_episodes.groupby("overlap_cluster").agg(
        first_signal_date=("signal_date", "min"), last_signal_date=("signal_date", "max"),
        positive_signal_dates=("signal_date", "size"),
    ).reset_index()
    if analog_events is not None:
        tables["analog_training_coverage"] = analog_training_coverage(
            analog_events, primary=primary, auxiliary=auxiliary,
            base_returns=pd.read_csv(base_artifact / "strategy_returns.csv"),
        )
    primary_heads = training_spec["primary_tail_loss_weight"] * (outer.primary_tail_loss + outer.primary_alpha_loss)
    auxiliary_heads = provenance["candidate"]["auxiliary_weight"] * (outer.auxiliary_tail_loss + outer.auxiliary_alpha_loss)
    tables["loss_scale_diagnostic"] = pd.DataFrame([
        {"component": "allocation", "last_epoch_scalar": outer.allocation_loss},
        {"component": "weighted_primary_BCE_heads", "last_epoch_scalar": primary_heads},
        {"component": "weighted_auxiliary_BCE_heads", "last_epoch_scalar": auxiliary_heads},
    ])
    drift = tables["feature_drift"]
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in provenance["source_sha256"]}
    if hashes != source_hashes_before:
        raise ValueError("blocked: audit source changed during fixed-snapshot reads")
    matches = {name: value == provenance["source_sha256"][name] for name, value in hashes.items()}
    reconstructed = (len(primary) == int(outer.primary_train_rows) and len(auxiliary) == int(outer.auxiliary_train_rows)
                     and bool(drift.retained_scaler_matches_reconstructed_training.all()))
    summary = {
        "status": "audited" if reconstructed else "needs_data_training_reconstruction_mismatch",
        "base_recorder_id": bundle["recorder_id"], "action_recorder_id": action_summary["recorder_id"],
        "training_information_cutoff": cutoff, "label_maturity": label_audit,
        "primary_actual_fit_rows": len(primary), "auxiliary_actual_fit_rows": len(auxiliary),
        "primary_tail_positive_rows": len(primary_tail_dates),
        "primary_tail_operational_overlap_clusters": int(primary_clusters.nunique()),
        "auxiliary_unique_dates": int(auxiliary.datetime.nunique()),
        "auxiliary_positive_crash_rows": int(auxiliary.synthetic_2x_crash_label.eq(1).sum()),
        "auxiliary_operational_overlap_clusters": len(tables["auxiliary_crash_overlap_clusters"]),
        "sampling": {"method": "uniform row sampling with replacement; normalized training_weight multiplies per-row BCE",
                     "epochs": training_spec["epochs"], "batch_size": training_spec["auxiliary_batch_size"],
                     "expected_total_auxiliary_draws": training_spec["epochs"] * min(training_spec["auxiliary_batch_size"], len(auxiliary)),
                     "loss_coefficient": provenance["candidate"]["auxiliary_weight"]},
        "action_training": action_audit,
        "feature_drift": {"target_start": target_start, "target_end": target_end, "target_rows": len(target),
                          "features": len(drift), "latest_features_outside_train_1_99": int(drift.target_latest_outside_q01_q99.fillna(False).sum()),
                          "latest_features_missing": int(drift.target_latest_value.isna().sum()),
                          "window_feature_cells_outside_train_1_99": int(drift.target_outside_q01_q99_rows.sum()),
                          "window_feature_cells_total": int(drift.target_rows.sum()),
                          "window_feature_cells_missing": int(drift.target_missing_rows.sum()),
                          "latest_features_clipped_by_retained_scaler": int(drift.target_latest_unclipped_z.abs().gt(8).sum()),
                          "all_scaler_statistics_reconstructed": bool(drift.retained_scaler_matches_reconstructed_training.all())},
        "source_manifest_hash_matches_retained_training": matches, "current_source_hashes": hashes,
        "source_audit": source_audit,
        "limits": ["No model is trained, selected, promoted or activated by this audit.",
                   "Cross-sectional rows and overlapping labels are not independent market episodes.",
                   "Operational overlap clusters use shared-calendar label overlap across all assets; not IID sample size.",
                   "Yearly overlap-cluster counts are not additive when a shared episode crosses a year boundary.",
                   "Nominal loss shares and last-epoch loss values do not identify gradient influence or final model causality.",
                   "Reported loss components are from the last epoch; the saved best-loss state may be earlier.",
                   "Synthetic 2x stock labels are training-only hypotheses, not observed leveraged-ETF performance.",
                   "Training-only quantile outliers are descriptive drift diagnostics, not proof of model failure.",
                   "Marginal P1-P99 support does not establish coverage of joint feature sequences or portfolio-controller paths.",
                   "Any resampling must keep related assets and synthetic descendants in the same held-out calendar episode."],
    }
    artifact_files = [base_artifact / name for name in ("provenance.json", "selected_model.json", "training_diagnostics.csv", "strategy_returns.csv")]
    artifact_files.extend(action_artifact / name for name in ("dataset_audit.json", "action_value_training_frame.csv", "portfolio_report.csv", "training_diagnostics.csv"))
    artifact_references = {
        str(path.relative_to(artifacts)): {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}
        for path in artifact_files
    }
    base_returns = pd.read_csv(base_artifact / "strategy_returns.csv")
    action_validation = pd.read_csv(action_artifact / "portfolio_report.csv")
    inputs = {
        "primary_training_features_and_labels": primary.loc[:, [*state["primary_features"], *LABELS]].copy(),
        "primary_target_features": target.loc[:, state["primary_features"]].copy(),
        "common_training_features": common.reindex(primary.index).loc[:, state["common_features"]].copy(),
        "common_target_features": common.reindex(target.index).loc[:, state["common_features"]].copy(),
        "auxiliary_effective_training_rows": auxiliary.loc[:, ["datetime", "instrument", "$pit_available", "training_weight",
            "synthetic_2x_crash_label", "synthetic_2x_upside_label", *COMMON_FEATURES]].copy(),
        "shared_training_calendar": calendar,
        "action_counterfactual_training_rows": pd.read_csv(action_artifact / "action_value_training_frame.csv"),
        "action_dataset_audit": _read(action_artifact / "dataset_audit.json"),
        "action_validation_state_rows": action_validation.loc[:, ["signal_date", "trade_date", "segment", "risk_state", "portfolio_drawdown", "current_beta_cap"]].copy(),
        "base_walk_forward_return_rows": base_returns.loc[base_returns.cost_bps.eq(10), ["signal_date", "trade_date", "exit_date", "segment", "cost_bps", "cash_weight", "strategy_return", "soxx_return"]].copy(),
        "retained_primary_scaler": state["primary_scaler"], "retained_common_scaler": state["common_scaler"],
        "training_spec": training_spec, "candidate": provenance["candidate"],
        "retained_artifact_references": artifact_references,
        "usage": "immutable audit inputs only; no training or production activation",
    }
    if analog_events is not None:
        inputs["analog_events"] = analog_events.copy()
    summary["immutable_input_artifacts"] = artifact_references
    return {"summary": summary, "tables": tables, "inputs": inputs}
