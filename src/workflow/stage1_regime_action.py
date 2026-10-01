"""Causal regime-conditioned execution value on the policy's settled holdings.

Qlib is the source of regime observations and the training Dataset. The economic
router remains responsible for legal targets, risk clocks, fee accounting and
conditional recovery. This module changes only whether an ordinary open target
is worth executing; required repairs always execute.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.workflow.stage1_economic_action import ECONOMIC_FEATURES, RISK_TOLERANCE, prepare_economic_open
from src.workflow.stage1_economic_training import (
    LABEL_COST_BPS, _Portfolio, _event_at, _finalize_label, _model_at,
    _ordered_frame, _record_day, _settle_branch,
)

REGIME_PROBABILITIES = ("regime_up_probability", "regime_range_probability", "regime_down_probability")
REGIME_FEATURES = (
    "regime_momentum_20", "regime_momentum_60", "regime_efficiency_20",
    "regime_volatility_20", "regime_drawdown_60",
    *REGIME_PROBABILITIES,
)
REGIME_ACTION_DELTAS = ("effective_beta_delta", "open_turnover", "soxl_weight_delta", "soxx_weight_delta")
REGIME_SCHEMA_VERSION = "causal_soxx_close_regime_v1"


@dataclass(frozen=True)
class RegimeActionSpec:
    name: str
    estimator: str = "ridge"
    use_regime: bool = True
    ridge_alpha: float = 10.0
    min_advantage_bps: float = 0.0
    protect_derisk: bool = True
    minimum_training_rows: int = 200
    max_leaf_nodes: int = 7
    max_iter: int = 80
    l2_regularization: float = 10.0

    def __post_init__(self):
        if not self.name or self.estimator not in {"ridge", "hist_gradient_boosting"}:
            raise ValueError("blocked: invalid preregistered regime estimator")
        values = (self.ridge_alpha, self.min_advantage_bps, self.l2_regularization)
        if not np.isfinite(values).all() or self.ridge_alpha <= 0 or self.min_advantage_bps < 0 or self.l2_regularization < 0:
            raise ValueError("blocked: invalid regime action parameters")
        if self.minimum_training_rows < 2 or not 2 <= self.max_leaf_nodes <= 15 or not 1 <= self.max_iter <= 200:
            raise ValueError("blocked: regime model complexity exceeds bounded contract")


def regime_feature_names(spec: RegimeActionSpec) -> tuple[str, ...]:
    if not spec.use_regime:
        return tuple(ECONOMIC_FEATURES)
    return (*ECONOMIC_FEATURES, *REGIME_FEATURES,
            *(f"{probability}__{delta}" for probability in REGIME_PROBABILITIES for delta in REGIME_ACTION_DELTAS))


def _validate_regime(frame: pd.DataFrame) -> None:
    missing = set(REGIME_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"blocked: missing causal regime features {sorted(missing)}")
    values = frame.loc[:, list(REGIME_FEATURES)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("blocked: missing or nonfinite causal regime observations; warmup is not complete")
    probabilities = frame.loc[:, list(REGIME_PROBABILITIES)].to_numpy(dtype=float)
    if (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-12, rtol=0):
        raise ValueError("blocked: invalid causal regime probabilities")


def compute_regime_features(observations: pd.DataFrame) -> pd.DataFrame:
    """Transform Qlib close observations with backward-only windows.

    Probabilities are descriptive soft states, not fitted future regime labels.
    The first 62 observations remain unavailable; there is no forward/back fill.
    """
    if not isinstance(observations.index, pd.DatetimeIndex) or observations.index.has_duplicates or not observations.index.is_monotonic_increasing:
        raise ValueError("blocked: regime observations need a unique ordered Qlib date index")
    if "$close" not in observations.columns:
        raise ValueError("blocked: regime observations require Qlib close")
    raw = observations.loc[:, ["$close"]].astype(float)
    if not np.isfinite(raw.to_numpy()).all() or (raw["$close"] <= 0).any():
        raise ValueError("blocked: invalid Qlib SOXX close observations")
    close = raw["$close"]
    log_close = np.log(close)
    returns = log_close.diff()
    volatility = returns.rolling(20, min_periods=20).std(ddof=0)
    momentum20, momentum60 = log_close.diff(20), log_close.diff(60)
    travel = returns.abs().rolling(20, min_periods=20).sum()
    efficiency = momentum20.abs() / travel.clip(lower=1e-12)
    trend = .5 * (momentum20 / (volatility.clip(lower=1e-8) * np.sqrt(20))
                  + momentum60 / (volatility.clip(lower=1e-8) * np.sqrt(60)))
    trend = trend.clip(-3., 3.)
    logits = np.column_stack([1.5 * trend, 1.25 * (1 - efficiency), -1.5 * trend])
    finite = np.isfinite(logits).all(axis=1)
    probabilities = np.full_like(logits, np.nan)
    stable = logits[finite] - logits[finite].max(axis=1, keepdims=True)
    probabilities[finite] = np.exp(stable) / np.exp(stable).sum(axis=1, keepdims=True)
    result = pd.DataFrame({
        "regime_momentum_20": momentum20, "regime_momentum_60": momentum60,
        "regime_efficiency_20": efficiency, "regime_volatility_20": volatility,
        "regime_drawdown_60": close / close.rolling(60, min_periods=60).max() - 1,
    }, index=raw.index)
    result.loc[:, list(REGIME_PROBABILITIES)] = pd.DataFrame(
        probabilities, index=raw.index, columns=REGIME_PROBABILITIES,
    ).rolling(3, min_periods=3).mean()
    result.index.name = "signal_date"
    return result


def load_regime_features(provider_uri: str | Path, start, end) -> pd.DataFrame:
    """Load SOXX through Qlib and retain exact signal dates with full warmup."""
    import qlib
    from qlib.data import D

    provider = Path(provider_uri).expanduser().resolve()
    if not provider.is_dir():
        raise ValueError(f"blocked: qlib provider unavailable: {provider}")
    start_at, end_at = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if pd.isna(start_at) or pd.isna(end_at) or start_at > end_at:
        raise ValueError("blocked: invalid regime data window")
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    calendar = pd.DatetimeIndex(D.calendar(end_time=end_at, freq="day"))
    prior = calendar[calendar < start_at]
    if len(prior) < 62:
        raise ValueError("blocked: fewer than 62 prior Qlib sessions for regime warmup")
    history_start = prior[-62]
    loaded = D.features(["SOXX"], ["$close"], start_time=history_start, end_time=end_at, freq="day")
    if loaded.empty or set(loaded.index.get_level_values("instrument")) != {"SOXX"}:
        raise ValueError("blocked: Qlib SOXX regime observations unavailable")
    raw = loaded.xs("SOXX", level="instrument").sort_index()
    expected = calendar[(calendar >= history_start) & (calendar <= end_at)]
    if not raw.index.equals(expected):
        raise ValueError("blocked: SOXX observations do not cover the Qlib calendar")
    result = compute_regime_features(raw).loc[start_at:end_at]
    if result.empty:
        raise ValueError("blocked: empty Qlib regime window")
    _validate_regime(result)
    result.attrs["regime_source"] = {
        "provider_uri": str(provider), "source_api": "qlib.data.D.features", "calendar_api": "qlib.data.D.calendar",
        "instrument": "SOXX", "fields": ["$close"], "schema_version": REGIME_SCHEMA_VERSION,
        "feature_scope": "price_only; canonical daily provider does not contain historical SOXX volume",
        "warmup_start": history_start.date().isoformat(), "signal_start": str(result.index.min().date()),
        "signal_end": str(result.index.max().date()), "future_fill": False,
        "source_sha256": hashlib.sha256(pd.util.hash_pandas_object(raw, index=True).values.tobytes()).hexdigest(),
    }
    return result


def attach_regime_features(frame: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Exact-date one-to-one join; unavailable dates or stale features block."""
    if "signal_date" not in frame or features.index.has_duplicates or features.empty:
        raise ValueError("blocked: invalid regime attachment dates")
    _validate_regime(features)
    dates = pd.to_datetime(frame.signal_date)
    if dates.isna().any() or dates.duplicated().any() or not dates.isin(features.index).all():
        raise ValueError("blocked: missing exact signal-date regime observations")
    if set(REGIME_FEATURES).intersection(frame.columns):
        raise ValueError("blocked: regime features already attached")
    result = frame.copy()
    result.loc[:, list(REGIME_FEATURES)] = features.reindex(pd.DatetimeIndex(dates)).loc[:, list(REGIME_FEATURES)].to_numpy()
    result.attrs.update(frame.attrs)
    result.attrs["regime_source"] = dict(features.attrs.get("regime_source", {}))
    return result


def prepare_regime_open(row: pd.Series, *, pretrade_weights: np.ndarray, equity: float, peak: float,
                        risk_state: int, risk_clock: int, history: list[float], recovery_spec,
                        model: Any | None, spec: RegimeActionSpec, days_since_trade: int = 0,
                        cost_bps: float = 10.) -> dict[str, Any]:
    opened = prepare_economic_open(
        row, pretrade_weights=pretrade_weights, equity=equity, peak=peak, risk_state=risk_state,
        risk_clock=risk_clock, history=history, recovery_spec=recovery_spec, model=None,
        policy="net_value", days_since_trade=days_since_trade, cost_bps=cost_bps,
    )
    features = dict(opened["features"])
    if spec.use_regime:
        _validate_regime(pd.DataFrame([row]))
        features.update({column: float(row[column]) for column in REGIME_FEATURES})
        features.update({f"{probability}__{delta}": features[probability] * features[delta]
                         for probability in REGIME_PROBABILITIES for delta in REGIME_ACTION_DELTAS})
    values = pd.DataFrame([features]).loc[:, list(regime_feature_names(spec))]
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("blocked: nonfinite regime action features")
    if model is not None and getattr(model, "regime_action_spec", asdict(spec)) != asdict(spec):
        raise ValueError("blocked: regime model and runtime specification differ")
    protected_derisk = bool(spec.protect_derisk and features["effective_beta_delta"] < -RISK_TOLERANCE)
    # A mandatory legal repair or explicitly protected cut does not need an
    # economic score. Estimator failures cannot suppress these actions; input
    # finiteness and sealed-model compatibility still fail closed above.
    prediction_evaluated = model is not None and not (opened["hard_risk_violation"] or protected_derisk)
    predicted = float(model.predict(values)[0]) if prediction_evaluated else np.nan
    if opened["hard_risk_violation"]:
        execute, reason = True, "hard_risk_repair"
    elif protected_derisk:
        execute, reason = True, "protected_beta_reduction"
    elif model is None:
        execute, reason = True, "model_unavailable_execute_target"
    elif not np.isfinite(predicted):
        execute, reason = True, "nonfinite_prediction_execute_target"
    else:
        execute = predicted > spec.min_advantage_bps
        reason = "regime_positive_net_advantage" if execute else "regime_insufficient_net_advantage"
    pretrade = np.asarray(pretrade_weights, dtype=float)
    execution = opened["candidate_execution_target_weights"]
    opened.update(
        policy="regime_net_value", features=features, execute_open_target=bool(execute), decision_reason=reason,
        predicted_net_advantage_bps=predicted, protected_derisk=protected_derisk,
        prediction_evaluated=prediction_evaluated,
        execution_target_weights=execution.copy() if execute else pretrade.copy(),
        open_weights=opened["desired_open_weights"].copy() if execute else pretrade.copy(),
        execution_turnover=float(np.abs(execution[:2] - pretrade[:2]).sum()) if execute else 0.,
        routing_eligible=not opened["hard_risk_violation"] and not protected_derisk,
        regime_action_spec=asdict(spec), regime_schema_version=REGIME_SCHEMA_VERSION,
    )
    return opened


def build_regime_dataset(training_rows: pd.DataFrame, *, spec: RegimeActionSpec):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    if training_rows.empty:
        raise ValueError("blocked: no own-policy regime training rows")
    ordered = training_rows.sort_values("signal_date")
    static = pd.concat({"feature": ordered.loc[:, list(regime_feature_names(spec))].astype(float),
                        "label": ordered.loc[:, ["execute_advantage_bps"]].astype(float),
                        "metadata": ordered.loc[:, ["label_available_at"]]}, axis=1)
    dates = pd.to_datetime(ordered.signal_date)
    static.index = pd.MultiIndex.from_arrays([dates, np.full(len(dates), "REGIME_OPEN")], names=["datetime", "instrument"])
    return DatasetH(handler=DataHandlerLP(instruments=None, start_time=dates.min(), end_time=dates.max(),
        data_loader=StaticDataLoader(config=static), infer_processors=[], learn_processors=[]),
        segments={"all": (dates.min(), dates.max())})


def mature_regime_rows(dataset, *, spec: RegimeActionSpec, model_year: int,
                       first_decision_at: pd.Timestamp) -> tuple[pd.DataFrame, pd.Timestamp]:
    prepared = dataset.prepare("all", col_set=["feature", "label", "metadata"])
    prepared.columns = [str(column[-1]) for column in prepared.columns]
    decision = pd.Timestamp(first_decision_at)
    if decision.tzinfo is None:
        raise ValueError("blocked: regime fit decision must be timezone-aware")
    cutoff = min(pd.Timestamp(f"{model_year}-01-01", tz="America/New_York").tz_convert("UTC"), decision.tz_convert("UTC"))
    available = pd.to_datetime(prepared.label_available_at, utc=True)
    years = pd.to_datetime(prepared.index.get_level_values("datetime")).year
    mature = prepared.loc[available.notna() & available.lt(cutoff) & (years < model_year)].dropna(subset=["execute_advantage_bps"])
    if not np.isfinite(mature.loc[:, [*regime_feature_names(spec), "execute_advantage_bps"]].to_numpy(dtype=float)).all():
        raise ValueError("blocked: nonfinite mature regime DatasetH values")
    return mature, cutoff


def _fit(rows: list[dict], *, spec: RegimeActionSpec, model_year: int, decision_at: pd.Timestamp):
    dataset = build_regime_dataset(pd.DataFrame(rows), spec=spec)
    prepared, cutoff = mature_regime_rows(dataset, spec=spec, model_year=model_year, first_decision_at=decision_at)
    model = None
    if len(prepared) >= spec.minimum_training_rows:
        if spec.estimator == "ridge":
            model = make_pipeline(StandardScaler(), Ridge(alpha=spec.ridge_alpha))
        else:
            model = HistGradientBoostingRegressor(max_leaf_nodes=spec.max_leaf_nodes, max_iter=spec.max_iter,
                l2_regularization=spec.l2_regularization, learning_rate=.05, min_samples_leaf=40,
                early_stopping=False, random_state=17)
        model.fit(prepared.loc[:, list(regime_feature_names(spec))], prepared.execute_advantage_bps)
        model.regime_action_spec = asdict(spec)
        model.regime_schema_version = REGIME_SCHEMA_VERSION
    audit = {
        "model_year": model_year, "training_rows": len(prepared), "model_available": model is not None,
        "training_signal_end": str(prepared.index.get_level_values("datetime").max().date()) if len(prepared) else None,
        "label_available_at_max": pd.to_datetime(prepared.label_available_at, utc=True).max().isoformat() if len(prepared) else None,
        "information_cutoff_exclusive": cutoff.isoformat(), "first_test_decision_at": decision_at.isoformat(),
        "training_state_source": "causal_own_policy_path", "dataset_class": "qlib.data.dataset.DatasetH",
        "selection_uses_future_year": False, "label_cost_bps": LABEL_COST_BPS, **asdict(spec),
        "training_sha256": hashlib.sha256(pd.util.hash_pandas_object(prepared, index=True).values.tobytes()).hexdigest(),
        "feature_schema_sha256": hashlib.sha256(json.dumps(regime_feature_names(spec)).encode()).hexdigest(),
    }
    if model is not None:
        model.regime_training_audit = audit
    return model, audit


def _run(frame, *, models, recovery_models, source_spec, selected_model, spec, cost_bps,
         startcash, reset_on_segment, update_through_year):
    provenance = dict(frame.attrs.get("regime_source", {}))
    frame = _ordered_frame(frame, allow_terminal_close=update_through_year is None)
    if spec.use_regime:
        _validate_regime(frame)
    if not np.isfinite(startcash) or startcash <= 0:
        raise ValueError("blocked: invalid regime simulation capital")
    source_spec = replace(source_spec, transaction_cost_bps=float(cost_bps))
    state = _Portfolio(float(startcash), float(startcash))
    rows, training_rows, diagnostics = [], [], []
    models = dict(models or {})
    fitted_years = {}
    pending = previous_segment = last_year = last_fit_year = last_model = None

    def open_at(row, *, model, cost):
        return prepare_regime_open(row, pretrade_weights=state.weights.copy(), equity=state.equity,
            peak=state.peak, risk_state=state.risk_state, risk_clock=state.risk_clock, history=list(state.history),
            recovery_spec=replace(source_spec, transaction_cost_bps=cost), model=model, spec=spec,
            days_since_trade=state.days_since_trade, cost_bps=cost)

    for _, row in frame.iterrows():
        if reset_on_segment and previous_segment is not None and row.segment != previous_segment:
            if pending is not None:
                pending[0]["label_status"] = "unavailable_segment_reset_boundary"
            pending = None
            state = _Portfolio(float(startcash), float(startcash))
        year = int(row.signal_date.year)
        if update_through_year is not None and year != last_year:
            if year <= update_through_year + 1 and training_rows:
                last_model, audit = _fit(training_rows, spec=spec, model_year=year, decision_at=_event_at(row.trade_date, minute=31))
                diagnostics.append(audit)
                last_fit_year = year
            models[year] = last_model
            fitted_years[year] = last_fit_year
        model = _model_at(models, year)
        fit_year = fitted_years.get(year)
        if update_through_year is None and model is not None:
            fit_year = min(int(key) for key, value in models.items() if value is model)
        age = state.days_since_trade
        opened = open_at(row, model=model, cost=cost_bps)
        recovery_model = _model_at(recovery_models, year)
        if update_through_year is not None:
            label_opened = open_at(row, model=model, cost=LABEL_COST_BPS)
            if pending is not None:
                _finalize_label(pending[0], pending[1], next_row=row, next_target=label_opened["desired_open_weights"])
            branches = {name: _settle_branch(row, pretrade=state.weights.copy(), opened=label_opened,
                execute=execute, cost_bps=LABEL_COST_BPS, recovery_model=recovery_model,
                source_spec=source_spec, selected_model=selected_model) for name, execute in (("execute", True), ("hold", False))}
            sample = {"signal_date": row.signal_date, "trade_date": row.trade_date, "exit_date": row.exit_date,
                "year": year, "segment": row.segment, **opened["features"], "decision_at": _event_at(row.trade_date, minute=31),
                "execute_advantage_bps": np.nan, "label_available_at": pd.NaT, "label_status": "pending_next_reset_target",
                "execute_net_return": branches["execute"]["settled"]["net_return"],
                "hold_net_return": branches["hold"]["settled"]["net_return"],
                "execute_recovery_action": bool(branches["execute"]["recovery"]["recovery_action"]),
                "hold_recovery_action": bool(branches["hold"]["recovery"]["recovery_action"]),
                "label_target_cost_bps": LABEL_COST_BPS,
                **{f"label_target_{asset}_weight": float(label_opened["desired_open_weights"][index]) for index, asset in enumerate(("soxl", "soxx", "cash"))}}
            training_rows.append(sample)
            pending = sample, branches
        branch = _settle_branch(row, pretrade=state.weights.copy(), opened=opened,
            execute=bool(opened["execute_open_target"]), cost_bps=cost_bps, recovery_model=recovery_model,
            source_spec=source_spec, selected_model=selected_model)
        settled = branch["settled"]
        state.risk_state, state.risk_clock = opened["risk_state"], opened["risk_clock"]
        state.weights = settled["ending_weights"].copy()
        state.history.append(float(settled["net_return"]))
        state.equity *= settled["net_equity"]
        state.peak = max(state.peak, state.equity)
        state.days_since_trade = 0 if settled["turnover"] > 1e-8 else state.days_since_trade + 1
        record = _record_day(row, opened=opened, branch=branch, state=state, cost_bps=cost_bps,
            decision_days_since_trade=age, model_year=fit_year)
        record.update({column: row.get(column, np.nan) for column in REGIME_FEATURES})
        record.update(protected_derisk=opened["protected_derisk"], regime_candidate=spec.name)
        if spec.use_regime:
            record["regime"] = ("up", "range", "down")[int(np.argmax([row[column] for column in REGIME_PROBABILITIES]))]
        rows.append(record)
        previous_segment, last_year = row.segment, year
    # A development-only notebook need not load any outer-period observation
    # merely to trigger the final annual refit. The conservative January 1
    # cutoff excludes all labels not already mature before the new year.
    if update_through_year is not None and last_year == update_through_year and training_rows:
        final_year = update_through_year + 1
        cutoff = pd.Timestamp(f"{final_year}-01-01", tz="America/New_York").tz_convert("UTC")
        last_model, audit = _fit(training_rows, spec=spec, model_year=final_year, decision_at=cutoff)
        audit.update(first_test_decision_at=None, fit_trigger="development_end_refit", fit_information_asof=cutoff.isoformat())
        diagnostics.append(audit)
        last_fit_year = final_year
        models[final_year] = last_model
        fitted_years[final_year] = final_year
    return {"replay": pd.DataFrame(rows), "models": models, "training_rows": pd.DataFrame(training_rows),
        "diagnostics": pd.DataFrame(diagnostics), "final_model": last_model, "final_model_year": last_fit_year,
        "dataset_audit": {"dataset_class": "qlib.data.dataset.DatasetH", "handler_class": "qlib.data.dataset.handler.DataHandlerLP",
            "loader_class": "qlib.data.dataset.loader.StaticDataLoader", "features": list(regime_feature_names(spec)),
            "training_state": "causal_own_policy_path", "evaluation_segment_reset": reset_on_segment,
            "training_history_reset_on_segment": False, "label_cost_bps": LABEL_COST_BPS,
            "label_proxy": "one session execute-versus-hold net value plus common next own-policy desired-target reset cost; not a multiday policy rollout",
            "maturity": "label_available_at strictly before both January 1 New York of fit year and first fit-year decision",
            "recovery_branches": "separately recomputed from each branch's marked holdings; same frozen open controller snapshot",
            "maximum_training_signal_year": update_through_year, "fit_year_by_decision_year": fitted_years,
            "regime_schema_version": REGIME_SCHEMA_VERSION, "regime_source": provenance, "spec": asdict(spec)}}


def train_regime_models(frame: pd.DataFrame, *, recovery_models: dict[int, Any], source_spec,
                        selected_model: dict, spec: RegimeActionSpec, update_through_year: int = 2022,
                        cost_bps: float = 10., startcash: float = 1., reset_on_segment: bool = True) -> dict[str, Any]:
    if update_through_year not in {2022, 2025}:
        raise ValueError("blocked: only preregistered 2022 or 2025 information windows are permitted")
    return _run(frame, models=None, recovery_models=recovery_models, source_spec=source_spec,
        selected_model=selected_model, spec=spec, cost_bps=cost_bps, startcash=startcash,
        reset_on_segment=reset_on_segment, update_through_year=update_through_year)


def simulate_regime_policy(frame: pd.DataFrame, *, models: dict[int, Any], recovery_models: dict[int, Any],
                           source_spec, selected_model: dict, spec: RegimeActionSpec, cost_bps: float = 10.,
                           startcash: float = 1., reset_on_segment: bool = True) -> pd.DataFrame:
    return _run(frame, models=models, recovery_models=recovery_models, source_spec=source_spec,
        selected_model=selected_model, spec=spec, cost_bps=cost_bps, startcash=startcash,
        reset_on_segment=reset_on_segment, update_through_year=None)["replay"]


__all__ = ["RegimeActionSpec", "REGIME_FEATURES", "REGIME_PROBABILITIES", "REGIME_SCHEMA_VERSION",
           "regime_feature_names", "compute_regime_features", "load_regime_features", "attach_regime_features",
           "prepare_regime_open", "build_regime_dataset", "mature_regime_rows", "train_regime_models", "simulate_regime_policy"]
