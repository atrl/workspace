"""Single-vintage layer attribution of the frozen f696 allocator + on-policy action/recovery.

Five pre-registered layers, each advancing its own equity/peak/history/risk-clock/cash/qty
state. Zero fits. The open/recovery steps use the on-policy helpers
(``prepare_on_policy_open`` / ``prepare_on_policy_recovery``) with only HOLD/recovery toggled;
``full_withhold_and_recovery`` must reproduce the e4 uniform ``active_hold_baseline`` day-by-day.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.workflow import coherent_price_replay_experiment as C
from src.workflow import feasible_action_experiment as X
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.workflow.stage1_open_action_value import (
    Stage1OpenActionValueSpec,
    _simulate_on_policy_router,
    prepare_on_policy_open,
    prepare_on_policy_recovery,
)

F696_DIR = ("research/pit_constituent_auxiliary_training_allocation_retrained_20260928/qlib_recorder/"
            "188464491611747257/f696f656b3d44e868588e09a5d97c905/artifacts/"
            "pit_constituent_auxiliary_training_allocation_retrained_20260928")
EXPECTED_RAW_SHA256 = "f532c84b5c3c05d181092044aa1bc6223e26e774d81a26f3bfa181828bdf342e"
LAYERS = ("raw_allocator_only", "open_risk_projection_nohold_no_recovery",
          "open_risk_projection_withhold_no_recovery", "full_withhold_and_recovery",
          "nohold_with_recovery_reference")
COSTS = (10.0, 25.0)
SEGMENTS = {"development_walk_forward": "development_walk_forward", "outer_diagnostic": "outer_reused_diagnostic"}
EXPECTED_ROWS = {"development_walk_forward": 1756, "outer_diagnostic": 934}
RAW_CONVENTION = "post_fee_desired"
LEGACY_CONVENTION = "pre_fee_account"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_raw_targets(root: Path) -> tuple[pd.DataFrame, dict]:
    """Read the full frozen inference snapshot; independently crosscheck original outputs."""
    original_path = root / F696_DIR / "predictions.csv"
    if _sha(original_path) != EXPECTED_RAW_SHA256:
        raise ValueError("blocked: original f696 prediction identity changed")
    prep = C.load_frozen(root)["inputs/prepared_inputs.pkl"]
    audit = prep["source_audit"]
    if not (audit["source_recorder_id"] == "f696f656b3d44e868588e09a5d97c905"
            and audit["source_model_frozen"] is True
            and audit["source_refresh_mode"] == "pinned_recorder_frozen_model_inference_only"
            and audit["source_training_end"] == "2022-12-30"
            and audit["source_prediction_end"] == "2026-09-23"):
        raise ValueError("blocked: unverified frozen inference extension provenance")
    raw = prep["frame"][["signal_date", "segment", "soxl_weight", "soxx_weight", "cash_weight"]].rename(
        columns={"soxl_weight": "raw_soxl", "soxx_weight": "raw_soxx", "cash_weight": "raw_cash"}).copy()
    original = pd.read_csv(original_path, parse_dates=["signal_date", "train_end"]).rename(
        columns={"soxl_weight": "raw_soxl", "soxx_weight": "raw_soxx", "cash_weight": "raw_cash"})
    raw["signal_date"] = pd.to_datetime(raw["signal_date"])
    if raw.signal_date.duplicated().any() or original.signal_date.duplicated().any():
        raise ValueError("blocked: duplicate raw target dates")
    overlap = original.merge(raw, on=["signal_date", "segment"], how="left", validate="one_to_one",
                             suffixes=("_original", ""), indicator=True)
    if len(overlap) != 2686 or not overlap._merge.eq("both").all():
        raise ValueError("blocked: original prediction overlap incomplete")
    columns = ["raw_soxl", "raw_soxx", "raw_cash"]
    maxerr = float(np.abs(overlap[columns].to_numpy(float) -
                         overlap[[c + "_original" for c in columns]].to_numpy(float)).max())
    if maxerr > 1e-12:
        raise ValueError("blocked: frozen snapshot disagrees with original raw outputs")
    raw = raw.merge(original[["signal_date", "segment", "train_end"]], on=["signal_date", "segment"],
                    how="left", validate="one_to_one")
    extension = sorted(raw.loc[raw.train_end.isna(), "signal_date"].dt.strftime("%Y-%m-%d"))
    if extension != ["2026-09-18", "2026-09-21", "2026-09-22", "2026-09-23"]:
        raise ValueError("blocked: frozen inference extension dates changed")
    raw.loc[raw.train_end.isna(), "train_end"] = pd.Timestamp(audit["source_training_end"])
    raw["target_source"] = np.where(raw.signal_date.dt.strftime("%Y-%m-%d").isin(extension),
                                   "parent17_frozen_inference_extension", "f696_original_overlap")
    raw_coverage(raw, prep["frame"])
    return raw, {"source": "parent17 inputs/prepared_inputs.pkl[frame]", "rows": len(raw),
                 "prepared_sha256": C.SOURCE_OBJECT_SHA256["inputs/prepared_inputs.pkl"],
                 "original_predictions_sha256": EXPECTED_RAW_SHA256, "original_rows": len(original),
                 "f696_crosscheck_maxerr": maxerr, "extension_dates": extension,
                 "extension_training_cutoff": audit["source_training_end"],
                 "inference_extension_to": audit["source_prediction_end"],
                 "original_train_cutoffs_retained": True}


def raw_coverage(raw: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """Block missing, extra, duplicate, illegal or late-trained targets before settlement."""
    columns = ["raw_soxl", "raw_soxx", "raw_cash"]
    weights = raw[columns].to_numpy(float)
    if not np.isfinite(weights).all() or (weights < -1e-12).any() or not np.allclose(
            weights.sum(1), 1.0, atol=1e-12, rtol=0):
        raise ValueError("blocked: illegal raw weights")
    out = {}
    for segment, count in EXPECTED_ROWS.items():
        market = panel[panel.segment.eq(segment)].sort_values("signal_date")
        targets = raw[raw.segment.eq(segment)].sort_values("signal_date")
        dates = pd.to_datetime(market.signal_date).to_numpy()
        if (len(market) != count or len(targets) != count or market.signal_date.duplicated().any()
                or targets.signal_date.duplicated().any()
                or not np.array_equal(dates, pd.to_datetime(targets.signal_date).to_numpy())):
            raise ValueError(f"blocked: incomplete exact raw calendar for {segment}")
        if not pd.to_datetime(targets.train_end).lt(pd.to_datetime(targets.signal_date)).all():
            raise ValueError("blocked: raw training cutoff not before signal")
        if not pd.to_datetime(market.signal_date).lt(pd.to_datetime(market.trade_date)).all():
            raise ValueError("blocked: signal not before execution")
        out[SEGMENTS[segment]] = {"expected": count, "raw": count, "missing": [], "complete": True}
    if set(raw.segment) != set(SEGMENTS) or set(panel.segment) != set(SEGMENTS):
        raise ValueError("blocked: unexpected segment")
    return out


def _metrics(replay: pd.DataFrame) -> dict:
    r = replay["strategy_return"].to_numpy(float)
    growth = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    peak = np.maximum.accumulate(growth)
    tail = np.sort(r)[:max(1, int(np.ceil(0.05 * r.size)))]
    rolling5 = pd.Series(r).rolling(5).apply(lambda x: float(np.prod(1.0 + x) - 1.0), raw=True)
    std = r.std(ddof=1)
    return {"sessions": int(r.size), "return_pct": 100.0 * float(growth[-1] - 1.0),
            "max_drawdown_pct": 100.0 * float((growth / peak - 1.0).min()),
            "cvar5_pct": 100.0 * float(tail.mean()), "worst_day_pct": 100.0 * float(r.min()),
            "worst_rolling_5d_pct": 100.0 * float(rolling5.min()),
            "sharpe": float(r.mean() / std * np.sqrt(252)) if std > 0 else None,
            "total_turnover": float(replay["turnover"].sum()),
            "avg_ending_cash_weight_pct": float(100.0 * replay["ending_cash_weight"].mean())
            if "ending_cash_weight" in replay else None}


def _layer_loop(market: pd.DataFrame, *, spec, recovery_spec, models: dict, recovery_models: dict,
                selected_model: dict, cost: float, use_hold: bool, use_recovery: bool,
                funding_convention: str, raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Replicate the on-policy router loop with only HOLD/recovery toggled.

    Records the exact pre-funding desired target, recovery target, decision flags and
    the priced asset/early returns so the fee ledger can re-settle the SAME path.
    """
    from src.backtest.position_accounting import settle_target_session
    from src.workflow.stage1_economic_action import fee_affordable_target

    equity = peak = 1.0
    state = clock = 0
    history: list[float] = []
    previous = np.array([0.0, 0.0, 1.0])
    rows = []
    raw_index = raw.set_index("signal_date") if raw is not None else None
    for _, row in market.sort_values("signal_date").iterrows():
        year = int(row["signal_date"].year)
        drawdown_before = max(0.0, 1.0 - equity / peak)
        state_before, clock_before = state, clock
        raw_beta = 3.0 * float(row["soxl_weight"]) + float(row["soxx_weight"])
        trace = {"drawdown_before": drawdown_before,
                 "risk_state_before": state, "risk_clock_before": clock,
                 "raw_beta": raw_beta, "pretrade_beta": float(3.0 * previous[0] + previous[1])}
        if raw is not None:
            r = raw_index
            if row["signal_date"] in r.index:
                desired = np.array([r.at[row["signal_date"], "raw_soxl"], r.at[row["signal_date"], "raw_soxx"],
                                    r.at[row["signal_date"], "raw_cash"]])
                execute, exec_reason = True, "raw_execute_target"
            else:
                raise ValueError("blocked: missing raw target at settlement")
            rec_target = rec_action = None
            state_after, clock_after = state, clock
            trace.update(current_beta_cap=None, risk_q=None,
                         stop_entered=False, emergency_active=False, trailing_loss_active=False,
                         clock_cap_binding=False, emergency_cap_binding=False,
                         trailing_cap_binding=False, max_soxl_binding=False, min_cash_binding=False)
        else:
            opened = prepare_on_policy_open(row, pretrade_weights=previous, equity=equity, peak=peak,
                                            risk_state=state, risk_clock=clock, history=history,
                                            recovery_spec=recovery_spec, model=(models.get(year) if use_hold else None),
                                            action_spec=spec)
            execute = bool(opened["execute_open_target"])  # honor helper flag (incl. repair/fallback)
            desired = opened["desired_open_weights"]
            exec_reason = opened.get("decision_reason", "on_policy_open")
            rec_target = rec_action = None
            if use_recovery:
                recovered = prepare_on_policy_recovery(row, open_state=opened, recovery_spec=recovery_spec,
                                                       model=recovery_models.get(year), selected_model=selected_model)
                rec_action = bool(recovered["recovery_action"])
                rec_target = recovered["target_weights"] if rec_action else None
            state_after, clock_after = opened["risk_state"], opened["risk_clock"]
            from src.workflow.intraday_recovery_direct_allocation import _advance_state
            check_state, check_clock, q = _advance_state(state_before, clock_before,
                                                        drawdown_before, recovery_spec)
            if (check_state, check_clock) != (state_after, clock_after):
                raise ValueError("blocked: trace does not match actual controller state")
            clock_cap = max(recovery_spec.defensive_beta, q * raw_beta)
            trailing_active = (len(history) >= recovery_spec.trailing_loss_days
                               and np.prod(1 + np.asarray(history[-recovery_spec.trailing_loss_days:])) - 1
                               <= -recovery_spec.trailing_loss_trigger)
            after_trailing = min(clock_cap, recovery_spec.trailing_loss_beta) if trailing_active else clock_cap
            emergency_active = drawdown_before >= recovery_spec.emergency_drawdown
            target_beta = min(raw_beta, max(0.0, opened["current_beta_cap"]))
            raw_risk = float(row["soxl_weight"] + row["soxx_weight"])
            target_risk = min(raw_risk, target_beta, 1 - recovery_spec.minimum_cash_weight)
            unconstrained_soxl = max(0.0, (target_beta - target_risk) / 2.0)
            trace.update(current_beta_cap=float(opened["current_beta_cap"]), risk_q=float(q),
                         stop_entered=bool(state_before == 0 and drawdown_before >= recovery_spec.stop_drawdown),
                         emergency_active=bool(emergency_active), trailing_loss_active=bool(trailing_active),
                         clock_cap_binding=bool(clock_cap < raw_beta - 1e-12),
                         trailing_cap_binding=bool(trailing_active and recovery_spec.trailing_loss_beta < clock_cap - 1e-12),
                         emergency_cap_binding=bool(emergency_active and recovery_spec.emergency_beta < after_trailing - 1e-12),
                         max_soxl_binding=bool(unconstrained_soxl > recovery_spec.maximum_soxl_weight + 1e-12),
                         min_cash_binding=bool(min(raw_risk, target_beta) > 1 - recovery_spec.minimum_cash_weight + 1e-12))
        trace.update(risk_state_after=state_after, risk_clock_after=clock_after,
                     desired_beta=float(3 * desired[0] + desired[1]))
        funded = None
        open_target = desired
        if desired is not None and funding_convention == "post_fee_desired":
            funded = fee_affordable_target(desired, pretrade_weights=previous, cost_bps=cost)
            open_target = funded
        settled = settle_target_session(
            previous, open_target_weights=(open_target if execute else None),
            asset_returns=[row["soxl_return"], row["soxx_return"], 0.0], cost_bps=cost,
            recovery_target_weights=rec_target,
            early_returns=[row["early_ret_soxl"], row["early_ret_soxx"], 0.0] if rec_target is not None else None)
        previous = settled["ending_weights"]
        strategy_return = settled["net_return"]
        history.append(strategy_return)
        equity *= 1.0 + strategy_return
        peak = max(peak, equity)
        state, clock = state_after, clock_after
        rows.append({**trace, "signal_date": row["signal_date"], "trade_date": row.get("trade_date"), "exit_date": row.get("exit_date"),
                     "segment": row["segment"],
                     "strategy_return": strategy_return, "turnover": settled["turnover"], "fees": settled["fees"],
                     "execute_open_target": bool(execute), "execute": bool(execute), "decision_reason": exec_reason,
                     "recovery_action": bool(rec_action),
                     "funding_convention": funding_convention,
                     "desired_soxl": (float(desired[0]) if desired is not None else np.nan),
                     "desired_soxx": (float(desired[1]) if desired is not None else np.nan),
                     "desired_cash": (float(desired[2]) if desired is not None else np.nan),
                     "funded_soxl": (float(funded[0]) if funded is not None else np.nan),
                     "funded_soxx": (float(funded[1]) if funded is not None else np.nan),
                     "funded_cash": (float(funded[2]) if funded is not None else np.nan),
                     "recovery_soxl": (float(rec_target[0]) if rec_target is not None else np.nan),
                     "recovery_soxx": (float(rec_target[1]) if rec_target is not None else np.nan),
                     "recovery_cash": (float(rec_target[2]) if rec_target is not None else np.nan),
                     "soxl_return": float(row["soxl_return"]), "soxx_return": float(row["soxx_return"]),
                     "early_ret_soxl": float(row["early_ret_soxl"]), "early_ret_soxx": float(row["early_ret_soxx"]),
                     "ending_soxl_weight": previous[0], "ending_soxx_weight": previous[1],
                     "ending_cash_weight": previous[2], "cost_bps": cost})
    return pd.DataFrame(rows)


def replay_all(panel: pd.DataFrame, raw: pd.DataFrame, *, root: Path, anchor: dict) -> dict:
    results = {}
    for cost in COSTS:
        spec = replace(Stage1OpenActionValueSpec(root=root), transaction_cost_bps=cost)
        recovery_spec = replace(anchor["source_spec"], transaction_cost_bps=cost)
        legacy = anchor["legacy_models"]
        recovery_models = anchor["recovery_models"]
        selected_model = anchor["selected_model"]
        for segment, span in SEGMENTS.items():
            market = panel[panel["segment"] == segment].reset_index(drop=True)
            raw_seg = raw[raw["segment"] == segment]
            key = f"{span}::{int(cost)}"
            results[f"raw_allocator_only::{key}"] = _layer_loop(market, spec=spec, recovery_spec=recovery_spec, models={}, recovery_models=recovery_models,
                                                               selected_model=selected_model, cost=cost, funding_convention=RAW_CONVENTION,
                                                               use_hold=False, use_recovery=False, raw=raw_seg)
            results[f"open_risk_projection_nohold_no_recovery::{key}"] = _layer_loop(
                market, spec=spec, recovery_spec=recovery_spec, models={}, recovery_models=recovery_models,
                selected_model=selected_model, cost=cost, funding_convention=LEGACY_CONVENTION, use_hold=True, use_recovery=False)
            results[f"open_risk_projection_withhold_no_recovery::{key}"] = _layer_loop(
                market, spec=spec, recovery_spec=recovery_spec, models=legacy, recovery_models=recovery_models,
                selected_model=selected_model, cost=cost, funding_convention=LEGACY_CONVENTION, use_hold=True, use_recovery=False)
            results[f"full_withhold_and_recovery::{key}"] = _layer_loop(
                market, spec=spec, recovery_spec=recovery_spec, models=legacy, recovery_models=recovery_models,
                selected_model=selected_model, cost=cost, funding_convention=LEGACY_CONVENTION, use_hold=True, use_recovery=True)
            results[f"nohold_with_recovery_reference::{key}"] = _layer_loop(
                market, spec=spec, recovery_spec=recovery_spec, models={}, recovery_models=recovery_models,
                selected_model=selected_model, cost=cost, funding_convention=LEGACY_CONVENTION, use_hold=True, use_recovery=True)
    return results


def full_parity(root: Path, panel: pd.DataFrame, *, anchor: dict, results: dict) -> dict:
    """Day-by-day parity of full_withhold_and_recovery vs e4 saved active_hold_baseline.

    Compares exact dates, return, all three ending weights, open/recovery decision flags
    and replay length; any mismatch blocks.
    """
    checks = {}
    e4 = anchor["replays"]
    for segment, span in SEGMENTS.items():
        for cost in COSTS:
            saved = e4[f"active_hold_baseline::{span}::{int(cost)}"].reset_index(drop=True)
            got = results[f"full_withhold_and_recovery::{span}::{int(cost)}"].reset_index(drop=True)
            ok = (len(got) == len(saved)
                  and np.array_equal(pd.to_datetime(got["signal_date"]).to_numpy(),
                                     pd.to_datetime(saved["signal_date"]).to_numpy())
                  and np.allclose(got["strategy_return"], saved["strategy_return"], rtol=0, atol=1e-12)
                  and all(np.allclose(got[c], saved[c], rtol=0, atol=1e-12) for c in
                          ("ending_soxl_weight", "ending_soxx_weight", "ending_cash_weight"))
                  and bool((got["execute"].to_numpy() == saved["execute_open_target"].astype(bool).to_numpy()).all())
                  and bool((got["recovery_action"].astype(bool).to_numpy()
                            == saved["recovery_action"].astype(bool).to_numpy()).all()))
            checks[f"{span}::{int(cost)}"] = bool(ok)
    if not all(checks.values()):
        raise ValueError(f"blocked: full layer does not reproduce e4 active_hold_baseline: {checks}")
    return checks


def soxx_benchmark(panel: pd.DataFrame) -> list[dict]:
    rows = []
    for cost in COSTS:
        path = X.soxx_buy_hold_path(panel, cost_bps=cost)
        for _, seg_label in SEGMENTS.items():
            seg = path[path["segment"] == seg_label]
            wealth = seg["net_equity"].to_numpy(float)
            rows.append({"segment": seg_label, "cost_bps": cost, "return_pct": 100.0 * float(wealth[-1] - 1.0),
                         "max_drawdown_pct": 100.0 * float((wealth / np.maximum.accumulate(wealth) - 1.0).min())})
    return rows


def _resettle_path(base: pd.DataFrame, *, cost: float, convention: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Re-settle the SAME recorded path from fresh cash at a given cost."""
    from src.backtest.position_accounting import settle_target_session
    from src.workflow.stage1_economic_action import fee_affordable_target
    previous = np.array([0.0, 0.0, 1.0])
    rets, fees, ends, currency_fees = [], [], [], []
    for _, r in base.iterrows():
        desired = (np.array([r["desired_soxl"], r["desired_soxx"], r["desired_cash"]])
                   if np.isfinite(r["desired_soxl"]) else None)
        target = desired
        if target is not None and convention == RAW_CONVENTION:
            target = fee_affordable_target(target, pretrade_weights=previous, cost_bps=cost)
        rec = (np.array([r["recovery_soxl"], r["recovery_soxx"], r["recovery_cash"]])
               if bool(r["recovery_action"]) else None)
        equity_before = float(np.prod(1.0 + np.array(rets))) if rets else 1.0
        settled = settle_target_session(
            previous, open_target_weights=(target if bool(r["execute"]) else None),
            asset_returns=[r["soxl_return"], r["soxx_return"], 0.0], cost_bps=cost,
            recovery_target_weights=rec,
            early_returns=[r["early_ret_soxl"], r["early_ret_soxx"], 0.0] if rec is not None else None)
        previous = settled["ending_weights"]
        rets.append(settled["net_return"]); fees.append(settled["fees"]); ends.append(previous.copy())
        currency_fees.append(settled["fees"] * equity_before)
    return np.array(rets), np.array(fees), np.array(ends), float(np.sum(currency_fees))


def fee_ledger(results: dict) -> list[dict]:
    """Direct fee effect. First re-settle at the base 10bp and prove exact day parity; then 0/25.

    ``fee_fraction_of_nav`` is the sum of NAV-normalized daily fee fractions (unitless);
    ``fee_currency_of_startcash`` scales each day by that day's pre-trade equity.
    """
    rows = []
    for layer in LAYERS:
        for segment, span in SEGMENTS.items():
            base = results[f"{layer}::{span}::10"].reset_index(drop=True)
            convention = base["funding_convention"].iloc[0]
            r10, f10, e10, c10 = _resettle_path(base, cost=10.0, convention=convention)
            base_ends = base[["ending_soxl_weight", "ending_soxx_weight", "ending_cash_weight"]].to_numpy(float)
            parity = (np.allclose(r10, base["strategy_return"], rtol=0, atol=1e-12)
                      and np.allclose(f10, base["fees"], rtol=0, atol=1e-12)
                      and np.allclose(e10, base_ends, rtol=0, atol=1e-12))
            if not parity:
                raise ValueError(f"blocked: fee-ledger 10bp day parity failed for {layer} {span}")
            for cost in (0.0, 25.0):
                rets, fees, ends, currency = _resettle_path(base, cost=cost, convention=convention)
                rows.append({"layer": layer, "segment": span, "cost_bps": cost, "status": "ok", "convention": convention,
                             "return_pct": 100.0 * float(np.prod(1.0 + rets) - 1.0),
                             "fee_fraction_of_nav_sum": float(np.sum(fees)),
                             "fee_currency_of_startcash": currency,
                             "fee_unit": "fee_fraction_of_nav_sum is unitless NAV fraction; fee_currency_of_startcash is in units of the 1.0 start cash",
                             "base_10bp_day_parity": bool(parity)})
    return rows


def risk_report(native_objects: dict, panel: pd.DataFrame) -> list[dict]:
    """Exact common execution-date native analysis; one benchmark entry fee per book."""
    from qlib.contrib.evaluate import risk_analysis
    benchmarks = {}
    for cost in COSTS:
        path = X.soxx_buy_hold_path(panel, cost_bps=cost)
        for segment, span in SEGMENTS.items():
            band = path[path.segment.eq(span)].sort_values("trade_date")
            wealth = band.net_equity.to_numpy(float)
            returns = np.concatenate([[wealth[0] - 1], wealth[1:] / wealth[:-1] - 1])
            benchmarks[(span, cost)] = pd.Series(returns, index=pd.DatetimeIndex(band.trade_date))
    rows = []
    for layer in LAYERS:
        for segment, span in SEGMENTS.items():
            expected = pd.DatetimeIndex(panel.loc[panel.segment.eq(segment), "trade_date"].sort_values())
            for cost in COSTS:
                key = f"{layer}::{span}::{int(cost)}"
                parts = [native_objects[f"{key}::{stage}_reconciliation"][["trade_date", "native_return"]]
                         for stage in ("prefix", "suffix") if f"{key}::{stage}_reconciliation" in native_objects]
                if not parts:
                    raise ValueError(f"blocked: native analysis missing {key}")
                daily = pd.concat(parts, ignore_index=True).sort_values("trade_date")
                dates = pd.DatetimeIndex(daily.trade_date)
                b = benchmarks[(span, cost)]
                if dates.has_duplicates or not dates.equals(expected) or not b.index.equals(expected):
                    raise ValueError(f"blocked: native/benchmark exact execution calendar mismatch {key}")
                r = pd.Series(daily.native_return.to_numpy(float), index=dates)
                if not np.isfinite(r).all() or not np.isfinite(b).all():
                    raise ValueError("blocked: nonfinite native/benchmark return")
                ra = risk_analysis(r, freq="day", mode="product")
                risk = {str(i): float(v) for i, v in ra.iloc[:, 0].items()}
                var = float(b.var(ddof=1))
                beta = float(r.cov(b) / var) if var > 0 else None
                annual = []
                for year, sr in r.groupby(r.index.year):
                    br = b.loc[sr.index]
                    sw, bw = float(np.prod(1 + sr)), float(np.prod(1 + br))
                    annual.append({"year": int(year), "sessions": len(sr),
                                   "strategy_return_pct": 100 * (sw - 1),
                                   "soxx_return_pct": 100 * (bw - 1),
                                   "relative_wealth_vs_soxx": sw / bw})
                rows.append({"layer": layer, "segment": span, "cost_bps": cost, "sessions": len(r),
                             **risk, "beta_vs_soxx": beta,
                             "relative_wealth_vs_soxx": float(np.prod(1 + r) / np.prod(1 + b)),
                             "annual": annual})
    return rows


def native_case_passed(entry: dict, *, prefix_sessions: int, suffix_sessions: int) -> bool:
    for stage, count in (("prefix", prefix_sessions), ("suffix", suffix_sessions)):
        evidence = entry.get(stage)
        if count and (not evidence or evidence.get("status") != "passed"
                      or evidence.get("sessions") != count):
            return False
        if not count and evidence is not None:
            return False
    return True


def native_reconcile(root: Path, panel: pd.DataFrame, results: dict) -> tuple[dict, dict]:
    """Real Qlib native prefix(day)+suffix(15min) reconciliation for every layer case."""
    coherent = (root / X.PROVIDERS["coherent_provider"]).resolve()
    execution = (root / X.PROVIDERS["execution_mark_provider"]).resolve()
    boundary = pd.Timestamp(json.loads((coherent / "qlib_store_manifest.json").read_text())["intraday_price_source_from"])
    spec = C.Spec(root=root) if hasattr(C, "Spec") else type("S", (), {"root": root})()
    out, objects = {}, {}
    for layer in LAYERS:
        conv = RAW_CONVENTION if layer == "raw_allocator_only" else LEGACY_CONVENTION
        for segment, span in SEGMENTS.items():
            for cost in COSTS:
                key = f"{layer}::{span}::{int(cost)}"
                r = results[key].reset_index(drop=True)
                schedule = pd.DataFrame({
                    "signal_date": r["signal_date"], "trade_date": r["trade_date"], "exit_date": r["exit_date"],
                    "strategy_return": r["strategy_return"],
                    "execute_open_target": r["execute"].astype(bool), "recovery_action": r["recovery_action"].astype(bool),
                    "open_soxl_weight": r["desired_soxl"].fillna(0.0), "open_soxx_weight": r["desired_soxx"].fillna(0.0),
                    "open_cash_weight": r["desired_cash"].fillna(1.0),
                    "soxl_weight": r["recovery_soxl"].where(r["recovery_action"], r["desired_soxl"]).fillna(0.0),
                    "soxx_weight": r["recovery_soxx"].where(r["recovery_action"], r["desired_soxx"]).fillna(0.0),
                    "cash_weight": r["recovery_cash"].where(r["recovery_action"], r["desired_cash"]).fillna(1.0)})
                try:
                    res, objs = C._native_reconcile(spec, panel, {key: schedule}, coherent, execution,
                                                    target_conventions={key: conv})
                    entry = res["results"][key]
                    prefix_count = int(pd.to_datetime(r.trade_date).lt(boundary).sum())
                    entry["all_stages_passed"] = native_case_passed(
                        entry, prefix_sessions=prefix_count, suffix_sessions=len(r) - prefix_count)
                    out[key] = entry
                    objects.update(objs)
                except ValueError as exc:
                    out[key] = {"status": "needs_data", "reason": str(exc), "target_convention": conv}
    passed = sum(1 for v in out.values() if v.get("all_stages_passed"))
    return {"status": "passed" if passed == len(out) else ("partial" if passed else "needs_data"),
            "coverage": {"cases": len(out), "native_passed": passed}, "results": out}, objects


def _anchor(root: Path) -> dict:
    from qlib.workflow import R
    import qlib
    frozen = C.load_frozen(root)
    prep = frozen["inputs/prepared_inputs.pkl"]
    qlib.init(provider_uri=str((root / "data/qlib/soxl_soxx_soxs").resolve()), region="us")
    with R.uri_context(str((root / "research/stage1_feasible_action_training_20261009/qlib_recorder").resolve())):
        rec = R.get_recorder(recorder_id="e4d888a42c914bf69660ab78a35ba6e7",
                             experiment_name="stage1_feasible_action_training_20261009")
        replays = rec.load_object("replays")
    return {"legacy_models": prep["legacy_models"], "recovery_models": prep["recovery_models"],
            "selected_model": prep["selected_model"],
            "source_spec": IntradayRecoveryDirectAllocationSpec(root=root), "replays": replays, "prep": prep}


def run_layer_attribution(*, root=None, card_dir=None) -> dict:
    C._quiet()
    spec_root = Path(root).resolve() if root else Path(".").resolve()
    card = Path(card_dir) if card_dir else (spec_root / "research" / "stage1_uniform_layer_attribution_20261010")
    card.mkdir(parents=True, exist_ok=True)
    plan_path = spec_root / "research" / "stage1_uniform_layer_attribution_20261010" / "preregister_plan.json"
    plan = json.loads(plan_path.read_text())
    providers = X._freeze_providers(spec_root)
    if {k: v["manifest_sha256"] for k, v in providers.items()} != plan["provider_manifest_sha256"]:
        raise ValueError("blocked: frozen provider identity changed")
    reference = plan["frozen_sources"]["full_parity_reference"]
    if _sha(spec_root / reference["path"]) != reference["sha256"]:
        raise ValueError("blocked: frozen full-parity reference changed")
    anchor = _anchor(spec_root)
    frame = anchor["prep"]["frame"]
    panel = C.rebuild_market_inputs(frame, coherent_provider=spec_root / X.PROVIDERS["coherent_provider"],
                                    intraday_provider=spec_root / X.PROVIDERS["intraday_provider"])
    raw, raw_audit = load_raw_targets(spec_root)
    rc = raw_coverage(raw, panel)
    results = replay_all(panel, raw, root=spec_root, anchor=anchor)
    parity = full_parity(spec_root, panel, anchor=anchor, results=results)
    native, native_objects = native_reconcile(spec_root, panel, results)
    if native["status"] != "passed" or native["coverage"] != {"cases": 20, "native_passed": 20}:
        raise ValueError("blocked: incomplete native layer reconciliation")
    metrics = []
    for layer in LAYERS:
        for segment, span in SEGMENTS.items():
            for cost in COSTS:
                metrics.append({"layer": layer, "segment": span, "cost_bps": cost,
                                **_metrics(results[f"{layer}::{span}::{int(cost)}"])})
    risk = risk_report(native_objects, panel)
    annual = [{"layer": r["layer"], "segment": r["segment"], "cost_bps": r["cost_bps"], **a}
              for r in risk for a in r.pop("annual")]
    from src.analysis.uniform_risk_mechanism import analyze_risk_mechanism
    mechanism = analyze_risk_mechanism(results)
    if len(mechanism) != 20 or any(m["status"] != "ok" for m in mechanism):
        raise ValueError("blocked: incomplete risk mechanism analysis")
    summary = {
        "experiment": "stage1_uniform_layer_attribution_20261010", "models_trained": 0, "new_parameters": 0,
        "decision": "layer_attribution_reported_diagnostic_only", "promotion_eligible": False,
        "plan_sha256": _sha(plan_path), "raw_targets_audit": raw_audit, "raw_targets_sha256": raw_audit["prepared_sha256"],
        "raw_convention": RAW_CONVENTION, "legacy_convention": LEGACY_CONVENTION,
        "full_parity_vs_e4_active_hold": parity,
        "native": {"status": native["status"], "coverage": native["coverage"], "results": native["results"]},
        "metrics": metrics, "layers": list(LAYERS), "raw_coverage": rc, "risk_report": risk, "annual_metrics": annual, "risk_mechanism": mechanism,
        "provider_lineage": providers, "region": "us", "as_of": "2026-09-25",
        "orders_submitted": 0, "new_candidates": 0,
        "source_snapshot_as_of": "2026-09-25",
        "price_vintage_as_of": {k: v["as_of"] for k, v in providers.items()},
        "risk_trace_scope": "exact legacy prepare_on_policy_open/_base_cap ordered local limiter flags; not the separate economic _risk_envelope",
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                            "recorder_only_artifacts": sorted([f"replay::{k}" for k in results] +
                                ["native::*", "layer_metrics", "risk_report", "annual_metrics", "fee_ledger", "soxx_benchmark", "risk_mechanism"]),
                            "slim_sidecars": ["summary.json", "workflow.yaml", "conclusion.md", "layer_metrics.csv",
                                              "soxx_benchmark.csv", "fee_ledger.csv", "risk_report.csv", "annual_metrics.csv", "risk_mechanism.csv"]},
        "soxx_benchmark": soxx_benchmark(panel), "fee_ledger": fee_ledger(results),
        "reporting": {
            "dev10_raw_to_full_pp": _metrics(results["raw_allocator_only::development_walk_forward::10"])["return_pct"]
            - _metrics(results["full_withhold_and_recovery::development_walk_forward::10"])["return_pct"]},
        "caveats": ["raw association diagnostic; not separable predictive info vs optimizer from a raw-only failure",
                    "ordered path comparison, not a unique additive causal contribution",
                    "no new alpha; no promotion; not production"],
        "qlib_docs_reference": C.QLIB_DOCS,
    }
    summary.update(_record_attribution(spec_root, card, results, native_objects, summary, plan_path))
    _write_attribution_card(summary, card)
    return summary


def _record_attribution(spec_root, card, results, native_objects, summary, plan_path):
    import os
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import logging
    import warnings
    for name in ("qlib", "mlflow", "qlib.data", "qlib.workflow"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    import qlib
    from qlib.workflow import R
    uri = (Path(card) / "qlib_recorder").resolve()
    uri.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=str((spec_root / "data/qlib/soxl_soxx_soxs").resolve()), region="us")
    saved = {f"replay::{k}": v for k, v in results.items()}
    saved["layer_metrics"] = pd.DataFrame(summary["metrics"])
    for name in ("risk_report", "annual_metrics", "fee_ledger", "soxx_benchmark", "risk_mechanism"):
        saved[name] = pd.DataFrame(summary[name])
    for name, value in native_objects.items():
        saved[f"native::{name}"] = value
    with R.start(experiment_name="stage1_uniform_layer_attribution_20261010", recorder_name="layer_attr", uri=str(uri)):
        rec = R.get_recorder()
        rec.log_params(schema_version="stage1_uniform_layer_attribution_v1", models_trained=0)
        rec.save_objects(**saved)
        rec.log_artifact(str(plan_path), artifact_path="plan")
        rid, eid = rec.id, str(rec.experiment_id)
    return {"recorder_id": rid, "experiment_id": eid, "recorder_uri": str(uri), "saved_objects": sorted(saved)}


def _write_attribution_card(summary, card):
    (card / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    pd.DataFrame(summary["metrics"]).to_csv(card / "layer_metrics.csv", index=False)
    pd.DataFrame(summary["soxx_benchmark"]).to_csv(card / "soxx_benchmark.csv", index=False)
    pd.DataFrame(summary["fee_ledger"]).to_csv(card / "fee_ledger.csv", index=False)
    for name in ("risk_report", "annual_metrics", "risk_mechanism"):
        pd.DataFrame(summary[name]).to_csv(card / (name + ".csv"), index=False)
    import yaml
    workflow = {"schema_version": "stage1_uniform_layer_attribution_v1", "experiment": summary["experiment"],
                "models_trained": 0, "layers": summary["layers"], "segments": list(SEGMENTS.values()),
                "costs_bps": list(COSTS), "raw_convention": summary["raw_convention"],
                "legacy_convention": summary["legacy_convention"], "plan_sha256": summary["plan_sha256"],
                "raw_targets_sha256": summary["raw_targets_sha256"], "full_parity_vs_e4_active_hold": summary["full_parity_vs_e4_active_hold"],
                "native": summary["native"]["coverage"], "artifact_policy": summary["artifact_policy"],
                "qlib_docs_reference": summary["qlib_docs_reference"]}
    (card / "workflow.yaml").write_text(yaml.safe_dump(workflow, sort_keys=False), encoding="utf-8")
    selected = {"selection": "none_frozen_source_reuse_only", "models_trained": 0,
                "allocator_recorder": "f696f656b3d44e868588e09a5d97c905",
                "action_model_source": "parent17 prepared_inputs.pkl[legacy_models]",
                "identity_vs_b472": "not_verified_not_claimed",
                "prepared_inputs_recorder": C.SOURCE_RECORDER,
                "prepared_inputs_sha256": summary["raw_targets_audit"]["prepared_sha256"],
                "raw_funding_convention": RAW_CONVENTION,
                "legacy_funding_convention": LEGACY_CONVENTION,
                "promotion_eligible": False}
    (card / "selected_model.json").write_text(json.dumps(selected, indent=2) + "\n")
    lines = ["# 冻结配置、风险投影与动作层诊断\n\n",
             "使用同一价格版本、五层独立账户、两窗口与10/25bp费用；零拟合、零新候选。\n\n",
             "完整目标读取已哈希parent17冻结推理快照；原f696预测2686行逐行对照，末尾四日来自已保存的冻结模型推理延展。未重新推理、未静默HOLD。\n\n",
             f"Qlib原生账户对账：{summary['native']['coverage']['native_passed']}/20；完整动作层与父e4逐日收益、三权重和动作一致。\n\n",
             "| 窗口/费用 | 层 | 收益% | 最大回撤% | CVaR5% | 相对SOXX财富 |\n",
             "|---|---|---:|---:|---:|---:|\n"]
    risk_lookup = {(r["layer"], r["segment"], r["cost_bps"]): r for r in summary["risk_report"]}
    for m in summary["metrics"]:
        r = risk_lookup[(m["layer"], m["segment"], m["cost_bps"])]
        lines.append(f"| {m['segment']} / {m['cost_bps']:g}bp | {m['layer']} | {m['return_pct']:.2f} | "
                     f"{m['max_drawdown_pct']:.2f} | {m['cvar5_pct']:.2f} | {r['relative_wealth_vs_soxx']:.4f} |\n")
    lookup = {(m["layer"], m["segment"], m["cost_bps"]): m for m in summary["metrics"]}
    lines.append("\nHOLD与恢复动作的交互（收益百分点，只是有序路径对照）：\n\n")
    for span in SEGMENTS.values():
        for cost in COSTS:
            def ret(layer):
                return lookup[(layer, span, cost)]["return_pct"]
            recovery_nohold = ret("nohold_with_recovery_reference") - ret("open_risk_projection_nohold_no_recovery")
            recovery_hold = ret("full_withhold_and_recovery") - ret("open_risk_projection_withhold_no_recovery")
            lines.append(f"- {span} / {cost:g}bp：无HOLD时加入恢复 {recovery_nohold:+.2f}pp；"
                         f"有HOLD时加入恢复 {recovery_hold:+.2f}pp。\n")
    lines.append("\n风险循环的实际账户统计：\n\n")
    for m in summary["risk_mechanism"]:
        if m["layer"] != "open_risk_projection_nohold_no_recovery":
            continue
        lines.append(f"- {m['segment']} / {m['cost_bps']:g}bp：stop事件 {m['stop_events']} 次；"
                     f"emergency最长连续 {m['emergency_longest_consecutive_days']} 日。"
                     f"详情、状态占比和分组配对log财富差见risk_mechanism.csv及Recorder。\n")
    lines += ["\n年度、beta及Qlib product模式风险统计见annual_metrics.csv、risk_report.csv。费用归因从10bp实际目标路径重新结算0/25bp，10bp自复现检查真执行；不是从净收益再次减费。\n\n",
              "风险投影减少回撤也牺牲上涨暴露；该比较不能证明约束过度，不能靠删除风控达成上线。"
              "raw使用post_fee_desired，旧动作层使用pre_fee_account，差异及账户反馈均包含在路径比较中。"
              "这些层不是唯一可加的因果贡献，条件分组不是可交易子策略。\n\n",
              "外窗已被反复查看，只是复用历史诊断，不是新鲜样本外。收益、尾部风险、年度稳定性和独立前向证据仍需分别达标。\n\n",
              "晋升决定：不晋升；本卡只定位机制。下一步依据冻结状态统计制定一个明确机制增量，再单独冻结验证；不得继续搜索失败的信息路线、改符号或放宽门槛。\n"]
    (card / "conclusion.md").write_text("".join(lines), encoding="utf-8")
