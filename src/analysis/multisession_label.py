"""Multi-session label audit (diagnostic only, no model fitting).

Reads the saved ``control_all_state`` annual models from the feasible-action
Recorder (frozen; NEVER re-fit; ``train_causal_economic_models`` is not called),
rebuilds the parent account day-by-day, and compares the original one-session
action label with a fixed 5-session counterfactual label where each branch owns
its cash/quantities/risk-clock/recovery.

Targets are economic ``post_fee_desired``; the ledger is open->next-open with
intraday recovery. Continuation freezes the label-start-year net_value and
recovery models. No common next-target reset. The full official calendar is kept
with explicit per-row status; unavailable windows are never dropped or filled.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.workflow import coherent_price_replay_experiment as C
from src.workflow import feasible_action_experiment as X
from src.workflow.intraday_recovery_direct_allocation import IntradayRecoveryDirectAllocationSpec
from src.workflow.stage1_economic_training import (
    LABEL_COST_BPS,
    _Portfolio,
    _event_at,
    _finalize_label,
    _model_at,
    _open,
    _settle_branch,
)

CONTROL_CARD = "research/stage1_feasible_action_training_20261009"
CONTROL_RECORDER = "e4d888a42c914bf69660ab78a35ba6e7"
CONTROL_EXPERIMENT = "stage1_feasible_action_training_20261009"
HORIZON = 5
BLOCK = 20
BOOTSTRAP = 2000
SEGMENT_LABELS = {"development_walk_forward": "development_walk_forward",
                  "outer_diagnostic": "outer_reused_diagnostic"}
EXPECTED_OBJECT_SHA256 = {
    "model_control_all_state_2016": "9c298d589a2158eb513cb52191144518a2acab2cb0c04f1df14fca0f712fa4a1",
    "model_control_all_state_2017": "194aeaf843787def0f77e8bccb005fbed3fc3bf0b24fe5107072bb8c4581705b",
    "model_control_all_state_2018": "e1ef03327b6f54fd9cf4e7b4ea1a21d961e4dd1595850f5997441f8660b67038",
    "model_control_all_state_2019": "78728ed47ff40a512e3d66ad8b187efbfed31d3c89518b6f7f5a5e0ab98152b5",
    "model_control_all_state_2020": "77710354e5aa02a23d34c8f67d10bb9c06c1bc6b57640e17467f8d1dfc9f334a",
    "model_control_all_state_2021": "96adcba2ac020c935e6c1dcb3aebbb4971fb08ccff9268064b63ceafe3c1e97d",
    "model_control_all_state_2022": "e194a5ff5d85676434c88248dd1367b9a051c72507f6f108932a1445754b0995",
    "model_control_all_state_2023": "770acef63a0c0b6fd47425dd14c2f450fc67fb264e78c7c3b6741dbb34ff625e",
    "model_control_all_state_2024": "770acef63a0c0b6fd47425dd14c2f450fc67fb264e78c7c3b6741dbb34ff625e",
    "model_control_all_state_2025": "770acef63a0c0b6fd47425dd14c2f450fc67fb264e78c7c3b6741dbb34ff625e",
    "model_control_all_state_2026": "770acef63a0c0b6fd47425dd14c2f450fc67fb264e78c7c3b6741dbb34ff625e",
    "replays": "510ae5e9c95df55759be5a4ac62b53bf06b70bf6bf4c106eb9e58a6e1780b17d",
    "training_rows_control": "ce49a677e18c3c2b4c09600378fa58ecd649118020fb987d654f2d7aeb7d5cc9",
    "diagnostics_control": "6dffebbe9b5f3887eb03fa3b08961573c1ea1c8bf915a0f64a949bdff536b1d4",
}
# Late-year keys reuse the frozen 2023 model (same object hash); recorded explicitly.
CARRY_FORWARD_FROM_YEAR = {2024: 2023, 2025: 2023, 2026: 2023}
EXPECTED_PROVIDER_SHA256 = {
    "stage1_uniform_intraday_20261009": "fb6afd75ea128999b1b87bd178e9d54eb7e00f1bd1ee3fc03d657db39c91274a",
    "stage1_uniform_execution_marks_20261009": "d565935ca518aecff338631ad993e8a9235ec1c9c39a44b62af75db4f84f27eb",
    "stage1_uniform_daily_marks_20261009": "48ec582f748d1c4f769aac2771569dd320ee4872813fc1f43e096717bde2081f",
}
EXPECTED_PREPARED_SHA256 = "8baa4f74449c1da41eb189f51c75074a61dacb8d7c8540710f4f67560d58d4aa"


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def _file_or_tree_sha(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    files = sorted(p for p in path.rglob("*") if p.is_file())
    _require(bool(files), f"saved object is empty: {path}")
    digest = hashlib.sha256()
    for entry in files:
        digest.update(hashlib.sha256(entry.read_bytes()).digest())
    return digest.hexdigest()


def load_frozen_control(root: Path) -> dict:
    from qlib.workflow import R

    # Verify every source/provider/object identity BEFORE reading or computing.
    frozen = C.load_frozen(root)  # hash-bound parent 17 object; verifies prepared object hash
    _require(C.SOURCE_OBJECT_SHA256["inputs/prepared_inputs.pkl"] == EXPECTED_PREPARED_SHA256,
             "parent prepared_inputs identity changed")
    for name, expected in EXPECTED_PROVIDER_SHA256.items():
        manifest = root / "data/qlib" / name / "qlib_store_manifest.json"
        _require(manifest.is_file() and hashlib.sha256(manifest.read_bytes()).hexdigest() == expected,
                 f"uniform provider {name} hash mismatch")
    base = root / CONTROL_CARD / "qlib_recorder"
    with R.uri_context(str(base)):
        recorder = R.get_recorder(recorder_id=CONTROL_RECORDER, experiment_name=CONTROL_EXPERIMENT)
        artifacts = Path(recorder.get_local_dir()) / "artifacts"
        observed = {name: _file_or_tree_sha(artifacts / name) for name in EXPECTED_OBJECT_SHA256}
        for name, expected in EXPECTED_OBJECT_SHA256.items():
            _require(observed[name] == expected, f"control object {name} hash mismatch")
        models = {year: recorder.load_object(f"model_control_all_state_{year}") for year in range(2016, 2027)}
        replays = recorder.load_object("replays")
        training_rows = recorder.load_object("training_rows_control")
        diagnostics = recorder.load_object("diagnostics_control")
    prepared = frozen["inputs/prepared_inputs.pkl"]
    panel = C.rebuild_market_inputs(prepared["frame"].copy(),
                                    coherent_provider=root / X.PROVIDERS["coherent_provider"],
                                    intraday_provider=root / X.PROVIDERS["intraday_provider"]).reset_index(drop=True)
    return {"panel": panel, "models": models, "replays": replays, "training_rows": training_rows,
            "diagnostics": diagnostics, "prepared_sha256": EXPECTED_PREPARED_SHA256,
            "object_sha256": observed, "provider_sha256": EXPECTED_PROVIDER_SHA256,
            "carry_forward_from_year": CARRY_FORWARD_FROM_YEAR,
            "common": dict(recovery_models=prepared["recovery_models"],
                           source_spec=IntradayRecoveryDirectAllocationSpec(root=root),
                           selected_model=prepared["selected_model"])}


def observed_parent(panel: pd.DataFrame, *, models: dict, common: dict, cost_bps: float = 10.0) -> pd.DataFrame:
    """Reconstruct the parent account and one-session labels WITHOUT any fitting.

    Every parent row carries an explicit ``label_status``; no row is silently
    filtered. ``peak`` starts at 1.0 and re-initialises per segment.
    """
    source_spec = replace(common["source_spec"], transaction_cost_bps=float(cost_bps))
    label_spec = replace(common["source_spec"], transaction_cost_bps=LABEL_COST_BPS)
    records = []
    state = _Portfolio(1.0, 1.0)
    pending = None
    previous_segment = None
    for _, row in panel.iterrows():
        if previous_segment is not None and row["segment"] != previous_segment:
            if pending is not None:
                pending[0]["label_status"] = "unavailable_segment_reset_boundary"
            pending = None
            state = _Portfolio(1.0, 1.0)
        previous_segment = row["segment"]
        year = int(row["signal_date"].year)
        model = _model_at(models, year)
        recovery_model = _model_at(common["recovery_models"], year)
        pretrade = state.weights.copy()
        opened = _open(row, state, model=model, source_spec=source_spec, policy="net_value")
        label_opened = _open(row, state, model=model, source_spec=label_spec, policy="net_value")
        if pending is not None:
            _finalize_label(pending[0], pending[1], next_row=row, next_target=label_opened["desired_open_weights"])
        branches = {name: _settle_branch(row, pretrade=pretrade, opened=label_opened, execute=execute,
                                         cost_bps=LABEL_COST_BPS, recovery_model=recovery_model,
                                         source_spec=common["source_spec"], selected_model=common["selected_model"])
                    for name, execute in (("execute", True), ("hold", False))}
        record = {"signal_date": row["signal_date"], "trade_date": row["trade_date"], "exit_date": row["exit_date"],
                  "segment": row["segment"], "year": year,
                  "predicted_net_advantage_bps": float(opened.get("predicted_net_advantage_bps", np.nan)),
                  "execute_open_target": bool(opened["execute_open_target"]),
                  "routing_eligible": bool(opened.get("routing_eligible", not opened.get("hard_risk_violation", False))),
                  "label_status": "pending_next_reset_target", "execute_advantage_bps": np.nan,
                  "label_available_at": pd.NaT,
                  "pre_equity": state.equity, "pre_peak": state.peak, "pre_history": list(state.history),
                  "pre_risk_state": state.risk_state, "pre_risk_clock": state.risk_clock,
                  "pre_days_since_trade": state.days_since_trade,
                  "pre_soxl_weight": float(pretrade[0]), "pre_soxx_weight": float(pretrade[1]),
                  "pre_cash_weight": float(pretrade[2])}
        records.append(record)
        pending = (record, branches)
        branch = _settle_branch(row, pretrade=pretrade, opened=opened, execute=bool(opened["execute_open_target"]),
                                cost_bps=cost_bps, recovery_model=recovery_model,
                                source_spec=common["source_spec"], selected_model=common["selected_model"])
        settled = branch["settled"]
        # Record the ACTUAL observed settlement so parity is checked against this,
        # not against a second independent simulation.
        record.update(observed_return=float(settled["net_return"]),
                      observed_end_soxl=float(settled["ending_weights"][0]),
                      observed_end_soxx=float(settled["ending_weights"][1]),
                      observed_end_cash=float(settled["ending_weights"][2]),
                      observed_turnover=float(settled["turnover"]),
                      post_risk_state=int(opened["risk_state"]), post_risk_clock=int(opened["risk_clock"]))
        state.risk_state, state.risk_clock = opened["risk_state"], opened["risk_clock"]
        state.weights = settled["ending_weights"].copy()
        state.history.append(float(settled["net_return"]))
        state.equity *= settled["net_equity"]
        state.peak = max(state.peak, state.equity)
        state.days_since_trade = 0 if settled["turnover"] > 1e-8 else state.days_since_trade + 1
    if pending is not None:
        # The saved source leaves the terminal row as pending; keep that status for
        # parity and expose explicit unknown identity separately.
        pending[0]["label_status"] = "pending_next_reset_target"
    frame = pd.DataFrame(records)
    identity = {"complete_one_session_common_next_target_reset_proxy": "complete",
                "unavailable_segment_reset_boundary": "unknown_segment_boundary",
                "pending_next_reset_target": "unknown_terminal",
                "unavailable_noncontiguous_next_reset": "unknown_noncontiguous"}
    frame["label_identity"] = frame["label_status"].map(identity)
    return frame


def verify_observed_parity(parents: pd.DataFrame, saved: pd.DataFrame, *, label: str) -> dict:
    """Exact observed-state parity (rtol=0) against the saved control replay."""
    a = parents.reset_index(drop=True)
    b = saved.reset_index(drop=True)
    checks = {"sessions_equal": bool(len(a) == len(b))}
    if not checks["sessions_equal"]:
        raise ValueError(f"blocked: {label} session count differs ({len(a)} vs {len(b)})")
    checks["trade_dates_equal"] = bool(np.array_equal(pd.to_datetime(a["trade_date"]).to_numpy(),
                                                      pd.to_datetime(b["trade_date"]).to_numpy()))
    for name, left, right in (("return", a["observed_return"], b["strategy_return"]),
                              ("end_soxl", a["observed_end_soxl"], b["ending_soxl_weight"]),
                              ("end_soxx", a["observed_end_soxx"], b["ending_soxx_weight"]),
                              ("end_cash", a["observed_end_cash"], b["ending_cash_weight"])):
        checks[name] = bool(np.allclose(left.to_numpy(float), right.to_numpy(float), rtol=0, atol=1e-12))
    ok = all(checks.values())
    if not ok:
        raise ValueError(f"blocked: {label} observed-parent parity failed: {checks}")
    return {"status": "exact", "rtol": 0, "atol": 1e-12, **checks}


def _state_from(record: pd.Series) -> _Portfolio:
    state = _Portfolio(float(record["pre_equity"]), float(record["pre_peak"]))
    state.weights = np.array([record["pre_soxl_weight"], record["pre_soxx_weight"], record["pre_cash_weight"]], float)
    state.risk_state, state.risk_clock = int(record["pre_risk_state"]), int(record["pre_risk_clock"])
    state.days_since_trade = int(record["pre_days_since_trade"])
    state.history = list(record["pre_history"])
    return state


def _advance(state: _Portfolio, opened: dict, settled: dict, *, fees: float) -> dict:
    state.risk_state, state.risk_clock = opened["risk_state"], opened["risk_clock"]
    state.weights = settled["ending_weights"].copy()
    state.history.append(float(settled["net_return"]))
    state.equity *= settled["net_equity"]
    state.peak = max(state.peak, state.equity)
    state.days_since_trade = 0 if settled["turnover"] > 1e-8 else state.days_since_trade + 1
    return {"risk_state": int(state.risk_state), "risk_clock": int(state.risk_clock),
            "soxl_weight": float(state.weights[0]), "soxx_weight": float(state.weights[1]),
            "cash_weight": float(state.weights[2]), "equity": float(state.equity), "peak": float(state.peak),
            "days_since_trade": int(state.days_since_trade), "fees": float(fees),
            "net_return": float(settled["net_return"]), "turnover": float(settled["turnover"])}


def _traj_row(*, name: str, offset: int, row: pd.Series, execute: bool, opened: dict, branch: dict) -> dict:
    settled, recovered = branch["settled"], branch["recovery"]
    target = recovered.get("target_weights")
    return {
        "offset": offset, "branch": name, "trade_date": row["trade_date"], "exit_date": row["exit_date"],
        "execute": bool(execute), "router_execute_open_target": bool(opened["execute_open_target"]),
        "recovery_action": bool(recovered["recovery_action"]),
        "recovery_target_soxl": (float(target[0]) if target is not None else None),
        "recovery_target_soxx": (float(target[1]) if target is not None else None),
        "recovery_target_cash": (float(target[2]) if target is not None else None),
        "entry_target_soxl": float(opened["desired_open_weights"][0]),
        "entry_target_soxx": float(opened["desired_open_weights"][1]),
        "entry_target_cash": float(opened["desired_open_weights"][2]),
        "fees": float(settled["fees"]), "turnover": float(settled["turnover"]),
        "net_return": float(settled["net_return"]),
        "soxl_return": float(row["soxl_return"]), "soxx_return": float(row["soxx_return"]),
        "predicted_bps": (None if not np.isfinite(opened.get("predicted_net_advantage_bps", np.nan))
                          else float(opened["predicted_net_advantage_bps"])),
    }


def five_day_labels(panel: pd.DataFrame, parents: pd.DataFrame, *, models: dict, common: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All parent rows get a status; only same-year, contiguous, in-segment windows compute a label."""
    label_spec = replace(common["source_spec"], transaction_cost_bps=LABEL_COST_BPS)
    rows = panel.reset_index(drop=True)
    parents = parents.reset_index(drop=True)
    segments = rows["segment"].to_numpy()
    signal_years = pd.to_datetime(rows["signal_date"]).dt.year.to_numpy()
    exit_dates = pd.to_datetime(rows["exit_date"]).to_numpy()
    trade_dates = pd.to_datetime(rows["trade_date"]).to_numpy()
    labels, trajectories = [], []
    for index in range(len(rows)):
        parent = parents.iloc[index]
        start_year = int(parent["year"])
        status = None
        if index + HORIZON - 1 >= len(rows):
            status = "unavailable_terminal_short"
        elif any(segments[index + k] != segments[index] for k in range(1, HORIZON)):
            status = "unavailable_segment_or_gap"
        elif not all(exit_dates[index + k] == trade_dates[index + k + 1] for k in range(HORIZON - 1)):
            status = "unavailable_noncontiguous"
        elif any(signal_years[index + k] != start_year for k in range(HORIZON)):
            status = "unavailable_cross_year_upstream_identity"
        entry = {"signal_date": parent["signal_date"], "trade_date": parent["trade_date"], "year": start_year,
                 "segment": parent["segment"], "routing_eligible": bool(parent["routing_eligible"]),
                 "predicted_net_advantage_bps": float(parent["predicted_net_advantage_bps"]),
                 "orig_label_bps": float(parent.get("execute_advantage_bps", np.nan)),
                 "orig_label_status": parent["label_status"],
                 "window_status": status if status else "available",
                 "label_5d_bps": np.nan, "risk_state_disagreement": 0, "risk_clock_disagreement": 0,
                 "recovery_disagreement": 0,
                 "horizon_available_at": pd.NaT}
        if status is None:
            frozen_model = _model_at(models, start_year)
            frozen_recovery = _model_at(common["recovery_models"], start_year)
            details = {}
            for name, execute in (("execute", True), ("hold", False)):
                state = _state_from(parent)
                opened = _open(rows.iloc[index], state, model=frozen_model, source_spec=label_spec, policy="net_value")
                branch = _settle_branch(rows.iloc[index], pretrade=state.weights.copy(), opened=opened, execute=execute,
                                        cost_bps=LABEL_COST_BPS, recovery_model=frozen_recovery,
                                        source_spec=common["source_spec"], selected_model=common["selected_model"])
                cumulative = branch["settled"]["net_equity"]
                day_path = [{**_advance(state, opened, branch["settled"], fees=branch["settled"]["fees"]),
                             **_traj_row(name=name, offset=0, row=rows.iloc[index], execute=execute,
                                         opened=opened, branch=branch)}]
                for k in range(1, HORIZON):
                    row = rows.iloc[index + k]
                    opened_k = _open(row, state, model=frozen_model, source_spec=label_spec, policy="net_value")
                    branch_k = _settle_branch(row, pretrade=state.weights.copy(), opened=opened_k,
                                              execute=bool(opened_k["execute_open_target"]), cost_bps=LABEL_COST_BPS,
                                              recovery_model=frozen_recovery, source_spec=common["source_spec"],
                                              selected_model=common["selected_model"])
                    cumulative *= branch_k["settled"]["net_equity"]
                    execute_k = bool(opened_k["execute_open_target"])
                    day_path.append({**_advance(state, opened_k, branch_k["settled"], fees=branch_k["settled"]["fees"]),
                                     **_traj_row(name=name, offset=k, row=row, execute=execute_k,
                                                 opened=opened_k, branch=branch_k)})
                details[name] = {"cum": float(cumulative), "path": day_path}
            entry["label_5d_bps"] = 10_000.0 * (details["execute"]["cum"] - details["hold"]["cum"])
            entry["risk_state_disagreement"] = int(any(a["risk_state"] != b["risk_state"]
                                                       for a, b in zip(details["execute"]["path"], details["hold"]["path"])))
            entry["risk_clock_disagreement"] = int(any(a["risk_clock"] != b["risk_clock"]
                                                       for a, b in zip(details["execute"]["path"], details["hold"]["path"])))
            entry["recovery_disagreement"] = int(any(a["recovery_action"] != b["recovery_action"]
                                                     for a, b in zip(details["execute"]["path"], details["hold"]["path"])))
            entry["horizon_available_at"] = _event_at(rows.iloc[index + HORIZON - 1]["exit_date"], minute=30)
            for day in details["execute"]["path"] + details["hold"]["path"]:
                trajectories.append({"signal_date": parent["signal_date"], **day})
        labels.append(entry)
    return pd.DataFrame(labels), pd.DataFrame(trajectories)


def block_bootstrap_full(calendar: pd.DataFrame, statistic, *, block: int = BLOCK, bootstrap: int = BOOTSTRAP) -> dict:
    """Fixed moving-block bootstrap over the FULL calendar; statistic filters inside."""
    n = len(calendar)
    point = statistic(calendar)
    if n < block or block < 2:
        return {"point": point, "ci_low": None, "ci_high": None, "blocks": 0, "status": "needs_data"}
    starts = np.arange(0, n - block + 1)
    rng = np.random.default_rng(20261010)
    draws = []
    for _ in range(bootstrap):
        chosen = rng.choice(starts, size=int(np.ceil(n / block)), replace=True)
        idx = np.concatenate([np.arange(s, s + block) for s in chosen])[:n]
        value = statistic(calendar.iloc[idx])
        if value is not None and np.isfinite(value):
            draws.append(float(value))
    if len(draws) < max(50, bootstrap // 20):
        return {"point": point, "ci_low": None, "ci_high": None, "blocks": int(len(starts)), "status": "needs_data"}
    return {"point": point, "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
            "blocks": int(len(starts)), "status": "ok"}


def _spearman(x, y) -> float | None:
    x = np.asarray(pd.Series(x, dtype="float64").to_numpy(float))
    y = np.asarray(pd.Series(y, dtype="float64").to_numpy(float))
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 5 or np.std(x[finite]) == 0 or np.std(y[finite]) == 0:
        return None
    return float(np.corrcoef(pd.Series(x[finite]).rank(), pd.Series(y[finite]).rank())[0, 1])


def _filtered(calendar: pd.DataFrame, *, segment: str | None = None) -> pd.DataFrame:
    pred = pd.to_numeric(calendar["predicted_net_advantage_bps"], errors="coerce").to_numpy(float)
    label = pd.to_numeric(calendar["label_5d_bps"], errors="coerce").to_numpy(float)
    mask = (calendar["routing_eligible"].to_numpy(bool) & (calendar["window_status"] == "available").to_numpy(bool)
            & np.isfinite(pred) & np.isfinite(label))
    if segment is not None:
        mask &= (calendar["segment"] == segment).to_numpy(bool)
    return calendar[mask]


def summarize(labels: pd.DataFrame, calendar: pd.DataFrame) -> dict:
    result = {"diagnostic_only": True, "models_trained": 0, "horizon_sessions": HORIZON,
              "window_status_counts": {k: int(v) for k, v in labels["window_status"].value_counts().items()},
              "orig_label_status_counts": {k: int(v) for k, v in labels["orig_label_status"].value_counts().items()},
              "routing_ineligible_saved": int((~labels["routing_eligible"]).sum()),
              "segments": {}}
    for segment, span in SEGMENT_LABELS.items():
        seg = labels[labels["segment"] == segment]
        available = seg[(seg["routing_eligible"]) & (seg["window_status"] == "available")
                        & pd.to_numeric(seg["label_5d_bps"], errors="coerce").notna()].reset_index(drop=True)
        entry = {"eligible_available_sessions": int(len(available)),
                 "eligible_available_by_year": {int(y): int(c) for y, c in available.groupby("year").size().items()}}
        finite = available[pd.to_numeric(available["orig_label_bps"], errors="coerce").notna()]
        if len(finite) == 0:
            entry["label_comparison"] = None
        else:
            orig = pd.to_numeric(finite["orig_label_bps"], errors="coerce").to_numpy(float)
            new = pd.to_numeric(finite["label_5d_bps"], errors="coerce").to_numpy(float)
            entry["label_comparison"] = {
                "n_with_orig": int(len(finite)),
                "sign_disagreement_rate": float(np.mean(np.sign(orig) != np.sign(new))),
                "orig_mean_bps": float(orig.mean()), "label_5d_mean_bps": float(new.mean()),
                "orig_std_bps": float(orig.std(ddof=1)), "label_5d_std_bps": float(new.std(ddof=1)),
                "pearson_corr": (float(np.corrcoef(orig, new)[0, 1]) if np.std(orig) > 0 and np.std(new) > 0 else None),
                "spearman_corr": _spearman(orig, new),
                "risk_state_disagreement_rate": float(available["risk_state_disagreement"].mean()),
                "risk_clock_disagreement_rate": float(available["risk_clock_disagreement"].mean()),
                "recovery_disagreement_rate": float(available["recovery_disagreement"].mean())}
        seg_calendar = calendar[calendar["segment"] == segment].reset_index(drop=True)

        def ic(frame, segment=segment):
            subset = _filtered(frame, segment=segment)
            if len(subset) < 5:
                return None
            return _spearman(subset["predicted_net_advantage_bps"], subset["label_5d_bps"])

        entry["oos_ic_overall"] = block_bootstrap_full(seg_calendar, ic)
        by_year = {}
        for year, section in available.groupby("year"):
            has_model = int(year) >= 2017
            subset = _filtered(labels[(labels["segment"] == segment) & (labels["year"] == int(year))], segment=segment)
            by_year[int(year)] = {"model_available": bool(has_model), "n_finite": int(len(subset)),
                                  "ic": (_spearman(subset["predicted_net_advantage_bps"], subset["label_5d_bps"]) if has_model else None)}
        entry["oos_ic_by_year"] = by_year
        entry["oos_ic_model_available_from"] = 2017
        result["segments"][span] = entry
    return result
