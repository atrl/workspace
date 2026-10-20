"""One frozen, two-phase SOXX signed-semivariance information screen."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.core.native_evidence import _atomic_json, native_dataset, prepare_groups, finalize_evidence
from src.objective.relative_wealth import stationary_block_bootstrap_indices

NAME = "signed_variance_information_20261020"
FEATURES = ("log_rv1", "log_rv5", "log_rv22", "prior_signed_return", "prior_negative_share")
FIELDS = ["$open", "$close", "$price_valid"]
DEPENDENCIES = ("src/core/native_evidence.py", "src/objective/relative_wealth.py", "src/data/nas_market_data.py")
BOUNDS = {"training": ("2018-10-12 09:31:00", "2022-12-30 16:00:00"),
          "held": ("2023-01-03 09:31:00", "2026-10-09 16:00:00")}
BOOTSTRAP = dict(mean_block_length=63, replicates=10000, seed=20261014, lower_quantile=.025 / 15)


def _require(condition, message):
    if not bool(condition):
        raise ValueError("needs_data: " + message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _journal(path, **changes):
    state = json.loads(Path(path).read_text())
    state.update(changes)
    _atomic_json(Path(path), state)


def _operation(path, phase, operation, callback):
    state = json.loads(Path(path).read_text())
    state["operations"].append(dict(phase=phase, operation=operation, status="started"))
    _atomic_json(Path(path), state)
    result = callback()
    state = json.loads(Path(path).read_text())
    state["operations"].append(dict(phase=phase, operation=operation, status="completed"))
    _atomic_json(Path(path), state)
    return result


def _notebook_source(path):
    return [{"cell_type": c["cell_type"], "source": "".join(c.get("source", []))}
            for c in json.loads(Path(path).read_text())["cells"]
            if c.get("metadata", {}).get("role") != "conclusion"]


def current_implementation_identity(root):
    root = Path(root)
    card = root / "research" / NAME
    return dict(implementation_sha256=_sha(root / "src/workflow/signed_variance_information.py"),
        notebook_sha256=_digest(_notebook_source(card / (NAME + ".ipynb"))),
        test_sha256=_sha(root / "tests/test_signed_variance_information.py"),
        implementation_dependencies_sha256={p: _sha(root / p) for p in DEPENDENCIES})


def _authorize(root, card, plan):
    go_path = card / "execution_authorization.json"
    _require(go_path.is_file(), "explicit root execution_authorization.json required")
    go = json.loads(go_path.read_text())
    identity = {**current_implementation_identity(root), "plan_sha256": _sha(card / "plan.json"),
                "design_review_sha256": _sha(card / "design_review.json")}
    _require(go.get("root_go") is True and go.get("approved") is True and go.get("owner") == "root", "root GO absent")
    _require(go.get("budget") == plan["two_phase_bounds"]["budgets"], "GO budget differs")
    for key, value in identity.items():
        _require(go.get(key) == value, "GO identity differs: " + key)
    design = json.loads((card / "design_review.json").read_text())
    _require(design.get("verdict") == "ACCEPT_ONE_BOUNDED_SIGNED_VARIANCE_INFORMATION_DESIGN"
             and design.get("plan_sha256") == identity["plan_sha256"], "design review stale or rejected")
    review_path = card / "implementation_review.json"
    _require(review_path.is_file() and go.get("implementation_review_sha256") == _sha(review_path), "implementation review absent/stale")
    review = json.loads(review_path.read_text())
    _require(review.get("verdict") == "APPROVE_BOUNDED_SIGNED_VARIANCE_INFORMATION_IMPLEMENTATION", "implementation not approved")
    for key, value in identity.items():
        _require(review.get(key) == value, "implementation review identity differs: " + key)
    _require(not (card / "run_state.json").exists(), "single attempt already exists; no automatic rerun")
    return go, identity


def schedule_snapshot(binding):
    """Recompute only versioned XNYS metadata; never opens market values."""
    import exchange_calendars as xc
    actual_versions = dict(python_version=platform.python_version(), qlib_version=version("pyqlib"),
                           library_version=version("exchange_calendars"))
    for key, value in actual_versions.items():
        _require(value == binding[key], "environment version mismatch: " + key)
    start, end = binding["inclusive_bounds"]
    exchange = xc.get_calendar("XNYS", start=start, end=end)
    sessions = exchange.sessions_in_range(start, end)
    rows = []
    for session in sessions:
        open_et = exchange.session_open(session).tz_convert(ZoneInfo("America/New_York"))
        close_et = exchange.session_close(session).tz_convert(ZoneInfo("America/New_York"))
        minutes = int((close_et - open_et).total_seconds() / 60)
        _require(minutes in (390, 210), "unexpected official session length")
        rows.append(dict(session=pd.Timestamp(session).strftime("%Y-%m-%d"),
            open_et=open_et.isoformat(timespec="seconds"), close_et=close_et.isoformat(timespec="seconds"), minutes=minutes))
    encoded = (json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
    seal = dict(**actual_versions, session_count=len(rows), first_session=rows[0]["session"], last_session=rows[-1]["session"],
        regular_390_minute_sessions=sum(r["minutes"] == 390 for r in rows),
        early_210_minute_sessions=sum(r["minutes"] == 210 for r in rows),
        canonical_schedule_sha256=hashlib.sha256(encoded).hexdigest())
    for key, value in seal.items():
        _require(value == binding[key], "bounded XNYS schedule mismatch: " + key)
    schedule = pd.DataFrame(rows).set_index("session")
    schedule.index = pd.DatetimeIndex(schedule.index, name="datetime")
    for column in ("open_et", "close_et"):
        schedule[column] = [pd.Timestamp(value).tz_convert("America/New_York") for value in schedule[column]]
    return schedule, seal, rows


def verify_source(root, plan):
    provider = Path(root) / plan["source"]["provider_uri"]
    result = {relative: _sha(provider / relative) for relative in plan["source"]["sha256"]}
    _require(result == plan["source"]["sha256"], "pinned source bytes changed")
    return result


def _no_writer():
    from qlib.workflow import R
    if R.__dict__.get("_provider") is not None:
        active = R.exp_manager.active_experiment
        _require(active is None or active.active_recorder is None, "source query with writable global Recorder")


def _normalize_bars(frame):
    _require(isinstance(frame.index, pd.MultiIndex) and set(frame.index.names) == {"datetime", "instrument"}, "native feature index differs")
    if list(frame.index.names) != ["datetime", "instrument"]:
        frame = frame.reorder_levels(["datetime", "instrument"])
    _require(set(frame.index.get_level_values("instrument").str.upper()) == {"SOXX"}, "unexpected feature instrument")
    out = frame.droplevel("instrument").sort_index().copy()
    out.columns = [str(c).lstrip("$") for c in out.columns]
    _require(out.index.tz is None and out.index.is_unique, "source minute labels must be unique naive ET right edges")
    return out


def measure_sessions(bars, schedule):
    """Keep every scheduled session; never bridge an absent intraday minute."""
    groups = {day: group for day, group in bars.groupby(bars.index.normalize())}
    rows = []
    for day, session in schedule.iterrows():
        expected = pd.date_range(session.open_et.tz_localize(None) + pd.Timedelta(minutes=1),
                                 session.close_et.tz_localize(None), freq="min")
        observed = groups.get(day, bars.iloc[:0])
        valid = observed.index.equals(expected)
        if valid:
            prices = observed[["open", "close"]].to_numpy(float)
            valid = bool(np.isfinite(prices).all() and (prices > 0).all() and observed.price_valid.eq(1).all())
        rv = share = signed = np.nan
        if valid:
            logprices = np.log(np.r_[observed.open.iloc[0], observed.close.to_numpy(float)])
            returns = np.diff(logprices)
            rv = float(np.square(returns).sum())
            valid = bool(np.isfinite(rv) and rv > 0)
            if valid:
                share = float(np.square(returns[returns < 0]).sum() / rv)
                signed = float(logprices[-1] - logprices[0])
        rows.append(dict(datetime=day, rv=rv if valid else np.nan, negative_share=share if valid else np.nan,
            signed_return=signed if valid else np.nan, session_known=int(valid),
            expected_minutes=int(session.minutes), observed_minutes=len(observed),
            completed_available_at=session.close_et + pd.Timedelta(seconds=60)))
    return pd.DataFrame(rows).set_index("datetime")


def _dataset(frame, *, features, labels=()):
    source = frame.copy()
    source.index = pd.MultiIndex.from_arrays([frame.index, ["SOXX"] * len(frame)], names=["datetime", "instrument"])
    source.columns = pd.MultiIndex.from_tuples([("feature" if c in features else "label" if c in labels else "metadata", c) for c in frame.columns])
    dataset = native_dataset(source, segments={"all": (None, None)})
    readback = prepare_groups(dataset, "all")
    _require(readback.index.equals(source.index) and readback.columns.equals(source.columns), "DatasetH changed source keys/schema")
    flattened = readback.copy()
    flattened.columns = flattened.columns.get_level_values(1)
    flattened.index = flattened.index.droplevel("instrument")
    flattened.attrs["native_qlib_dataset"] = "DatasetH"
    return dataset, flattened


def _read_phase(root, plan, schedule, phase, journal):
    import qlib
    from qlib.data import D
    _no_writer()
    start, end = BOUNDS[phase]
    qlib.init(provider_uri=str(Path(root) / plan["source"]["provider_uri"]), region="us", kernels=1)
    native_calendar = pd.DatetimeIndex(_operation(journal, phase, "D.calendar", lambda: D.calendar(start_time=start, end_time=end, freq="1min")))
    phase_schedule = schedule.loc[pd.Timestamp(start).normalize():pd.Timestamp(end).normalize()]
    _require(native_calendar.normalize().unique().equals(phase_schedule.index), "Qlib and XNYS source session dates differ")
    pool = D.instruments("all")
    instruments = _operation(journal, phase, "D.list_instruments", lambda: D.list_instruments(pool, start_time=start, end_time=end, freq="1min", as_list=True))
    _require("SOXX" in {str(s).upper() for s in instruments}, "native provider lacks SOXX")
    raw = _operation(journal, phase, "D.features", lambda: D.features(["SOXX"], FIELDS, start_time=start, end_time=end, freq="1min"))
    bars = _normalize_bars(raw)
    _require(bars.index.min() >= pd.Timestamp(start) and bars.index.max() <= pd.Timestamp(end), "numeric source escaped phase")
    daily = measure_sessions(bars, phase_schedule)
    dataset, daily = _dataset(daily, features=("rv", "negative_share", "signed_return"))
    return daily, dataset


def target_rows(daily, schedule, *, start, end):
    """Use 22 consecutive official lag sessions, preserving incomplete targets."""
    dates = schedule.loc[start:end].index
    rows = []
    for target in dates:
        pos = schedule.index.get_loc(target)
        lag_dates = schedule.index[max(0, pos - 22):pos]
        window = daily.reindex(lag_dates)
        target_row = daily.reindex([target]).iloc[0]
        lag_known = (len(lag_dates) == 22 and window.session_known.eq(1).all()
                     and np.isfinite(window.rv).all() and window.rv.gt(0).all())
        label_known = bool(target_row.session_known == 1 and np.isfinite(target_row.rv) and target_row.rv > 0)
        available = window.completed_available_at.iloc[-1] if len(window) else pd.NaT
        decision = schedule.loc[target, "open_et"] - pd.Timedelta(minutes=1)
        clocks_ok = bool(pd.notna(available) and available < decision)
        known = bool(lag_known and clocks_ok)
        prior = window.iloc[-1] if len(window) else None
        feature_values = [np.log(prior.rv), np.log(window.rv.iloc[-5:].mean()), np.log(window.rv.mean()),
                          prior.signed_return, prior.negative_share] if known else [np.nan] * len(FEATURES)
        _require(not known or np.isfinite(feature_values).all(), "invalid lag features")
        rows.append(dict(datetime=target, **dict(zip(FEATURES, feature_values)), target_rv=float(target_row.rv) if label_known else np.nan,
            feature_known=int(known), label_known=int(label_known), required_lag_sessions=len(lag_dates),
            known_lag_sessions=int(window.session_known.eq(1).sum()),
            feature_available_at=available, decision_at=decision,
            label_mature_at=schedule.loc[target, "close_et"] + pd.Timedelta(seconds=60)))
    return pd.DataFrame(rows).set_index("datetime")


def _complete(frame):
    return (frame.feature_known.eq(1) & frame.label_known.eq(1)
            & np.isfinite(frame[[*FEATURES, "target_rv"]].to_numpy(float)).all(axis=1) & frame.target_rv.gt(0))


@dataclass(frozen=True)
class VarianceOLS:
    arm: str
    features: tuple
    mean: tuple
    std: tuple
    coefficients: tuple
    smearing: float
    training_rows: int
    training_frame_sha256: str

    def predict(self, frame):
        values = frame.loc[:, self.features].to_numpy(float)
        design = np.column_stack([np.ones(len(values)), (values - self.mean) / self.std])
        forecast = np.exp(design @ np.asarray(self.coefficients)) * self.smearing
        _require(np.isfinite(forecast).all() and (forecast > 0).all(), "nonpositive or nonfinite variance forecast")
        return forecast

    @property
    def gamma(self):
        return float(self.coefficients[-1] / self.std[-1]) if self.arm == "C" else None


def _frame_hash(frame):
    return hashlib.sha256(frame.to_json(orient="split", date_format="iso", date_unit="ns").encode()).hexdigest()


def fit_model(training, *, arm):
    _require(arm in ("B", "C"), "unknown OLS arm")
    _require(training.attrs.get("native_qlib_dataset") == "DatasetH", "fit requires native DatasetH readback")
    _require(training.index.is_unique and training.index.is_monotonic_increasing, "unordered training sessions")
    _require(training.index.min() >= pd.Timestamp("2019-01-02") and training.index.max() <= pd.Timestamp("2022-12-30"), "fit escaped training target dates")
    _require(len(training) >= 200 and _complete(training).all(), "training row qualification failed")
    features = FEATURES[:-1] if arm == "B" else FEATURES
    x, y = training[list(features)].to_numpy(float), np.log(training.target_rv.to_numpy(float))
    mean, std = x.mean(axis=0), x.std(axis=0, ddof=0)
    _require(np.isfinite(std).all() and (std > 0).all(), "invalid training scale")
    design = np.column_stack([np.ones(len(x)), (x - mean) / std])
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    _require(rank == design.shape[1], "rank-deficient training model")
    smear = float(np.exp(y - design @ coefficients).mean())
    _require(np.isfinite(smear) and smear > 0, "invalid training-only smearing factor")
    return VarianceOLS(arm, tuple(features), tuple(map(float, mean)), tuple(map(float, std)),
        tuple(map(float, coefficients)), smear, len(training), _frame_hash(training[[*FEATURES, "target_rv"]]))


def qlike(actual, forecast):
    actual, forecast = np.asarray(actual, float), np.asarray(forecast, float)
    _require(np.isfinite(actual).all() and (actual > 0).all() and np.isfinite(forecast).all() and (forecast > 0).all(), "QLIKE inputs invalid")
    return np.log(forecast) + actual / forecast


def evaluate(held, models, expected_index, *, bootstrap=BOOTSTRAP):
    _require(held.index.equals(expected_index) and held.index.is_unique, "held target calendar changed or compacted")
    _require(_complete(held).all(), "held target or required 22-session lag unknown; no QLIKE/bootstrap permitted")
    _require(set(models) == {"B", "C"}, "exact frozen B/C models required")
    pred_b, pred_c = models["B"].predict(held), models["C"].predict(held)
    loss_b, loss_c = qlike(held.target_rv, pred_b), qlike(held.target_rv, pred_c)
    delta = loss_b - loss_c
    indices = stationary_block_bootstrap_indices(observation_count=len(delta), path_length=len(delta),
        path_count=int(bootstrap["replicates"]), expected_block_length=int(bootstrap["mean_block_length"]), seed=int(bootstrap["seed"]))
    draws = delta[indices].mean(axis=1)
    lower = float(np.quantile(draws, bootstrap["lower_quantile"], method="linear"))
    paired = pd.DataFrame(dict(target_rv=held.target_rv, forecast_B=pred_b, forecast_C=pred_c,
        qlike_B=loss_b, qlike_C=loss_c, paired_improvement=delta), index=held.index)
    yearly = {str(year): dict(sessions=len(part), mean_qlike_B=float(part.qlike_B.mean()), mean_qlike_C=float(part.qlike_C.mean()),
                            mean_paired_improvement=float(part.paired_improvement.mean())) for year, part in paired.groupby(paired.index.year)}
    gamma, improvement = models["C"].gamma, float(delta.mean())
    return dict(gamma=gamma, held_sessions=len(held), mean_paired_qlike_improvement=improvement,
        stationary_bootstrap_lower=lower, bootstrap=dict(bootstrap), yearly=yearly,
        passed=bool(gamma > 0 and improvement > 0 and lower > 0), paired=paired, bootstrap_draws=draws)


def _transport(root, card):
    source = json.loads((root / "research/constituent_intraday_history_20261014/nas_binding.json").read_text())
    _atomic_json(card / "nas_binding.json", dict(schema_version="recorder_evidence_binding_v1", authority=source["authority"],
        ssh=source["ssh"], source_id=NAME, source_plan="plan.json"))
    return card / "nas_binding.json"


def run(*, root):
    """Execute the notebook's single reviewed attempt; no pre-GO numeric reads."""
    root = Path(root).resolve()
    card = root / "research" / NAME
    plan = json.loads((card / "plan.json").read_text())
    go, identity = _authorize(root, card, plan)
    journal = card / "run_state.json"
    with journal.open("x") as stream:
        json.dump(dict(status="authorized", identity=identity, authorization_sha256=_sha(card / "execution_authorization.json"),
            operations=[], fits_completed=[], fit_attempts=0, bootstrap_attempts=0, bootstrap_completed=0,
            accounts=0, automatic_resume_forbidden=True), stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    from qlib.workflow import R
    from qlib.workflow.recorder import Recorder
    recorder = None
    try:
        schedule, calendar_seal, schedule_rows = schedule_snapshot(plan["calendar_and_measurement"]["schedule_binding"])
        source_seal = verify_source(root, plan)
        _journal(journal, status="training_pre_read_sealed", training_pre_read_seal=dict(calendar=calendar_seal, source=source_seal))
        daily, daily_dataset = _read_phase(root, plan, schedule, "training", journal)
        rows = target_rows(daily, schedule, start="2019-01-02", end="2022-12-30")
        training_dataset, rows = _dataset(rows, features=FEATURES, labels=("target_rv",))
        training = rows.loc[_complete(rows)].copy()
        training.attrs["native_qlib_dataset"] = "DatasetH"
        _require(len(training) >= 200, "fewer than 200 common qualified training sessions")
        tail = daily.tail(22).copy()
        _require(tail.index.equals(schedule.loc[:"2022-12-30"].index[-22:]), "training tail is not the exact final 22 native sessions")
        tail_dataset, tail = _dataset(tail, features=("rv", "negative_share", "signed_return"))
        source_seal = verify_source(root, plan)
        training_audit = dict(target_sessions=len(rows), qualified_sessions=len(training), excluded_sessions=int((~_complete(rows)).sum()),
                              measured_sessions=len(daily), unknown_source_sessions=int(daily.session_known.eq(0).sum()))
        _journal(journal, status="training_sources_complete", training_audit=training_audit)
        with R.start(experiment_name=NAME, recorder_name="single_fixed_signed_variance", uri=str(card / "qlib_recorder")):
            recorder = R.get_recorder()
            _journal(journal, status="training_models", recorder_id=str(recorder.id), experiment_id=str(recorder.experiment_id))
            recorder.save_objects(artifact_path="inputs", plan=plan, authorization=go, training_dataset=training_dataset,
                training_daily_dataset=daily_dataset, training_audit=training_audit, bounded_schedule=schedule_rows, pre_read_calendar_seal=calendar_seal)
            recorder.save_objects(artifact_path="sources", implementation_snapshot={p: (root / p).read_bytes() for p in
                ("src/workflow/signed_variance_information.py", "tests/test_signed_variance_information.py", *DEPENDENCIES)},
                normalized_notebook=_notebook_source(card / (NAME + ".ipynb")), identity=identity)
            models = {}
            for arm in ("B", "C"):
                _journal(journal, fit_attempts=len(models) + 1)
                def fit_and_save():
                    model = fit_model(training, arm=arm)
                    recorder.save_objects(artifact_path="models", **{arm: asdict(model)})
                    return model
                model = _operation(journal, "training", "OLS_" + arm, fit_and_save)
                models[arm] = model
                _journal(journal, fits_completed=list(models), model_sha256={key: _digest(asdict(value)) for key, value in models.items()})
            receipt = dict(identity=identity, models={key: asdict(value) for key, value in models.items()},
                training_rows_sha256=_frame_hash(training), tail_sha256=_frame_hash(tail), tail_sessions=22,
                source_sha256=source_seal, calendar=calendar_seal, held_numeric_not_read=True)
            receipt_hash = _digest(receipt)
            recorder.save_objects(artifact_path="freeze", receipt=receipt, receipt_sha256=receipt_hash, lag_tail_dataset=tail_dataset)
            recorder.log_metrics(training_qualified_sessions=len(training), information_fits=2)
        _require(recorder.status == Recorder.STATUS_FI, "training writer not FINISHED at barrier")
        _no_writer()
        _require(_digest(recorder.load_object("freeze/receipt")) == receipt_hash, "freeze receipt readback mismatch")
        for arm in ("B", "C"):
            _require(recorder.load_object("models/" + arm) == asdict(models[arm]), "model readback mismatch")
        cached = prepare_groups(recorder.load_object("freeze/lag_tail_dataset"), "all")
        cached.columns = cached.columns.get_level_values(1)
        cached.index = cached.index.droplevel("instrument")
        _require(_frame_hash(cached) == receipt["tail_sha256"], "sealed 22-session tail readback mismatch")
        _journal(journal, status="models_and_tail_sealed", freeze_receipt_sha256=receipt_hash)
        held_schedule, held_seal, _ = schedule_snapshot(plan["calendar_and_measurement"]["schedule_binding"])
        _require(held_seal == calendar_seal, "bounded schedule changed between phases")
        held_source_seal = verify_source(root, plan)
        _journal(journal, status="held_pre_read_sealed", held_pre_read_seal=dict(calendar=held_seal, source=held_source_seal))
        held_daily, held_daily_dataset = _read_phase(root, plan, held_schedule, "held", journal)
        verify_source(root, plan)
        held = target_rows(pd.concat([cached, held_daily]), held_schedule, start="2023-01-03", end="2026-10-09")
        held_dataset, held = _dataset(held, features=FEATURES, labels=("target_rv",))
        held_audit = dict(target_sessions=len(held), qualified_sessions=int(_complete(held).sum()),
            unknown_target_sessions=int(held.label_known.eq(0).sum()), unknown_lag_windows=int(held.feature_known.eq(0).sum()))
        _journal(journal, status="held_source_complete", held_audit=held_audit)
        recorder.start_run()
        recorder.save_objects(artifact_path="held", dataset=held_dataset, daily_dataset=held_daily_dataset, audit=held_audit)
        _require(_complete(held).all(), "held target or required lag unknown; abort before QLIKE/bootstrap")
        _journal(journal, status="held_evaluation", bootstrap_attempts=1)
        result = _operation(journal, "held", "QLIKE_stationary_bootstrap", lambda: evaluate(held, models, held_schedule.loc["2023-01-03":"2026-10-09"].index))
        _journal(journal, bootstrap_completed=1)
        recorder.save_objects(artifact_path="analysis", result=result)
        scalar = {key: value for key, value in result.items() if key not in ("paired", "bootstrap_draws")}
        recorder.log_metrics(q_coefficient=result["gamma"], held_mean_paired_qlike=result["mean_paired_qlike_improvement"],
                             stationary_bootstrap_lower=result["stationary_bootstrap_lower"], information_screen_pass=int(result["passed"]))
        verify_source(root, plan)
        summary = dict(experiment=NAME, decision="information_screen_pass" if result["passed"] else "not_promoted", promotion_decision="not_promoted",
            claim_scope=plan["claim_scope"], independent_validation=False, production_changed=False, information_screen=scalar,
            plan_sha256=identity["plan_sha256"], implementation_identity=identity, freeze_receipt_sha256=receipt_hash,
            training=training_audit, held=held_audit, actual_counts=json.loads(journal.read_text()),
            new_information_fits=2, new_bootstraps=1, new_native_accounts=0,
            recorder_uri=str(card / "qlib_recorder"), region="us", workflow_config="workflow.yaml",
            artifact_policy=dict(policy="slim_card_recorder_first", workflow_config="workflow.yaml",
                recorder_only_artifacts=["inputs", "sources", "models", "freeze", "held", "analysis/result"]))
        recorder.save_objects(artifact_path="analysis", summary=summary)
        recorder.end_run(status=Recorder.STATUS_FI)
        summary = finalize_evidence(recorder, card=card, summary=summary, transport_binding=_transport(root, card), workspace_root=root)
        _journal(journal, status="completed", decision=summary["decision"])
        return summary
    except BaseException as error:
        _journal(journal, status="needs_data" if isinstance(error, ValueError) else "execution_failed",
            failure=dict(type=type(error).__name__, message=str(error)), automatic_resume_forbidden=True)
        if recorder is not None:
            try:
                if recorder.status != Recorder.STATUS_R:
                    recorder.start_run()
                state = json.loads(journal.read_text())
                recorder.save_objects(artifact_path="execution", failure=state)
                recorder.log_metrics(actual_completed_fits=len(state["fits_completed"]), actual_bootstrap_attempts=state["bootstrap_attempts"])
                recorder.end_run(status=Recorder.STATUS_FA)
                from src.data.nas_market_data import archive_nas_recorder
                evidence = archive_nas_recorder(_transport(root, card), workspace_root=root, card=card,
                    recorder_id=str(recorder.id), experiment_id=str(recorder.experiment_id))
                _journal(journal, failure_nas_evidence=evidence)
            except BaseException as archive_error:
                _journal(journal, failure_archive_error=dict(type=type(archive_error).__name__, message=str(archive_error)))
        raise
