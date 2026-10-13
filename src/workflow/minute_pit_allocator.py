"""Notebook-owned, bounded minute-information campaign for the current PIT MLP."""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import QlibProviderConfig, build_qlib_store_from_frame, load_qlib_feature_frame
from src.data.minute_allocation_features import completed_session_features, label_maturity
from src.data.minute_intake import atomic_json
from src.data.nas_minute_bundle import load_nas_minute_bundle
from src.data.nas_market_data import archive_nas_recorder, materialize_nas_snapshot
from src.workflow.direct_continuous_soxx_allocation import DirectContinuousAllocationSpec, _load_frame
from src.workflow.pit_allocation_retrain import _training_auxiliary, validate_frozen_bundle
from src.workflow.pit_constituent_auxiliary_training_allocation import (
    COMMON_FEATURES, PitAuxiliaryAllocationSpec, _soxx_common_training_frame,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset(features, labels, common, metadata, segments):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    frame = pd.concat({"feature": features, "label": labels, "common": common, "meta": metadata}, axis=1)
    frame.index = pd.MultiIndex.from_arrays([frame.index, ["SOXL_SOXX_CASH"] * len(frame)],
                                           names=["datetime", "instrument"])
    return DatasetH(handler=DataHandlerLP.from_df(frame), segments=segments)


def _auxiliary_dataset(frame, observations_end):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    eligible = frame.loc[frame["$pit_available"].gt(.5) & frame["training_eligible"]].copy()
    required = [*COMMON_FEATURES, "synthetic_2x_crash_label", "synthetic_2x_upside_label", "training_weight"]
    eligible = eligible.dropna(subset=required)
    features = eligible.loc[:, list(COMMON_FEATURES)]
    labels = eligible.loc[:, required[len(COMMON_FEATURES):]]
    metadata = pd.DataFrame({"observations_end": pd.Timestamp(observations_end)}, index=eligible.index)
    static = pd.concat({"feature": features, "label": labels, "meta": metadata}, axis=1)
    static.index = pd.MultiIndex.from_arrays([eligible.datetime, eligible.instrument], names=["datetime", "instrument"])
    return DatasetH(handler=DataHandlerLP.from_df(static), segments={"train": (None, observations_end)})


def prepare_minute_allocator_inputs(*, root: Path, card: Path, plan: dict) -> dict:
    """Freeze Qlib-read inputs before training; do not mutate runtime providers."""
    import exchange_calendars as xc
    from qlib.data import D
    from qlib.workflow import R

    binding = json.loads((card / plan["nas_binding"]).read_text())
    if binding["source_id"] != plan["source_id"]:
        raise ValueError("blocked: minute source differs from the frozen plan")
    materialize_nas_snapshot(binding, workspace_root=root)
    manifest = root / binding["source_manifest_path"]
    if _hash(manifest) != plan["source_id"]:
        raise ValueError("blocked: raw source identity differs")
    raw, _ = load_nas_minute_bundle(binding, workspace_root=root)
    provider_root = root / "data/qlib" / plan["experiment"]
    fields = ["open", "high", "low", "close", "volume", "price_valid"]
    minute_audits = {}
    native = {}
    for key, symbols, frequency in [("core", ["SOXL", "SOXX"], "1min"),
                                    ("peer", ["QQQ", "SMH", "NVDA", "AMD"], "5min")]:
        rows = raw.loc[raw.symbol.isin(symbols)].rename(columns={"symbol": "instrument", "timestamp": "datetime"})
        provider = provider_root / (key + "_" + frequency)
        minute_audits[key] = build_qlib_store_from_frame(rows, provider_uri=provider, freq=frequency,
            region="us", as_of=plan["as_of"], field_map={"$" + name: name for name in fields},
            source="NAS raw NONE snapshot sha256=" + plan["source_id"])
        native[key] = load_qlib_feature_frame(QlibProviderConfig(provider, freq=frequency),
            instruments=symbols, fields=["$" + name for name in fields], start=plan["source_start"],
            end=plan["as_of"] + " 16:00:00")
    source_rows = len(raw)
    del raw
    schedule = xc.get_calendar("XNYS", start=plan["source_start"], end=plan["as_of"]).schedule
    minute, coverage = completed_session_features(native["core"], native["peer"], schedule,
        core_names=plan["feature_groups"]["core"], peer_names=plan["feature_groups"]["peer"],
        peer_symbols=["QQQ", "SMH", "NVDA", "AMD"])
    del native

    before = {name: _hash(root / name) for name in plan["daily_source_manifest_sha256"]}
    if before != plan["daily_source_manifest_sha256"]:
        raise ValueError("blocked: daily provider identity changed after the plan was frozen")
    frame, base_audit, features = _load_frame(DirectContinuousAllocationSpec(root=root), include_incomplete_latest=True)
    if frame.index.max() > pd.Timestamp(plan["as_of"]):
        raise ValueError("blocked: daily observations exceed the frozen as_of")
    frame = frame.loc[pd.Timestamp(plan["train_start"]):].copy()
    common = _soxx_common_training_frame(PitAuxiliaryAllocationSpec(root=root), end=plan["as_of"])
    prices = load_qlib_feature_frame(QlibProviderConfig(root / "data/qlib/soxl_soxx_soxs"),
        instruments=["SOXL", "SOXX"], fields=["$open", "$close"],
        start="2010-03-11", end=plan["as_of"])
    calendar = pd.DatetimeIndex(D.calendar(start_time="2010-03-11", end_time=plan["as_of"], freq="day"))
    daily_provider = provider_root / "daily_prices"
    price_manifest = build_qlib_store_from_frame(prices.reset_index(), provider_uri=daily_provider,
        freq="day", region="us", as_of=plan["as_of"], field_map={c: c for c in prices},
        source="Qlib observed open/close execution marks only; source manifests pinned by plan.json")
    timing = label_maturity(frame, calendar)
    augmented = frame.join(minute.drop(columns="minute_feature_available_at"), how="left")
    augmented.loc[:, plan["availability_features"]] = augmented.loc[:, plan["availability_features"]].fillna(0.)
    available = minute.minute_feature_available_at.reindex(frame.index)
    if (available.dropna() > timing.decision_at.reindex(available.dropna().index)).any():
        raise ValueError("blocked: minute features exceed the declared decision clock")
    numeric = list(dict.fromkeys([*features, *plan["availability_features"],
                                  *plan["feature_groups"]["core"], *plan["feature_groups"]["peer"]]))
    feature_rows = augmented.loc[:, numeric].reset_index().rename(columns={augmented.index.name: "datetime"})
    feature_rows["instrument"] = "SOXL_SOXX_CASH"
    feature_provider = provider_root / "allocation_features"
    feature_manifest = build_qlib_store_from_frame(feature_rows, provider_uri=feature_provider, region="us",
        freq="day", as_of=plan["as_of"], field_map={"$" + name: name for name in numeric},
        source="native daily inputs plus completed NAS NONE minute sessions; source=" + plan["source_id"])
    observed = load_qlib_feature_frame(QlibProviderConfig(feature_provider), instruments=["SOXL_SOXX_CASH"],
        fields=["$" + name for name in numeric], start=plan["train_start"], end=plan["as_of"])
    observed = observed.xs("SOXL_SOXX_CASH", level="instrument").rename(columns=lambda c: c.removeprefix("$")).astype(float)
    if not observed.index.equals(frame.index):
        raise ValueError("blocked: new features changed the full daily calendar")

    model_parameters = {name: plan[name] for name in ["hidden_units", "tail_hidden_units", "epochs", "learning_rate",
        "weight_decay", "maximum_soxl_weight", "minimum_cash_weight", "primary_tail_loss_weight",
        "turnover_loss_weight", "training_cost_bps", "auxiliary_batch_size", "purge_rows", "smoothing_span", "rebalance_threshold"]}
    spec = PitAuxiliaryAllocationSpec(root=root, experiment=plan["experiment"], train_start=plan["train_start"],
        development_years=tuple(plan["development_years"]), outer=tuple(plan["outer_window"]), **model_parameters)
    train_ends = {}
    auxiliary, auxiliary_audits = {}, {}
    for year in plan["development_years"]:
        end_position = calendar.searchsorted(pd.Timestamp(f"{year}-01-01")) - plan["purge_rows"] - 1
        train_ends[str(year)] = str(calendar[end_position].date())
    train_ends["outer"] = "2022-12-30"
    for fold, end in train_ends.items():
        values, audit = _training_auxiliary(replace(spec, development_end=end))
        auxiliary[fold] = values
        auxiliary_audits[fold] = audit
    if {_name: _hash(root / _name) for _name in before} != before:
        raise ValueError("blocked: daily source changed during input preparation")
    parent = root / plan["parent_release"]["allocation"]["card_path"]
    if _hash(parent / "frozen_bundle_manifest.json") != plan["parent_bundle_manifest_sha256"]:
        raise ValueError("blocked: current model bundle manifest changed")
    validate_frozen_bundle(parent)
    artifact = plan["parent_model_artifact"]
    relative = artifact["path"].split("/artifacts/", 1)[1]
    with R.uri_context(str(parent / "qlib_recorder")):
        recorder = R.get_recorder(recorder_id=plan["parent_release"]["allocation"]["recorder_id"], experiment_name=parent.name)
        downloaded = Path(recorder.download_artifact(relative))
        if _hash(downloaded) != artifact["sha256"]:
            raise ValueError("blocked: parent model object changed")
        frozen_models = pickle.loads(downloaded.read_bytes())
    audit = {"source_observations": source_rows, "minute_source_id": plan["source_id"],
        "minute_price_basis": "raw NONE within-session", "minute_sessions": len(minute),
        "valid_core_sessions": int(minute.minute_core_ready.sum()), "valid_peer_sessions": int(minute.minute_peer_ready.sum()),
        "full_daily_rows": len(frame), "daily_first_signal": str(frame.index.min().date()),
        "daily_last_signal": str(frame.index.max().date()), "original_features": features,
        "native_minute_store_audits": minute_audits, "native_daily_price_audit": price_manifest,
        "native_feature_store_audit": feature_manifest, "base_input_audit": base_audit,
        "auxiliary_audits": auxiliary_audits, "train_ends": train_ends,
        "canonical_missing_minute_features_preserved": True, "runtime_providers_mutated": False,
        "new_minute_source_uses_future_adjustment_coefficients": False,
        "historical_timing_is_live_receipt_certification": False}
    return {"frame": frame, "features": observed, "base_features": features,
        "common": common.reindex(frame.index).loc[:, list(COMMON_FEATURES)], "timing": timing,
        "minute": minute, "coverage": coverage, "calendar": calendar, "prices": prices,
        "auxiliary": auxiliary, "frozen_models": frozen_models, "spec": spec,
        "daily_provider": daily_provider, "audit": audit}


def _prediction_dates(frame):
    result = frame.copy()
    if isinstance(result.index, pd.MultiIndex):
        result.index = result.index.get_level_values("datetime")
    return result.sort_index()


def _primary_dataset(inputs, *, train_end, predict_bounds):
    from src.model.minute_pit_allocator import TARGETS

    return _dataset(inputs["features"], inputs["frame"].loc[:, list(TARGETS)], inputs["common"],
        inputs["timing"], {"train": (inputs["spec"].train_start, train_end), "predict": tuple(predict_bounds)})


def _market(inputs, *, window, plan):
    frame, timing = inputs["frame"], inputs["timing"]
    known = frame.trade_date.notna() & frame.exit_date.notna()
    if window == "development":
        known &= frame.index.year.isin(plan["development_years"])
        known &= timing.return_available_at.lt(pd.Timestamp(plan["development_return_cutoff"]))
    else:
        known &= frame.index >= pd.Timestamp(plan["outer_window"][0])
    result = frame.loc[known, ["trade_date", "exit_date", "soxl_forward_open_return", "soxx_forward_open_return"]].copy()
    result = result.rename(columns={"soxl_forward_open_return": "soxl_return", "soxx_forward_open_return": "soxx_return"})
    result["signal_date"] = result.index
    result["decision_at"], result["return_available_at"] = timing.loc[known, "decision_at"], timing.loc[known, "return_available_at"]
    return result.reset_index(drop=True)


def _frozen_predictions(inputs, *, window, plan):
    from src.model.minute_pit_allocator import _scaled
    from src.workflow.pit_constituent_auxiliary_training_allocation import PitAuxiliaryAllocationNet, _project_legal_weights
    import torch

    parts = []
    folds = [str(y) for y in plan["development_years"]] if window == "development" else ["outer"]
    for fold in folds:
        state = inputs["frozen_models"]["outer_refit" if fold == "outer" else "fold_" + fold]
        take = (inputs["features"].index >= pd.Timestamp(plan["outer_window"][0])) if fold == "outer" else (inputs["features"].index.year == int(fold))
        feature = inputs["features"].loc[take]
        common = inputs["common"].loc[take]
        network = PitAuxiliaryAllocationNet(primary_feature_count=len(state["primary_features"]),
            common_feature_count=len(state["common_features"]), hidden_units=plan["hidden_units"],
            tail_hidden_units=plan["tail_hidden_units"])
        network.load_state_dict({key: torch.tensor(value) for key, value in state["model_state"].items()})
        network.eval()
        with torch.no_grad():
            logits, tail, alpha = network(_scaled(feature, state["primary_features"], state["primary_scaler"]),
                                          _scaled(common, state["common_features"], state["common_scaler"]))
            values = _project_legal_weights(logits, maximum_soxl_weight=plan["maximum_soxl_weight"],
                                            minimum_cash_weight=plan["minimum_cash_weight"]).numpy().astype(float)
        values /= values.sum(axis=1, keepdims=True)
        output = pd.DataFrame(values, index=feature.index, columns=["soxl_weight", "soxx_weight", "cash_weight"])
        output["tail_probability"], output["alpha_probability"] = torch.sigmoid(tail).numpy(), torch.sigmoid(alpha).numpy()
        parts.append(output)
    return pd.concat(parts).sort_index()


def _ensemble(predictions, family, seeds):
    parts = [predictions[(family, seed)] for seed in seeds]
    if any(not p.index.equals(parts[0].index) for p in parts):
        raise ValueError("blocked: paired seed prediction calendars differ")
    result = sum(parts) / len(parts)
    names = ["soxl_weight", "soxx_weight", "cash_weight"]
    result.loc[:, names] = result.loc[:, names].div(result.loc[:, names].sum(axis=1), axis=0)
    return result


def run_minute_allocator_campaign(*, root: Path, card: Path, plan: dict, inputs: dict) -> dict:
    """Train the preregistered groups once; archive all seeds and native books."""
    import qlib
    import yaml
    from qlib.workflow import R
    from src.model.minute_pit_allocator import MinutePitAllocator
    from src.backtest.minute_allocator import native_allocator_book
    from src.analysis.minute_allocator import compare_native_books, development_selection, feature_information_audit
    from src.analysis.integrated_allocation import annual_paired_metrics, paired_metrics

    if (card / "summary.json").exists():
        existing = json.loads((card / "summary.json").read_text())
        if existing["plan_sha256"] != _hash(card / "plan.json"):
            raise ValueError("blocked: finalized campaign plan differs")
        return existing
    plan_hash = _hash(card / "plan.json")
    original_release_hash = _hash(root / "production/stage1_futu_sim_v1/runtime_release.json")
    spec = {**asdict(inputs["spec"]), "root": str(root)}
    masks = {**{name: "minute_core_ready" for name in plan["feature_groups"]["core"]},
             **{name: "minute_peer_ready" for name in plan["feature_groups"]["peer"]}}
    auxiliary_datasets = {fold: _auxiliary_dataset(frame, inputs["audit"]["train_ends"][fold])
                          for fold, frame in inputs["auxiliary"].items()}
    datasets = {str(year): _primary_dataset(inputs, train_end=inputs["audit"]["train_ends"][str(year)],
                                            predict_bounds=(f"{year}-01-01", f"{year}-12-31"))
                for year in plan["development_years"]}
    tracking = card / "qlib_recorder"
    qlib.init(provider_uri=str(inputs["daily_provider"]), region="us", kernels=1)
    training_rows, predictions, fitted_models = [], {}, {}
    with R.start(experiment_name=plan["experiment"], recorder_name="current_pit_mlp_minute_information", uri=str(tracking)):
        recorder = R.get_recorder()
        recorder_id, experiment_id = recorder.id, str(recorder.experiment_id)
        recorder.log_params(plan_sha256=plan_hash, source_id=plan["source_id"], model_family="PitAuxiliaryAllocationNet",
                            maximum_model_fits=plan["maximum_model_fits"], maximum_native_cases=plan["maximum_native_cases"])
        qualified_aux = {fold: values.loc[values["$pit_available"].gt(.5)].copy() for fold, values in inputs["auxiliary"].items()}
        recorder.save_objects(artifact_path="inputs", daily_frame=inputs["frame"], features=inputs["features"],
            common=inputs["common"], timing=inputs["timing"], minute_features=inputs["minute"],
            coverage=inputs["coverage"], native_prices=inputs["prices"], auxiliary=qualified_aux,
            frozen_parent_models=inputs["frozen_models"], audit=inputs["audit"], plan=plan)
        recorder.log_metrics(registered_artifact_count=11, completed_model_fits=0)
        atomic_json({"recorder_id": recorder_id, "experiment_id": experiment_id, "plan_sha256": plan_hash,
                     "status": "training", "completed_model_fits": 0}, card / "run_state.json")
        for year in plan["development_years"]:
            for family in plan["candidate_families"]:
                for seed in plan["seeds"]:
                    key = f"{family}_{year}_{seed}"
                    model = MinutePitAllocator(base_features=inputs["base_features"], extra_features=plan["family_features"][family],
                        feature_masks=masks, spec=spec, seed=seed + year, cutoff=f"{year}-01-01T00:00:00-05:00",
                        train_end=inputs["audit"]["train_ends"][str(year)],
                        minimum_observations=plan["minimum_observed_minute_training_sessions"])
                    model.fit(datasets[str(year)], auxiliary_dataset=auxiliary_datasets[str(year)])
                    predicted = _prediction_dates(model.predict(datasets[str(year)]))
                    predictions.setdefault((family, seed), []).append(predicted)
                    fitted_models[key] = model
                    training_rows.append({"family": family, "fold": year, "seed": seed, **model.audit})
                    recorder.save_objects(artifact_path="models", **{key: model})
                    recorder.log_metrics(completed_model_fits=len(fitted_models))
                    print(f"fit {len(fitted_models)}/{plan['maximum_model_fits']} {key} n={model.audit['training_rows']}", flush=True)
        predictions = {key: pd.concat(parts).sort_index() for key, parts in predictions.items()}
        recorder.save_objects(artifact_path="analysis", development_predictions=predictions, training_audit=training_rows)
    information = feature_information_audit(inputs["features"], inputs["frame"], inputs["timing"],
        groups=plan["feature_groups"], years=plan["development_years"], cutoff=plan["development_return_cutoff"])
    market = {window: _market(inputs, window=window, plan=plan) for window in ("development", "reused_diagnostic")}
    books = {}

    def add_book(window, fee, family, prediction, *, benchmark=False):
        if len(books) >= plan["maximum_native_cases"]:
            raise ValueError("blocked: frozen native account budget exceeded")
        key = f"{window}_{fee}_{family}"
        books[key] = native_allocator_book(market[window], prediction, provider_uri=inputs["daily_provider"],
            cost_bps=fee, smoothing_span=plan["smoothing_span"], rebalance_threshold=plan["rebalance_threshold"], benchmark=benchmark)
        print(f"native {len(books)}/{plan['maximum_native_cases']} {key}", flush=True)

    current = _frozen_predictions(inputs, window="development", plan=plan)
    for fee in plan["costs_bps"]:
        for family in plan["candidate_families"]:
            add_book("development", fee, family, _ensemble(predictions, family, plan["seeds"]))
        add_book("development", fee, "frozen_current", current)
        add_book("development", fee, "soxx", None, benchmark=True)
    numerical_families = ["core_path", "peer_context", "core_peer"]
    selected, selection, uncertainty = development_selection(books, candidate_families=numerical_families,
                                                               costs=plan["costs_bps"], bootstrap=plan["bootstrap"])
    atomic_json({"selected_family": selected, "selection_uses_outer": False, "plan_sha256": plan_hash,
                 "selection_window": "matured development 2020-2022", "production_enabled": False}, card / "selected_model.json")
    seed_rows = []
    for fee in plan["costs_bps"]:
        for family in [selected, "base", "availability_control"]:
            for seed in plan["seeds"]:
                add_book("development", fee, family + "_seed_" + str(seed), predictions[(family, seed)])
        for seed in plan["seeds"]:
            candidate = books[f"development_{fee}_{selected}_seed_{seed}"]["authority"]
            for reference in ("base", "availability_control"):
                comparator = books[f"development_{fee}_{reference}_seed_{seed}"]["authority"]
                seed_rows.append({"seed": seed, "cost_bps": fee, "reference": reference,
                                  **paired_metrics(candidate, comparator)})

    outer_dataset = _primary_dataset(inputs, train_end=inputs["audit"]["train_ends"]["outer"],
                                     predict_bounds=plan["outer_window"])
    outer_predictions = {}
    qlib.init(provider_uri=str(inputs["daily_provider"]), region="us", kernels=1)
    with R.start(experiment_name=plan["experiment"], recorder_id=recorder_id, uri=str(tracking), resume=True):
        recorder = R.get_recorder()
        if recorder.id != recorder_id:
            raise ValueError("blocked: candidate recorder identity changed during resume")
        for family in [selected, "base", "availability_control"]:
            for seed in plan["seeds"]:
                if len(fitted_models) >= plan["maximum_model_fits"]:
                    raise ValueError("blocked: frozen model-fit budget exceeded")
                model = MinutePitAllocator(base_features=inputs["base_features"], extra_features=plan["family_features"][family],
                    feature_masks=masks, spec=spec, seed=seed, cutoff=plan["development_return_cutoff"],
                    train_end=inputs["audit"]["train_ends"]["outer"],
                    minimum_observations=plan["minimum_observed_minute_training_sessions"])
                model.fit(outer_dataset, auxiliary_dataset=auxiliary_datasets["outer"])
                outer_predictions[(family, seed)] = _prediction_dates(model.predict(outer_dataset))
                key = f"{family}_outer_{seed}"
                fitted_models[key] = model
                training_rows.append({"family": family, "fold": "outer", "seed": seed, **model.audit})
                recorder.save_objects(artifact_path="models", **{key: model})
                recorder.log_metrics(completed_model_fits=len(fitted_models))
                print(f"fit {len(fitted_models)}/{plan['maximum_model_fits']} {key}", flush=True)
        recorder.save_objects(artifact_path="analysis", outer_predictions=outer_predictions,
                              development_selection=selection, uncertainty=uncertainty, feature_information=information)
    current_outer = _frozen_predictions(inputs, window="reused_diagnostic", plan=plan)
    for fee in plan["costs_bps"]:
        for family in [selected, "base", "availability_control"]:
            add_book("reused_diagnostic", fee, family, _ensemble(outer_predictions, family, plan["seeds"]))
        add_book("reused_diagnostic", fee, "frozen_current", current_outer)
        add_book("reused_diagnostic", fee, "soxx", None, benchmark=True)
    references = ["base", "availability_control", "frozen_current", "soxx"]
    development, dev_annual = compare_native_books(books, window="development", costs=plan["costs_bps"],
        families=[*plan["candidate_families"], "frozen_current", "soxx"], references=references)
    diagnostic, outer_annual = compare_native_books(books, window="reused_diagnostic", costs=plan["costs_bps"],
        families=[selected, "base", "availability_control", "frozen_current", "soxx"], references=references)
    comparison = pd.concat([development, diagnostic], ignore_index=True)
    annual = pd.concat([dev_annual, outer_annual], ignore_index=True)
    seed_table = pd.DataFrame(seed_rows)
    chosen = selection.loc[selection.family.eq(selected)].iloc[0]
    seed_pass = bool(seed_table.groupby(["cost_bps", "reference"]).relative_wealth.apply(lambda x: (x > 1.).sum()).ge(2).all())
    year_counts = annual.loc[annual.window.eq("development") & annual.family.eq(selected) & annual.reference.isin(["base", "availability_control"])]
    year_pass = bool(year_counts.groupby(["cost_bps", "reference"]).relative_wealth.apply(lambda x: (x > 1.).sum()).ge(2).all())
    candidate_rows = comparison.loc[comparison.family.eq(selected)]
    benchmark_pass = bool(candidate_rows.wealth_ratio_vs_soxx.gt(1.).all()
        and candidate_rows.mdd_no_worse_than_soxx.all() and candidate_rows.cvar_no_worse_than_soxx.all())
    incremental_pass = bool(chosen.positive_both_costs_and_references and chosen.risk_no_worse_than_matched_controls
                            and chosen.family_adjusted_bounds_positive and seed_pass and year_pass)
    latest = _ensemble(outer_predictions, selected, plan["seeds"]).tail(1)
    policy = {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
        "recorder_only_artifacts": ["inputs/", "models/", "analysis/development_predictions", "analysis/outer_predictions",
            "analysis/native/", "analysis/uncertainty", "analysis/feature_information"],
        "recorder_storage": "NAS content-addressed archive", "nas_binding": "nas_binding.json",
        "large_data_git_delivery": False}
    summary = {"experiment": plan["experiment"], "plan_sha256": plan_hash,
        "decision": "not_promoted_pending_independent_forward" if incremental_pass and benchmark_pass else "not_promoted_minute_mlp_gates_failed",
        "selected_family": selected, "selected_for_diagnostic_only": True, "production_enabled": False,
        "objective_achieved": False, "forward_qualified_sessions": 0,
        "model_class": "MinutePitAllocator wrapping the current PitAuxiliaryAllocationNet",
        "model_fits": len(fitted_models), "native_cases": len(books), "seeds": plan["seeds"],
        "as_of": plan["as_of"], "data_authority": "yanxuan_nas", "source_id": plan["source_id"],
        "source_binding": "nas_binding.json", "provider_uri": str(inputs["daily_provider"]),
        "region": "us", "market": "all", "recorder_id": recorder_id, "experiment_id": experiment_id,
        "recorder_uri": str(tracking), "recorder_local_dir": str(tracking / experiment_id / recorder_id),
        "artifact_policy": policy, "input_audit": inputs["audit"],
        "incremental_development_gate": incremental_pass, "paired_seed_gate": seed_pass,
        "positive_development_year_gate": year_pass, "benchmark_and_risk_gate": benchmark_pass,
        "full_source_live_receipt_qualified": False, "outer_uses_reused_history": True,
        "latest_inference": {"signal_date": str(latest.index[-1].date()), "weights": latest.iloc[-1].to_dict(),
                             "label_known": False, "broker_target_emitted": False},
        "current_release_changed": False, "active_stage1_controller_parity_claimed": False,
        "frozen_current_reference": "original model state on new frozen inputs; independently settled daily policy, not the active action controller",
        "known_baseline_failures_not_retested": "four legacy production checks from the prior task: execution-mark expectations and absent legacy Recorder",
        "qlib_docs_reference": ["https://qlib.readthedocs.io/en/latest/component/data.html",
            "https://qlib.readthedocs.io/en/latest/component/model.html", "https://qlib.readthedocs.io/en/latest/component/recorder.html",
            "https://qlib.readthedocs.io/en/latest/component/strategy.html"]}
    if _hash(root / "production/stage1_futu_sim_v1/runtime_release.json") != original_release_hash:
        raise ValueError("blocked: active release changed during the campaign")
    qlib.init(provider_uri=str(inputs["daily_provider"]), region="us", kernels=1)
    with R.start(experiment_name=plan["experiment"], recorder_id=recorder_id, uri=str(tracking), resume=True):
        recorder = R.get_recorder()
        for key, book in books.items():
            recorder.save_objects(artifact_path="analysis/native", **{key: book})
        recorder.save_objects(artifact_path="analysis", comparison=comparison, annual=annual,
                              seed_comparison=seed_table, training_audit=training_rows, summary=summary)
        recorder.log_metrics(completed_model_fits=len(fitted_models), completed_native_cases=len(books),
                             incremental_development_gate=int(incremental_pass), benchmark_gate=int(benchmark_pass),
                             registered_artifact_count=11 + len(fitted_models) + len(books) + 10)
    for name, table in {"backtest_benchmark_comparison": comparison, "regime_metrics": annual,
        "model_selection": selection, "uncertainty_summary": uncertainty, "feature_information": information,
        "seed_comparison": seed_table, "training_summary": pd.DataFrame(training_rows),
        "minute_coverage_summary": inputs["coverage"]}.items():
        if name == "training_summary":
            table = table.drop(columns=["extra_scaler", "observed_minute_training_rows"], errors="ignore")
        table.to_csv(card / (name + ".csv"), index=False)
    summary["compact_metrics"] = candidate_rows.loc[:, ["window", "cost_bps", "return_pct", "cagr_pct",
        "max_drawdown_pct", "cvar5_pct", "wealth_ratio_vs_base", "wealth_ratio_vs_availability_control",
        "wealth_ratio_vs_frozen_current", "wealth_ratio_vs_soxx", "turnover", "trade_days"]].to_dict("records")
    atomic_json(summary, card / "summary.json")
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": plan["experiment"], "as_of": plan["as_of"],
        "plan": "plan.json", "source_binding": "nas_binding.json", "provider_uri": summary["provider_uri"],
        "recorder_uri": str(tracking), "model_class": summary["model_class"], "artifact_policy": policy}, allow_unicode=True, sort_keys=False))
    conclusion = f"""# 中文结论

- 执行：NAS raw NONE 的全天分钟信息直接进入现役同结构的 PIT 辅助神经分配器；原有 {len(inputs['base_features'])} 项输入保留。固定五组输入、三个 seed、{len(fitted_models)} 次拟合、{len(books)} 个原生 Qlib 账户。
- 数据：分钟源 {inputs['audit']['source_observations']:,} 条观察；{len(inputs['frame'])} 个日频信号保留 2011 年以来完整历史。有效核心/上下文交易日分别 {inputs['audit']['valid_core_sessions']}/{inputs['audit']['valid_peer_sessions']}；缺分钟通道为 NaN 和可用标记，仅在模型内部中性编码。源异常压力日没有从账户日历删去。
- 特征：核心 12 项、上下文 8 项，使用同一交易时段的不复权价格比率，无 VWAP 特征和未来复权系数。分组 IC 仅为诊断，未按单项历史相关性挑选特征。
- 训练：固定末轮 checkpoint、300 epochs；20 个原生交易日 purge，主标签真实 H5 成熟时点严格早于 cutoff；辅助标签从每个 cutoff 之前的 Qlib 原始观察重建。共同原网络初始化，新输入初始权重为零。
- 窗口：2020–2022 成熟开发期选择 {selected}；2023–2026 为复用诊断，账户最终退出日 {market['reused_diagnostic'].exit_date.iloc[-1].date()}。最新 {latest.index[-1].date()} 信号保留未成熟标签，可生成预测，未向 broker 发出目标。
- 费用：10/25bp 各自拥有现金、持仓、EMA/HOLD 决策与实际费用；每个路径经原生 Qlib Position/Account 读回核对。冻结父模型只是同输入的神经网络参考，不代表现役 Stage1 动作层路径。
- 结果：增量开发门={incremental_pass}，配对 seed 门={seed_pass}，开发年份门={year_pass}，SOXX 收益与风险门={benchmark_pass}。各费用收益、回撤、CVaR、年度差异见 backtest_benchmark_comparison.csv、regime_metrics.csv；家庭校正区块区间见 uncertainty_summary.csv。
- 晋升：{summary['decision']}；独立合格前向=0，objective_achieved=false，现役模型/动作层未切换。历史提升与稳定前向超额是不同结论。
- 风险：历史首次收到行情和完整时点资格未认证；每日开盘历史执行不声称真实分钟成交；原有日频来源、PIT 辅助样本与现金零利息口径继承原模型，不以本研究补发来源资格。
- 下一步：结束本次固定特征路线，保留全部失败与对照；只有具备新信息或独立前向证据的新授权研究才继续，不追加参数搜索。
- 原生 Recorder：{recorder_id}；行情与大研究对象留在 NAS，Git 只交代码、notebook、瘦卡与引用。
"""
    (card / "conclusion.md").write_text(conclusion)
    local = tracking / experiment_id / recorder_id
    for name in ("code_cached.txt", "code_diff.txt", "code_status.txt"):
        (local / "artifacts" / name).unlink(missing_ok=True)
    entries = [{"path": str(path.relative_to(card)), "sha256": _hash(path), "bytes": path.stat().st_size}
               for path in sorted((local / "artifacts").rglob("*")) if path.is_file()]
    atomic_json({"recorder_id": recorder_id, "entries": entries}, card / "verification_manifest.json")
    evidence = archive_nas_recorder(card / "nas_binding.json", workspace_root=root, card=card,
                                    recorder_id=recorder_id, experiment_id=experiment_id)
    summary["nas_evidence"] = evidence
    policy["recorder_archive_sha256"] = evidence["sha256"]
    atomic_json(summary, card / "summary.json")
    atomic_json({"recorder_id": recorder_id, "experiment_id": experiment_id, "plan_sha256": plan_hash,
                "status": "completed", "completed_model_fits": len(fitted_models), "completed_native_cases": len(books)}, card / "run_state.json")
    return summary
