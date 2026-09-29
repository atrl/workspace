"""Two fixed PIT allocator refits with different history windows; no selection."""
from __future__ import annotations

import hashlib
import json
import pickle
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.position_accounting import settle_target_session
from src.workflow.direct_continuous_soxx_allocation import DirectContinuousAllocationSpec, _load_frame, _prepare
from src.workflow.intraday_recovery_direct_allocation import _frozen_outer_predictions, _risk_metrics, project_legal_weights
from src.workflow.pit_allocation_retrain import (
    PRIMARY_LABELS, _mature_training_labels, _training_auxiliary, validate_frozen_bundle,
)
from src.workflow.pit_constituent_auxiliary_training_allocation import (
    CANDIDATES, COMMON_FEATURES, PitAuxiliaryAllocationSpec,
    _build_auxiliary_dataset, _build_primary_dataset, _fit_predict, _prepare_auxiliary,
    _soxx_common_training_frame,
)
from src.workflow.soxl_objective_feasibility_audit import _json_ready
from src.workflow.stage1_economic_action import fee_affordable_target
from src.workflow.trainable_regime_policy_nested_wf import _remove_qlib_code_snapshot_artifacts


EXPERIMENT = "pit_training_window_20260929"
BASE_CARD = "pit_constituent_auxiliary_training_allocation_retrained_20260928"
WINDOWS = {"expanding": "2011-01-03", "rolling8": "2018-01-01"}
CUTOFF = "2025-12-31"
AS_OF = "2026-09-25"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_completed_card(card: Path, plan_hash: str) -> dict | None:
    """Reopening the dated notebook must not refit against a changing live store."""
    if not (card / "summary.json").exists():
        return None
    summary = json.loads((card / "summary.json").read_text())
    if summary["source_audit"]["plan_sha256"] != plan_hash:
        raise ValueError("blocked: completed experiment plan changed; use a new card identity")
    manifest = json.loads((card / "frozen_bundle_manifest.json").read_text())
    if manifest["recorder_id"] != summary["recorder_id"]:
        raise ValueError("blocked: completed card and retained recorder identities differ")
    for entry in manifest["artifacts"]:
        path = card / entry["path"]
        if not path.exists():
            path = Path(manifest["canonical_card_dir"]) / entry["path"]
        if not path.is_file() or _sha(path) != entry["sha256"]:
            raise ValueError(f"blocked: retained training-window artifact changed: {entry['path']}")
    return summary


def filter_training_window(frame: pd.DataFrame, *, start: str, cutoff: str, date_column: str | None = None) -> pd.DataFrame:
    """Filter fit rows, while permitting earlier data only for feature warmup."""
    dates = pd.to_datetime(frame[date_column] if date_column else frame.index)
    selected = frame.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(cutoff))].copy()
    if selected.empty:
        raise ValueError("blocked: empty fixed training window")
    return selected


def _effective_auxiliary(frame: pd.DataFrame) -> pd.DataFrame:
    required = [*COMMON_FEATURES, "synthetic_2x_crash_label", "synthetic_2x_upside_label", "training_weight"]
    return frame.dropna(subset=required).loc[lambda table: table["$pit_available"].gt(.5)].copy()


def evaluate_base_allocator(market: pd.DataFrame, prediction: pd.DataFrame, *, cost_bps: float) -> tuple[dict, pd.DataFrame]:
    """Apply identical static constraints and net-cash fees to each base model."""
    previous = np.array([0., 0., 1.])
    equity = 1.0
    rows = []
    for date, weights in prediction.sort_index().iterrows():
        row = market.loc[date]
        desired = project_legal_weights(
            float(weights.soxl_weight), float(weights.soxx_weight), 1.45,
            maximum_soxl_weight=.30, minimum_cash_weight=.15,
        )
        raw = weights[["soxl_weight", "soxx_weight", "cash_weight"]].to_numpy(dtype=float)
        execution = fee_affordable_target(desired, pretrade_weights=previous, cost_bps=cost_bps)
        settled = settle_target_session(
            previous, open_target_weights=execution,
            asset_returns=[row.soxl_forward_open_return, row.soxx_forward_open_return, 0.0],
            cost_bps=cost_bps,
        )
        after_fee = settled["open_net_weights"]
        if after_fee[0] > .30 + 1e-12 or after_fee[2] < .15 - 1e-12 or 3 * after_fee[0] + after_fee[1] > 1.45 + 1e-12:
            raise ValueError("blocked: fee-aware base allocation violates static limits")
        equity *= 1.0 + settled["net_return"]
        previous = settled["ending_weights"]
        rows.append({
            "signal_date": date, "trade_date": row.trade_date, "exit_date": row.exit_date,
            "raw_soxl_weight": raw[0], "raw_soxx_weight": raw[1], "raw_cash_weight": raw[2],
            "soxl_weight": desired[0], "soxx_weight": desired[1], "cash_weight": desired[2],
            "static_projection_changed": not np.allclose(raw, desired, rtol=0, atol=1e-10),
            "strategy_return": settled["net_return"], "turnover": settled["turnover"],
            "fees": settled["fees"], "equity": equity,
            "open_trade": settled["open_turnover"] > 1e-8,
            "ending_soxl_weight": previous[0], "ending_soxx_weight": previous[1], "ending_cash_weight": previous[2],
            "cost_bps": cost_bps,
        })
    replay = pd.DataFrame(rows)
    metrics = _risk_metrics(replay.strategy_return.to_numpy())
    path = np.r_[1.0, replay.equity.to_numpy()]
    metrics["max_drawdown_pct"] = float(100 * np.min(path / np.maximum.accumulate(path) - 1))
    metrics.update(total_turnover=float(replay.turnover.sum()), open_trade_days=int(replay.open_trade.sum()),
                   average_cash_weight_pct=float(100 * replay.cash_weight.mean()),
                   static_projection_rows=int(replay.static_projection_changed.sum()))
    return metrics, replay


def run_pit_training_window(root: Path | str) -> dict:
    from qlib.workflow import R
    from qlib.contrib.evaluate import risk_analysis

    root = Path(root).resolve()
    card = root / "research" / EXPERIMENT
    plan_path = card / "selection_plan.json"
    plan = json.loads(plan_path.read_text())
    if plan["candidate"] != "pit_only_aux_005" or plan["training_information_cutoff"] != CUTOFF:
        raise ValueError("blocked: fixed training-window plan changed")
    if plan["variants"] != {name: [start, CUTOFF] for name, start in WINDOWS.items()}:
        raise ValueError("blocked: training windows differ from the registered variants")
    plan_hash = _sha(plan_path)
    completed = _read_completed_card(card, plan_hash)
    if completed is not None:
        return completed
    candidate = next(value for value in CANDIDATES if value.name == "pit_only_aux_005")
    base_spec = PitAuxiliaryAllocationSpec(
        root=root, experiment=EXPERIMENT, development_end=CUTOFF,
        outer=("2026-01-02", AS_OF),
    )
    source_files = [root / "data/qlib" / name / "qlib_store_manifest.json" for name in (
        "soxl", "soxl_soxx_soxs", "dynamic_soxx_soxl_cash_allocation_wf_v1", "soxx_pit_leveraged_training_v1")]
    source_files += [root / "src/workflow/pit_constituent_auxiliary_training_allocation.py",
                     root / "src/workflow/stage1_economic_action.py"]
    source_hashes = {str(path.relative_to(root)): _sha(path) for path in source_files}
    print("Loading frozen Qlib inputs for the two fixed outer refits", flush=True)
    market, market_audit, features = _load_frame(
        DirectContinuousAllocationSpec(root=root, train_start=base_spec.train_start,
                                       development_end=CUTOFF, outer=base_spec.outer),
        include_incomplete_latest=True,
    )
    market = market.loc[market.index <= pd.Timestamp(AS_OF)].copy()
    matured, maturity_audit = _mature_training_labels(market, CUTOFF)
    primary_dataset, primary_audit = _build_primary_dataset(matured, features, base_spec)
    common = _soxx_common_training_frame(base_spec, end=AS_OF).reindex(market.index)
    auxiliary, auxiliary_audit = _training_auxiliary(base_spec)
    auxiliary_dataset, auxiliary_dataset_audit = _build_auxiliary_dataset(auxiliary, base_spec)
    complete_auxiliary = _prepare_auxiliary(auxiliary_dataset, "all")
    complete_primary = _prepare(primary_dataset, (base_spec.train_start, CUTOFF))
    test = _prepare(primary_dataset, "outer")
    old_card = root / "research" / BASE_CARD
    old_manifest = validate_frozen_bundle(old_card)
    old_entry = next(entry for entry in old_manifest["artifacts"] if entry["path"].endswith("models/pit_only_aux_005.pkl"))
    old_model_path = old_card / old_entry["path"]
    with old_model_path.open("rb") as handle:
        old_state = pickle.load(handle)["outer_refit"]
    if list(old_state["primary_features"]) != list(features):
        raise ValueError("blocked: primary feature contract differs from the fixed reference")
    models, inputs, predictions, fit_rows = {}, {}, {}, []
    old_spec = replace(base_spec, development_end="2022-12-30")
    predictions["old_frozen_2022"] = _frozen_outer_predictions(
        market=market, common=common, state=old_state, spec=old_spec,
    )
    models["old_frozen_2022"] = old_state
    for name, start in WINDOWS.items():
        spec = replace(base_spec, train_start=start)
        train = filter_training_window(complete_primary, start=start, cutoff=CUTOFF)
        aux_train = filter_training_window(complete_auxiliary, start=start, cutoff=CUTOFF, date_column="datetime")
        effective_train = train.dropna(subset=PRIMARY_LABELS)
        effective_aux = _effective_auxiliary(aux_train)
        if effective_train.index.max() > pd.Timestamp(CUTOFF) or effective_aux.datetime.max() > pd.Timestamp(CUTOFF):
            raise ValueError("blocked: a post-cutoff row entered fitting")
        print(f"Fitting {name}: primary={len(effective_train)}, auxiliary={len(effective_aux)}, seed={spec.seed}", flush=True)
        initial_predictions, state, diagnostics = _fit_predict(
            train=train, test=test, soxx_common=common, auxiliary=aux_train,
            primary_features=features, candidate=candidate, spec=spec, seed=spec.seed,
        )
        # The fitting helper drops test rows with incomplete five-day labels.
        # Re-infer from saved weights using only known features/settled returns.
        full_predictions = _frozen_outer_predictions(market=market, common=common, state=state, spec=spec)
        models[name] = state
        predictions[name] = full_predictions
        inputs[name] = {
            "actual_primary_fit_rows": effective_train,
            "actual_common_fit_rows": common.reindex(effective_train.index),
            "actual_auxiliary_fit_rows": effective_aux,
            "training_spec": asdict(spec) | {"root": str(root)},
        }
        realized_dates = pd.to_datetime(market.exit_date).shift(-4).reindex(effective_train.index)
        if realized_dates.isna().any() or realized_dates.max() > pd.Timestamp(CUTOFF):
            raise ValueError("blocked: primary label has not matured by the training cutoff")
        fit_rows.append({
            "variant": name, "train_start": start, "training_information_cutoff": CUTOFF,
            "first_actual_fit_signal": effective_train.index.min(), "last_actual_fit_signal": effective_train.index.max(),
            "latest_primary_label_realization": realized_dates.max(),
            "first_auxiliary_fit_date": effective_aux.datetime.min(), "last_auxiliary_fit_date": effective_aux.datetime.max(),
            "initial_test_prediction_rows": len(initial_predictions), "full_settled_prediction_rows": len(full_predictions),
            "test_tail_rows_restored_by_frozen_inference": len(full_predictions) - len(initial_predictions),
            "seed": spec.seed, "epochs": spec.epochs, **diagnostics,
        })
    common_dates = predictions["old_frozen_2022"].index
    if any(not frame.index.equals(common_dates) for frame in predictions.values()):
        raise ValueError("blocked: the three models do not share exactly the same diagnostic dates")
    matched = market.loc[common_dates]
    comparisons, replays, risk_tables = [], [], []
    for variant, prediction in predictions.items():
        for cost in (10.0, 25.0):
            metrics, replay = evaluate_base_allocator(matched, prediction, cost_bps=cost)
            comparisons.append({"variant": variant, "cost_bps": cost, **metrics})
            replays.append(replay.assign(variant=variant))
            risk = risk_analysis(pd.Series(replay.strategy_return.to_numpy(), index=pd.to_datetime(replay.trade_date)), freq="day", mode="product").reset_index()
            risk_tables.append(risk.assign(variant=variant, cost_bps=cost))
    comparison = pd.DataFrame(comparisons)
    fits = pd.DataFrame(fit_rows)
    for cost in (10., 25.):
        baseline_return = comparison.loc[(comparison.variant == "old_frozen_2022") & (comparison.cost_bps == cost), "return_pct"].iloc[0]
        comparison.loc[comparison.cost_bps == cost, "return_delta_vs_old_pct_points"] = comparison.loc[comparison.cost_bps == cost, "return_pct"] - baseline_return
    if _sha(plan_path) != plan_hash or any(_sha(root / name) != value for name, value in source_hashes.items()):
        raise ValueError("blocked: plan, sources, or evaluation implementation changed during fitting")
    provenance = {
        "plan_sha256": plan_hash, "source_sha256": source_hashes, "candidate": asdict(candidate),
        "primary_dataset": primary_audit, "auxiliary_dataset": auxiliary_dataset_audit,
        "primary_label_maturity": maturity_audit, "auxiliary_audit": auxiliary_audit, "market_audit": market_audit,
        "old_reference": {"card": BASE_CARD, "recorder_id": old_manifest["recorder_id"], "model_sha256": old_entry["sha256"]},
    }
    canonical_card = Path("/Users/rziiiii/dev/workspace/research") / EXPERIMENT
    R.set_uri(str(canonical_card / "qlib_recorder"))
    with tempfile.TemporaryDirectory(prefix="pit_training_window_") as temporary, R.start(experiment_name=EXPERIMENT):
        recorder = R.get_recorder()
        R.log_params(candidate=candidate.name, model_count=2, training_cutoff=CUTOFF,
                     seed=base_spec.seed, selection_uses_2026="false", policy_layers="base_allocator_only")
        R.log_metrics(new_model_count=2.0, reference_model_count=1.0,
                      diagnostic_interval_count=float(len(common_dates)),
                      expanding_primary_fit_rows=float(fits.loc[fits.variant.eq("expanding"), "primary_train_rows"].iloc[0]),
                      rolling8_primary_fit_rows=float(fits.loc[fits.variant.eq("rolling8"), "primary_train_rows"].iloc[0]))
        R.save_objects(artifact_path="inputs", **{
            "market_frame.pkl": market, "primary_matured_frame.pkl": matured,
            "common_frame.pkl": common, "training_inputs_by_variant.pkl": inputs,
            "candidate.pkl": asdict(candidate), "provenance.pkl": provenance,
        })
        R.save_objects(artifact_path="analysis", **{
            "predictions_by_variant.pkl": predictions, "portfolio_replays.pkl": pd.concat(replays, ignore_index=True),
            "comparison.pkl": comparison, "fit_diagnostics.pkl": fits,
            "qlib_risk_analysis.pkl": pd.concat(risk_tables, ignore_index=True),
        })
        model_artifacts = {}
        for name, state in models.items():
            path = Path(temporary) / f"{name}.pkl"
            with path.open("wb") as handle:
                pickle.dump({"outer_refit": state}, handle)
            recorder.log_artifact(str(path), artifact_path="models")
            model_artifacts[name] = {"artifact": f"artifacts/models/{name}.pkl", "sha256": _sha(path), "promotion_status": "not_promoted"}
        recorder.log_artifact(str(plan_path), artifact_path="inputs")
        pd.testing.assert_frame_equal(comparison, recorder.load_object("analysis/comparison.pkl"))
        recorder_dir = Path(recorder.get_local_dir())
        recorder_info = {"recorder_id": recorder.id, "recorder_local_dir": str(recorder_dir),
                         "recorder_relative_dir": str(recorder_dir.relative_to(canonical_card))}
    _remove_qlib_code_snapshot_artifacts(recorder_dir)
    summary = {
        "experiment": EXPERIMENT, "decision": "diagnostic_only_no_winner_selected", "promotion_status": "not_promoted",
        "system_of_record": "qlib", "provider_uri": "data/qlib/soxl", "region": "us", "market": "all", "as_of": AS_OF,
        "candidate": candidate.name, "architecture_or_objective_changed": False, "active_configuration_changed": False,
        "trained_models": list(WINDOWS), "training_information_cutoff": CUTOFF, "fit_diagnostics": fits.to_dict("records"),
        "actual_common_window": {"signal_start": common_dates.min(), "signal_end": common_dates.max(),
                                 "trade_start": matched.trade_date.min(), "trade_end": matched.trade_date.max(),
                                 "exit_start": matched.exit_date.min(), "exit_end": matched.exit_date.max(),
                                 "interval_count": len(common_dates), "terminal_basis": "final observed exit OPEN, not the 9/25 close"},
        "comparisons": comparison.to_dict("records"), "source_audit": provenance,
        "runtime_models": model_artifacts, "selection_uses_2026": False, "research_only": True,
        "evaluation_contract": plan["evaluation_contract"],
        "limitations": ["2026 was already repeatedly viewed and is diagnostic, not clean OOS.", "Only base allocator outputs are evaluated; results do not represent Stage1/HOLD or actual fills.", "All three smoothers start on the same 2026 diagnostic window, so this is a controlled window comparison, not a claim of exact production-path parity."],
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml", "recorder_only_artifacts": ["inputs/", "models/", "analysis/predictions_by_variant.pkl", "analysis/portfolio_replays.pkl", "analysis/comparison.pkl", "analysis/fit_diagnostics.pkl", "analysis/qlib_risk_analysis.pkl"]},
        "qlib_api": ["D.features", "DatasetH", "DataHandlerLP", "R.start", "R.save_objects", "Recorder.load_object"],
        "qlib_docs_reference": ["https://qlib.readthedocs.io/en/latest/component/data.html", "https://qlib.readthedocs.io/en/latest/component/recorder.html"],
        "sidecars": ["selection_plan.json", "summary.json", "selected_model.json", "conclusion.md", "workflow.yaml", "backtest_benchmark_comparison.csv", "training_summary.csv", "risk_analysis.csv", "frozen_bundle_manifest.json"],
        **recorder_info,
    }
    comparison.to_csv(card / "backtest_benchmark_comparison.csv", index=False)
    fits.to_csv(card / "training_summary.csv", index=False)
    pd.concat(risk_tables, ignore_index=True).to_csv(card / "risk_analysis.csv", index=False)
    (card / "summary.json").write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2) + "\n")
    (card / "selected_model.json").write_text(json.dumps({"selected": None, "selection_performed": False, "models": model_artifacts}, indent=2) + "\n")
    (card / "workflow.yaml").write_text(f"experiment: {EXPERIMENT}\nselection_plan: selection_plan.json\nprovider_uri: data/qlib/soxl\nregion: us\nmarket: all\nas_of: {AS_OF}\ntraining_cutoff: {CUTOFF}\nseed: 71\nmodel_count: 2\nselection_uses_2026: false\npromotion_status: not_promoted\nartifact_policy: slim_card_recorder_first\nrecorder_relative_dir: {recorder_info['recorder_relative_dir']}\n")
    lines = ["# 中文结论", "", "固定pit_only_aux_005、架构、损失、seed=71和300 epochs，仅重训扩展2011起与滚动2018起两个outer模型，标签信息截至2025-12-31。旧2022冻结基模仅作同窗对照，不改active。",
             f"实际共同信号区间{common_dates.min().date()}–{common_dates.max().date()}，执行区间{matched.trade_date.min().date()}–{matched.trade_date.max().date()}，最后结算为{matched.exit_date.max().date()}开盘；没有声称完整到9/25收盘。",
             "三模型统一原平滑/执行带、静态SOXL≤30%和现金≥15%投影、费用可负担目标、逐笔现金净费结算；没有Stage1回撤控制器或HOLD。", ""]
    for row in comparisons:
        lines.append(f"- {row['variant']} / {row['cost_bps']:.0f}bp：收益{row['return_pct']:+.2f}%，MaxDD {row['max_drawdown_pct']:.2f}%，CVaR5 {row['cvar5_pct']:.2f}%，平均现金{row['average_cash_weight_pct']:.2f}%，交易日{row['open_trade_days']}。")
    lines += ["", "2026已经被查看，只展示全部结果，不选赢家、不与新HOLD自动组合，也不构成晋升证据。完整输入、两个新模型和旧参考模型均入QLib recorder，后续须做新的前向验证。", f"QLib recorder：{recorder_info['recorder_id']}。"]
    (card / "conclusion.md").write_text("\n".join(lines) + "\n")
    files = [path for path in recorder_dir.rglob("*") if path.is_file()] + [recorder_dir.parent / "meta.yaml"]
    manifest = {"version": 1, "experiment": EXPERIMENT, "recorder_id": recorder_info["recorder_id"],
                "canonical_card_dir": str(canonical_card), "promotion_status": "not_promoted",
                "artifacts": [{"path": str(path.relative_to(canonical_card)), "sha256": _sha(path), "size": path.stat().st_size}
                              for path in sorted(set(files)) if path.is_file()]}
    (card / "frozen_bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return _json_ready(summary)
