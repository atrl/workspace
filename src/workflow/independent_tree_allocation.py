"""Frozen XGB/LGB unrestricted versus constrained native-account comparison."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import xgboost  # Import before Qlib/PyTorch to avoid the macOS OpenMP crash.
import numpy as np
import pandas as pd

from src.analysis.integrated_allocation import annual_paired_metrics, paired_metrics, stationary_bootstrap_excess
from src.backtest.full_path_tiered import replay_full_path
from src.backtest.stage1_ibkr import native_tiered_stage1
from src.backtest.qlib_continuous_position import QlibContinuousPosition
from src.data.minute_intake import atomic_json
from src.data.nas_market_data import archive_nas_recorder
from src.model.independent_tree_allocation import IndependentTreeAllocation
from src.workflow.minute_action_value import _source


PRICE_COLUMNS = ("soxl_open", "soxx_open", "raw_soxl_open", "raw_soxx_open",
                 "soxl_exit_open", "soxx_exit_open", "raw_soxl_exit_open", "raw_soxx_exit_open",
                 "trade_date", "exit_date")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset(frame: pd.DataFrame, columns: list[str], cutoff: str):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    joined = pd.concat({"feature": frame[columns], "price": frame[list(PRICE_COLUMNS)],
        "meta": frame[["label_available_at", "decision_at", "feature_available_at"]]}, axis=1)
    joined.index = pd.MultiIndex.from_arrays([frame.index, ["SOXL_SOXX_CASH"] * len(frame)],
                                              names=["datetime", "instrument"])
    return DatasetH(handler=DataHandlerLP.from_df(joined),
        segments={"train": (None, pd.Timestamp(cutoff) - pd.Timedelta(nanoseconds=1)),
                  "test": (pd.Timestamp(cutoff), None)})


def run(*, root: Path, card: Path, plan: dict) -> dict:
    import qlib
    import yaml
    from qlib.workflow import R

    root, card = Path(root).resolve(), Path(card).resolve()
    digest = _sha(card / "plan.json")
    if (card / "summary.json").exists():
        previous = json.loads((card / "summary.json").read_text())
        if previous["plan_sha256"] != digest:
            raise ValueError("blocked: frozen plan changed")
        return previous
    source_card = root / plan["source"]["card"]
    if _sha(source_card / "plan.json") != plan["source"]["plan_sha256"]:
        raise ValueError("blocked: full-path source plan changed")
    provider = root / plan["provider_uri"]
    if _sha(provider / "qlib_store_manifest.json") != plan["provider_manifest_sha256"]:
        raise ValueError("blocked: Qlib provider manifest changed")
    qlib.init(provider_uri=str(provider), region="us", kernels=1)
    objects, audit = _source(root, plan["source"], ["inputs/frame", "analysis/native/reused_outer_soxx_ensemble_0"])
    frame = objects["inputs/frame"]
    if not frame.signal_date.between(plan["source_start"], plan["source_end"]).all():
        raise ValueError("blocked: source frame scope changed")
    if set(plan["features"]) - set(frame.columns) or set(PRICE_COLUMNS) - set(frame.columns):
        raise ValueError("needs_data: frozen features or account marks absent")
    dataset = _dataset(frame, plan["features"], plan["cutoff"])
    tracking = card / "qlib_recorder"
    progress_path = card / "run_state.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        "plan_sha256": digest, "fit_attempts": [], "account_attempts": []}
    if progress["plan_sha256"] != digest:
        raise ValueError("blocked: previous attempt has a different plan")
    kwargs = {"experiment_name": card.name, "uri": str(tracking)}
    if progress.get("recorder_id"):
        kwargs.update(recorder_id=progress["recorder_id"], resume=True)
    with R.start(**kwargs):
        recorder = R.get_recorder()
        progress.update(recorder_id=recorder.id, experiment_id=str(recorder.experiment_id))
        atomic_json(progress, progress_path)
        recorder.save_objects(artifact_path="inputs", plan=plan, source_audit=audit)
        recorder.save_objects(artifact_path="sources", model=(root / "src/model/independent_tree_allocation.py").read_text(),
            workflow=Path(__file__).read_text())
        models = {}
        for family in plan["families"]:
            if family in progress.get("fits_done", []):
                model = recorder.load_object("models/" + family)
            else:
                if family in progress["fit_attempts"] and family not in progress.get("retryable_runtime_fits", []):
                    raise ValueError("blocked: prior model attempt has no saved result")
                if family not in progress["fit_attempts"]:
                    if len(progress["fit_attempts"]) >= plan["budget"]["model_fits"]:
                        raise ValueError("blocked: model fit budget consumed")
                    progress["fit_attempts"].append(family)
                    atomic_json(progress, progress_path)
                model = IndependentTreeAllocation(family=family, columns=plan["features"],
                    cutoff=plan["cutoff"], tariff=plan["tariff"], initial_cash=plan["initial_cash"],
                    seed=plan["seed"]).fit(dataset)
                recorder.save_objects(artifact_path="models", **{family: model})
                progress.setdefault("fits_done", []).append(family)
                atomic_json(progress, progress_path)
            models[family] = model
            recorder.log_metrics(completed_model_fits=len(models))
        test = frame.loc[frame.signal_date.between(plan["test_start"], plan["test_end"])].copy()
        if len(test) != plan["expected_test_rows"]:
            raise ValueError("blocked: test calendar differs")
        benchmark = objects["analysis/native/reused_outer_soxx_ensemble_0"]
        if not test.trade_date.reset_index(drop=True).equals(benchmark["authority"].trade_date.reset_index(drop=True)):
            raise ValueError("blocked: SOXX account calendar differs")
    cases = {}
    # Predictions are fitted once per family. Both policy arms share those bytes.
    for family, model in models.items():
        model.operational_fee_reserve = plan["operational_fee_reserve"]
        for arm in ("unrestricted", "constrained"):
            key = f"{family}_{arm}_0"
            if key not in progress["account_attempts"]:
                if len(progress["account_attempts"]) >= plan["budget"]["native_accounts"]:
                    raise ValueError("blocked: native-account budget consumed")
                progress["account_attempts"].append(key)
                atomic_json(progress, progress_path)
            model.constrained = arm == "constrained"
            if key in progress.get("accounts_done", []):
                book = recorder.load_object("analysis/native/" + key)
            else:
                replay = replay_full_path(test, models=[model], plan=plan, slippage_bps=0)
                book = native_tiered_stage1(test, replay, provider_uri=provider, tariff=plan["tariff"],
                    slippage_bps=0, initial_cash=plan["initial_cash"], freq="1min",
                        opening_offset=pd.Timedelta(hours=9, minutes=47), position_type=QlibContinuousPosition)
                book["targets"] = replay["targets"]
                recorder.save_objects(artifact_path="analysis/native", **{key: book})
                progress.setdefault("accounts_done", []).append(key)
                atomic_json(progress, progress_path)
            cases[key] = book
            recorder.log_metrics(completed_native_accounts=len(cases))
            print("native", key, flush=True)
    class ConstantTarget:
        def __init__(self, weights):
            self.weights = np.asarray(weights, dtype=float)

        def target(self, row, q, cash, known_prices, peak):
            return self.weights.copy()

    for name, weights in (("half_soxl_cash", [0.5, 0., 0.5]),
                          ("full_soxl", [1. - plan["operational_fee_reserve"], 0., plan["operational_fee_reserve"]])):
        key = f"reference_{name}_0"
        if key not in progress["account_attempts"]:
            if len(progress["account_attempts"]) >= plan["budget"]["native_accounts"]:
                raise ValueError("blocked: native-account budget consumed")
            progress["account_attempts"].append(key)
            atomic_json(progress, progress_path)
        if key in progress.get("accounts_done", []):
            book = recorder.load_object("analysis/native/" + key)
        else:
            replay = replay_full_path(test, models=[ConstantTarget(weights)], plan=plan, slippage_bps=0)
            book = native_tiered_stage1(test, replay, provider_uri=provider, tariff=plan["tariff"],
                slippage_bps=0, initial_cash=plan["initial_cash"], freq="1min",
                    opening_offset=pd.Timedelta(hours=9, minutes=47), position_type=QlibContinuousPosition)
            book["targets"] = replay["targets"]
            recorder.save_objects(artifact_path="analysis/native", **{key: book})
            progress.setdefault("accounts_done", []).append(key)
            atomic_json(progress, progress_path)
        cases[key] = book
        print("native", key, flush=True)
    rows, years, windows, uncertainty = [], [], [], []
    for key, book in cases.items():
        path, base = book["authority"], benchmark["authority"]
        paired = paired_metrics(path, base)
        annual = annual_paired_metrics(path, base)
        annual["case"] = key
        log_excess = np.log1p(path.strategy_return.to_numpy(float)) - np.log1p(base.strategy_return.to_numpy(float))
        rolling = pd.Series(log_excess).rolling(126).sum().dropna()
        complete = annual.loc[annual.year.isin(plan["complete_years"])]
        prelim = (paired["relative_wealth"] > 1 + 1e-8 and paired["max_drawdown_pct"] >= -30
            and paired["cvar5_pct"] >= paired["benchmark_cvar5_pct"]
            and float(rolling.gt(1e-8).mean()) >= .7
            and len(complete) >= 2 and float(complete.relative_wealth.gt(1).mean()) >= .7)
        stats = stationary_bootstrap_excess(path, base, mean_block=63, resamples=10000,
            alpha=.025, seed=20261014) if prelim else {"lower_bound_positive": False, "skipped": "preliminary_gate_failed"}
        rows.append({"case": key, **paired, "monitored_drawdown_pct": -100 * path.monitored_drawdown.max(),
            "positive_126d_fraction": float(rolling.gt(1e-8).mean()),
            "complete_year_positive_fraction": float(complete.relative_wealth.gt(1).mean()),
            "orders": len(book["orders"]), "average_cash_pct": 100 * path.ending_cash_weight.mean(),
            "average_soxl_pct": 100 * path.ending_soxl_weight.mean(),
            "average_nominal_beta": float((3 * path.ending_soxl_weight + path.ending_soxx_weight).mean()),
            "preliminary_gate": bool(prelim), "bootstrap_positive": bool(stats["lower_bound_positive"])})
        years.append(annual)
        windows.append({"case": key, "total": len(rolling), "positive": int(rolling.gt(1e-8).sum())})
        uncertainty.append({"case": key, **stats})
    result = pd.DataFrame(rows)
    candidate = result.loc[~result.case.str.startswith("reference_")]
    passed = candidate.loc[candidate.preliminary_gate & candidate.bootstrap_positive
        & result.monitored_drawdown_pct.ge(-30)]
    summary = {"experiment": card.name, "plan_sha256": digest,
        "decision": "0bp_pass_needs_2_5bp" if len(passed) else "not_promoted_0bp_gate_failed",
        "objective_achieved": False, "historical_research_passed": False,
        "new_model_fits": len(models), "new_native_accounts": len(cases),
        "source_recorder_id": plan["source"]["recorder_id"], "source_audit": audit,
        "recorder_id": recorder.id, "experiment_id": str(recorder.experiment_id),
        "recorder_uri": str(tracking), "recorder_local_dir": str(tracking / str(recorder.experiment_id) / recorder.id),
        "provider_uri": str(provider), "result": result.to_dict("records"),
        "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": ["inputs/*", "sources/*", "models/*", "analysis/*"]}}
    recorder.save_objects(artifact_path="analysis", summary=summary, comparison=result,
        annual=pd.concat(years), windows=pd.DataFrame(windows), uncertainty=pd.DataFrame(uncertainty))
    recorder.log_metrics(cases_passing_0bp=float(len(passed)))
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": card.name, "provider_uri": str(provider),
        "recorder_uri": str(tracking), "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    compact = result[["case", "return_pct", "relative_wealth", "max_drawdown_pct", "positive_126d_fraction", "average_soxl_pct"]]
    table = "|" + "|".join(compact.columns) + "|\n|" + "|".join(["---"] * len(compact.columns)) + "|\n"
    table += "".join("|" + "|".join(f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
        for value in row) + "|\n" for row in compact.itertuples(index=False, name=None))
    (card / "conclusion.md").write_text("# 独立树模型三资产配置\n\n" +
        f"冻结同一训练标签和输入，分别拟合 XGB/LGB；每个模型共用预测，比较放开限制与加回限制的自身完整账户。0bp仍收Tiered费用。\n\n"
        + table
        + f"\n\n决定：`{summary['decision']}`。2023—2026 是反复研究历史，不构成独立前向；旧 v2 的简化3bp结果不与本账户混用。\n")
    atomic_json(summary, card / "summary.json")
    summary["nas_evidence"] = archive_nas_recorder(card / "nas_binding.json", workspace_root=root,
        card=card, recorder_id=summary["recorder_id"], experiment_id=summary["experiment_id"])
    atomic_json(summary, card / "summary.json")
    return summary
