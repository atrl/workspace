"""Frozen two-route local-value information screen on exact-parent Qlib evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.minute_intake import atomic_json
from src.model.f696_joint_local_value import F696JointLocalValue
from src.workflow.minute_action_value import _source


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset(frame: pd.DataFrame, *, features: list[str], cutoff: str):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    joined = pd.concat({"feature": frame[features], "label": frame[["local_advantage"]],
        "meta": frame[["feature_available_at", "decision_at", "label_available_at"]]}, axis=1)
    joined.index = pd.MultiIndex.from_arrays([frame.trade_date, ["SOXL_SOXX_CASH"] * len(frame)],
        names=["datetime", "instrument"])
    return DatasetH(handler=DataHandlerLP.from_df(joined),
        segments={"train": (None, pd.Timestamp(cutoff) - pd.Timedelta(nanoseconds=1)),
                  "test": (pd.Timestamp(cutoff), None)})


def _stationary_lower(values: np.ndarray, *, mean_block: int, repetitions: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    size = len(values)
    indices = rng.integers(size, size=repetitions)
    means = np.zeros(repetitions)
    for step in range(size):
        if step:
            start = rng.random(repetitions) < 1 / mean_block
            indices = np.where(start, rng.integers(size, size=repetitions), (indices + 1) % size)
        means += values[indices]
    return float(np.quantile(means / size, .025))


def _scores(actual: np.ndarray, candidate: np.ndarray, control: np.ndarray, mean: float) -> dict:
    binary = actual > 0
    auc = float(roc_auc_score(binary, candidate)) if len(np.unique(binary)) == 2 else float("nan")
    return {"rows": len(actual), "candidate_mse": float(np.mean((actual - candidate) ** 2)),
        "control_mse": float(np.mean((actual - control) ** 2)),
        "train_mean_mse": float(np.mean((actual - mean) ** 2)),
        "candidate_auc": auc, "predicted_positive": int((candidate > 0).sum()),
        "predicted_negative": int((candidate < 0).sum()),
        "realized_positive": int(binary.sum()), "realized_negative": int((~binary).sum())}


def run(*, root: Path, card: Path, plan: dict) -> dict:
    import qlib
    import yaml
    from qlib.workflow import R

    root, card = Path(root).resolve(), Path(card).resolve()
    digest = _sha(card / "plan.json")
    if (card / "summary.json").exists():
        old = json.loads((card / "summary.json").read_text())
        if old["plan_sha256"] != digest:
            raise ValueError("blocked: fixed joint information plan changed")
        return old
    for role in ("labels", "frame"):
        source = plan["sources"][role]
        if _sha(root / source["card"] / "summary.json") != source["summary_sha256"]:
            raise ValueError("blocked: joint information source changed: " + role)
    if _sha(root / plan["provider_uri"] / "qlib_store_manifest.json") != plan["provider_manifest_sha256"]:
        raise ValueError("blocked: Qlib provider identity differs")
    qlib.init(provider_uri=str(root / plan["provider_uri"]), region="us", kernels=1)
    labels, label_audit = _source(root, plan["sources"]["labels"], ["analysis/labels"])
    source, frame_audit = _source(root, plan["sources"]["frame"], ["inputs/covered_frame"])
    label_frame = labels["analysis/labels"]
    market = source["inputs/covered_frame"][["signal_date", *plan["market_features"], "feature_available_at"]]
    combined = label_frame.merge(market, on="signal_date", how="left", validate="one_to_one")
    if len(combined) != plan["expected_reference_rows"] or combined[plan["market_features"]].isna().any().any():
        raise ValueError("needs_data: frozen market features incomplete")
    if not pd.to_datetime(combined.feature_available_at).lt(pd.to_datetime(combined.decision_at)).all():
        raise ValueError("blocked: information not available at opening decision")
    tracking = card / "qlib_recorder"
    progress_path = card / "run_state.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
        "plan_sha256": digest, "fit_attempts": [], "fits_done": []}
    if progress["plan_sha256"] != digest:
        raise ValueError("blocked: prior fit attempt has different plan")
    kwargs = {"experiment_name": card.name, "uri": str(tracking)}
    if progress.get("recorder_id"):
        kwargs.update(recorder_id=progress["recorder_id"], resume=True)
    with R.start(**kwargs):
        recorder = R.get_recorder()
        progress.update(recorder_id=recorder.id, experiment_id=str(recorder.experiment_id))
        atomic_json(progress, progress_path)
        recorder.save_objects(artifact_path="inputs", plan=plan,
            source_audits={"labels": label_audit, "frame": frame_audit})
        recorder.save_objects(artifact_path="sources", workflow=Path(__file__).read_text(),
            model=(root / "src/model/f696_joint_local_value.py").read_text())
        rows, predictions, audits, gates = [], {}, [], {}
        for route, config in plan["routes"].items():
            data = combined.loc[combined.eligible & combined.route.eq(route)].sort_values("trade_date").copy()
            if len(data) < config["minimum_total_rows"]:
                raise ValueError("needs_data: route source support differs: " + route)
            cutoff = config["cutoff"]
            cohorts = {}
            for arm, features in (("control", plan["state_features"]),
                                  ("candidate", [*plan["state_features"], *plan["market_features"]])):
                name = route + "_" + arm
                dataset = _dataset(data, features=features, cutoff=cutoff)
                if name in progress["fits_done"]:
                    model = recorder.load_object("models/" + name)
                else:
                    if name in progress["fit_attempts"] or len(progress["fit_attempts"]) >= plan["budget"]["model_fit_attempts"]:
                        raise ValueError("blocked: model fit attempt consumed")
                    progress["fit_attempts"].append(name)
                    atomic_json(progress, progress_path)
                    model = F696JointLocalValue(features=features, cutoff=cutoff, seed=plan["seed"]).fit(dataset)
                    recorder.save_objects(artifact_path="models", **{name: model})
                    progress["fits_done"].append(name)
                    atomic_json(progress, progress_path)
                score = model.predict(dataset)
                cohorts[arm] = (model, score)
                audits.append({"route": route, "arm": arm, **model.audit})
                recorder.log_metrics(completed_model_fits=float(len(progress["fits_done"])))
            truth = data.loc[data.trade_date.ge(pd.Timestamp(cutoff)), ["trade_date", "local_advantage"]].copy()
            index = pd.MultiIndex.from_arrays([truth.trade_date, ["SOXL_SOXX_CASH"] * len(truth)], names=["datetime", "instrument"])
            control = cohorts["control"][1].reindex(index).to_numpy(float)
            candidate = cohorts["candidate"][1].reindex(index).to_numpy(float)
            actual = truth.local_advantage.to_numpy(float)
            if len(actual) != config["expected_evaluation_rows"] or not np.isfinite(np.c_[actual, control, candidate]).all():
                raise ValueError("blocked: evaluation cohort or predictions differ: " + route)
            mean = cohorts["control"][0].training_mean
            pooled = _scores(actual, candidate, control, mean)
            improvement = (actual - control) ** 2 - (actual - candidate) ** 2
            lower = _stationary_lower(improvement, mean_block=21, repetitions=3000,
                                      seed=plan["seed"] + config["seed_offset"])
            years = []
            for year, part in truth.groupby(truth.trade_date.dt.year):
                if year not in config["required_years"]:
                    continue
                positions = truth.trade_date.dt.year.eq(year).to_numpy()
                score = _scores(actual[positions], candidate[positions], control[positions], mean)
                years.append({"route": route, "year": int(year), **score})
            route_pass = (pooled["candidate_mse"] < pooled["control_mse"]
                and pooled["candidate_mse"] < pooled["train_mean_mse"]
                and pooled["candidate_auc"] > .5 and pooled["predicted_positive"] >= 5
                and pooled["predicted_negative"] >= 5 and lower > 0
                and all(row["rows"] >= config["minimum_year_rows"]
                    and row["candidate_mse"] < row["control_mse"]
                    and row["candidate_auc"] > .5 for row in years)
                and len(years) == len(config["required_years"]))
            gates[route] = {"passed": bool(route_pass), "pooled": pooled,
                "bootstrap_lower_mse_gain": lower, "required_years": config["required_years"]}
            rows.extend(years)
            predictions[route] = pd.DataFrame({"trade_date": truth.trade_date.to_numpy(),
                "actual": actual, "control": control, "candidate": candidate})
        passed = all(value["passed"] for value in gates.values())
        summary = {"experiment": card.name, "plan_sha256": digest,
            "decision": "joint_information_pass_needs_0bp_account_design" if passed
                        else "not_promoted_fixed_joint_information_gate_failed",
            "objective_achieved": False, "historical_research_passed": False,
            "gates": gates, "new_model_fits": len(progress["fits_done"]), "new_native_accounts": 0,
            "recorder_id": recorder.id, "experiment_id": str(recorder.experiment_id),
            "recorder_uri": str(tracking),
            "recorder_local_dir": str(tracking / str(recorder.experiment_id) / recorder.id),
            "provider_uri": str(root / plan["provider_uri"]),
            "artifact_policy": {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                "recorder_only_artifacts": ["inputs/*", "sources/*", "models/*", "analysis/predictions", "analysis/*"]}}
        recorder.save_objects(artifact_path="analysis", summary=summary, predictions=predictions,
            annual=pd.DataFrame(rows), training_audit=pd.DataFrame(audits))
        recorder.log_metrics(information_gate_pass=float(passed))
    annual = pd.DataFrame(rows)
    annual.to_csv(card / "annual_information.csv", index=False)
    (card / "workflow.yaml").write_text(yaml.safe_dump({"experiment": card.name,
        "provider_uri": summary["provider_uri"], "recorder_uri": str(tracking),
        "artifact_policy": summary["artifact_policy"]}, sort_keys=False))
    (card / "conclusion.md").write_text("# 精确f696双层局部价值信息门\n\n"
        "固定四次LightGBM拟合，两条接收任务各比较自身状态控制组与自身状态加六项决策前市场坐标。"
        "按预先冻结的时间切分、MSE、正优势AUC、逐年及21个合格样本平均区块bootstrap下界判断。"
        f"结果：`{summary['decision']}`。逐年与完整门值见 `annual_information.csv`、`summary.json`。"
        "这仍是2018–2022开发期信息检验，未运行2023+完整候选账户，不能宣告SOXX经济成功。"
        "若任一路线失败，关闭固定版本；不改符号、年份、特征、树深或阈值救援。\n")
    atomic_json(summary, card / "summary.json")
    return summary
