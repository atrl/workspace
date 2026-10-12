"""One notebook-owned selector ablation on frozen native Qlib candidates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from src.core.recorder_run import archive_evidence, publish_evidence_manifest, start_evidence_run
from src.data.convex_allocation import LABEL_COLUMNS
from src.model.generic_horizon_estimator import fit_horizon_estimator, fit_horizon_prior
from src.model.relative_loss_selection import rescore_frozen_annual_candidates


def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def verify_selection_inputs(*, evidence_root, source_root, plan):
    """Check explicit source and code bytes without creating a writable run."""
    from src.data.provider_content_hash import provider_content_hash

    evidence_root, source_root = Path(evidence_root), Path(source_root)
    _require(Path(__file__).resolve().is_relative_to(source_root.resolve()), "imported source root differs")
    for relative, digest in plan["frozen_code_sha256"].items():
        _require(hashlib.sha256((source_root / relative).read_bytes()).hexdigest() == digest,
                 "frozen code differs: " + relative)
    for relative, digest in plan["frozen_evidence_files_sha256"].items():
        _require(hashlib.sha256((evidence_root / relative).read_bytes()).hexdigest() == digest,
                 "frozen evidence differs: " + relative)
    for identity in plan["providers"].values():
        _require(provider_content_hash(evidence_root / identity["provider_uri"])[0] == identity["content_sha256"],
                 "frozen provider differs")
    _require(plan["horizon_sessions"] == 5 and plan["costs_bps"] == [10., 25.]
             and plan["robust_radius"] == 1., "matched reference method differs")
    _require(plan["candidate_names"] == ["prior", "ridge_10", "ridge_100", "ridge_1000"]
             and plan["tie_priority"] == ["prior", "ridge_1000", "ridge_100", "ridge_10"],
             "fixed candidate inventory differs")
    _require(plan["maximum_new_fits"] == len(plan["outer_years"])
             and plan["fixed_new_inner_fits"] == 0, "fit budget differs")
    _require(plan["authorization"]["orders"] is False and plan["promotion_eligible"] is False,
             "historical research cannot grant orders or promotion")


def load_frozen_selection_inputs(*, evidence_root, source_root, plan):
    """Restore the parent's price-only DatasetH and exact prior/Ridge fits."""
    from src.core.recorder_sources import load_verified_recorder_objects
    from src.data.information_increment import bind_matched_information_datasets

    verify_selection_inputs(evidence_root=evidence_root, source_root=source_root, plan=plan)
    binding = plan["parent"]
    objects, audit = load_verified_recorder_objects(
        Path(evidence_root) / binding["card_path"], expected_recorder_id=binding["recorder_id"],
        artifacts=list(binding["objects_sha256"]), expected_artifact_sha256=binding["objects_sha256"],
        expected_manifest_sha256=binding["verification_manifest_sha256"])
    bundle = bind_matched_information_datasets(objects["inputs/inputs.pkl"], as_of=plan["as_of"])
    _require(tuple(bundle["control_features"]) == tuple(plan["feature_columns"])
             and "information_value" not in plan["feature_columns"], "price-only feature schema differs")
    selections = objects["aggregate/selections.pkl"]["control"]
    _require(set(selections) == set(plan["outer_years"]), "parent selection-year inventory differs")
    return dict(bundle=bundle, parent_selections=selections, source_audit=audit)


def _load_parent_cases(evidence_root, plan):
    from qlib.workflow import R
    from src.core.recorder_artifacts import load_compressed_object

    binding = plan["parent"]
    card = Path(evidence_root) / binding["card_path"]
    summary = json.loads((card / "summary.json").read_text())
    cases = {}
    with R.uri_context(summary["recorder_uri"]):
        recorder = R.get_recorder(experiment_name=summary["experiment"], recorder_id=binding["recorder_id"])
        _require(recorder.client.get_run(recorder.id).info.status == "FINISHED", "parent run is unfinished")
        for key, identity in binding["case_artifacts"].items():
            _require(summary["compressed_case_artifacts"][key] == identity, "parent case identity differs")
            result = load_compressed_object(recorder, identity)
            _require(result["native"]["actual_position_audit"]["status"] == "passed", "inherited native QA failed")
            cases[key] = result
    _require(len(cases) == plan["fixed_inherited_native_cases"], "inherited case coverage differs")
    return cases


def run_relative_loss_experiment(*, evidence_root, source_root, card, plan_path, inputs):
    """Score prior-year labels once, refit only changed winners, then replay."""
    from src.analysis.excess_resolution import apply_excess_resolution
    from src.analysis.forecast_information_increment import paired_forecast_information
    from src.backtest.generic_horizon_window import native_verify_window, replay_horizon_window
    from src.workflow.benchmark_evidence import evaluate_registered_paths

    evidence_root, source_root, card = map(Path, (evidence_root, source_root, card))
    plan_bytes = Path(plan_path).read_bytes()
    plan = json.loads(plan_bytes)
    verify_selection_inputs(evidence_root=evidence_root, source_root=source_root, plan=plan)
    _require(inputs["source_audit"]["recorder_id"] == plan["parent"]["recorder_id"], "loaded parent differs")
    attempt = card / "execution_attempt.json"
    _require(not attempt.exists(), "fixed selector experiment already attempted")
    parent_cases = _load_parent_cases(evidence_root, plan)
    bundle = inputs["bundle"]
    frame, calendar, covariance = bundle["frame"], bundle["calendar"], bundle["covariance"]
    parent_selections = inputs["parent_selections"]
    common = dict(horizon_sessions=plan["horizon_sessions"], feature_columns=plan["feature_columns"],
                  label_columns=LABEL_COLUMNS, availability_column=plan["availability_column"],
                  eligibility_column=plan["eligibility_column"], train_start=plan["train_start"],
                  minimum_training_rows=plan["minimum_training_rows"], hac_lags=plan["hac_lags"])
    with attempt.open("x") as stream:
        json.dump(dict(status="starting", new_fit_calls=0), stream)
    identity = start_evidence_run(
        experiment=plan["experiment"], uri=card / "execution_recorder", name="fixed_relative_selector",
        objects={"plan.pkl": plan, "bundle.pkl": bundle, "parent_source_audit.pkl": inputs["source_audit"]},
        params=dict(mechanism="relative_validation_loss_only", maximum_new_fits=plan["maximum_new_fits"]))
    selections, score_frames, selection_rows, new_fits = {}, [], [], {}
    cases, case_artifacts = {}, {}
    try:
        # Every year's selector is frozen before any outer return is evaluated.
        for year in plan["outer_years"]:
            result = rescore_frozen_annual_candidates(
                bundle["control"], frame, parent_selections[year], outer_year=year,
                candidate_names=plan["candidate_names"], tie_priority=plan["tie_priority"],
                minimum_validation_rows=plan["minimum_validation_rows"], **common)
            selections[year] = result
            score_frames.append(result["candidate_scores"])
            selection_rows.append(dict(year=year, parent_selected=result["parent_selected_name"],
                                       relative_selected=result["selected_name"], changed=result["requires_refit"],
                                       validation_rows=result["audit"]["validation_rows"],
                                       selection_uses_outer_outcomes=False))
            print(f"{year}: {result['parent_selected_name']} -> {result['selected_name']}", flush=True)
        archive_evidence(identity, {"selector_decisions.pkl": selections})
        for year, selection in selections.items():
            name = selection["selected_name"]
            if selection["requires_refit"]:
                model = (fit_horizon_prior(bundle["control"], year=year, **common) if name == "prior"
                         else fit_horizon_estimator(bundle["control"], year=year,
                                                   ridge_alpha=float(name.split("_")[1]), **common))
                new_fits[year] = model
                archive_evidence(identity, {"new_fits.pkl": new_fits})
            else:
                model = parent_selections[year]["selected_fit"]
            _require(model["audit"]["training_frame_sha256"] == selection["outer_training_frame_sha256"],
                     "selected outer training bytes differ")
            selection["selected_fit"] = model
        _require(len(new_fits) == sum(row["changed"] for row in selection_rows)
                 and len(new_fits) <= plan["maximum_new_fits"], "actual fit budget differs")
        unsettled = pd.to_datetime(frame.exit_date).isna().to_numpy()
        _require(int(unsettled.sum()) == 2 and unsettled[-2:].all(), "unsettled tail differs")
        windows = {"development": plan["development_years"], "reused_diagnostic": plan["diagnostic_years"]}
        paths, forecasts, native_rows, usage_rows = {}, {}, [], []
        for window, years in windows.items():
            mask = frame.index.year.isin(years) & ~unsettled
            part = frame.loc[mask].reset_index(drop=True)
            window_changed = any(selections[y]["requires_refit"] for y in years)
            for cost in plan["costs_bps"]:
                for arm in ("control", "soxx", "candidate"):
                    key = f"{window}::{arm}::{int(cost)}"
                    inherited = arm != "candidate" or not window_changed
                    source_key = f"{window}::{'control' if arm == 'candidate' else arm}::{int(cost)}"
                    if inherited:
                        result = parent_cases[source_key]
                    else:
                        result = replay_horizon_window(
                            part, selections, covariance[mask], calendar=calendar, cost_bps=cost,
                            horizon_sessions=plan["horizon_sessions"], robust_radius=plan["robust_radius"],
                            initial_cash=plan["initial_cash"], feature_columns=plan["feature_columns"],
                            availability_column=plan["availability_column"], eligibility_column=plan["eligibility_column"])
                        result = native_verify_window(part, result,
                            provider_uri=evidence_root / plan["providers"]["execution"]["provider_uri"],
                            cost_bps=cost, initial_cash=plan["initial_cash"])
                    _require(result["native"]["actual_position_audit"]["status"] == "passed", "native QA failed")
                    cases[key] = result
                    paths[(window, arm, cost)] = result["path"]
                    if arm != "soxx":
                        if (window, arm) in forecasts:
                            pd.testing.assert_frame_equal(forecasts[(window, arm)], result["forecast"])
                        forecasts[(window, arm)] = result["forecast"]
                    native_rows.append(dict(case=key, origin="inherited" if inherited else "recomputed",
                        parent_case=source_key if inherited else "", **result["native"]["actual_position_audit"]))
                    path = result["path"]
                    usage_rows.append(dict(window=window, arm=arm, cost_bps=cost, sessions=len(path),
                        soxl_positive_sessions=int((path.ending_soxl_weight > 1e-8).sum()),
                        maximum_soxl_weight=float(path.ending_soxl_weight.max()),
                        cash_positive_sessions=int((path.ending_cash_weight > 1e-8).sum()),
                        mean_cash_weight=float(path.ending_cash_weight.mean())))
                    leaf = key.replace("::", "__") + ".pkl.gz"
                    case_artifacts[key] = archive_evidence(identity, {leaf: result}, artifact_path="cases", compressed=True)[leaf]
                    print(f"native {key}: {'inherited' if inherited else 'recomputed'} passed", flush=True)
        _require(len(cases) == len(case_artifacts) == 12, "case inventory differs")
        combined = {(arm, cost): pd.concat([paths[(w, arm, cost)] for w in windows], ignore_index=True)
                    for arm in ("candidate", "control", "soxx") for cost in plan["costs_bps"]}
        registered = dict(development_years=plan["development_years"], costs_bps=plan["costs_bps"],
                          as_of=plan["as_of"], variants=[{"name": "candidate"}, {"name": "control"}],
                          candidate_budget=1, gates=plan["gates"])
        tables, evidence = evaluate_registered_paths(combined, plan=registered, calendar=calendar,
                                                     candidate_controls={"candidate": "control"})
        resolved = apply_excess_resolution(evidence["candidate"],
            {c: paths[("development", "candidate", c)] for c in plan["costs_bps"]},
            {c: paths[("development", "soxx", c)] for c in plan["costs_bps"]},
            log_excess_floor=plan["log_excess_floor"])
        tables.update({"raw_gates.csv": tables["gates.csv"], "gates.csv": resolved["gates"],
                       "resolved_annual_metrics.csv": resolved["annual"],
                       "bootstrap_analysis.csv": resolved["bootstrap"],
                       "rolling_summary.csv": resolved["rolling_summary"]})
        diagnostic = tables["backtest_benchmark_comparison.csv"].query("strategy=='candidate' and window=='reused_diagnostic'")
        diagnostic_pass = bool((diagnostic.log_excess_sum > plan["log_excess_floor"]).all()
            and (diagnostic.max_drawdown_pct >= diagnostic.benchmark_max_drawdown_pct - 1e-10).all()
            and (diagnostic.cvar5_pct >= diagnostic.benchmark_cvar5_pct - 1e-10).all())
        portfolio_pass = bool((tables["ablation_comparison.csv"].log_excess_sum > plan["log_excess_floor"]).all())
        info, information_pass = paired_forecast_information(
            pd.concat([forecasts[(w, "candidate")] for w in windows]),
            pd.concat([forecasts[(w, "control")] for w in windows]),
            cutoff=pd.Timestamp(plan["as_of"] + " 16:15", tz="America/New_York"), windows=windows,
            resolution=plan["log_excess_floor"])
        tables.update({"forecast_information.csv": info, "model_selection.csv": pd.DataFrame(selection_rows),
                       "validation_scores.csv": pd.concat(score_frames, ignore_index=True),
                       "native_audit.csv": pd.DataFrame(native_rows), "position_usage.csv": pd.DataFrame(usage_rows)})
        tables["risk_analysis.csv"] = pd.DataFrame([dict(case=k, metric=str(m), value=float(row["risk"]))
            for k, r in cases.items() for m, row in r["native"]["risk_analysis_next_open"].iterrows()])
        verify_selection_inputs(evidence_root=evidence_root, source_root=source_root, plan=plan)
        _require(Path(plan_path).read_bytes() == plan_bytes, "frozen plan changed during execution")
        all_pass = bool(resolved["historical_gate_pass"] and diagnostic_pass and portfolio_pass and information_pass)
        policy = dict(policy="slim_card_recorder_first", workflow_config="workflow.yaml",
                      recorder_only_artifacts=["inputs/*.pkl", "aggregate/selector_decisions.pkl",
                          "aggregate/new_fits.pkl", "aggregate/selections.pkl", "cases/*.pkl.gz",
                          "aggregate/tables.pkl", "aggregate/raw_evidence.pkl", "aggregate/resolved_evidence.pkl",
                          "aggregate/final_summary.pkl"])
        summary = dict(identity, status="complete_not_promoted", promotion_decision="not_promoted",
            promotion_eligible=False, scientific_promotion=False, new_fit_calls=len(new_fits), new_inner_fit_calls=0,
            changed_selection_years=[r["year"] for r in selection_rows if r["changed"]],
            native_cases=12, inherited_native_cases=sum(r["origin"] == "inherited" for r in native_rows),
            recomputed_native_cases=sum(r["origin"] == "recomputed" for r in native_rows),
            historical_gate_pass=bool(resolved["historical_gate_pass"]), diagnostic_gate_pass=diagnostic_pass,
            incremental_information_pass=information_pass, incremental_portfolio_pass=portfolio_pass,
            route_decision="historical_pass_requires_independent_forward" if all_pass else "end_relative_selector_route_no_further_search",
            plan_sha256=hashlib.sha256(plan_bytes).hexdigest(), source=plan["providers"],
            as_of=plan["as_of"], region="us", source_binding=plan["parent"],
            recorder_status="FINISHED", compressed_case_artifacts=case_artifacts, artifact_policy=policy,
            scientific_scope=plan["scientific_scope"], qlib_docs=plan["qlib_docs"])
        archive_evidence(identity, {"selections.pkl": selections, "new_fits.pkl": new_fits, "tables.pkl": tables,
            "raw_evidence.pkl": evidence, "resolved_evidence.pkl": resolved, "final_summary.pkl": summary})
        for name, table in tables.items():
            table.to_csv(card / name, index=False)
        publish_evidence_manifest(card, summary)
        (card / "workflow.yaml").write_text(yaml.safe_dump(dict(experiment=plan["experiment"],
            provider_uri=plan["providers"]["price"]["provider_uri"], source=plan["providers"],
            recorder_id=summary["recorder_id"], recorder_uri=summary["recorder_uri"],
            artifact_policy=policy, selection_loss="relative_response_mse", segments=windows), sort_keys=False))
        (card / "selected_model.json").write_text(json.dumps(dict(model="native_prior_ridge_relative_loss_selection",
            selections=selection_rows, promotion_decision="not_promoted"), indent=2) + "\n")
        lines = ["# 超额误差选模型：冻结对照结果", "",
            f"复用既有四候选的内层拟合，仅改变选模型损失。{len(new_fits)} 次新外层拟合，零次新内层拟合；"
            f"原生账户场景12/12通过（继承{summary['inherited_native_cases']}，重算{summary['recomputed_native_cases']}）。",
            f"选择改变年份：{summary['changed_selection_years']}。H5、价格特征、日风险预算、稳健半径和10/25bp费用保持配对。", ""]
        for row in tables["backtest_benchmark_comparison.csv"].query("strategy=='candidate'").itertuples():
            lines.append(f"- {row.window}/{row.cost_bps:g}bp：收益 {row.return_pct:.2f}%，相对SOXX终值 {row.relative_wealth:.6f}，MaxDD {row.max_drawdown_pct:.2f}%。")
        lines += ["", f"历史稳定门={summary['historical_gate_pass']}；复用诊断门={diagnostic_pass}；"
            f"超额预测MSE增量={information_pass}；对匹配控制组净组合增量={portfolio_pass}。",
            f"晋升决定：不晋升；路线：{summary['route_decision']}。",
            "该实验只检验选模型的评价口径；固定alpha下线性Ridge直接拟合收益差，与两腿预测相减等价，不凭标签换名制造信息增量。",
            "匹配样本继承父价格控制组的来源资格掩码；未使用原始指引数值。历史已反复查看，属于复用诊断，不能再称新样本外。",
            "费用各自重跑配置与结算；继承场景读取原生数量/现金/费用证据，不宣称重新回放。SOXL与现金实际使用见position_usage.csv。",
            "下一步：若本机制失败，封存该选择口径路线；新增机制须先提出可验证信息与目标的关系并冻结，再取得独立前向证据。",
            f"Qlib Recorder {summary['recorder_id']}；FINISHED。"]
        (card / "conclusion.md").write_text("\n".join(lines) + "\n")
        attempt.write_text(json.dumps(dict(status="completed", new_fit_calls=len(new_fits), **identity), indent=2) + "\n")
        return summary
    except Exception as error:
        failure = dict(status="blocked", error=repr(error), new_fit_calls=len(new_fits), completed_cases=list(cases), **identity)
        archive_evidence(identity, {"failure.pkl": failure, "new_fits.pkl": new_fits, "partial_selections.pkl": selections})
        attempt.write_text(json.dumps(failure, indent=2) + "\n")
        raise
