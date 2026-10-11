"""H20 matched-horizon experiment workflow (code only; runs only under authorization).

One-shot fixed budget: 2016..2026 annual selection (5 fits/year = 55), one H20 selected
forecast, two costs (10/25), robust_radius 1, the two independent windows each reset from
cash, and a SOXX buy-and-hold baseline. Both candidate and baseline come from root's
``src/backtest/generic_horizon_window`` two-step API: ``replay_horizon_window`` then
``native_verify_window`` (which replaces strategy_return with actual native next-open
returns and appends the native cash/quantity QA). No callback is invented here.

Recorder: open one Recorder up front binding source/plan/Dataset and the 55-fit budget,
then run; on failure archive the failure and completed fits/cases and stop blocked. On
success archive all models/forecasts/paths/native objects and only then publish the summary.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.convex_allocation import FEATURE_COLUMNS
from src.data.h20_native_dataset import build_h20_native_dataset, load_parent_plan

EXPERIMENT = "h20_matched_forecast_execution_20261010"
SCHEMA_VERSION = "h20_matched_experiment_v3"
LABEL_COLUMNS = ("soxl_return_h20", "soxx_return_h20")
SELECTION_YEARS = tuple(range(2016, 2027))
FITS_PER_YEAR = 5
FIXED_FIT_BUDGET = 55
FIXED_NATIVE_CASES = 8
COSTS_BPS = (10.0, 25.0)
ROBUST_RADIUS = 1.0
INITIAL_CASH = 100_000.0
WINDOW_MODULE = "src.backtest.generic_horizon_window"
ARTIFACT_POLICY = {"policy": "slim_card_recorder_first", "workflow_config": "workflow.yaml",
                   "recorder_only_artifacts": ["inputs/inputs.pkl", "aggregate/case_results.pkl", "aggregate/selections.pkl",
                       "aggregate/window_paths.pkl", "aggregate/raw_evidence.pkl", "aggregate/resolved_evidence.pkl", "aggregate/final_summary.pkl"]}



def _require(condition, message):
    if not condition:
        raise ValueError("blocked: " + message)


def _strict_int(value, name):
    _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)), f"{name} must be an int")
    return int(value)


def load_execution_plan(path) -> dict:
    plan = json.loads(Path(path).read_text())
    required = {"experiment", "source", "selection_years", "costs_bps", "robust_radius", "windows",
                "authorization", "development_years", "horizon_sessions", "gates"}
    _require(not (required - set(plan)), f"execution plan missing {sorted(required - set(plan))}")
    _require(tuple(_strict_int(y, "selection_year") for y in plan["selection_years"]) == SELECTION_YEARS, "selection years must be exactly 2016..2026")
    _require(tuple(float(c) for c in plan["costs_bps"]) == COSTS_BPS, "costs must be exactly 10/25")
    _require(float(plan["robust_radius"]) == ROBUST_RADIUS, "robust_radius must be 1")
    _require(_strict_int(plan["horizon_sessions"], "horizon_sessions") == 20, "horizon must be 20")
    src = plan["source"]
    _require(src.get("source_sha256") and src.get("preregister_sha256") and src.get("providers"),
             "execution plan must freeze source_sha256, preregister_sha256 and both providers")
    auth = plan["authorization"]
    _require(auth.get("training_authorized") is True and auth.get("replay_authorized") is True,
             "this workflow requires an authorized one-shot execution plan")
    _require(tuple(plan["windows"]) == ("development", "reused_diagnostic"), "exactly two registered windows")
    for key, expected in (("fixed_fit_budget", FIXED_FIT_BUDGET), ("fixed_native_cases", FIXED_NATIVE_CASES),
                          ("candidate_budget", 1)):
        _require(_strict_int(plan.get(key), key) == expected, f"{key} must be {expected}")
    for key, expected in (("minimum_training_rows", 252), ("minimum_validation_rows", 126), ("hac_lags", 5)):
        _require(_strict_int(plan.get(key), key) == expected, "registered " + key + " differs")
    _require(plan.get("train_start") == "2011-01-03" and plan.get("initial_cash") == 100000., "registered training start or cash differs")
    _require(tuple(plan["development_years"]) == tuple(range(2016, 2023)), "development years differ")
    _require(not isinstance(plan["robust_radius"], bool), "boolean robust radius invalid")
    _require(set(src["providers"]) == {"published", "execution"} and src.get("prefit"), "both frozen providers and prefit evidence required")
    return plan


def fit_budget(plan: dict) -> int:
    return len(tuple(plan["selection_years"])) * FITS_PER_YEAR


def _providers(root: Path, plan: dict) -> dict:
    return {key: Path(root) / rel for key, rel in plan["source"]["providers"].items()}


def _source_freeze(root: Path, card: Path, plan: dict) -> dict:
    """PIT + provider freeze: src hashes, preregister content, both provider content, prefit QA."""
    from src.data.provider_content_hash import provider_content_hash
    from src.core.recorder_sources import load_verified_recorder_objects
    frozen = dict(plan["source"]["source_sha256"])
    verified = {}
    for rel, expected in frozen.items():
        path = Path(root) / rel
        _require(path.exists(), f"frozen source {rel} is missing")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(actual == expected, f"frozen source {rel} differs from the execution plan")
        verified[rel] = actual
    prereg = plan["source"].get("preregister_plan")
    if prereg:
        path = Path(root) / prereg
        _require(hashlib.sha256(path.read_bytes()).hexdigest() == plan["source"]["preregister_sha256"],
                 "preregister content differs from the execution plan freeze")
    providers = {}
    for key, store in _providers(root, plan).items():
        digest, inventory = provider_content_hash(store)
        _require(inventory, f"empty provider inventory {key}")
        _require(digest == plan["source"]["provider_sha256"][key], f"provider content differs: {key}")
        providers[key] = digest
    prefit = plan["source"].get("prefit")
    if prefit:
        load_verified_recorder_objects(Path(root) / prefit["card_dir"], expected_recorder_id=prefit["recorder_id"],
                                       artifacts=list(prefit["artifacts"]),
                                       expected_artifact_sha256=prefit.get("artifact_sha256"))
    return {"source_sha256": verified, "provider_content_sha256": providers}


def require_all_native_passed(native_audits: dict, expected_keys) -> None:
    """Exactly the 8 complete cases, every actual-position audit passed. No default/skip."""
    if set(native_audits) != set(expected_keys) or len(native_audits) != FIXED_NATIVE_CASES:
        raise ValueError("blocked: exactly 8 complete native cases are required before gates")
    failed = {k: v.get("status") for k, v in native_audits.items() if v.get("status") != "passed"}
    if failed:
        raise ValueError(f"blocked: actual cash/quantity native audit must pass for all 8 cases: {failed}")


def _window_frames(frame: pd.DataFrame, covariance, plan: dict) -> tuple[dict, dict]:
    """Exact dates; the unsettled NaT exit must be exactly the last 2 rows, never an interior row."""
    market = frame.copy()
    exit_na = pd.to_datetime(market["exit_date"]).isna().to_numpy()
    _require(int(exit_na.sum()) == 2 and bool(exit_na[-2:].all()) and not bool(exit_na[:-2].any()),
             "unsettled NaT exit rows must be exactly the last 2 sessions")
    years = pd.DatetimeIndex(market["signal_date"]).year
    dev_years = tuple(int(y) for y in plan["development_years"])
    covariance = np.asarray(covariance)
    _require(covariance.shape[0] == len(market), "covariance must align with the full frame")
    windows = {}
    for name in plan["windows"]:
        mask = (np.isin(years, dev_years) if name == "development" else years > max(dev_years)) & ~exit_na
        _require(bool(mask.any()), f"window {name} is empty")
        windows[name] = {"frame": market.loc[mask].reset_index(drop=True), "covariance": covariance[mask]}
    return windows, {"full_rows": int(len(market)), "excluded_unsettled_terminal_rows": 2}


def _assemble_registered_paths(window_paths: dict, plan: dict, *, dev_only: bool = False) -> dict:
    strategies = ("h20", "soxx")
    windows = ("development",) if dev_only else tuple(plan["windows"])
    registered = {}
    for strategy in strategies:
        for cost in COSTS_BPS:
            parts = [window_paths[(w, strategy, cost)] for w in windows]
            registered[(strategy, cost)] = pd.concat(parts, ignore_index=True).sort_values("signal_date").reset_index(drop=True)
    return registered


def _open_inputs_recorder(root: Path, card: Path, plan: dict, *, plan_path, inputs) -> tuple[str, str, str]:
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import qlib
    from qlib.workflow import R
    provider = str(_providers(root, plan)["published"] if "published" in plan["source"]["providers"]
                   else next(iter(_providers(root, plan).values())))
    uri = (Path(card) / "qlib_recorder").resolve()
    uri.mkdir(parents=True, exist_ok=True)
    qlib.init(provider_uri=provider, region=plan["source"].get("region", "us"))
    with R.start(experiment_name=plan["experiment"], recorder_name="h20_matched", uri=str(uri)):
        record = R.get_recorder()
        record.log_params(schema_version=SCHEMA_VERSION, fit_budget=FIXED_FIT_BUDGET,
                          native_cases=FIXED_NATIVE_CASES,
                          plan_sha256=hashlib.sha256(Path(plan_path).read_bytes()).hexdigest())
        record.save_objects(artifact_path="inputs", **{"inputs.pkl": {**inputs}})
        recorder_id, local_dir = record.id, record.get_local_dir()
    _require(record.status == "FINISHED", "input Recorder did not finish")
    return recorder_id, str(uri), local_dir


def _archive(uri, recorder_id, experiment, *, objects: dict, tables: dict | None = None):
    """Archive aggregate leaf .pkl objects (artifact_path='aggregate') under the SAME recorder id."""
    from qlib.workflow import R
    _require(objects and all(str(name).endswith(".pkl") for name in objects), "archive leaves must be leaf .pkl names")
    with R.uri_context(str(uri)):
        with R.start(experiment_name=experiment, recorder_id=recorder_id, uri=str(uri), resume=True):
            record = R.get_recorder()
            _require(record.id == recorder_id, "resumed recorder id changed")
            record.save_objects(artifact_path="aggregate", **objects)
            for name, table in (tables or {}).items():
                record.save_objects(artifact_path="tables", **{Path(name).stem + ".pkl": table})
            recorder_id = record.id
        _require(record.status == "FINISHED", "Recorder did not finish after archive")   # checked OUTSIDE the run
        return recorder_id


def _small_gate_records(raw_evidence: dict, resolved: dict) -> list:
    """Only small scalar gate fields enter the summary; full tables go to the Recorder."""
    records = []
    for source, table in (("raw", raw_evidence.get("gates")), ("resolved", resolved.get("excess_resolution"))):
        if hasattr(table, "items"):
            for key, value in table.items():
                if isinstance(value, (bool, int, float, str)) or value is None:
                    records.append({"source": source, "name": key, "value": value})
    return records


def _load_frozen_attempt(plan):
    from qlib.workflow import R
    source = plan['recovery']
    with R.uri_context(source['recorder_uri']):
        rec = R.get_recorder(recorder_id=source['recorder_id'],experiment_name=plan['experiment'])
        _require(rec.status == 'FINISHED', 'frozen attempt Recorder not finished')
        _require(rec.list_params()['plan_sha256'] == source['original_execution_plan_sha256'], 'original training plan identity differs')
        result = {}
        for name,digest in source['artifact_sha256'].items():
            path = Path(rec.get_local_dir())/'artifacts'/name
            _require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, 'frozen attempt artifact changed: ' + name)
            result[name] = rec.load_object(name)
    selected = result['aggregate/partial_selections.pkl']
    _require(set(selected) == set(plan['selection_years']), 'frozen model years differ')
    _require(sum(s['audit']['model_fit_calls'] for s in selected.values()) == 55, 'inherited fit budget differs')
    return result


def run_h20_matched_experiment(*, root, card, execution_plan_path, write=True):
    """One real attempt; archive intermediate evidence before any failure propagates."""
    root, card = Path(root).resolve(), Path(card).resolve()
    plan = load_execution_plan(execution_plan_path)
    recovery = 'recovery' in plan
    attempt = card / ('replay_recovery_attempt.json' if recovery else 'execution_attempt.json')
    _require(not attempt.exists(), 'one-shot attempt already exists; read its Recorder instead of refitting')
    os.environ.setdefault('MLFLOW_ALLOW_FILE_STORE', 'true')
    import qlib
    qlib.init(provider_uri=str(_providers(root, plan)['published']), region=plan['source']['region'], kernels=1)
    freeze_start = _source_freeze(root, card, plan)
    frozen = _load_frozen_attempt(plan) if recovery else None
    if recovery:
        inputs = frozen['inputs/inputs.pkl']
        bundle = {**inputs, 'dataset':None}
    else:
        parent_plan = load_parent_plan(root, plan['source']['parent_plan'])
        bundle = build_h20_native_dataset(root,parent_plan,horizon_sessions=20)
    dataset,frame,calendar = bundle['dataset'],bundle['frame'],bundle['calendar']
    windows, window_audit = _window_frames(frame, bundle['covariance'], plan)
    with attempt.open('x') as f:
        json.dump(dict(status='starting', plan_sha256=hashlib.sha256(Path(execution_plan_path).read_bytes()).hexdigest()),f)
    recorder_id, uri, local_dir = _open_inputs_recorder(root,card,plan,plan_path=execution_plan_path,
        inputs=dict(frame=frame, covariance=bundle['covariance'], calendar=calendar, prepared=bundle['prepared'],
                    audit=bundle['audit'], source_freeze=freeze_start, plan=plan, window_audit=window_audit))
    attempt.write_text(json.dumps(dict(status='running',recorder_id=recorder_id,recorder_uri=uri),indent=2)+'\n')
    selections = frozen['aggregate/partial_selections.pkl'] if recovery else {}
    fit_calls = 55 if recovery else 0
    cases = frozen['aggregate/partial_case_results.pkl'] if recovery else {}
    window_paths = {}
    stage = 'fit'
    try:
        from src.model.generic_annual_ridge_selection import select_annual_horizon
        from src.backtest.generic_horizon_window import replay_horizon_window, native_verify_window
        for year in (() if recovery else plan['selection_years']):
            result = select_annual_horizon(dataset,frame,horizon_sessions=20,feature_columns=FEATURE_COLUMNS,
                label_columns=LABEL_COLUMNS,availability_column='label_h20_available_at',outer_year=year,
                minimum_training_rows=252,minimum_validation_rows=126,hac_lags=5,train_start=plan['train_start'])
            selections[year] = result
            fit_calls += result['audit']['model_fit_calls']
            attempt.write_text(json.dumps(dict(status='running',stage=f'fit_{year}',fit_calls=fit_calls,
                native_cases=0,recorder_id=recorder_id,recorder_uri=uri),indent=2)+'\n')
            print(f'H20 selected {year}: {result["selected_name"]}; cumulative fits {fit_calls}',flush=True)
        _require(fit_calls == FIXED_FIT_BUDGET, 'fit budget differs')
        # Keep fitted models even if the first native case fails.
        _archive(uri,recorder_id,plan['experiment'],objects={'selections.pkl':selections})
        expected = {f'{w}::{s}::{int(c)}' for w in plan['windows'] for s in ('h20','soxx') for c in COSTS_BPS}
        for window, parts in windows.items():
            for cost in COSTS_BPS:
                for strategy, baseline in (('h20',False),('soxx',True)):
                    key = f'{window}::{strategy}::{int(cost)}'
                    stage = key
                    result = cases.get(key)
                    if result is None:
                        result = replay_horizon_window(parts['frame'],selections,parts['covariance'],calendar=calendar,
                            cost_bps=cost,horizon_sessions=20,robust_radius=1.,initial_cash=100000.,benchmark_only=baseline)
                    cases[key] = result
                    result = native_verify_window(parts['frame'],result,provider_uri=_providers(root,plan)['execution'],
                                                  cost_bps=cost,initial_cash=100000.)
                    cases[key] = result
                    window_paths[(window,strategy,cost)] = result['path']
                    print(f'H20 native passed: {key}, {len(result["path"])} sessions',flush=True)
                    _archive(uri,recorder_id,plan['experiment'],objects={'case_results.pkl':cases})
                    attempt.write_text(json.dumps(dict(status='running',stage=key,fit_calls=fit_calls,
                        native_cases=len(window_paths),recorder_id=recorder_id,recorder_uri=uri),indent=2)+'\n')
        audits = {key:result['native']['actual_position_audit'] for key,result in cases.items()}
        require_all_native_passed(audits,expected)
        stage = 'statistics'
        from src.workflow.benchmark_evidence import evaluate_registered_paths
        from src.analysis.excess_resolution import apply_excess_resolution
        full = _assemble_registered_paths(window_paths,plan)
        dev = _assemble_registered_paths(window_paths,plan,dev_only=True)
        registered = dict(development_years=plan['development_years'],costs_bps=list(COSTS_BPS),
            as_of=plan['source']['as_of'],variants=[{'name':'h20'}],candidate_budget=1,gates=plan['gates'])
        tables,evidence = evaluate_registered_paths(full,plan=registered,calendar=calendar,candidate_controls={'h20':'soxx'})
        resolved = apply_excess_resolution(evidence['h20'],{c:dev[('h20',c)] for c in COSTS_BPS},
            {c:dev[('soxx',c)] for c in COSTS_BPS},log_excess_floor=plan['numerical_log_excess_floor'])
        for filename,key in (('gates.csv','gates'),('rolling_summary.csv','rolling_summary'),('bootstrap_analysis.csv','bootstrap')):
            tables['raw_'+filename] = tables[filename].copy()
            tables[filename] = resolved[key].assign(strategy='h20')
        tables['resolved_annual_metrics.csv'] = resolved['annual'].assign(strategy='h20')
        tables['resolved_metrics.csv'] = resolved['metrics'].assign(strategy='h20')
        tables['native_audit.csv'] = pd.DataFrame([dict(case=key,**audit) for key,audit in audits.items()])
        risk_rows = []
        for key,result in cases.items():
            for metric,row in result['native']['risk_analysis_next_open'].iterrows():
                risk_rows.append(dict(case=key,mark='native_next_open_net',metric=str(metric),value=float(row['risk'])))
        tables['risk_analysis.csv'] = pd.DataFrame(risk_rows)
        freeze_end = _source_freeze(root,card,plan)
        _require(freeze_start == freeze_end,'source changed during experiment')
        fingerprint = lambda x:hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
        raw_gate, gate = bool(evidence['h20']['historical_gate_pass']),bool(resolved['historical_gate_pass'])
        diagnostic = tables['backtest_benchmark_comparison.csv'].query("window=='reused_diagnostic' and strategy=='h20'")
        diagnostic_pass = bool((diagnostic.log_excess_sum > plan['numerical_log_excess_floor']).all()
            and (diagnostic.max_drawdown_pct >= diagnostic.benchmark_max_drawdown_pct - 1e-10).all()
            and (diagnostic.cvar5_pct >= diagnostic.benchmark_cvar5_pct - 1e-10).all())
        summary = dict(experiment=plan['experiment'],schema_version=SCHEMA_VERSION,status='complete_not_promoted',
            promotion_decision='not_promoted',promotion_eligible=False,scientific_promotion=False,
            raw_historical_gate_pass=raw_gate,historical_gate_pass=gate,diagnostic_gate_pass=diagnostic_pass,
            route_decision='historical_pass_requires_new_forward' if gate and diagnostic_pass else 'end_h20_route_no_further_search',
            fit_calls=fit_calls,new_fit_calls=0 if recovery else fit_calls,inherited_model_recorder_id=plan.get('recovery',{}).get('recorder_id'),fixed_fit_budget=55,native_cases=len(cases),costs_bps=list(COSTS_BPS),
            source=dict(provider_uri=plan['source']['providers']['published'],execution_provider_uri=plan['source']['providers']['execution'],
                        region=plan['source']['region'],as_of=plan['source']['as_of']),
            source_identity_sha256_before=fingerprint(freeze_start),source_identity_sha256_after=fingerprint(freeze_end),
            execution_plan_sha256=hashlib.sha256(Path(execution_plan_path).read_bytes()).hexdigest(),
            native_actual_position_audit=audits,window_audit=window_audit,
            scientific_scope=plan['scientific_scope'],recorder_id=recorder_id,recorder_uri=uri,recorder_local_dir=local_dir,
            recorder_status='FINISHED',artifact_policy=ARTIFACT_POLICY)
        _archive(uri,recorder_id,plan['experiment'],objects={'case_results.pkl':cases,'selections.pkl':selections,
            'window_paths.pkl':window_paths,'raw_evidence.pkl':evidence['h20'],'resolved_evidence.pkl':resolved,
            'final_summary.pkl':summary},tables=tables)
        if write:
            _write_card(card,summary,selections,tables)
        attempt.write_text(json.dumps(dict(status='complete',recorder_id=recorder_id,recorder_uri=uri),indent=2)+'\n')
        return summary
    except Exception as exc:
        _archive(uri,recorder_id,plan['experiment'],objects={'partial_selections.pkl':selections,'partial_case_results.pkl':cases,
            'partial_window_paths.pkl':window_paths,'failure.pkl':dict(stage=stage,error=repr(exc),fit_calls=fit_calls)})
        attempt.write_text(json.dumps(dict(status='blocked',stage=stage,error=repr(exc),recorder_id=recorder_id,
                                           recorder_uri=uri,fit_calls=fit_calls),indent=2)+'\n')
        raise


def _write_card(card,summary,selections,tables):
    for name,table in tables.items():
        table.to_csv(card/name,index=False)
    pd.DataFrame([dict(year=year,selected=result['selected_name']) for year,result in selections.items()]).to_csv(card/'model_selection.csv',index=False)
    pd.concat([result['candidate_scores'].assign(year=year) for year,result in selections.items()],ignore_index=True).to_csv(card/'selection_scores.csv',index=False)
    entries = []
    artifact_root = Path(summary['recorder_local_dir'])/'artifacts'
    for path in sorted(artifact_root.rglob('*')):
        if path.is_file() and path.name not in ('code_diff.txt','code_cached.txt','code_status.txt'):
            entries.append(dict(path=str(path.relative_to(card)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size))
    (card/'verification_manifest.json').write_text(json.dumps(dict(recorder_id=summary['recorder_id'],entries=entries),indent=2)+'\n')
    (card/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    (card/'workflow.yaml').write_text(yaml.safe_dump(dict(experiment=summary['experiment'],source=summary['source'],
        recorder_id=summary['recorder_id'],recorder_uri=summary['recorder_uri'],artifact_policy=ARTIFACT_POLICY),sort_keys=False))
    (card/'selected_model.json').write_text(json.dumps(dict(model='native_nested_annual_prior_ridge',horizon_sessions=20,
        selected_by_year={str(y):r['selected_name'] for y,r in selections.items()},promotion_decision='not_promoted'),indent=2)+'\n')
    lines=['# H20 匹配期限实验','',f'实际执行 {summary["fit_calls"]} 次拟合和 {summary["native_cases"]} 个原生回放；现金、数量、费用和三个资产权重对账通过。','']
    for row in tables['backtest_benchmark_comparison.csv'].query("strategy=='h20'").itertuples():
        lines.append(f'- {row.window}/{row.cost_bps:g}bp：收益 {row.return_pct:.2f}%，相对 SOXX 终值 {row.relative_wealth:.4f}，MaxDD {row.max_drawdown_pct:.2f}%，CVaR5 {row.cvar5_pct:.2f}%。')
    lines += ['',f'历史门：{summary["historical_gate_pass"]}；诊断门：{summary["diagnostic_gate_pass"]}。不晋升，路线决定：{summary["route_decision"]}。',
        '这仍是已复用历史；H5 经济参考未核验，不能把差异归因于单独期限。固定 HAC5 不覆盖所有 H20 重叠标签依赖；历史开盘是报价成交代理，并非真实 09:31 成交。',
        '若历史及诊断门通过，需新的冻结身份及至少 60 个独立合格前向日；否则停止 H20，不重搜期限、alpha 或阈值。',f'Recorder `{summary["recorder_id"]}`，FINISHED。']
    (card/'conclusion.md').write_text('\n'.join(lines)+'\n')
