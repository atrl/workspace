from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import QlibProviderConfig, load_qlib_feature_frame
from src.workflow.intraday_recovery_direct_allocation import (
    IntradayRecoveryDirectAllocationSpec,
    _build_teacher_training_frame,
    _load_source_frame,
    _risk_metrics,
    _simulate_policy,
    project_legal_weights,
)


EXPERIMENT = "stage1_open15m_defensive_sleeve_v1"
SOURCE_EXPERIMENT = "intraday_on_policy_staged_validation_v1"
SOURCE_RECORDER_ID = "f6b09330175e41719886d58ee2d11bec"

FEATURE_COLUMNS = (
    "first_ret_soxl",
    "first_ret_soxx",
    "first_dislocation",
    "first_range_soxl",
    "first_range_soxx",
    "first_close_pos_soxl",
    "first_close_pos_soxx",
    "gap_soxl",
    "gap_soxx",
    "gap_dislocation",
    "first_rvol20_soxl",
    "first_rvol20_soxx",
    "first_volume_ratio",
    "open_soxl_weight",
    "open_soxx_weight",
    "open_cash_weight",
    "current_beta_cap",
)


@dataclass(frozen=True)
class Stage1Open15mDefensiveSleeveSpec:
    root: Path
    experiment: str = EXPERIMENT
    source_experiment: str = SOURCE_EXPERIMENT
    source_recorder_id: str = SOURCE_RECORDER_ID
    development_years: tuple[int, ...] = (2020, 2021, 2022)
    outer_years: tuple[int, ...] = (2023, 2024, 2025, 2026)
    first_action_year: int = 2020
    teacher_beta: float = 0.20
    transaction_cost_bps: float = 10.0
    stress_cost_bps: float = 25.0

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def intraday_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl_intraday_15min"

    @property
    def source_card_dir(self) -> Path:
        return self.root / "research" / self.source_experiment


def run_stage1_open15m_defensive_sleeve_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    return _run_card(Stage1Open15mDefensiveSleeveSpec(root=_resolve_root(root)))


def _resolve_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "SPEC.md").exists():
            return current
        current = current.parent
    raise ValueError("could not resolve repository root")


def _source_selection(spec: Stage1Open15mDefensiveSleeveSpec) -> dict[str, Any]:
    summary = json.loads((spec.source_card_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("recorder_id") != spec.source_recorder_id:
        raise ValueError("blocked: frozen Stage1 recorder drifted")
    if not str(summary.get("decision", "")).startswith("promoted_research_candidate_stage1"):
        raise ValueError("blocked: source Stage1 is not promoted")
    if bool(summary["selected_model"].get("stage2_enabled")):
        raise ValueError("blocked: comparison requires Stage2 disabled")
    return dict(summary["selected_model"])


def _stage1_replay(
    spec: Stage1Open15mDefensiveSleeveSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    recovery_spec = IntradayRecoveryDirectAllocationSpec(root=spec.root)
    frame, source_audit = _load_source_frame(recovery_spec)
    teacher = _build_teacher_training_frame(
        frame,
        spec=recovery_spec,
        teacher_beta=recovery_spec.teacher_recovery_beta,
    )
    selected = _source_selection(spec)
    replay = _simulate_policy(
        frame.reset_index(),
        spec=recovery_spec,
        teacher_frame=teacher,
        c_value=float(selected["logistic_c"]),
        threshold=float(selected["probability_threshold"]),
        action_beta=float(selected["action_beta"]),
        first_action_year=spec.first_action_year,
        model_cache={},
        cost_bps=spec.transaction_cost_bps,
    )
    replay["trade_date"] = pd.to_datetime(replay["trade_date"])
    return replay, source_audit


def _opening_features(spec: Stage1Open15mDefensiveSleeveSpec) -> pd.DataFrame:
    qframe = load_qlib_feature_frame(
        QlibProviderConfig(
            provider_uri=spec.intraday_provider_uri,
            market="all",
            region="us",
            freq="15min",
            kernels=1,
        ),
        instruments=["SOXL", "SOXX"],
        fields=["$open", "$high", "$low", "$close", "$volume", "$bar_count"],
        start="2018-12-31 09:45:00",
        end="2099-12-31 16:00:00",
    ).reset_index()
    qframe["datetime"] = pd.to_datetime(qframe["datetime"])
    qframe["date"] = qframe["datetime"].dt.normalize()
    qframe = qframe.sort_values(["instrument", "date", "datetime"])
    regular = qframe[
        qframe["datetime"].dt.time.between(
            pd.Timestamp("09:45").time(), pd.Timestamp("16:00").time()
        )
    ].copy()
    first = regular.groupby(["instrument", "date"], group_keys=False).head(1).copy()
    first["first_ret"] = first["$close"] / first["$open"] - 1.0
    first["first_range"] = (first["$high"] - first["$low"]) / first["$open"]
    spread = (first["$high"] - first["$low"]).replace(0.0, np.nan)
    first["first_close_pos"] = ((first["$close"] - first["$low"]) / spread).fillna(0.5)
    first["first_volume_median20"] = first.groupby("instrument")["$volume"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=10).median()
    )
    first["first_rvol20"] = first["$volume"] / first["first_volume_median20"]

    first4 = regular.groupby(["instrument", "date"], group_keys=False).head(4)
    first4_grouped = first4.groupby(["instrument", "date"]).agg(
        first_hour_open=("$open", "first"),
        first_hour_close=("$close", "last"),
    )
    first4_grouped["first_hour_ret"] = (
        first4_grouped["first_hour_close"] / first4_grouped["first_hour_open"] - 1.0
    )
    last3 = first4_grouped[["first_hour_ret"]].copy()

    daily = regular.groupby(["instrument", "date"]).agg(
        day_open=("$open", "first"),
        day_close=("$close", "last"),
    )
    daily["previous_close"] = daily.groupby(level="instrument")["day_close"].shift(1)
    daily["gap"] = daily["day_open"] / daily["previous_close"] - 1.0

    compact = first.set_index(["instrument", "date"])[
        ["first_ret", "first_range", "first_close_pos", "first_rvol20", "$volume"]
    ].join(last3).join(daily[["gap"]])
    compact["next45_ret"] = (
        (1.0 + compact["first_hour_ret"]) / (1.0 + compact["first_ret"]) - 1.0
    )
    wide = compact.reset_index().pivot(
        index="date",
        columns="instrument",
        values=[
            "first_ret",
            "first_range",
            "first_close_pos",
            "first_rvol20",
            "$volume",
            "gap",
            "first_hour_ret",
            "next45_ret",
        ],
    )
    wide.columns = [
        f"{field.lstrip('$')}_{str(instrument).lower()}"
        for field, instrument in wide.columns
    ]
    wide["first_dislocation"] = wide["first_ret_soxl"] - 3.0 * wide["first_ret_soxx"]
    wide["gap_dislocation"] = wide["gap_soxl"] - 3.0 * wide["gap_soxx"]
    wide["first_volume_ratio"] = np.log1p(wide["volume_soxl"]) - np.log1p(
        wide["volume_soxx"]
    )
    return wide.sort_index()


def _comparison_frame(
    spec: Stage1Open15mDefensiveSleeveSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    replay, source_audit = _stage1_replay(spec)
    opening = _opening_features(spec)
    frame = replay.merge(
        opening,
        left_on="trade_date",
        right_index=True,
        how="inner",
        validate="many_to_one",
    )
    frame["year"] = frame["trade_date"].dt.year
    frame["baseline_beta"] = (
        3.0 * frame["open_soxl_weight"] + frame["open_soxx_weight"]
    )
    return frame.sort_values("trade_date").reset_index(drop=True), source_audit


def _overlay_delta(
    row: pd.Series,
    *,
    action_beta: float,
    cost_bps: float,
) -> tuple[float, float, float]:
    open_weights = np.array(
        [row.open_soxl_weight, row.open_soxx_weight, row.open_cash_weight],
        dtype="float64",
    )
    close_weights = np.array([row.soxl_weight, row.soxx_weight, row.cash_weight])
    cut_weights = project_legal_weights(
        float(row.open_soxl_weight),
        float(row.open_soxx_weight),
        action_beta,
    )
    first = np.array([row.first_ret_soxl, row.first_ret_soxx, 0.0], dtype="float64")
    next45 = np.array([row.next45_ret_soxl, row.next45_ret_soxx, 0.0], dtype="float64")
    hour = np.array(
        [row.first_hour_ret_soxl, row.first_hour_ret_soxx, 0.0], dtype="float64"
    )

    baseline_value_15 = 1.0 + open_weights @ first
    baseline_drift_15 = open_weights * (1.0 + first) / baseline_value_15
    overlay_value_60 = baseline_value_15 * (1.0 + cut_weights @ next45)
    overlay_drift_60 = cut_weights * (1.0 + next45) / (1.0 + cut_weights @ next45)

    baseline_value_60 = 1.0 + open_weights @ hour
    baseline_drift_60 = open_weights * (1.0 + hour) / baseline_value_60
    post_factor = (1.0 + float(row.gross_return)) / baseline_value_60
    gross_delta = (overlay_value_60 - baseline_value_60) * post_factor

    overlay_turnover = np.abs(cut_weights[:2] - baseline_drift_15[:2]).sum()
    overlay_turnover += np.abs(close_weights[:2] - overlay_drift_60[:2]).sum()
    baseline_cut_turnover = (
        np.abs(close_weights[:2] - baseline_drift_60[:2]).sum()
        if bool(row.recovery_action)
        else 0.0
    )
    incremental_turnover = max(0.0, overlay_turnover - baseline_cut_turnover)
    net_delta = gross_delta - incremental_turnover * cost_bps / 10_000.0
    return float(net_delta), float(gross_delta), float(incremental_turnover)


def _teacher_frame(
    frame: pd.DataFrame, spec: Stage1Open15mDefensiveSleeveSpec
) -> pd.DataFrame:
    teacher = frame.copy()
    deltas = teacher.apply(
        lambda row: _overlay_delta(
            row,
            action_beta=spec.teacher_beta,
            cost_bps=spec.transaction_cost_bps,
        ),
        axis=1,
        result_type="expand",
    )
    deltas.columns = ["teacher_net_delta", "teacher_gross_delta", "teacher_turnover"]
    teacher = pd.concat([teacher, deltas], axis=1)
    teacher["teacher_target"] = teacher["teacher_net_delta"].gt(0.0).astype(int)
    return teacher


def _build_dataset(teacher: pd.DataFrame):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    indexed = teacher.set_index("trade_date").sort_index()
    static = pd.concat(
        {
            "feature": indexed.loc[:, FEATURE_COLUMNS],
            "label": indexed.loc[:, ["teacher_target"]],
        },
        axis=1,
    )
    static.index = pd.MultiIndex.from_arrays(
        [static.index, np.full(len(static), "STAGE1_OPEN15M")],
        names=["datetime", "instrument"],
    )
    segments = {
        f"train_{year}": ("2019-01-01", f"{year - 1}-12-31")
        for year in range(2020, 2027)
    }
    handler = DataHandlerLP(
        instruments=None,
        start_time=static.index.get_level_values("datetime").min(),
        end_time=static.index.get_level_values("datetime").max(),
        data_loader=StaticDataLoader(config=static),
    )
    dataset = DatasetH(handler=handler, segments=segments)
    audit = {
        "api": ["DatasetH", "DataHandlerLP", "StaticDataLoader"],
        "segments": {name: list(bounds) for name, bounds in segments.items()},
        "segment_rows": {
            name: int(len(dataset.prepare(name))) for name in segments
        },
        "label": "teacher_net_delta_after_round_trip_cost_gt_0",
    }
    return dataset, audit


def _fit_model(dataset, *, year: int, c_value: float):
    prepared = dataset.prepare(f"train_{year}", col_set=["feature", "label"])
    prepared = prepared.droplevel("instrument").sort_index()
    prepared.columns = [str(column[-1]) for column in prepared.columns]
    training = prepared[prepared["teacher_target"].notna()]
    if len(training) < 100 or training["teacher_target"].nunique() < 2:
        return None
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=c_value, class_weight="balanced", max_iter=2000),
    )
    model.fit(training.loc[:, FEATURE_COLUMNS], training["teacher_target"])
    return model


def _simulate(
    teacher: pd.DataFrame,
    *,
    dataset,
    c_value: float,
    threshold: float,
    action_beta: float,
    cost_bps: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    models: dict[int, Any] = {}
    for _, row in teacher.iterrows():
        year = int(row.year)
        model = models.get(year)
        if year not in models:
            model = (
                _fit_model(dataset, year=year, c_value=c_value)
                if year >= 2020
                else None
            )
            models[year] = model
        probability = (
            float(model.predict_proba(pd.DataFrame([row.loc[list(FEATURE_COLUMNS)]]))[0, 1])
            if model is not None
            else np.nan
        )
        action = bool(
            year >= 2020
            and pd.notna(probability)
            and probability >= threshold
            and float(row.baseline_beta) > action_beta + 1e-12
        )
        delta, gross_delta, incremental_turnover = (
            _overlay_delta(row, action_beta=action_beta, cost_bps=cost_bps)
            if action
            else (0.0, 0.0, 0.0)
        )
        rows.append(
            {
                "signal_date": row.signal_date,
                "trade_date": row.trade_date,
                "year": year,
                "segment": row.segment,
                "baseline_return": float(row.strategy_return),
                "sleeve_return": float(row.strategy_return) + delta,
                "incremental_return": delta,
                "incremental_gross_return": gross_delta,
                "incremental_turnover": incremental_turnover,
                "open15_probability": probability,
                "open15_action": action,
                "action_beta": action_beta,
                "cost_bps": cost_bps,
            }
        )
    return pd.DataFrame(rows)


def _candidate_metrics(replay: pd.DataFrame, years: tuple[int, ...]) -> dict[str, float]:
    section = replay[replay["year"].isin(years)]
    baseline = _risk_metrics(section["baseline_return"].to_numpy())
    sleeve = _risk_metrics(section["sleeve_return"].to_numpy())
    return {
        "return_pct": sleeve["return_pct"],
        "max_drawdown_pct": sleeve["max_drawdown_pct"],
        "worst_day_pct": sleeve["worst_day_pct"],
        "cvar5_pct": sleeve["cvar5_pct"],
        "worst_rolling_5d_pct": sleeve["worst_rolling_5d_pct"],
        "baseline_return_pct": baseline["return_pct"],
        "baseline_max_drawdown_pct": baseline["max_drawdown_pct"],
        "return_delta_pct_points": sleeve["return_pct"] - baseline["return_pct"],
        "max_drawdown_delta_pct_points": (
            sleeve["max_drawdown_pct"] - baseline["max_drawdown_pct"]
        ),
        "action_count": int(section["open15_action"].sum()),
        "positive_action_rate": float(
            section.loc[section["open15_action"], "incremental_return"].gt(0.0).mean()
        )
        if section["open15_action"].any()
        else 0.0,
    }


def _select_candidate(
    teacher: pd.DataFrame, dataset, spec: Stage1Open15mDefensiveSleeveSpec
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for c_value in (0.03, 0.10, 0.30):
        for threshold in (0.55, 0.60, 0.65, 0.70):
            for action_beta in (0.20, 0.40, 0.60):
                replay = _simulate(
                    teacher,
                    dataset=dataset,
                    c_value=c_value,
                    threshold=threshold,
                    action_beta=action_beta,
                    cost_bps=spec.transaction_cost_bps,
                )
                metrics = _candidate_metrics(replay, spec.development_years)
                rows.append(
                    {
                        "segment": "development_walk_forward",
                        "selection_uses_outer_test": False,
                        "logistic_c": c_value,
                        "probability_threshold": threshold,
                        "action_beta": action_beta,
                        **metrics,
                    }
                )
    table = pd.DataFrame(rows)
    eligible = table[
        table["action_count"].ge(8)
        & table["max_drawdown_delta_pct_points"].ge(-1.0)
    ].copy()
    if eligible.empty:
        eligible = table.copy()
    selected = eligible.sort_values(
        ["return_pct", "max_drawdown_pct", "positive_action_rate"],
        ascending=False,
    ).iloc[0].to_dict()
    return selected, table


def _record(
    spec: Stage1Open15mDefensiveSleeveSpec,
    *,
    selected: dict[str, Any],
    comparison: pd.DataFrame,
    model_selection: pd.DataFrame,
    feature_audit: pd.DataFrame,
) -> dict[str, str]:
    from qlib.workflow import R

    recorder_uri = spec.card_dir / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(prefix=f"{spec.experiment}_") as tmp:
        tmp_dir = Path(tmp)
        comparison.to_csv(tmp_dir / "comparison_daily.csv", index=False)
        model_selection.to_csv(tmp_dir / "model_selection_full.csv", index=False)
        feature_audit.to_csv(tmp_dir / "feature_audit.csv", index=False)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                source_experiment=spec.source_experiment,
                source_recorder_id=spec.source_recorder_id,
                decision_time="09:45_ET_after_first_15m_bar_close",
                restore_time="10:30_ET_before_existing_stage1_recovery",
                selection_uses_outer_test="false",
            )
            R.log_metrics(
                development_return_delta_pct_points=float(
                    selected["return_delta_pct_points"]
                ),
                development_action_count=float(selected["action_count"]),
            )
            for filename in (
                "comparison_daily.csv",
                "model_selection_full.csv",
                "feature_audit.csv",
            ):
                recorder.log_artifact(
                    (tmp_dir / filename).as_posix(), artifact_path=spec.experiment
                )
            result = {
                "recorder_id": recorder.id,
                "recorder_uri": recorder_uri.resolve().as_posix(),
                "recorder_local_dir": Path(recorder.get_local_dir()).resolve().as_posix(),
            }
    artifacts = Path(result["recorder_local_dir"]) / "artifacts"
    for filename in ("code_cached.txt", "code_diff.txt", "code_status.txt"):
        (artifacts / filename).unlink(missing_ok=True)
    return result


def _run_card(spec: Stage1Open15mDefensiveSleeveSpec) -> dict[str, Any]:
    spec.card_dir.mkdir(parents=True, exist_ok=True)
    frame, source_audit = _comparison_frame(spec)
    teacher = _teacher_frame(frame, spec)
    dataset, dataset_audit = _build_dataset(teacher)
    selected, model_selection = _select_candidate(teacher, dataset, spec)
    comparison = _simulate(
        teacher,
        dataset=dataset,
        c_value=float(selected["logistic_c"]),
        threshold=float(selected["probability_threshold"]),
        action_beta=float(selected["action_beta"]),
        cost_bps=spec.transaction_cost_bps,
    )
    stress = _simulate(
        teacher,
        dataset=dataset,
        c_value=float(selected["logistic_c"]),
        threshold=float(selected["probability_threshold"]),
        action_beta=float(selected["action_beta"]),
        cost_bps=spec.stress_cost_bps,
    )
    development = _candidate_metrics(comparison, spec.development_years)
    outer = _candidate_metrics(comparison, spec.outer_years)
    outer_stress = _candidate_metrics(stress, spec.outer_years)
    decision = (
        "candidate_for_forward_paper_validation_not_promoted"
        if development["return_delta_pct_points"] > 0
        and development["action_count"] >= 8
        else "not_promoted_development_gate_failed"
    )
    feature_audit = teacher[
        ["signal_date", "trade_date", "year", *FEATURE_COLUMNS, "teacher_target"]
    ]
    recorder = _record(
        spec,
        selected=selected,
        comparison=comparison,
        model_selection=model_selection,
        feature_audit=feature_audit,
    )

    selected_payload = {
        "selection_uses_outer_test": False,
        "selected_from_segment": "development_walk_forward_2020_2022",
        "outer_role": "reused_diagnostic_only_not_model_selection",
        "teacher_beta": spec.teacher_beta,
        "logistic_c": float(selected["logistic_c"]),
        "probability_threshold": float(selected["probability_threshold"]),
        "action_beta": float(selected["action_beta"]),
    }
    summary = {
        "experiment": spec.experiment,
        "card_type": "stage1_open15m_defensive_sleeve_research_card",
        "decision": decision,
        "source_stage1": {
            "experiment": spec.source_experiment,
            "recorder_id": spec.source_recorder_id,
            "frozen": True,
            "stage2_enabled": False,
        },
        "provider_uri": spec.intraday_provider_uri.resolve().as_posix(),
        "qlib_api": [
            "qlib.init",
            "qlib.data.D.features",
            "DatasetH",
            "DataHandlerLP",
            "qlib.workflow.R",
        ],
        "dataset_audit": dataset_audit,
        "qlib_docs_reference": [
            "https://qlib.readthedocs.io/en/latest/component/data.html",
            "https://qlib.readthedocs.io/en/stable/component/recorder.html",
        ],
        "timing": {
            "signal": "09:45_ET_after_first_15m_bar_close",
            "temporary_defensive_window": "09:45_to_10:30_ET",
            "restore": "10:30_ET_before_existing_stage1_recovery",
            "overnight_stage1_weights_changed": False,
        },
        "features": {
            "columns": list(FEATURE_COLUMNS),
            "feature_count": len(FEATURE_COLUMNS),
            "path_aware": True,
            "rvol_policy": "same_09_30_09_45_slot_trailing20_median_shift1",
        },
        "selected_model": selected_payload,
        "development_metrics": development,
        "outer_diagnostic_metrics": outer,
        "outer_25bps_diagnostic_metrics": outer_stress,
        "source_audit": source_audit,
        **recorder,
        "sidecars": {
            "selected_model": "selected_model.json",
            "model_selection": "model_selection.csv",
            "comparison": "backtest_benchmark_comparison.csv",
            "analysis_summary": "analysis_summary.json",
            "conclusion": "conclusion.md",
        },
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": [
                f"artifacts/{spec.experiment}/comparison_daily.csv",
                f"artifacts/{spec.experiment}/model_selection_full.csv",
                f"artifacts/{spec.experiment}/feature_audit.csv",
            ],
        },
    }
    comparison_sidecar = pd.DataFrame(
        [
            {"segment": "development_2020_2022", **development},
            {"segment": "outer_reused_diagnostic_2023_2026", **outer},
            {"segment": "outer_reused_diagnostic_25bps", **outer_stress},
        ]
    )
    model_selection.sort_values(
        ["return_pct", "max_drawdown_pct"], ascending=False
    ).head(12).to_csv(spec.card_dir / "model_selection.csv", index=False)
    comparison_sidecar.to_csv(
        spec.card_dir / "backtest_benchmark_comparison.csv", index=False
    )
    (spec.card_dir / "selected_model.json").write_text(
        json.dumps(selected_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (spec.card_dir / "analysis_summary.json").write_text(
        json.dumps(
            {
                "decision": decision,
                "development": development,
                "outer_diagnostic": outer,
                "outer_25bps_diagnostic": outer_stress,
                "interpretation": (
                    "The sleeve changes only the 09:45-10:30 intraday path; "
                    "outer years are reused diagnostics and cannot promote it."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (spec.card_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    workflow = {
        "experiment": spec.experiment,
        "provider_uri": spec.intraday_provider_uri.resolve().as_posix(),
        "source_stage1_recorder_id": spec.source_recorder_id,
        "selection_uses_outer_test": False,
        "timing": summary["timing"],
        "features": summary["features"],
        "artifact_policy": summary["artifact_policy"],
    }
    (spec.card_dir / "workflow.yaml").write_text(
        yaml.safe_dump(workflow, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    conclusion = f"""# 中文结论

- 做了什么：冻结现有 Stage1 与 10:30 恢复逻辑，只增加 09:45–10:30 的临时防守 sleeve；09:45 使用第一根完整 15m 数据，10:30 无条件回到原 Stage1 路径。
- 因子：首根 15m 收益/振幅/区间位置、隔夜缺口、SOXL-3xSOXX 偏离、同时间槽 trailing-20 RVOL、原 Stage1 开盘权重与 beta cap。
- 开发期 2020–2022：增量收益={development['return_delta_pct_points']:.4f} 个百分点，动作={development['action_count']} 次，MaxDD 变化={development['max_drawdown_delta_pct_points']:.4f} 个百分点。
- 复用外层 2023–2026（只诊断）：增量收益={outer['return_delta_pct_points']:.4f} 个百分点，动作={outer['action_count']} 次，正增量动作比例={outer['positive_action_rate']:.2%}。
- 25bp 外层压力：增量收益={outer_stress['return_delta_pct_points']:.4f} 个百分点，MaxDD 变化={outer_stress['max_drawdown_delta_pct_points']:.4f} 个百分点。
- 决策：{decision}。外层年份已经反复查看，不用于选模，也不能据此替换生产 Stage1；只有新的连续前向模拟可以晋升。
- qlib recorder：{recorder['recorder_id']}
- 不构成买卖建议。
"""
    (spec.card_dir / "conclusion.md").write_text(conclusion, encoding="utf-8")
    return summary
