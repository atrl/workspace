from __future__ import annotations

import json
import pickle
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd

from src.data import QlibProviderConfig, init_qlib_provider
from src.workflow.auxiliary_3x_training_data_allocation import (
    NORMALIZATION_SUFFIXES,
    _attach_generic_targets,
    _prepare_panel,
    _training_end,
)
from src.workflow.cross_pair_auxiliary_training import (
    CrossPairCandidate,
    CrossPairTrainingSpec,
    _fit_candidate,
)
from src.workflow.homologous_3x_1x_cash_generalization import (
    _build_dataset,
    _evaluate_pair,
    _pair_market,
)
from src.workflow.multi_3x_etf_normalized_nested_wf import (
    _attach_normalized_features,
    _load_pair_frame,
    _model_feature_columns,
)
from src.workflow.soxl_objective_feasibility_audit import _json_ready
from src.workflow.trainable_regime_policy_nested_wf import (
    _remove_qlib_code_snapshot_artifacts,
)


EXPERIMENT = "standalone_tqqq_qqq_cash_training_v1"
PAIR_NAME = "TQQQ_QQQ_CASH"
PAIR_DEFINITION = {"leveraged": "TQQQ", "underlying": "QQQ"}
FORBIDDEN_TRAINING_ASSETS = ("SOXL", "SOXX")
TURNOVER_PENALTIES = (0.0, 0.005, 0.02, 0.05, 0.10)
CANDIDATES = tuple(
    CrossPairCandidate(
        name=(
            "tqqq_own_mse"
            if penalty == 0.0
            else f"tqqq_own_turnover_{str(penalty).replace('.', 'p')}"
        ),
        auxiliary_weight=0.0,
        turnover_penalty=penalty,
    )
    for penalty in TURNOVER_PENALTIES
)
RECORDER_ONLY_ARTIFACTS = (
    f"artifacts/{EXPERIMENT}/models/",
    f"artifacts/{EXPERIMENT}/predictions.csv",
    f"artifacts/{EXPERIMENT}/positions.csv",
    f"artifacts/{EXPERIMENT}/strategy_returns.csv",
    f"artifacts/{EXPERIMENT}/training_diagnostics.csv",
)


@dataclass(frozen=True)
class StandaloneTqqqSpec:
    root: Path
    experiment: str = EXPERIMENT
    train_start: str = "2011-01-03"
    development_years: tuple[int, ...] = tuple(range(2016, 2023))
    development_end: str = "2022-12-30"
    outer: tuple[str, str] = ("2023-01-03", "2026-07-24")
    purge_rows: int = 20
    hidden_units: int = 16
    epochs: int = 300
    learning_rate: float = 0.003
    weight_decay: float = 0.05
    maximum_leveraged_weight: float = 0.30
    minimum_cash_weight: float = 0.15
    costs_bps: tuple[int, ...] = (10, 25)
    seed: int = 113
    as_of: str = "2026-07-24"

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "tqqq"

    @property
    def model_spec(self) -> CrossPairTrainingSpec:
        return CrossPairTrainingSpec(
            root=self.root,
            experiment=self.experiment,
            train_start=self.train_start,
            development_years=self.development_years,
            development_end=self.development_end,
            outer=self.outer,
            purge_rows=self.purge_rows,
            hidden_units=self.hidden_units,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            maximum_leveraged_weight=self.maximum_leveraged_weight,
            minimum_cash_weight=self.minimum_cash_weight,
            costs_bps=self.costs_bps,
            seed=self.seed,
            as_of=self.as_of,
        )


def run_standalone_tqqq_qqq_cash_training_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    repo_root = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[2]
    )
    return _run_card(StandaloneTqqqSpec(root=repo_root))


def _load_tqqq_only_panel(
    spec: StandaloneTqqqSpec,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    config = QlibProviderConfig(
        provider_uri=spec.provider_uri,
        market="all",
        region="us",
        kernels=1,
    )
    provider_meta = init_qlib_provider(config)
    start = max(
        pd.Timestamp(spec.train_start),
        pd.Timestamp(provider_meta["calendar_start"]),
    ).date().isoformat()
    end = min(
        pd.Timestamp(spec.as_of),
        pd.Timestamp(provider_meta["calendar_end"]),
    ).date().isoformat()
    raw = _load_pair_frame(
        config,
        etf="TQQQ",
        underlying="QQQ",
        start=start,
        end=end,
    )
    if set(raw["instrument"].unique()) != {"TQQQ"}:
        raise ValueError("blocked: non-TQQQ instrument in standalone raw data")
    normalized = _attach_normalized_features(raw)
    panel = _attach_generic_targets(normalized)
    features = [
        column
        for column in _model_feature_columns(panel)
        if column.endswith(NORMALIZATION_SUFFIXES)
    ]
    features = list(dict.fromkeys(features))
    if not features:
        raise ValueError("blocked: no TQQQ-only normalized features")
    forbidden_identity = [
        column
        for column in features
        if any(
            token in column.lower()
            for token in ("instrument", "asset", "symbol")
        )
    ]
    if forbidden_identity:
        raise ValueError(
            f"blocked: identity features entered TQQQ model "
            f"{forbidden_identity}"
        )
    forbidden_rows = int(
        panel["instrument"].isin(FORBIDDEN_TRAINING_ASSETS).sum()
    )
    if forbidden_rows:
        raise ValueError("blocked: SOXL/SOXX entered TQQQ-only panel")
    audit = {
        "source": "qlib.data.D.features",
        "provider_uri": provider_meta["provider_uri"],
        "calendar_start": provider_meta["calendar_start"],
        "calendar_end": provider_meta["calendar_end"],
        "used_start": start,
        "used_end": end,
        "requested_instruments": ["TQQQ", "QQQ"],
        "model_instrument": "TQQQ",
        "underlying_instrument": "QQQ",
        "panel_instruments": sorted(
            panel["instrument"].unique().tolist()
        ),
        "forbidden_training_assets": list(FORBIDDEN_TRAINING_ASSETS),
        "forbidden_training_rows": forbidden_rows,
        "feature_count": len(features),
        "feature_identity_columns": forbidden_identity,
        "training_policy": "TQQQ_QQQ_history_only",
        "selection_uses_outer_test": False,
    }
    return panel, features, audit


def _candidate_predictions(
    *,
    candidate: CrossPairCandidate,
    dataset,
    panel: pd.DataFrame,
    features: list[str],
    spec: StandaloneTqqqSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    development_source = _prepare_panel(
        dataset, (spec.train_start, spec.development_end)
    )
    outer_source = _prepare_panel(dataset, "outer")
    if set(development_source["instrument"].unique()) != {"TQQQ"}:
        raise ValueError("blocked: non-TQQQ development rows")
    if set(outer_source["instrument"].unique()) != {"TQQQ"}:
        raise ValueError("blocked: non-TQQQ outer rows")
    development_predictions: list[pd.DataFrame] = []
    states: dict[str, Any] = {}
    diagnostics: list[dict[str, Any]] = []
    for year in spec.development_years:
        train_end = _training_end(panel, year, spec.purge_rows)
        prediction, state, fit = _fit_candidate(
            train=development_source.loc[
                development_source["datetime"].le(train_end)
            ],
            test=development_source.loc[
                development_source["datetime"].dt.year.eq(year)
            ],
            features=features,
            primary_symbol="TQQQ",
            candidate=candidate,
            spec=spec.model_spec,
            seed=spec.seed + year,
        )
        prediction["pair"] = PAIR_NAME
        prediction["candidate"] = candidate.name
        prediction["segment"] = "development_walk_forward"
        prediction["fold_year"] = year
        development_predictions.append(prediction)
        states[f"fold_{year}"] = state
        diagnostics.append(
            {
                "candidate": candidate.name,
                "segment": "development_walk_forward",
                "fold_year": year,
                "train_end": train_end.date().isoformat(),
                "training_assets": "TQQQ",
                "underlying_data": "QQQ",
                "test_assets": "TQQQ",
                "forbidden_training_rows": 0,
                "turnover_penalty": candidate.turnover_penalty,
                "selection_uses_outer_test": False,
                **fit,
            }
        )
    outer_prediction, outer_state, outer_fit = _fit_candidate(
        train=development_source,
        test=outer_source,
        features=features,
        primary_symbol="TQQQ",
        candidate=candidate,
        spec=spec.model_spec,
        seed=spec.seed,
    )
    outer_prediction["pair"] = PAIR_NAME
    outer_prediction["candidate"] = candidate.name
    outer_prediction["segment"] = "outer_diagnostic"
    outer_prediction["fold_year"] = outer_prediction.index.year
    states["outer_refit"] = outer_state
    diagnostics.append(
        {
            "candidate": candidate.name,
            "segment": "outer_diagnostic",
            "fold_year": "outer_refit",
            "train_end": spec.development_end,
            "training_assets": "TQQQ",
            "underlying_data": "QQQ",
            "test_assets": "TQQQ",
            "forbidden_training_rows": 0,
            "turnover_penalty": candidate.turnover_penalty,
            "selection_uses_outer_test": False,
            **outer_fit,
        }
    )
    return (
        pd.concat(development_predictions).sort_index(),
        outer_prediction.sort_index(),
        states,
        pd.DataFrame(diagnostics),
    )


def _evaluate_candidates(
    *,
    panel: pd.DataFrame,
    development_predictions: dict[str, pd.DataFrame],
    outer_predictions: dict[str, pd.DataFrame],
    spec: StandaloneTqqqSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    market = _pair_market(panel, "TQQQ")
    comparisons: list[pd.DataFrame] = []
    annual_frames: list[pd.DataFrame] = []
    return_frames: list[pd.DataFrame] = []
    position_frames: list[pd.DataFrame] = []
    for candidate in CANDIDATES:
        for segment, predictions in (
            (
                "development_walk_forward",
                development_predictions[candidate.name],
            ),
            ("outer_diagnostic", outer_predictions[candidate.name]),
        ):
            comparison, annual, returns, positions = _evaluate_pair(
                pair_name=PAIR_NAME,
                definition=PAIR_DEFINITION,
                market=market,
                prediction=predictions,
                segment=segment,
                costs_bps=spec.costs_bps,
            )
            for frame in (comparison, annual, returns, positions):
                frame["candidate"] = candidate.name
                frame["turnover_penalty"] = candidate.turnover_penalty
            comparisons.append(comparison)
            annual_frames.append(annual)
            return_frames.append(returns)
            position_frames.append(positions)
    return (
        pd.concat(comparisons, ignore_index=True),
        pd.concat(annual_frames, ignore_index=True),
        pd.concat(return_frames, ignore_index=True),
        pd.concat(position_frames, ignore_index=True),
    )


def _select_candidate(
    comparison: pd.DataFrame, annual: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        selector = (
            comparison["candidate"].eq(candidate.name)
            & comparison["segment"].eq("development_walk_forward")
        )
        model10 = comparison.loc[
            selector
            & comparison["strategy"].eq("model")
            & comparison["cost_bps"].eq(10)
        ].iloc[0]
        model25 = comparison.loc[
            selector
            & comparison["strategy"].eq("model")
            & comparison["cost_bps"].eq(25)
        ].iloc[0]
        matched10 = comparison.loc[
            selector
            & comparison["strategy"].eq("matched_static")
            & comparison["cost_bps"].eq(10)
        ].iloc[0]
        annual10 = annual.loc[
            annual["candidate"].eq(candidate.name)
            & annual["segment"].eq("development_walk_forward")
            & annual["strategy"].eq("model")
            & annual["cost_bps"].eq(10)
        ]
        row = {
            "candidate": candidate.name,
            "training_assets": "TQQQ,QQQ",
            "forbidden_training_assets": "SOXL,SOXX",
            "forbidden_training_rows": 0,
            "turnover_penalty": candidate.turnover_penalty,
            "ratio_10bps": float(
                model10["terminal_wealth_ratio_vs_soxx"]
            ),
            "ratio_25bps": float(
                model25["terminal_wealth_ratio_vs_soxx"]
            ),
            "dynamic_vs_matched_static": float(
                model10["terminal_wealth"]
                / matched10["terminal_wealth"]
            ),
            "max_drawdown_pct": float(model10["max_drawdown_pct"]),
            "worst_day_pct": float(model10["worst_day_pct"]),
            "cvar5_pct": float(model10["cvar5_pct"]),
            "worst_rolling_5d_pct": float(
                model10["worst_rolling_5d_pct"]
            ),
            "avg_cash_weight_pct": float(
                model10["avg_cash_weight_pct"]
            ),
            "upside_capture_pct": float(
                model10["upside_capture_pct"]
            ),
            "total_turnover": float(model10["total_turnover"]),
            "positive_relative_years": int(
                annual10["terminal_wealth_ratio_vs_soxx"].gt(1.0).sum()
            ),
            "selection_uses_outer_test": False,
        }
        row.update(
            {
                "return_gate": row["ratio_10bps"] > 1.0,
                "cost_gate": row["ratio_25bps"] > 1.0,
                "dynamic_gate": (
                    row["dynamic_vs_matched_static"] > 1.0
                ),
                "maxdd_gate": row["max_drawdown_pct"] >= -30.0,
                "worst_day_gate": row["worst_day_pct"] >= -12.0,
                "cvar_gate": row["cvar5_pct"] >= -8.0,
                "worst_5d_gate": (
                    row["worst_rolling_5d_pct"] >= -12.0
                ),
                "cash_gate": row["avg_cash_weight_pct"] >= 15.0,
                "capture_gate": row["upside_capture_pct"] >= 60.0,
                "year_gate": row["positive_relative_years"] >= 5,
            }
        )
        rows.append(row)
    selection = pd.DataFrame(rows)
    gates = [
        "return_gate",
        "cost_gate",
        "dynamic_gate",
        "maxdd_gate",
        "worst_day_gate",
        "cvar_gate",
        "worst_5d_gate",
        "cash_gate",
        "capture_gate",
        "year_gate",
    ]
    selection["gate_count"] = selection[gates].sum(axis=1)
    selection["all_gate"] = selection[gates].all(axis=1)
    selection["selection_score"] = (
        np.log(selection["ratio_10bps"].clip(lower=1e-12))
        + 0.50
        * np.log(
            selection["dynamic_vs_matched_static"].clip(lower=1e-12)
        )
        + 0.25
        * np.log(selection["ratio_25bps"].clip(lower=1e-12))
        + 0.02 * selection["positive_relative_years"]
    )
    selected_index = selection.sort_values(
        ["gate_count", "selection_score", "candidate"],
        ascending=[False, False, True],
    ).index[0]
    selection["selected"] = selection.index == selected_index
    selection["selection_policy"] = (
        "TQQQ_only_development_gate_count_then_relative_wealth"
    )
    return selection


def _outer_audit(
    *,
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
    selected_candidate: str,
) -> pd.DataFrame:
    model10 = comparison.loc[
        comparison["candidate"].eq(selected_candidate)
        & comparison["segment"].eq("outer_diagnostic")
        & comparison["strategy"].eq("model")
        & comparison["cost_bps"].eq(10)
    ].iloc[0]
    model25 = comparison.loc[
        comparison["candidate"].eq(selected_candidate)
        & comparison["segment"].eq("outer_diagnostic")
        & comparison["strategy"].eq("model")
        & comparison["cost_bps"].eq(25)
    ].iloc[0]
    matched10 = comparison.loc[
        comparison["candidate"].eq(selected_candidate)
        & comparison["segment"].eq("outer_diagnostic")
        & comparison["strategy"].eq("matched_static")
        & comparison["cost_bps"].eq(10)
    ].iloc[0]
    annual10 = annual.loc[
        annual["candidate"].eq(selected_candidate)
        & annual["segment"].eq("outer_diagnostic")
        & annual["strategy"].eq("model")
        & annual["cost_bps"].eq(10)
    ]
    relative_years = int(
        annual10["terminal_wealth_ratio_vs_soxx"].gt(1.0).sum()
    )
    risk_years = int(
        (
            annual10["max_drawdown_pct"].ge(-30.0)
            & annual10["worst_day_pct"].ge(-12.0)
            & annual10["cvar5_pct"].ge(-8.0)
            & annual10["worst_rolling_5d_pct"].ge(-12.0)
        ).sum()
    )
    checks = {
        "TQQQ_QQQ_data_only": True,
        "forbidden_training_rows_zero": True,
        "return_10bps_vs_QQQ": (
            float(model10["terminal_wealth_ratio_vs_soxx"]) > 1.0
        ),
        "return_25bps_vs_QQQ": (
            float(model25["terminal_wealth_ratio_vs_soxx"]) > 1.0
        ),
        "dynamic_vs_matched_static": (
            float(model10["terminal_wealth"])
            > float(matched10["terminal_wealth"])
        ),
        "maxdd": float(model10["max_drawdown_pct"]) >= -30.0,
        "worst_day": float(model10["worst_day_pct"]) >= -12.0,
        "cvar5": float(model10["cvar5_pct"]) >= -8.0,
        "worst_rolling_5d": (
            float(model10["worst_rolling_5d_pct"]) >= -12.0
        ),
        "average_cash": (
            float(model10["avg_cash_weight_pct"]) >= 15.0
        ),
        "annual_relative_stability": relative_years >= 3,
        "annual_risk_stability": risk_years >= 3,
    }
    return pd.DataFrame(
        [
            {
                "gate": gate,
                "required": True,
                "passed": bool(passed),
                "selected_candidate": selected_candidate,
                "outer_ratio_10bps_vs_QQQ": float(
                    model10["terminal_wealth_ratio_vs_soxx"]
                ),
                "outer_ratio_25bps_vs_QQQ": float(
                    model25["terminal_wealth_ratio_vs_soxx"]
                ),
                "dynamic_vs_matched_static": float(
                    model10["terminal_wealth"]
                    / matched10["terminal_wealth"]
                ),
                "max_drawdown_pct": float(
                    model10["max_drawdown_pct"]
                ),
                "worst_rolling_5d_pct": float(
                    model10["worst_rolling_5d_pct"]
                ),
                "avg_cash_weight_pct": float(
                    model10["avg_cash_weight_pct"]
                ),
                "positive_relative_years": relative_years,
                "risk_passing_years": risk_years,
                "selection_uses_outer_test": False,
            }
            for gate, passed in checks.items()
        ]
    )


def _write_notebook(card: Path, summary: dict[str, Any]) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            "# Standalone TQQQ / QQQ / cash training\n\n"
            "只使用 TQQQ 与 QQQ 自身历史训练；SOXL/SOXX 禁止进入数据集。"
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import sys\n"
            "import qlib\n"
            "from qlib.data import D\n"
            "from qlib.data.dataset import DatasetH\n"
            "from qlib.workflow import R\n"
            "repo_root = Path.cwd().parents[1]\n"
            "if str(repo_root) not in sys.path:\n"
            "    sys.path.insert(0, str(repo_root))\n"
            "qlib.init(provider_uri=str(repo_root / 'data/qlib/tqqq'), "
            "region='us', kernels=1, skip_if_reg=False)\n"
            "calendar = D.calendar(start_time='2011-01-03', "
            "end_time='2026-07-24', freq='day')\n"
            "from src.workflow.standalone_tqqq_qqq_cash_training "
            "import run_standalone_tqqq_qqq_cash_training_card\n"
            "summary = run_standalone_tqqq_qqq_cash_training_card("
            "root=repo_root)\n"
            "summary['promotion_decision']"
        ),
        nbformat.v4.new_code_cell(
            "import pandas as pd\n"
            "card = Path.cwd()\n"
            "pd.read_csv(card / 'model_selection.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'backtest_benchmark_comparison.csv').query("
            "\"segment == 'outer_diagnostic' and cost_bps == 10\")"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'training_data_audit.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'objective_gate_audit.csv')"
        ),
        nbformat.v4.new_markdown_cell(
            "## 中文结论\n\n"
            + (card / "conclusion.md").read_text(encoding="utf-8")
        ),
    ]
    nbformat.write(
        notebook, card / f"{summary['experiment']}.ipynb"
    )


def _write_card(
    *,
    card: Path,
    summary: dict[str, Any],
    selection: pd.DataFrame,
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
    diagnostics: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    selection.to_csv(card / "model_selection.csv", index=False)
    comparison.to_csv(
        card / "backtest_benchmark_comparison.csv", index=False
    )
    annual.to_csv(card / "backtest_walk_forward_metrics.csv", index=False)
    diagnostics.to_csv(card / "training_data_audit.csv", index=False)
    audit.to_csv(card / "objective_gate_audit.csv", index=False)
    selected = selection.loc[selection["selected"]].iloc[0]
    selected_name = str(selected["candidate"])
    selected_outer = comparison.loc[
        comparison["candidate"].eq(selected_name)
        & comparison["segment"].eq("outer_diagnostic")
        & comparison["cost_bps"].isin([10, 25])
    ].copy()
    selected_outer.to_csv(card / "risk_analysis.csv", index=False)
    selected_payload = {
        "candidate": selected_name,
        "training_assets": ["TQQQ", "QQQ"],
        "forbidden_training_assets": list(FORBIDDEN_TRAINING_ASSETS),
        "forbidden_training_rows": 0,
        "selection_uses_outer_test": False,
        "diagnostic_only": not bool(summary["objective_pass"]),
        "development_metrics": _json_ready(selected.to_dict()),
    }
    (card / "selected_model.json").write_text(
        json.dumps(selected_payload, indent=2), encoding="utf-8"
    )
    model10 = selected_outer.loc[
        selected_outer["strategy"].eq("model")
        & selected_outer["cost_bps"].eq(10)
    ].iloc[0]
    model25 = selected_outer.loc[
        selected_outer["strategy"].eq("model")
        & selected_outer["cost_bps"].eq(25)
    ].iloc[0]
    matched10 = selected_outer.loc[
        selected_outer["strategy"].eq("matched_static")
        & selected_outer["cost_bps"].eq(10)
    ].iloc[0]
    analysis = {
        "selected_candidate": selected_name,
        "training_assets": ["TQQQ", "QQQ"],
        "forbidden_training_rows": 0,
        "outer_model_10bps": _json_ready(model10.to_dict()),
        "outer_model_25bps": _json_ready(model25.to_dict()),
        "outer_matched_static_10bps": _json_ready(
            matched10.to_dict()
        ),
        "promotion_decision": summary["promotion_decision"],
    }
    (card / "analysis_summary.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )
    (card / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2), encoding="utf-8"
    )
    failed = audit.loc[~audit["passed"], "gate"].tolist()
    (card / "failure_modes.md").write_text(
        "# Failure modes\n\n"
        + (
            "\n".join(f"- {gate}" for gate in failed)
            if failed
            else "- none"
        )
        + "\n",
        encoding="utf-8",
    )
    conclusion = (
        "# 结论\n\n"
        "- 本实验只从 `data/qlib/tqqq` 读取 TQQQ 与 QQQ；"
        "训练、开发和outer数据中 SOXL/SOXX 行数均为0。\n"
        f"- 开发期选择 `{selected_name}`；outer 10bp收益 "
        f"{float(model10['return_pct']):.2f}%，QQQ "
        f"{float(model10['soxx_return_pct']):.2f}%，财富比 "
        f"{float(model10['terminal_wealth_ratio_vs_soxx']):.4f}；"
        f"25bp财富比 "
        f"{float(model25['terminal_wealth_ratio_vs_soxx']):.4f}。\n"
        f"- MaxDD {float(model10['max_drawdown_pct']):.2f}%，"
        f"最差五日 {float(model10['worst_rolling_5d_pct']):.2f}%，"
        f"平均现金 {float(model10['avg_cash_weight_pct']):.2f}%，"
        f"换手 {float(model10['total_turnover']):.2f}；"
        f"动态/同平均仓位静态财富比 "
        f"{float(model10['terminal_wealth'] / matched10['terminal_wealth']):.4f}。\n"
        f"- 开发期完整门：{'通过' if bool(selected['all_gate']) else '未通过'}；"
        f"晋升：`{summary['promotion_decision']}`。\n"
        "- Outer仅报告，未参与候选选择；任何outer更好的未选候选"
        "都不能事后替换开发期选择。\n"
        f"- Qlib recorder：`{summary['recorder_id']}`。\n"
    )
    (card / "conclusion.md").write_text(
        conclusion, encoding="utf-8"
    )
    workflow = (
        f"experiment: {summary['experiment']}\n"
        f"provider_uri: {summary['provider_uri']}\n"
        f"recorder_uri: {summary['recorder_uri']}\n"
        "market: US\n"
        "region: us\n"
        f"as_of: {summary['as_of']}\n"
        "assets: [TQQQ, QQQ, cash]\n"
        "training_assets: [TQQQ, QQQ]\n"
        "forbidden_training_assets: [SOXL, SOXX]\n"
        "model_selection: TQQQ_only_development_2016_2022\n"
        "outer_role: report_only\n"
        "allocation_output: direct_cash_constrained_weights\n"
        "artifact_policy:\n"
        "  policy: slim_card_recorder_first\n"
    )
    (card / "workflow.yaml").write_text(
        workflow, encoding="utf-8"
    )
    _write_notebook(card, summary)


def _run_card(spec: StandaloneTqqqSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    panel, features, source_audit = _load_tqqq_only_panel(spec)
    dataset, dataset_audit = _build_dataset(
        panel,
        features,
        spec.model_spec,
    )
    if set(dataset_audit["instruments"]) != {"TQQQ"}:
        raise ValueError("blocked: non-TQQQ instrument in DatasetH")
    development_predictions: dict[str, pd.DataFrame] = {}
    outer_predictions: dict[str, pd.DataFrame] = {}
    states: dict[str, Any] = {}
    diagnostic_frames: list[pd.DataFrame] = []
    for candidate in CANDIDATES:
        development, outer, candidate_states, diagnostics = (
            _candidate_predictions(
                candidate=candidate,
                dataset=dataset,
                panel=panel,
                features=features,
                spec=spec,
            )
        )
        development_predictions[candidate.name] = development
        outer_predictions[candidate.name] = outer
        states[candidate.name] = candidate_states
        diagnostic_frames.append(diagnostics)
    comparison, annual, strategy_returns, positions = (
        _evaluate_candidates(
            panel=panel,
            development_predictions=development_predictions,
            outer_predictions=outer_predictions,
            spec=spec,
        )
    )
    selection = _select_candidate(comparison, annual)
    selected = selection.loc[selection["selected"]].iloc[0]
    selected_name = str(selected["candidate"])
    audit = _outer_audit(
        comparison=comparison,
        annual=annual,
        selected_candidate=selected_name,
    )
    development_pass = bool(selected["all_gate"])
    outer_pass = bool(audit.loc[audit["required"], "passed"].all())
    objective_pass = bool(development_pass and outer_pass)
    decision = (
        "validated_standalone_tqqq_qqq_cash"
        if objective_pass
        else "not_promoted_standalone_tqqq_gates_failed"
    )
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    predictions = pd.concat(
        [
            *development_predictions.values(),
            *outer_predictions.values(),
        ]
    ).sort_index()

    from qlib.workflow import R

    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(
        prefix=f"{spec.experiment}_"
    ) as temporary:
        temp = Path(temporary)
        predictions.reset_index(names="signal_date").to_csv(
            temp / "predictions.csv", index=False
        )
        positions.to_csv(temp / "positions.csv", index=False)
        strategy_returns.to_csv(
            temp / "strategy_returns.csv", index=False
        )
        diagnostics.to_csv(
            temp / "training_diagnostics.csv", index=False
        )
        models_dir = temp / "models"
        models_dir.mkdir()
        for candidate_name, candidate_states in states.items():
            with (models_dir / f"{candidate_name}.pkl").open(
                "wb"
            ) as handle:
                pickle.dump(candidate_states, handle)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                assets="TQQQ,QQQ,cash",
                training_assets="TQQQ,QQQ",
                forbidden_training_assets="SOXL,SOXX",
                forbidden_training_rows=0,
                qlib_dataset="DatasetH",
                model_selection="TQQQ_only_development_2016_2022",
                outer_role="report_only",
            )
            R.log_metrics(
                development_pass=float(development_pass),
                outer_pass=float(outer_pass),
                objective_pass=float(objective_pass),
            )
            for artifact in temp.glob("*.csv"):
                recorder.log_artifact(
                    artifact.as_posix(), artifact_path=spec.experiment
                )
            for artifact in models_dir.glob("*.pkl"):
                recorder.log_artifact(
                    artifact.as_posix(),
                    artifact_path=f"{spec.experiment}/models",
                )
            recorder_id = recorder.id
            recorder_local_dir = Path(
                recorder.get_local_dir()
            ).resolve()
            summary = {
                "experiment": spec.experiment,
                "card_type": "qlib_standalone_tqqq_training_card",
                "primary_notebook": f"{spec.experiment}.ipynb",
                "as_of": spec.as_of,
                "training_end": spec.development_end,
                "outer": list(spec.outer),
                "provider_uri": spec.provider_uri.resolve().as_posix(),
                "source_audit": source_audit,
                "dataset_audit": dataset_audit,
                "spec": {
                    **asdict(spec),
                    "root": spec.root.resolve().as_posix(),
                },
                "assets": ["TQQQ", "QQQ", "cash"],
                "training_assets": ["TQQQ", "QQQ"],
                "forbidden_training_assets": list(
                    FORBIDDEN_TRAINING_ASSETS
                ),
                "forbidden_training_rows": 0,
                "candidates": [
                    asdict(candidate) for candidate in CANDIDATES
                ],
                "feature_count": len(features),
                "selection_uses_outer_test": False,
                "outer_role": "report_only",
                "selected_candidate": selected_name,
                "development_pass": development_pass,
                "outer_pass": outer_pass,
                "objective_pass": objective_pass,
                "promotion_decision": decision,
                "qlib_docs": [
                    "https://qlib.readthedocs.io/en/latest/component/data.html",
                    "https://qlib.readthedocs.io/en/latest/component/workflow.html",
                ],
                "recorder_id": recorder_id,
                "recorder_uri": recorder_uri.resolve().as_posix(),
                "recorder_local_dir": recorder_local_dir.as_posix(),
                "sidecars": {
                    "selected_model": "selected_model.json",
                    "model_selection": "model_selection.csv",
                    "backtest_benchmark_comparison": (
                        "backtest_benchmark_comparison.csv"
                    ),
                    "backtest_walk_forward_metrics": (
                        "backtest_walk_forward_metrics.csv"
                    ),
                    "risk_analysis": "risk_analysis.csv",
                    "training_data_audit": "training_data_audit.csv",
                    "objective_gate_audit": "objective_gate_audit.csv",
                    "analysis_summary": "analysis_summary.json",
                    "failure_modes": "failure_modes.md",
                    "conclusion": "conclusion.md",
                    "workflow_config": "workflow.yaml",
                },
                "artifact_policy": {
                    "policy": "slim_card_recorder_first",
                    "workflow_config": "workflow.yaml",
                    "recorder_only_artifacts": list(
                        RECORDER_ONLY_ARTIFACTS
                    ),
                },
            }
            _write_card(
                card=card,
                summary=summary,
                selection=selection,
                comparison=comparison,
                annual=annual,
                diagnostics=diagnostics,
                audit=audit,
            )
            for artifact_name in summary["sidecars"].values():
                recorder.log_artifact(
                    (card / artifact_name).as_posix(),
                    artifact_path=spec.experiment,
                )
            recorder.log_artifact(
                (card / f"{spec.experiment}.ipynb").as_posix(),
                artifact_path=spec.experiment,
            )
            _remove_qlib_code_snapshot_artifacts(recorder_local_dir)
    return _json_ready(summary)


if __name__ == "__main__":
    print(
        json.dumps(
            run_standalone_tqqq_qqq_cash_training_card(),
            indent=2,
        )
    )
