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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

from src.data import (
    QlibProviderConfig,
    build_taleb_fragility_qlib_store,
    load_qlib_feature_frame,
)
from src.factor import (
    EMPIRICAL_TAIL_FEATURES,
    IMPLIED_TAIL_FEATURES,
    STRUCTURAL_FRAGILITY_FEATURES,
    TALEB_FRAGILITY_FEATURES,
)
from src.factor.taleb_fragility import FEATURE_DEFINITIONS
from src.workflow.direct_continuous_soxx_allocation import (
    DEVELOPMENT_YEARS,
    OUTER_START,
    DirectContinuousAllocationSpec,
    _annual_metrics,
    _evaluate,
    _load_frame,
    _prepare,
    _regime_metrics,
)
from src.workflow.multitask_classification_allocation import (
    BASE_FEATURES,
    LABEL_COLUMNS,
    TARGET_WEIGHT_COLUMNS,
    AllocationCandidate,
    _attach_multitask_targets,
    _build_dataset,
    _classification_rows,
    _classification_summary,
    _development_selection,
    _fit_predict,
    _objective_audit,
)
from src.workflow.soxl_objective_feasibility_audit import _json_ready
from src.workflow.trainable_regime_policy_nested_wf import (
    _remove_qlib_code_snapshot_artifacts,
)


EXPERIMENT = "taleb_fragility_allocation_v1"
RECORDER_ONLY_ARTIFACTS = (
    f"artifacts/{EXPERIMENT}/models/",
    f"artifacts/{EXPERIMENT}/predictions.csv",
    f"artifacts/{EXPERIMENT}/positions.csv",
    f"artifacts/{EXPERIMENT}/strategy_returns.csv",
    f"artifacts/{EXPERIMENT}/training_diagnostics.csv",
)

CANDIDATES = (
    AllocationCandidate(
        name="tabular_base_multitask",
        encoder="tabular",
        feature_set="base",
        multitask=True,
    ),
    AllocationCandidate(
        name="tabular_base_empirical_tail",
        encoder="tabular",
        feature_set="base_empirical",
        multitask=True,
    ),
    AllocationCandidate(
        name="tabular_base_implied_tail",
        encoder="tabular",
        feature_set="base_implied",
        multitask=True,
    ),
    AllocationCandidate(
        name="tabular_base_structural_fragility",
        encoder="tabular",
        feature_set="base_structural",
        multitask=True,
    ),
    AllocationCandidate(
        name="tabular_base_all_taleb",
        encoder="tabular",
        feature_set="base_all_taleb",
        multitask=True,
    ),
)

CONTINUOUS_TAIL_TARGETS = (
    "soxx_future_adverse_5d",
    "soxl_future_adverse_5d",
)


@dataclass(frozen=True)
class TalebFragilityAllocationSpec:
    root: Path
    experiment: str = EXPERIMENT
    train_start: str = "2011-01-03"
    development_years: tuple[int, ...] = DEVELOPMENT_YEARS
    development_end: str = "2022-12-30"
    outer: tuple[str, str] = (OUTER_START, "2026-07-02")
    purge_rows: int = 20
    hidden_units: int = 24
    epochs: int = 350
    learning_rate: float = 0.003
    weight_decay: float = 0.03
    maximum_soxl_weight: float = 0.30
    costs_bps: tuple[int, ...] = (10, 25)
    seed: int = 29
    as_of: str = "2026-07-02"

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def daily_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"

    @property
    def feature_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "taleb_fragility_v1"


def run_taleb_fragility_allocation_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    repo_root = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[2]
    )
    return _run_card(TalebFragilityAllocationSpec(root=repo_root))


def _load_taleb_features(
    spec: TalebFragilityAllocationSpec,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = build_taleb_fragility_qlib_store(
        root=spec.root, as_of=spec.as_of
    )
    qlib_frame = load_qlib_feature_frame(
        QlibProviderConfig(
            provider_uri=spec.feature_provider_uri,
            market="all",
            region="us",
        ),
        instruments=["TALEB"],
        fields=[f"${feature}" for feature in TALEB_FRAGILITY_FEATURES],
        start="2010-03-11",
        end=spec.as_of,
    ).reset_index()
    qlib_frame["datetime"] = pd.to_datetime(
        qlib_frame["datetime"], errors="raise"
    ).dt.normalize()
    qlib_frame = qlib_frame.rename(
        columns={
            f"${feature}": feature
            for feature in TALEB_FRAGILITY_FEATURES
        }
    )
    return (
        qlib_frame.set_index("datetime")
        .loc[:, list(TALEB_FRAGILITY_FEATURES)]
        .sort_index(),
        manifest,
    )


def _feature_bundles(base: list[str]) -> dict[str, list[str]]:
    return {
        "base": list(base),
        "base_empirical": list(
            dict.fromkeys([*base, *EMPIRICAL_TAIL_FEATURES])
        ),
        "base_implied": list(
            dict.fromkeys([*base, *IMPLIED_TAIL_FEATURES])
        ),
        "base_structural": list(
            dict.fromkeys([*base, *STRUCTURAL_FRAGILITY_FEATURES])
        ),
        "base_all_taleb": list(
            dict.fromkeys([*base, *TALEB_FRAGILITY_FEATURES])
        ),
    }


def _candidate_features(
    candidate: AllocationCandidate,
    bundles: dict[str, list[str]],
) -> list[str]:
    if candidate.feature_set not in bundles:
        raise ValueError(
            f"blocked: unknown Taleb feature set {candidate.feature_set}"
        )
    return bundles[candidate.feature_set]


def _development_candidate(
    *,
    dataset,
    frame: pd.DataFrame,
    features: list[str],
    candidate: AllocationCandidate,
    spec: TalebFragilityAllocationSpec,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    states: dict[str, Any] = {}
    diagnostics: list[dict[str, Any]] = []
    for year in spec.development_years:
        test = _prepare(dataset, f"fold_{year}")
        test = test.loc[test.index.year == year]
        start_position = frame.index.searchsorted(
            pd.Timestamp(f"{year}-01-01")
        )
        train_end_position = start_position - spec.purge_rows - 1
        if train_end_position < 0:
            raise ValueError(
                f"blocked: insufficient history for fold {year}"
            )
        train_end = frame.index[train_end_position]
        train = _prepare(dataset, (spec.train_start, train_end))
        output, state, fit_diagnostics = _fit_predict(
            full_frame=frame,
            train=train,
            test=test,
            features=features,
            candidate=candidate,
            spec=spec,
            seed=spec.seed + year,
        )
        output["candidate"] = candidate.name
        output["segment"] = "development_walk_forward"
        output["fold_year"] = year
        output["train_end"] = train_end
        predictions.append(output)
        states[f"fold_{year}"] = state
        diagnostics.append(
            {
                "candidate": candidate.name,
                "fold_year": year,
                "train_end": train_end.date().isoformat(),
                "train_rows": int(len(train)),
                "test_rows": int(len(output)),
                "feature_count": len(features),
                "selection_uses_outer_test": False,
                **fit_diagnostics,
            }
        )
    combined = pd.concat(predictions).sort_index()
    classification = _classification_rows(
        combined,
        candidate=candidate.name,
        segment="development_walk_forward",
    )
    return (
        combined,
        states,
        pd.DataFrame(diagnostics),
        classification,
    )


def _outer_candidate(
    *,
    dataset,
    frame: pd.DataFrame,
    features: list[str],
    candidate: AllocationCandidate,
    spec: TalebFragilityAllocationSpec,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], pd.DataFrame]:
    train = _prepare(dataset, (spec.train_start, spec.development_end))
    test = _prepare(dataset, "outer")
    output, state, diagnostics = _fit_predict(
        full_frame=frame,
        train=train,
        test=test,
        features=features,
        candidate=candidate,
        spec=spec,
        seed=spec.seed,
    )
    output["candidate"] = candidate.name
    output["segment"] = "outer_diagnostic"
    output["fold_year"] = output.index.year
    output["train_end"] = pd.Timestamp(spec.development_end)
    classification = _classification_rows(
        output,
        candidate=candidate.name,
        segment="outer_diagnostic",
    )
    return output, state, diagnostics, classification


def _candidate_backtests(
    *,
    frame: pd.DataFrame,
    prediction_map: dict[str, pd.DataFrame],
    segment: str,
    spec: TalebFragilityAllocationSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparisons: list[dict[str, Any]] = []
    annual_frames: list[pd.DataFrame] = []
    stored_returns: list[pd.DataFrame] = []
    stored_positions: list[pd.DataFrame] = []
    for candidate in CANDIDATES:
        predictions = prediction_map[candidate.name]
        for cost_bps in spec.costs_bps:
            metrics, result, benchmark = _evaluate(
                market=frame,
                predictions=predictions,
                profile=candidate,
                cost_bps=cost_bps,
            )
            comparisons.append(
                {
                    "segment": segment,
                    "candidate": candidate.name,
                    "cost_bps": cost_bps,
                    "selection_uses_outer_test": False,
                    **metrics,
                }
            )
            annual_frames.append(
                _annual_metrics(
                    result,
                    benchmark,
                    {
                        "segment": segment,
                        "candidate": candidate.name,
                        "cost_bps": cost_bps,
                        "selection_uses_outer_test": False,
                    },
                )
            )
            stored_returns.append(
                result.assign(
                    segment=segment,
                    candidate=candidate.name,
                    cost_bps=cost_bps,
                )
            )
            stored_positions.append(
                result.loc[
                    :,
                    [
                        "signal_date",
                        "trade_date",
                        "exit_date",
                        "soxl_weight",
                        "soxx_weight",
                        "cash_weight",
                        "turnover",
                    ],
                ].assign(
                    segment=segment,
                    candidate=candidate.name,
                    cost_bps=cost_bps,
                )
            )
    return (
        pd.DataFrame(comparisons),
        pd.concat(annual_frames, ignore_index=True),
        pd.concat(stored_returns, ignore_index=True),
        pd.concat(stored_positions, ignore_index=True),
    )


def _continuous_tail_learnability(
    *,
    dataset,
    frame: pd.DataFrame,
    bundles: dict[str, list[str]],
    spec: TalebFragilityAllocationSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        features = _candidate_features(candidate, bundles)
        for year in spec.development_years:
            test = _prepare(dataset, f"fold_{year}")
            test = test.loc[test.index.year == year]
            start_position = frame.index.searchsorted(
                pd.Timestamp(f"{year}-01-01")
            )
            train_end = frame.index[
                start_position - spec.purge_rows - 1
            ]
            train = _prepare(dataset, (spec.train_start, train_end))
            medians = train.loc[:, features].median()
            train_x = train.loc[:, features].fillna(medians)
            test_x = test.loc[:, features].fillna(medians)
            for target in CONTINUOUS_TAIL_TARGETS:
                train_mask = train[target].notna()
                test_mask = test[target].notna()
                model = HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=0.10,
                    learning_rate=0.05,
                    max_iter=120,
                    max_leaf_nodes=15,
                    min_samples_leaf=30,
                    l2_regularization=1.0,
                    random_state=spec.seed + year,
                )
                model.fit(
                    train_x.loc[train_mask],
                    train.loc[train_mask, target],
                )
                actual = test.loc[test_mask, target].astype(float)
                predicted = pd.Series(
                    model.predict(test_x.loc[test_mask]),
                    index=actual.index,
                )
                baseline_prediction = float(
                    train.loc[train_mask, target].quantile(0.10)
                )
                model_loss = float(
                    mean_pinball_loss(
                        actual, predicted, alpha=0.10
                    )
                )
                baseline_loss = float(
                    mean_pinball_loss(
                        actual,
                        np.full(len(actual), baseline_prediction),
                        alpha=0.10,
                    )
                )
                cutoff = predicted.quantile(0.10)
                predicted_tail = actual.loc[predicted.le(cutoff)]
                rows.append(
                    {
                        "candidate": candidate.name,
                        "feature_set": candidate.feature_set,
                        "fold_year": year,
                        "target": target,
                        "pinball_loss": model_loss,
                        "baseline_pinball_loss": baseline_loss,
                        "pinball_skill": (
                            1.0 - model_loss / baseline_loss
                            if baseline_loss > 0.0
                            else float("nan")
                        ),
                        "spearman": float(
                            actual.corr(predicted, method="spearman")
                        ),
                        "predicted_tail_actual_mean": float(
                            predicted_tail.mean()
                        ),
                        "actual_mean": float(actual.mean()),
                        "selection_uses_outer_test": False,
                    }
                )
    metrics = pd.DataFrame(rows)
    summary = (
        metrics.groupby(
            ["candidate", "feature_set", "target"], as_index=False
        )
        .agg(
            mean_pinball_skill=("pinball_skill", "mean"),
            median_pinball_skill=("pinball_skill", "median"),
            positive_skill_years=(
                "pinball_skill",
                lambda values: int(pd.Series(values).gt(0.0).sum()),
            ),
            mean_spearman=("spearman", "mean"),
            mean_predicted_tail_actual=(
                "predicted_tail_actual_mean",
                "mean",
            ),
        )
    )
    return metrics, summary


def _summary(
    *,
    spec: TalebFragilityAllocationSpec,
    source_audit: dict[str, Any],
    tail_manifest: dict[str, Any],
    dataset_audit: dict[str, Any],
    bundles: dict[str, list[str]],
    selection: pd.DataFrame,
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
    classification: pd.DataFrame,
    development_classification: pd.DataFrame,
    outer_classification: pd.DataFrame,
    continuous_tail_summary: pd.DataFrame,
    regimes: pd.DataFrame,
    objective_audit: pd.DataFrame,
    objective_pass: bool,
    decision: str,
    recorder_id: str,
    recorder_uri: Path,
    recorder_local_dir: Path,
) -> dict[str, Any]:
    selected = selection.loc[selection["diagnostic_selected"]].iloc[0]
    selected_name = str(selected["candidate"])
    outer10 = comparison.loc[
        comparison["segment"].eq("outer_diagnostic")
        & comparison["candidate"].eq(selected_name)
        & comparison["cost_bps"].eq(10)
    ].iloc[0]
    return {
        "experiment": spec.experiment,
        "system_of_record": "qlib",
        "provider_uri": spec.feature_provider_uri.resolve().as_posix(),
        "execution_provider_uri": source_audit["execution_provider_uri"],
        "market": "US",
        "region": "us",
        "as_of": spec.as_of,
        "data_range": [
            source_audit["calendar_start"],
            spec.as_of,
        ],
        "benchmark": "SOXX",
        "assets": ["SOXL", "SOXX", "cash"],
        "hypothesis": (
            "Taleb-style empirical tail shape, implied tail pricing, and "
            "nonlinear leveraged-ETF fragility improve tail-label "
            "learnability and direct allocation"
        ),
        "architecture": {
            "base_position": None,
            "explicit_prior": None,
            "candidate_weight_grid": False,
            "allocation_output": (
                "SOXL=cap*sigmoid(a); cash=(1-SOXL)*sigmoid(b); "
                "SOXX=residual"
            ),
            "candidates": [asdict(candidate) for candidate in CANDIDATES],
            "auxiliary_tasks": list(LABEL_COLUMNS),
            "final_weights_are_direct_model_outputs": True,
        },
        "feature_sets": bundles,
        "taleb_feature_definitions": dict(FEATURE_DEFINITIONS),
        "labels": {
            "tail_1d": "future SOXL adverse 1d <= -8%",
            "tail_3d": "future SOXX adverse 3d <= -4%",
            "tail_5d": (
                "future SOXL adverse 5d <= -12% or SOXX adverse 5d <= -6%"
            ),
            "relative_alpha": (
                "risk-adjusted 5d/20d SOXL utility exceeds SOXX by 2%"
            ),
            "recovery": (
                "known stress context and future SOXX 5d >= 4% "
                "with adverse path >= -4%"
            ),
        },
        "data_audit": {
            "base": source_audit,
            "taleb_provider": tail_manifest,
            "dataset": dataset_audit,
        },
        "development": {
            "years": list(spec.development_years),
            "selection_uses_outer_test": False,
            "diagnostic_selected": _json_ready(selected.to_dict()),
        },
        "outer_diagnostic": {
            "start": OUTER_START,
            "selection_uses_outer_test": False,
            "selected_candidate_metrics_10bps": _json_ready(
                outer10.to_dict()
            ),
            "all_candidate_metrics": comparison.loc[
                comparison["segment"].eq("outer_diagnostic")
            ].to_dict(orient="records"),
        },
        "objective_pass": objective_pass,
        "promotion_decision": decision,
        "objective_gate_audit": objective_audit.to_dict(orient="records"),
        "model_selection": selection.to_dict(orient="records"),
        "annual_metrics": annual.to_dict(orient="records"),
        "classification_metrics": classification.to_dict(orient="records"),
        "development_classification_summary": (
            development_classification.to_dict(orient="records")
        ),
        "outer_classification_summary": outer_classification.to_dict(
            orient="records"
        ),
        "continuous_tail_learnability": continuous_tail_summary.to_dict(
            orient="records"
        ),
        "regime_metrics": regimes.to_dict(orient="records"),
        "qlib_docs": [
            "https://qlib.readthedocs.io/en/latest/component/data.html",
            "https://qlib.readthedocs.io/en/latest/component/model.html",
            "https://qlib.readthedocs.io/en/latest/component/workflow.html",
            "https://qlib.readthedocs.io/en/latest/component/recorder.html",
        ],
        "recorder_id": recorder_id,
        "recorder_uri": recorder_uri.resolve().as_posix(),
        "recorder_local_dir": recorder_local_dir.as_posix(),
        "sidecars": {
            "selected_model": "selected_model.json",
            "model_selection": "model_selection.csv",
            "model_comparison": "model_comparison.csv",
            "classification_metrics": "classification_metrics.csv",
            "classification_summary": "classification_summary.csv",
            "continuous_tail_metrics": "continuous_tail_metrics.csv",
            "continuous_tail_summary": "continuous_tail_summary.csv",
            "backtest_benchmark_comparison": (
                "backtest_benchmark_comparison.csv"
            ),
            "backtest_walk_forward_metrics": (
                "backtest_walk_forward_metrics.csv"
            ),
            "risk_analysis": "risk_analysis.csv",
            "regime_metrics": "regime_metrics.csv",
            "training_summary": "training_summary.csv",
            "feature_manifest": "feature_manifest.csv",
            "objective_gate_audit": "objective_gate_audit.csv",
            "analysis_summary": "analysis_summary.json",
            "failure_modes": "failure_modes.md",
            "conclusion": "conclusion.md",
            "workflow_config": "workflow.yaml",
        },
        "artifact_policy": {
            "policy": "slim_card_recorder_first",
            "workflow_config": "workflow.yaml",
            "recorder_only_artifacts": list(RECORDER_ONLY_ARTIFACTS),
        },
    }


def _write_notebook(card: Path, summary: dict[str, Any]) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            "# Taleb fragility feature allocation\n\n"
            "固定同一tabular多任务直接仓位模型，仅消融经验尾部、"
            "隐含尾部和结构脆弱度特征。"
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import sys\n"
            "import qlib\n"
            "from qlib.data import D\n"
            "from qlib.workflow import R\n"
            "repo_root = Path.cwd().parents[1]\n"
            "if str(repo_root) not in sys.path:\n"
            "    sys.path.insert(0, str(repo_root))\n"
            "qlib.init(provider_uri=str(repo_root / 'data/qlib/soxl'), "
            "region='us', kernels=1, skip_if_reg=False)\n"
            "calendar = D.calendar(start_time='2011-01-03', "
            "end_time='2026-07-02', freq='day')\n"
            "from src.workflow.taleb_fragility_allocation "
            "import run_taleb_fragility_allocation_card\n"
            "summary = run_taleb_fragility_allocation_card(root=repo_root)\n"
            "summary['promotion_decision']"
        ),
        nbformat.v4.new_code_cell(
            "import pandas as pd\n"
            "card = Path.cwd()\n"
            "pd.read_csv(card / 'model_selection.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'classification_summary.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'continuous_tail_summary.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'backtest_benchmark_comparison.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'objective_gate_audit.csv')"
        ),
        nbformat.v4.new_markdown_cell(
            (card / "conclusion.md").read_text(encoding="utf-8")
        ),
    ]
    nbformat.write(notebook, card / f"{summary['experiment']}.ipynb")


def _write_card(
    *,
    card: Path,
    summary: dict[str, Any],
    bundles: dict[str, list[str]],
    selection: pd.DataFrame,
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
    classification: pd.DataFrame,
    development_classification: pd.DataFrame,
    outer_classification: pd.DataFrame,
    continuous_tail_metrics: pd.DataFrame,
    continuous_tail_summary: pd.DataFrame,
    regimes: pd.DataFrame,
    objective_audit: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    selected = selection.loc[selection["diagnostic_selected"]].iloc[0]
    selected_name = str(selected["candidate"])
    outer10 = comparison.loc[
        comparison["segment"].eq("outer_diagnostic")
        & comparison["candidate"].eq(selected_name)
        & comparison["cost_bps"].eq(10)
    ].iloc[0]
    (card / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2), encoding="utf-8"
    )
    (card / "selected_model.json").write_text(
        json.dumps(_json_ready(selected.to_dict()), indent=2),
        encoding="utf-8",
    )
    selection.to_csv(card / "model_selection.csv", index=False)
    comparison.to_csv(card / "model_comparison.csv", index=False)
    comparison.to_csv(
        card / "backtest_benchmark_comparison.csv", index=False
    )
    annual.to_csv(card / "backtest_walk_forward_metrics.csv", index=False)
    classification.to_csv(card / "classification_metrics.csv", index=False)
    pd.concat(
        [
            development_classification.assign(
                summary_segment="development_walk_forward"
            ),
            outer_classification.assign(
                summary_segment="outer_diagnostic"
            ),
        ],
        ignore_index=True,
    ).to_csv(card / "classification_summary.csv", index=False)
    continuous_tail_metrics.to_csv(
        card / "continuous_tail_metrics.csv", index=False
    )
    continuous_tail_summary.to_csv(
        card / "continuous_tail_summary.csv", index=False
    )
    comparison.loc[
        :,
        [
            "segment",
            "candidate",
            "cost_bps",
            "max_drawdown_pct",
            "worst_day_pct",
            "cvar5_pct",
            "worst_rolling_5d_pct",
            "soxx_max_drawdown_pct",
        ],
    ].to_csv(card / "risk_analysis.csv", index=False)
    regimes.to_csv(card / "regime_metrics.csv", index=False)
    diagnostics.to_csv(card / "training_summary.csv", index=False)
    feature_to_sets: dict[str, list[str]] = {}
    for feature_set, features in bundles.items():
        for feature in features:
            feature_to_sets.setdefault(feature, []).append(feature_set)
    pd.DataFrame(
        [
            {
                "feature": feature,
                "feature_sets": ",".join(feature_sets),
                "family": (
                    "empirical_tail"
                    if feature in EMPIRICAL_TAIL_FEATURES
                    else "implied_tail"
                    if feature in IMPLIED_TAIL_FEATURES
                    else "structural_fragility"
                    if feature in STRUCTURAL_FRAGILITY_FEATURES
                    else "base"
                ),
                "definition": FEATURE_DEFINITIONS.get(
                    feature, "existing PIT base feature"
                ),
                "point_in_time": True,
                "base_or_prior_feature": False,
            }
            for feature, feature_sets in feature_to_sets.items()
        ]
    ).to_csv(card / "feature_manifest.csv", index=False)
    objective_audit.to_csv(card / "objective_gate_audit.csv", index=False)
    analysis = {
        "objective_pass": summary["objective_pass"],
        "promotion_decision": summary["promotion_decision"],
        "diagnostic_selected": selected_name,
        "development_ratio_10bps": selected["ratio_10bps"],
        "development_ratio_25bps": selected["ratio_25bps"],
        "development_max_drawdown_pct": selected["max_drawdown_pct"],
        "development_tail_1d_auc": selected["tail_1d_label_auc"],
        "development_relative_alpha_auc": selected[
            "relative_alpha_label_auc"
        ],
        "outer_ratio_10bps": outer10[
            "terminal_wealth_ratio_vs_soxx"
        ],
        "outer_max_drawdown_pct": outer10["max_drawdown_pct"],
        "selection_uses_outer_test": False,
    }
    (card / "analysis_summary.json").write_text(
        json.dumps(_json_ready(analysis), indent=2), encoding="utf-8"
    )
    failed = objective_audit.loc[
        objective_audit["required"] & ~objective_audit["passed"]
    ]
    (card / "failure_modes.md").write_text(
        "# Failure modes\n\n"
        + (
            "\n".join(
                f"- {row.gate}: {row.evidence}"
                for row in failed.itertuples()
            )
            if not failed.empty
            else "- none"
        )
        + "\n",
        encoding="utf-8",
    )
    base = selection.loc[
        selection["candidate"].eq("tabular_base_multitask")
    ].iloc[0]
    continuous_best = continuous_tail_summary.sort_values(
        ["mean_pinball_skill", "candidate"],
        ascending=[False, True],
    ).iloc[0]
    conclusion = (
        "# Taleb肥尾与脆弱度特征结论\n\n"
        "- 跑了什么：固定同一tabular多任务直接连续仓位模型，分别"
        "加入经验尾部、期权隐含尾部、结构脆弱度和全部Taleb特征；"
        "没有底仓、先验或仓位网格。\n"
        f"- 基准特征：开发期10bp财富/SOXX={base['ratio_10bps']:.4f}，"
        f"MaxDD={base['max_drawdown_pct']:.2f}%，tail1 AUC="
        f"{base['tail_1d_label_auc']:.4f}。\n"
        f"- 开发期诊断优胜：{selected_name}；10bp财富/SOXX="
        f"{selected['ratio_10bps']:.4f}，25bp="
        f"{selected['ratio_25bps']:.4f}，MaxDD="
        f"{selected['max_drawdown_pct']:.2f}%，tail1 AUC="
        f"{selected['tail_1d_label_auc']:.4f}。\n"
        f"- 连续尾部诊断最佳：{continuous_best['candidate']} / "
        f"{continuous_best['target']}，平均pinball skill="
        f"{continuous_best['mean_pinball_skill']:.4f}，正skill年份="
        f"{int(continuous_best['positive_skill_years'])}/7。\n"
        f"- Outer仅诊断：10bp财富/SOXX="
        f"{outer10['terminal_wealth_ratio_vs_soxx']:.4f}，MaxDD="
        f"{outer10['max_drawdown_pct']:.2f}%。\n"
        f"- 晋升决定：{summary['promotion_decision']}。\n"
        "- 风险：Hill/ES等滚动尾部估计仍有参数不确定性；成分股"
        "回填和2024年才开始的外部15分钟数据均未参与选择；"
        "outer没有参与特征或模型选择。\n"
        "- 下一步：只有开发期标签可学习性和风险收益同时改善，"
        "才保留对应特征族；否则停止日线尾部工程并转向真正PIT"
        "盘前或更长历史的期权截面。\n"
    )
    (card / "conclusion.md").write_text(conclusion, encoding="utf-8")
    workflow = (
        f"experiment: {EXPERIMENT}\n"
        "system_of_record: qlib\n"
        f"provider_uri: {summary['provider_uri']}\n"
        f"execution_provider_uri: {summary['execution_provider_uri']}\n"
        "benchmark: SOXX\n"
        "assets: [SOXL, SOXX, cash]\n"
        "allocation_output: direct_continuous_factorized\n"
        "feature_ablation: [base, base_empirical, base_implied, "
        "base_structural, base_all_taleb]\n"
        "base_position: none\n"
        "explicit_prior: none\n"
        "candidate_weight_grid: false\n"
        "development: [2016, 2022]\n"
        f"outer_diagnostic: [2023, {summary['data_range'][1]}]\n"
        "selection_uses_outer_test: false\n"
        "artifact_policy: slim_card_recorder_first\n"
    )
    (card / "workflow.yaml").write_text(workflow, encoding="utf-8")
    _write_notebook(card, summary)


def _run_card(spec: TalebFragilityAllocationSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    source_frame, source_audit, _ = _load_frame(
        DirectContinuousAllocationSpec(root=spec.root)
    )
    taleb_features, tail_manifest = _load_taleb_features(spec)
    frame = source_frame.join(taleb_features, how="left")
    frame, base_features, _ = _attach_multitask_targets(frame)
    base_features = [
        feature for feature in BASE_FEATURES if feature in frame
    ]
    bundles = _feature_bundles(base_features)
    all_features = list(
        dict.fromkeys(
            feature
            for features in bundles.values()
            for feature in features
        )
    )
    dataset, dataset_audit = _build_dataset(
        frame,
        all_features,
        spec,
        extra_labels=CONTINUOUS_TAIL_TARGETS,
    )

    development_predictions: dict[str, pd.DataFrame] = {}
    outer_predictions: dict[str, pd.DataFrame] = {}
    model_states: dict[str, Any] = {}
    diagnostics: list[pd.DataFrame] = []
    classification_rows: list[pd.DataFrame] = []
    for candidate in CANDIDATES:
        features = _candidate_features(candidate, bundles)
        development, states, fit_diagnostics, classification = (
            _development_candidate(
                dataset=dataset,
                frame=frame,
                features=features,
                candidate=candidate,
                spec=spec,
            )
        )
        outer, outer_state, outer_fit, outer_classification = (
            _outer_candidate(
                dataset=dataset,
                frame=frame,
                features=features,
                candidate=candidate,
                spec=spec,
            )
        )
        development_predictions[candidate.name] = development
        outer_predictions[candidate.name] = outer
        states["outer_refit"] = outer_state
        model_states[candidate.name] = states
        diagnostics.append(fit_diagnostics)
        diagnostics.append(
            pd.DataFrame(
                [
                    {
                        "candidate": candidate.name,
                        "fold_year": "outer_refit",
                        "train_end": spec.development_end,
                        "train_rows": int(
                            len(
                                _prepare(
                                    dataset,
                                    (
                                        spec.train_start,
                                        spec.development_end,
                                    ),
                                )
                            )
                        ),
                        "test_rows": int(len(outer)),
                        "feature_count": len(features),
                        "selection_uses_outer_test": False,
                        **outer_fit,
                    }
                ]
            )
        )
        classification_rows.extend(
            [classification, outer_classification]
        )

    classification = pd.concat(classification_rows, ignore_index=True)
    development_classification = _classification_summary(
        classification, "development_walk_forward"
    )
    outer_classification = _classification_summary(
        classification, "outer_diagnostic"
    )
    continuous_tail_metrics, continuous_tail_summary = (
        _continuous_tail_learnability(
            dataset=dataset,
            frame=frame,
            bundles=bundles,
            spec=spec,
        )
    )
    (
        development_comparison,
        development_annual,
        development_returns,
        development_positions,
    ) = _candidate_backtests(
        frame=frame,
        prediction_map=development_predictions,
        segment="development_walk_forward",
        spec=spec,
    )
    outer_comparison, outer_annual, outer_returns, outer_positions = (
        _candidate_backtests(
            frame=frame,
            prediction_map=outer_predictions,
            segment="outer_diagnostic",
            spec=spec,
        )
    )
    selection = _development_selection(
        development_comparison,
        development_annual,
        development_classification,
    )
    selected = selection.loc[selection["diagnostic_selected"]].iloc[0]
    selected_name = str(selected["candidate"])
    outer_selected = outer_comparison.loc[
        outer_comparison["candidate"].eq(selected_name)
        & outer_comparison["cost_bps"].eq(10)
    ].iloc[0]
    objective_audit = _objective_audit(selected, outer_selected)
    objective_pass = bool(
        objective_audit.loc[
            objective_audit["required"], "passed"
        ].all()
    )
    decision = (
        "validated_taleb_fragility_direct_allocation"
        if objective_pass
        else "not_promoted_taleb_features_or_risk_gates_failed"
    )
    selected_outer_result = outer_returns.loc[
        outer_returns["candidate"].eq(selected_name)
        & outer_returns["cost_bps"].eq(10)
    ].copy()
    regimes = _regime_metrics(frame, selected_outer_result)
    comparison = pd.concat(
        [development_comparison, outer_comparison], ignore_index=True
    )
    annual = pd.concat(
        [development_annual, outer_annual], ignore_index=True
    )
    all_predictions = pd.concat(
        [
            *development_predictions.values(),
            *outer_predictions.values(),
        ]
    ).sort_index()
    strategy_returns = pd.concat(
        [development_returns, outer_returns], ignore_index=True
    )
    positions = pd.concat(
        [development_positions, outer_positions], ignore_index=True
    )
    training_diagnostics = pd.concat(diagnostics, ignore_index=True)

    from qlib.workflow import R

    recorder_uri = card / "qlib_recorder"
    recorder_uri.mkdir(parents=True, exist_ok=True)
    R.set_uri(recorder_uri.resolve().as_posix())
    with tempfile.TemporaryDirectory(
        prefix=f"{spec.experiment}_"
    ) as temporary:
        temp = Path(temporary)
        all_predictions.reset_index(names="signal_date").to_csv(
            temp / "predictions.csv", index=False
        )
        positions.to_csv(temp / "positions.csv", index=False)
        strategy_returns.to_csv(
            temp / "strategy_returns.csv", index=False
        )
        training_diagnostics.to_csv(
            temp / "training_diagnostics.csv", index=False
        )
        models_dir = temp / "models"
        models_dir.mkdir()
        for candidate_name, states in model_states.items():
            with (
                models_dir / f"{candidate_name}.pkl"
            ).open("wb") as handle:
                pickle.dump(states, handle)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                benchmark="SOXX",
                assets="SOXL,SOXX,cash",
                candidate_count=len(CANDIDATES),
                feature_ablation=(
                    "base,empirical,implied,structural,all_taleb"
                ),
                allocation_output="direct_continuous_factorized",
                selection_uses_outer_test="false",
            )
            R.log_metrics(
                development_ratio_10bps=float(
                    selected["ratio_10bps"]
                ),
                development_max_drawdown_pct=float(
                    selected["max_drawdown_pct"]
                ),
                outer_ratio_10bps=float(
                    outer_selected["terminal_wealth_ratio_vs_soxx"]
                ),
                outer_max_drawdown_pct=float(
                    outer_selected["max_drawdown_pct"]
                ),
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
            summary = _summary(
                spec=spec,
                source_audit=source_audit,
                tail_manifest=tail_manifest,
                dataset_audit=dataset_audit,
                bundles=bundles,
                selection=selection,
                comparison=comparison,
                annual=annual,
                classification=classification,
                development_classification=development_classification,
                outer_classification=outer_classification,
                continuous_tail_summary=continuous_tail_summary,
                regimes=regimes,
                objective_audit=objective_audit,
                objective_pass=objective_pass,
                decision=decision,
                recorder_id=recorder_id,
                recorder_uri=recorder_uri,
                recorder_local_dir=recorder_local_dir,
            )
            _write_card(
                card=card,
                summary=summary,
                bundles=bundles,
                selection=selection,
                comparison=comparison,
                annual=annual,
                classification=classification,
                development_classification=development_classification,
                outer_classification=outer_classification,
                continuous_tail_metrics=continuous_tail_metrics,
                continuous_tail_summary=continuous_tail_summary,
                regimes=regimes,
                objective_audit=objective_audit,
                diagnostics=training_diagnostics,
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
            run_taleb_fragility_allocation_card(), indent=2
        )
    )
