from __future__ import annotations

import json
import math
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class QuantBenchGate:
    """SOXL-specific acceptance gate derived from the QuantBench evaluation stack."""

    max_drawdown_pct: float = 20.0
    min_stress_relative_return: float = 30.0
    min_stress_avg_pos: float = 0.10
    max_stress_avg_pos: float = 0.45
    min_bull_capture: float = 0.50
    min_proxy_return: float = 0.0


@dataclass(frozen=True)
class SummaryRecord:
    profile: str
    source_path: str
    test_proxy: dict[str, float]
    slices: dict[str, dict[str, float]]


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _numeric_metrics(payload: dict[str, Any] | None) -> dict[str, float]:
    payload = payload or {}
    return {str(key): _as_float(value) for key, value in payload.items() if isinstance(value, (int, float))}


def paper_requirement_matrix() -> list[dict[str, str]]:
    """Map QuantBench paper layers to the concrete SOXL harness workstream.

    The paper is a benchmark architecture, not a single trading model. This
    matrix is intentionally explicit so notebook screens can report coverage
    without implying that every family has already been trained.
    """
    return [
        {
            "key": "data_preparation",
            "paper_basis": "Unified data preparation with breadth, depth, alignment, aggregation, normalization, and imputation.",
            "soxl_mapping": "SOXL/SOXX/VIX daily panel, qlib binary store, broker/intraday feeds, and lagged feature packs.",
            "next_artifact": "runtime/data or runtime/model_selection notebook data audit.",
        },
        {
            "key": "factor_mining",
            "paper_basis": "Formula-based factor mining and learned factor discovery before modeling.",
            "soxl_mapping": "Feature pack diagnostics with IC, standalone backtest, correlation budget, and SHAP blind-spot checks.",
            "next_artifact": "runtime/factor/card/<pack>/<pack>.ipynb plus factor DB row.",
        },
        {
            "key": "modeling_tree",
            "paper_basis": "Tree models can outperform deep nets on strong engineered features.",
            "soxl_mapping": "Add non-RL tabular baselines such as XGBoost/LightGBM-style classifiers when dependencies are available.",
            "next_artifact": "runtime/model_selection/card/quantbench_paper_soxl_v1 tree baseline screen.",
        },
        {
            "key": "modeling_sequence",
            "paper_basis": "Temporal models such as LSTM are benchmarked against feature sets and objectives.",
            "soxl_mapping": "Existing PPO actor surfaces include MLP fallback, GRU, attention, TiDE, and MoE variants.",
            "next_artifact": "network/profile matrix and smoke/full retrain comparison rows.",
        },
        {
            "key": "modeling_relational",
            "paper_basis": "Relational or graph models test whether market structure adds information.",
            "soxl_mapping": "Single-SOXL setting maps this to SOXL/SOXX/VIX/sector-relative relation features before graph models.",
            "next_artifact": "SOXX-relative and multi-asset relation feature screen before any RGCN-style model.",
        },
        {
            "key": "training_objectives",
            "paper_basis": "Classification, IC, MSE, ranking, utility, and RL objectives can lead to different outcomes.",
            "soxl_mapping": "Compare return-prediction labels, IC/ranking proxies, BC, PPO rewards, and execution utility gates.",
            "next_artifact": "objective-family screen with fixed-split and walk-forward evidence.",
        },
        {
            "key": "rolling_continual_learning",
            "paper_basis": "Rolling updates mitigate alpha decay and distribution shift.",
            "soxl_mapping": "Use existing walk-forward folds plus 3/6/12-month rolling update diagnostics.",
            "next_artifact": "runtime/model_selection rolling cadence notebook and summary.json.",
        },
        {
            "key": "ensemble_robustness",
            "paper_basis": "Seed ensembles reduce low signal-to-noise overfitting.",
            "soxl_mapping": "Aggregate low-correlated actors/signals only after per-candidate drawdown and stress gates pass.",
            "next_artifact": "ensemble candidate summary with pairwise correlation and drawdown stability.",
        },
        {
            "key": "evaluation_metrics",
            "paper_basis": "Evaluation should include signal, portfolio, execution, robustness, correlation, and decay metrics.",
            "soxl_mapping": "Gate on IC/ICIR, return, Sharpe, MaxDD, turnover, benchmark-relative return, stress retention, bull capture, and alpha decay.",
            "next_artifact": "quantbench_screen summary.json plus candidate ranking CSV.",
        },
    ]


def quantbench_model_objective_coverage() -> list[dict[str, str]]:
    """Return source-owned coverage status for paper model/objective families."""
    return [
        {
            "family": "tree_xgboost_lightgbm",
            "paper_family": "XGBoost/tree model",
            "current_soxl_surface": "sklearn tree/boosted supervised baselines implemented; XGBoost/LightGBM dependency-specific runs pending",
            "coverage": "implemented_partial",
            "next_step": "promote only if QuantBench gate passes; otherwise test rolling/ensemble or external GBDT variants",
        },
        {
            "family": "sequence_lstm_gru",
            "paper_family": "LSTM/temporal sequence",
            "current_soxl_surface": "GRU actor exists; supervised TorchLSTM MSE baseline is implemented alongside lagged sequence regression",
            "coverage": "implemented_partial",
            "next_step": "compare recurrent supervised signals with GRU PPO actors under the same QuantBench gate",
        },
        {
            "family": "attention_tide_mixer",
            "paper_family": "deep sequence / mixer style",
            "current_soxl_surface": "attention and TiDE actor surfaces exist; supervised TorchTCN temporal-convolution baseline implemented; MLP-Mixer missing",
            "coverage": "implemented_partial",
            "next_step": "compare TorchTCN and mixer-style signals under the same QuantBench gate before any actor retrain",
        },
        {
            "family": "relational_rgcn",
            "paper_family": "RGCN/relational model",
            "current_soxl_surface": "SOXX-relative and multi-asset relation features plus supervised MarketGraphRidge relational baseline implemented",
            "coverage": "implemented_partial",
            "next_step": "promote only if MarketGraphRidge or graph-derived blends satisfy the QuantBench gate; otherwise test richer graph learners",
        },
        {
            "family": "objectives_clf_ic_mse_ranking",
            "paper_family": "CLF / IC / MSE / Ranking",
            "current_soxl_surface": "common supervised target builder and CLF/IC/MSE/ranking screen implemented in quantbench_supervised",
            "coverage": "implemented",
            "next_step": "use objective metrics and risk-gate failures to choose the next macro-risk or rolling screen",
        },
        {
            "family": "utility_rl_execution",
            "paper_family": "utility maximization / RL / execution",
            "current_soxl_surface": "PPO reward, risk budget, execution overlays exist",
            "coverage": "implemented_partial",
            "next_step": "use QuantBench gate for promotion selection",
        },
        {
            "family": "rolling_continual",
            "paper_family": "3/6/12 month rolling updates",
            "current_soxl_surface": "rolling supervised refit paths and rolling hazard refits exist with non-leaking fold ledgers",
            "coverage": "implemented_partial",
            "next_step": "keep cadence only if QuantBench gate improves; otherwise use rolling diagnostics to adjust risk objective",
        },
        {
            "family": "seed_ensemble",
            "paper_family": "seed ensemble for overfit mitigation",
            "current_soxl_surface": "low-correlation signal blend paths and macro path-ensemble gates exist in quantbench_ensemble",
            "coverage": "implemented_partial",
            "next_step": "keep only blends that improve the QuantBench gate; add seed/checkpoint actor ensembles after individual gates pass",
        },
    ]


def summary_record_from_payload(payload: dict[str, Any], source_path: str = "") -> SummaryRecord:
    profile = str(payload.get("profile") or payload.get("name") or Path(source_path).parent.name)
    raw_slices = payload.get("slices") or {}
    slices = {
        str(name): _numeric_metrics(metrics if isinstance(metrics, dict) else {})
        for name, metrics in raw_slices.items()
    }
    return SummaryRecord(
        profile=profile,
        source_path=source_path,
        test_proxy=_numeric_metrics(payload.get("test_proxy") if isinstance(payload.get("test_proxy"), dict) else {}),
        slices=slices,
    )


def _display_path(path: Path, root: Path | None = None) -> str:
    if root is None:
        return str(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def load_summary_records(paths: Iterable[Path], root: Path | None = None) -> list[SummaryRecord]:
    records: list[SummaryRecord] = []
    for path in sorted(paths):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        records.append(summary_record_from_payload(payload, source_path=_display_path(path, root=root)))
    return records


def discover_quantbench_summary_paths(root: Path) -> list[Path]:
    """Find runtime summaries that use the app-layer training/evaluation schema."""
    summary_paths: list[Path] = []
    for relative in (
        "runtime/model/card",
        "runtime/execution/card",
        "runtime/reward/card",
    ):
        base = root / relative
        if not base.exists():
            continue
        summary_paths.extend(
            path
            for path in base.rglob("*.json")
            if path.name.endswith("_summary.json") or path.name == "summary.json"
        )
    return sorted(summary_paths)


def _capture_ratio(strategy_return: float, benchmark_return: float) -> float:
    if not math.isfinite(strategy_return) or not math.isfinite(benchmark_return) or abs(benchmark_return) < 1e-8:
        return float("nan")
    if benchmark_return <= 0:
        return float("nan")
    return strategy_return / benchmark_return


def screen_quantbench_candidates(
    records: Iterable[SummaryRecord],
    gate: QuantBenchGate = QuantBenchGate(),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        proxy = record.test_proxy
        stress = record.slices.get("test_2022", {})
        bull = record.slices.get("test_2023_2024", {})

        proxy_return = _as_float(proxy.get("return"))
        proxy_max_dd = _as_float(proxy.get("max_dd"))
        stress_relative = _as_float(stress.get("relative_return"))
        stress_max_dd = _as_float(stress.get("max_dd"))
        stress_window_avg_pos = _as_float(stress.get("avg_pos"))
        stress_regime_avg_pos = _as_float(stress.get("stress_regime_avg_pos"))
        stress_avg_pos = stress_regime_avg_pos if math.isfinite(stress_regime_avg_pos) else stress_window_avg_pos
        bull_return = _as_float(bull.get("return"))
        bull_bnh = _as_float(bull.get("buy_and_hold_return"))
        bull_capture = _capture_ratio(bull_return, bull_bnh)

        failed_checks: list[str] = []
        if not (math.isfinite(proxy_max_dd) and proxy_max_dd <= gate.max_drawdown_pct):
            failed_checks.append("max_drawdown")
        if not (math.isfinite(proxy_return) and proxy_return >= gate.min_proxy_return):
            failed_checks.append("proxy_return")
        if not (math.isfinite(stress_relative) and stress_relative >= gate.min_stress_relative_return):
            failed_checks.append("stress_relative_return")
        if not (math.isfinite(stress_avg_pos) and gate.min_stress_avg_pos <= stress_avg_pos <= gate.max_stress_avg_pos):
            failed_checks.append("stress_avg_pos")
        if not (math.isfinite(stress_max_dd) and stress_max_dd <= gate.max_drawdown_pct):
            failed_checks.append("stress_max_drawdown")
        if not (math.isfinite(bull_capture) and bull_capture >= gate.min_bull_capture):
            failed_checks.append("bull_capture")

        eligible = not failed_checks
        score = (
            (proxy_return if math.isfinite(proxy_return) else -1000.0)
            + (stress_relative if math.isfinite(stress_relative) else -1000.0)
            + (100.0 * bull_capture if math.isfinite(bull_capture) else -1000.0)
            - 4.0 * (proxy_max_dd if math.isfinite(proxy_max_dd) else 100.0)
            - 20.0 * len(failed_checks)
        )
        rows.append(
            {
                "profile": record.profile,
                "source_path": record.source_path,
                "eligible": eligible,
                "score": round(score, 6),
                "failed_checks": failed_checks,
                "proxy_return": proxy_return,
                "proxy_max_dd": proxy_max_dd,
                "stress_relative_return": stress_relative,
                "stress_max_dd": stress_max_dd,
                "stress_avg_pos": stress_avg_pos,
                "stress_window_avg_pos": stress_window_avg_pos,
                "stress_regime_avg_pos": stress_regime_avg_pos,
                "bull_return": bull_return,
                "bull_buy_and_hold_return": bull_bnh,
                "bull_capture": bull_capture,
            }
        )
    return sorted(rows, key=lambda row: (row["eligible"], row["score"]), reverse=True)


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_quantbench_screen_report(
    root: Path,
    card_dir: Path,
    gate: QuantBenchGate = QuantBenchGate(),
) -> dict[str, Any]:
    """Build the QuantBench-to-SOXL screen report and persist sidecar artifacts."""
    root = root.resolve()
    card_dir = card_dir.resolve()
    card_dir.mkdir(parents=True, exist_ok=True)

    requirement_rows = paper_requirement_matrix()
    coverage_rows = quantbench_model_objective_coverage()
    from .quantbench_objectives import quantbench_objective_families

    objective_rows = quantbench_objective_families()
    summary_paths = discover_quantbench_summary_paths(root)
    records = load_summary_records(summary_paths, root=root)
    screen_rows = screen_quantbench_candidates(records, gate=gate)
    eligible_rows = [row for row in screen_rows if row["eligible"]]

    requirement_path = card_dir / "paper_requirement_matrix.csv"
    coverage_path = card_dir / "paper_model_objective_coverage.csv"
    objective_path = card_dir / "paper_objective_families.csv"
    screen_path = card_dir / "candidate_screen.csv"
    summary_path = card_dir / "summary.json"
    _write_rows_csv(requirement_path, requirement_rows)
    _write_rows_csv(coverage_path, coverage_rows)
    _write_rows_csv(objective_path, objective_rows)
    _write_rows_csv(screen_path, screen_rows)

    def _rel(path: Path) -> str:
        return _display_path(path, root=root)

    report = {
        "paper": {
            "title": "QuantBench: Benchmarking AI Methods for Quantitative Investment",
            "arxiv_pdf": "https://arxiv.org/pdf/2504.18600",
            "mapped_claims": [
                "full-pipeline benchmark coverage",
                "tree-vs-deep model comparison",
                "CLF/IC/MSE/ranking objective comparison",
                "rolling update / alpha decay",
                "validation-set selection",
                "seed ensemble robustness",
            ],
        },
        "gate": gate.__dict__,
        "artifacts": {
            "summary_json": _rel(summary_path),
            "candidate_screen_csv": _rel(screen_path),
            "paper_requirement_matrix_csv": _rel(requirement_path),
            "paper_model_objective_coverage_csv": _rel(coverage_path),
            "paper_objective_families_csv": _rel(objective_path),
        },
        "screen": {
            "summary_files": len(summary_paths),
            "records": len(records),
            "eligible_count": len(eligible_rows),
            "eligible_profiles": [row["profile"] for row in eligible_rows[:10]],
            "best_rows": screen_rows[:10],
        },
        "decision": "eligible_candidates_found"
        if eligible_rows
        else "no_current_model_satisfies_quantbench_soxl_20dd_gate",
        "next_action": (
            "Compare the supervised screen sidecars to PPO/GRU summaries under the same gate, "
            "then test rolling or ensemble variants only when the risk failures justify it."
        ),
    }
    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    return report


__all__ = [
    "QuantBenchGate",
    "SummaryRecord",
    "build_quantbench_screen_report",
    "discover_quantbench_summary_paths",
    "load_summary_records",
    "paper_requirement_matrix",
    "quantbench_model_objective_coverage",
    "screen_quantbench_candidates",
    "summary_record_from_payload",
]
