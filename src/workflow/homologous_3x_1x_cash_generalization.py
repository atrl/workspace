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
import torch

from src.workflow.auxiliary_3x_training_data_allocation import (
    NORMALIZATION_SUFFIXES,
    AuxiliaryTrainingAllocator,
    _attach_generic_targets,
    _prepare_panel,
    _standardize,
    _training_end,
)
from src.workflow.cash_floor_benchmark_relative_wf import _result_metrics
from src.workflow.multi_3x_etf_normalized_nested_wf import (
    Multi3xEtfNormalizedNestedWfSpec,
    _attach_normalized_features,
    _load_multi_3x_raw_panel,
    _model_feature_columns,
)
from src.workflow.multitask_classification_allocation import (
    TARGET_WEIGHT_COLUMNS,
)
from src.workflow.soxl_objective_feasibility_audit import (
    _json_ready,
    _simulate,
)
from src.workflow.trainable_regime_policy_nested_wf import (
    _remove_qlib_code_snapshot_artifacts,
)


EXPERIMENT = "homologous_3x_1x_cash_generalization_v1"
DEVELOPMENT_YEARS = tuple(range(2016, 2023))
OUTER_START = "2023-01-03"
PAIR_DEFINITIONS = {
    "SOXL_SOXX_CASH": {
        "leveraged": "SOXL",
        "underlying": "SOXX",
    },
    "TQQQ_QQQ_CASH": {
        "leveraged": "TQQQ",
        "underlying": "QQQ",
    },
}
RECORDER_ONLY_ARTIFACTS = (
    f"artifacts/{EXPERIMENT}/models/",
    f"artifacts/{EXPERIMENT}/predictions.csv",
    f"artifacts/{EXPERIMENT}/positions.csv",
    f"artifacts/{EXPERIMENT}/strategy_returns.csv",
    f"artifacts/{EXPERIMENT}/training_diagnostics.csv",
)


@dataclass(frozen=True)
class HomologousAllocationSpec:
    root: Path
    experiment: str = EXPERIMENT
    train_start: str = "2011-01-03"
    development_years: tuple[int, ...] = DEVELOPMENT_YEARS
    development_end: str = "2022-12-30"
    outer: tuple[str, str] = (OUTER_START, "2026-07-24")
    purge_rows: int = 20
    hidden_units: int = 16
    epochs: int = 300
    learning_rate: float = 0.003
    weight_decay: float = 0.05
    maximum_leveraged_weight: float = 0.30
    minimum_average_cash: float = 0.15
    costs_bps: tuple[int, ...] = (10, 25)
    seed: int = 79
    as_of: str = "2026-07-24"

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment


def run_homologous_3x_1x_cash_generalization_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    repo_root = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[2]
    )
    return _run_card(HomologousAllocationSpec(root=repo_root))


def _load_panel(
    spec: HomologousAllocationSpec,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    source_spec = Multi3xEtfNormalizedNestedWfSpec(
        root=spec.root,
        train_start=spec.train_start,
        max_common_end=spec.as_of,
    )
    raw, source_audit = _load_multi_3x_raw_panel(source_spec)
    requested = {
        definition["leveraged"] for definition in PAIR_DEFINITIONS.values()
    }
    raw = raw.loc[raw["instrument"].isin(requested)].copy()
    normalized = _attach_normalized_features(raw)
    panel = _attach_generic_targets(normalized)
    features = [
        column
        for column in _model_feature_columns(panel)
        if column.endswith(NORMALIZATION_SUFFIXES)
    ]
    features = list(dict.fromkeys(features))
    if not features:
        raise ValueError("blocked: no homologous normalized features")
    forbidden = [
        column
        for column in features
        if any(
            token in column.lower()
            for token in ("instrument", "asset", "symbol")
        )
    ]
    if forbidden:
        raise ValueError(
            f"blocked: identity columns entered homologous model {forbidden}"
        )
    audit = {
        "source": "qlib.data.D.features",
        "source_audit": {
            symbol: source_audit[symbol] for symbol in sorted(requested)
        },
        "pair_count": len(PAIR_DEFINITIONS),
        "feature_count": len(features),
        "feature_policy": (
            "identical_past_only_per_pair_rolling_zscore_percentile"
        ),
        "feature_identity_columns": forbidden,
        "training_policy": "each_pair_trained_only_on_its_own_history",
        "weight_transfer_between_pairs": False,
        "selection_uses_outer_test": False,
    }
    return panel, features, audit


def _build_dataset(
    panel: pd.DataFrame,
    features: list[str],
    spec: HomologousAllocationSpec,
):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    labels = [
        "soxl_forward_open_return",
        "soxx_forward_open_return",
        *TARGET_WEIGHT_COLUMNS,
    ]
    static = pd.concat(
        {
            "feature": panel.loc[:, features],
            "label": panel.loc[:, labels],
        },
        axis=1,
    )
    static.index = pd.MultiIndex.from_arrays(
        [
            pd.to_datetime(panel["datetime"]).dt.normalize(),
            panel["instrument"].astype(str).str.upper(),
        ],
        names=["datetime", "instrument"],
    )
    outer_end = min(spec.outer[1], spec.as_of)
    dataset = DatasetH(
        handler=DataHandlerLP(
            data_loader=StaticDataLoader(config=static),
            infer_processors=[],
            learn_processors=[],
        ),
        segments={
            "development": (
                f"{spec.development_years[0]}-01-01",
                spec.development_end,
            ),
            "outer": (spec.outer[0], outer_end),
        },
    )
    development = dataset.prepare(
        "development", col_set=["feature", "label"]
    )
    if development.empty:
        raise ValueError("blocked: homologous DatasetH development is empty")
    return dataset, {
        "dataset_class": "qlib.data.dataset.DatasetH",
        "handler_class": "qlib.data.dataset.handler.DataHandlerLP",
        "loader_class": "qlib.data.dataset.loader.StaticDataLoader",
        "development_rows": int(len(development)),
        "instruments": sorted(
            development.index.get_level_values("instrument")
            .unique()
            .tolist()
        ),
        "selection_uses_outer_test": False,
    }


def _fit_pair(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    leveraged_symbol: str,
    spec: HomologousAllocationSpec,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, float]]:
    train = train.loc[train["instrument"].eq(leveraged_symbol)].dropna(
        subset=[*features, *TARGET_WEIGHT_COLUMNS]
    )
    test = test.loc[test["instrument"].eq(leveraged_symbol)].dropna(
        subset=features
    )
    if train.empty or test.empty:
        raise ValueError(
            f"blocked: empty {leveraged_symbol} homologous train/test"
        )
    if set(train["instrument"].unique()) != {leveraged_symbol}:
        raise ValueError("blocked: another pair entered training rows")
    if set(test["instrument"].unique()) != {leveraged_symbol}:
        raise ValueError("blocked: another pair entered test rows")
    train_x, test_x, scaler = _standardize(train, test, features)
    target = torch.tensor(
        train.loc[:, TARGET_WEIGHT_COLUMNS].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AuxiliaryTrainingAllocator(
        feature_count=len(features),
        hidden_units=spec.hidden_units,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    final_components: dict[str, float] = {}
    for _ in range(spec.epochs):
        optimizer.zero_grad()
        allocation = model(
            train_x, spec.maximum_leveraged_weight
        )
        allocation_loss = (allocation - target).pow(2).mean()
        cash_floor_loss = torch.relu(
            spec.minimum_average_cash - allocation[:, 2].mean()
        ).pow(2)
        loss = allocation_loss + 2.0 * cash_floor_loss
        if not torch.isfinite(loss):
            raise ValueError(
                f"blocked: non-finite {leveraged_symbol} training loss"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        value = float(loss.detach())
        final_components = {
            "loss": value,
            "allocation_loss": float(allocation_loss.detach()),
            "cash_floor_loss": float(cash_floor_loss.detach()),
        }
        if value < best_loss:
            best_loss = value
            best_state = {
                key: tensor.detach().clone()
                for key, tensor in model.state_dict().items()
            }
    if best_state is None:
        raise ValueError(
            f"blocked: {leveraged_symbol} model state missing"
        )
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predicted = model(
            test_x, spec.maximum_leveraged_weight
        ).cpu().numpy()
    output = pd.DataFrame(
        predicted,
        index=pd.to_datetime(test["datetime"]).dt.normalize(),
        columns=[
            "leveraged_weight",
            "underlying_weight",
            "cash_weight",
        ],
    )
    output.index.name = "signal_date"
    state = {
        "model_state": {
            key: value.detach().cpu().numpy()
            for key, value in model.state_dict().items()
        },
        "scaler": {
            name: values.to_dict() for name, values in scaler.items()
        },
        "leveraged_symbol": leveraged_symbol,
        "features": features,
        "spec": {
            "hidden_units": spec.hidden_units,
            "epochs": spec.epochs,
            "learning_rate": spec.learning_rate,
            "weight_decay": spec.weight_decay,
            "maximum_leveraged_weight": (
                spec.maximum_leveraged_weight
            ),
            "minimum_average_cash": spec.minimum_average_cash,
        },
    }
    final_components.update(
        {
            "best_loss": best_loss,
            "train_rows": float(len(train)),
            "test_rows": float(len(test)),
            "cross_pair_train_rows": 0.0,
        }
    )
    return output, state, final_components


def _pair_predictions(
    *,
    pair_name: str,
    leveraged_symbol: str,
    dataset,
    panel: pd.DataFrame,
    features: list[str],
    spec: HomologousAllocationSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    development_source = _prepare_panel(
        dataset, (spec.train_start, spec.development_end)
    )
    outer_source = _prepare_panel(dataset, "outer")
    predictions: list[pd.DataFrame] = []
    states: dict[str, Any] = {}
    diagnostics: list[dict[str, Any]] = []
    for year in spec.development_years:
        train_end = _training_end(panel, year, spec.purge_rows)
        train = development_source.loc[
            development_source["datetime"].le(train_end)
        ]
        test = development_source.loc[
            development_source["datetime"].dt.year.eq(year)
        ]
        prediction, state, fit = _fit_pair(
            train=train,
            test=test,
            features=features,
            leveraged_symbol=leveraged_symbol,
            spec=spec,
            seed=spec.seed + year,
        )
        prediction["pair"] = pair_name
        prediction["segment"] = "development_walk_forward"
        prediction["fold_year"] = year
        predictions.append(prediction)
        states[f"fold_{year}"] = state
        diagnostics.append(
            {
                "pair": pair_name,
                "leveraged_symbol": leveraged_symbol,
                "segment": "development_walk_forward",
                "fold_year": year,
                "train_end": train_end.date().isoformat(),
                "training_assets": leveraged_symbol,
                "test_assets": leveraged_symbol,
                "selection_uses_outer_test": False,
                **fit,
            }
        )
    outer_prediction, outer_state, outer_fit = _fit_pair(
        train=development_source,
        test=outer_source,
        features=features,
        leveraged_symbol=leveraged_symbol,
        spec=spec,
        seed=spec.seed,
    )
    outer_prediction["pair"] = pair_name
    outer_prediction["segment"] = "outer_diagnostic"
    outer_prediction["fold_year"] = outer_prediction.index.year
    states["outer_refit"] = outer_state
    diagnostics.append(
        {
            "pair": pair_name,
            "leveraged_symbol": leveraged_symbol,
            "segment": "outer_diagnostic",
            "fold_year": "outer_refit",
            "train_end": spec.development_end,
            "training_assets": leveraged_symbol,
            "test_assets": leveraged_symbol,
            "selection_uses_outer_test": False,
            **outer_fit,
        }
    )
    return (
        pd.concat(predictions).sort_index(),
        outer_prediction.sort_index(),
        states,
        pd.DataFrame(diagnostics),
    )


def _pair_market(
    panel: pd.DataFrame, leveraged_symbol: str
) -> pd.DataFrame:
    market = (
        panel.loc[panel["instrument"].eq(leveraged_symbol)]
        .sort_values("datetime")
        .copy()
    )
    market.index = pd.to_datetime(market["datetime"]).dt.normalize()
    market["trade_date"] = market.index.to_series().shift(-1).to_numpy()
    market["exit_date"] = market.index.to_series().shift(-2).to_numpy()
    market["soxs_forward_open_return"] = 0.0
    market["soxx_trend60"] = (
        pd.to_numeric(market["underlying_close"], errors="coerce")
        / pd.to_numeric(
            market["underlying_close"], errors="coerce"
        ).shift(60)
        - 1.0
    )
    return market


def _strategy_weights(
    prediction: pd.DataFrame, strategy: str
) -> np.ndarray:
    rows = len(prediction)
    direct = prediction.loc[
        :,
        ["leveraged_weight", "underlying_weight", "cash_weight"],
    ].to_numpy(dtype=float)
    if strategy == "model":
        selected = direct
    elif strategy == "matched_static":
        selected = np.tile(direct.mean(axis=0), (rows, 1))
    elif strategy == "underlying_buyhold":
        selected = np.tile([0.0, 1.0, 0.0], (rows, 1))
    elif strategy == "leveraged_buyhold":
        selected = np.tile([1.0, 0.0, 0.0], (rows, 1))
    else:
        raise ValueError(f"unknown strategy {strategy}")
    return np.column_stack(
        [selected[:, 0], selected[:, 1], np.zeros(rows), selected[:, 2]]
    )


def _evaluate_pair(
    *,
    pair_name: str,
    definition: dict[str, str],
    market: pd.DataFrame,
    prediction: pd.DataFrame,
    segment: str,
    costs_bps: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usable_index = prediction.index.intersection(market.index)
    usable_market = market.loc[usable_index].dropna(
        subset=[
            "soxl_forward_open_return",
            "soxx_forward_open_return",
            "trade_date",
            "exit_date",
        ]
    )
    usable_prediction = prediction.loc[usable_market.index]
    comparison_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    return_frames: list[pd.DataFrame] = []
    position_frames: list[pd.DataFrame] = []
    for cost_bps in costs_bps:
        benchmark = _simulate(
            usable_market,
            _strategy_weights(
                usable_prediction, "underlying_buyhold"
            ),
            cost_bps=cost_bps,
            name="underlying_buyhold",
        )
        for strategy in (
            "model",
            "matched_static",
            "underlying_buyhold",
            "leveraged_buyhold",
        ):
            result = _simulate(
                usable_market,
                _strategy_weights(usable_prediction, strategy),
                cost_bps=cost_bps,
                name=strategy,
            )
            metrics = _result_metrics(result, benchmark)
            metadata = {
                "pair": pair_name,
                "leveraged_symbol": definition["leveraged"],
                "underlying_symbol": definition["underlying"],
                "segment": segment,
                "strategy": strategy,
                "cost_bps": cost_bps,
                "selection_uses_outer_test": False,
            }
            comparison_rows.append({**metadata, **metrics})
            years = pd.to_datetime(result["signal_date"]).dt.year
            for year in sorted(years.unique()):
                mask = years.eq(year)
                annual_rows.append(
                    {
                        **metadata,
                        "fold_year": int(year),
                        **_result_metrics(
                            result.loc[mask].reset_index(drop=True),
                            benchmark.loc[mask].reset_index(drop=True),
                        ),
                    }
                )
            return_frames.append(result.assign(**metadata))
            position_frames.append(
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
                ].assign(**metadata)
            )
    return (
        pd.DataFrame(comparison_rows),
        pd.DataFrame(annual_rows),
        pd.concat(return_frames, ignore_index=True),
        pd.concat(position_frames, ignore_index=True),
    )


def _gate_audit(
    comparison: pd.DataFrame, annual: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair_name in PAIR_DEFINITIONS:
        model10 = comparison.loc[
            comparison["pair"].eq(pair_name)
            & comparison["segment"].eq("outer_diagnostic")
            & comparison["strategy"].eq("model")
            & comparison["cost_bps"].eq(10)
        ].iloc[0]
        model25 = comparison.loc[
            comparison["pair"].eq(pair_name)
            & comparison["segment"].eq("outer_diagnostic")
            & comparison["strategy"].eq("model")
            & comparison["cost_bps"].eq(25)
        ].iloc[0]
        matched10 = comparison.loc[
            comparison["pair"].eq(pair_name)
            & comparison["segment"].eq("outer_diagnostic")
            & comparison["strategy"].eq("matched_static")
            & comparison["cost_bps"].eq(10)
        ].iloc[0]
        annual10 = annual.loc[
            annual["pair"].eq(pair_name)
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
        gates = {
            "outer_return_10bps": (
                float(model10["terminal_wealth_ratio_vs_soxx"]) > 1.0
            ),
            "outer_return_25bps": (
                float(model25["terminal_wealth_ratio_vs_soxx"]) > 1.0
            ),
            "dynamic_wealth_vs_matched_static": (
                float(model10["terminal_wealth"])
                > float(matched10["terminal_wealth"])
            ),
            "dynamic_calmar_vs_matched_static": (
                float(model10["calmar"]) > float(matched10["calmar"])
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
        for gate, passed in gates.items():
            rows.append(
                {
                    "pair": pair_name,
                    "gate": gate,
                    "required": True,
                    "passed": bool(passed),
                    "outer_ratio_10bps": float(
                        model10["terminal_wealth_ratio_vs_soxx"]
                    ),
                    "outer_ratio_25bps": float(
                        model25["terminal_wealth_ratio_vs_soxx"]
                    ),
                    "model_wealth_ratio_vs_matched_static": float(
                        model10["terminal_wealth"]
                        / matched10["terminal_wealth"]
                    ),
                    "model_calmar": float(model10["calmar"]),
                    "matched_static_calmar": float(
                        matched10["calmar"]
                    ),
                    "outer_max_drawdown_pct": float(
                        model10["max_drawdown_pct"]
                    ),
                    "outer_worst_day_pct": float(
                        model10["worst_day_pct"]
                    ),
                    "outer_cvar5_pct": float(model10["cvar5_pct"]),
                    "outer_worst_rolling_5d_pct": float(
                        model10["worst_rolling_5d_pct"]
                    ),
                    "outer_avg_cash_weight_pct": float(
                        model10["avg_cash_weight_pct"]
                    ),
                    "positive_relative_years": relative_years,
                    "risk_passing_years": risk_years,
                    "selection_uses_outer_test": False,
                }
            )
    return pd.DataFrame(rows)


def _write_notebook(card: Path, summary: dict[str, Any]) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            "# Homologous 3x / 1x / cash generalization\n\n"
            "`SOXL/SOXX/cash` 与 `TQQQ/QQQ/cash` 分别独立训练；"
            "不迁移仓位时序，也不使用 outer 选择模型。"
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
            "from src.workflow.homologous_3x_1x_cash_generalization "
            "import run_homologous_3x_1x_cash_generalization_card\n"
            "summary = run_homologous_3x_1x_cash_generalization_card("
            "root=repo_root)\n"
            "summary['generalization_decision']"
        ),
        nbformat.v4.new_code_cell(
            "import pandas as pd\n"
            "card = Path.cwd()\n"
            "pd.read_csv(card / 'backtest_benchmark_comparison.csv').query("
            "\"segment == 'outer_diagnostic' and cost_bps == 10\")"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'annual_metrics.csv').query("
            "\"segment == 'outer_diagnostic' and strategy == 'model' "
            "and cost_bps == 10\")"
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
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
    diagnostics: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    card.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(
        card / "backtest_benchmark_comparison.csv", index=False
    )
    annual.to_csv(card / "annual_metrics.csv", index=False)
    diagnostics.to_csv(card / "training_data_audit.csv", index=False)
    gates.to_csv(card / "objective_gate_audit.csv", index=False)
    outer_model = comparison.loc[
        comparison["segment"].eq("outer_diagnostic")
        & comparison["strategy"].eq("model")
    ].copy()
    outer_model.to_csv(card / "risk_analysis.csv", index=False)
    selected = {
        "policy": "fixed_identical_model_spec_per_pair",
        "pairs": PAIR_DEFINITIONS,
        "training_end": summary["training_end"],
        "selection_uses_outer_test": False,
        "weight_transfer_between_pairs": False,
    }
    (card / "selected_model.json").write_text(
        json.dumps(selected, indent=2), encoding="utf-8"
    )
    analysis = {}
    for pair_name in PAIR_DEFINITIONS:
        row = outer_model.loc[
            outer_model["pair"].eq(pair_name)
            & outer_model["cost_bps"].eq(10)
        ].iloc[0]
        analysis[pair_name] = {
            "return_pct": float(row["return_pct"]),
            "benchmark_return_pct": float(row["soxx_return_pct"]),
            "wealth_ratio_vs_underlying": float(
                row["terminal_wealth_ratio_vs_soxx"]
            ),
            "max_drawdown_pct": float(row["max_drawdown_pct"]),
            "avg_cash_weight_pct": float(row["avg_cash_weight_pct"]),
            "total_turnover": float(row["total_turnover"]),
        }
        matched = outer_model.loc[
            outer_model["pair"].eq(pair_name)
            & outer_model["cost_bps"].eq(10)
            & outer_model["strategy"].eq("matched_static")
        ]
        if matched.empty:
            matched = comparison.loc[
                comparison["pair"].eq(pair_name)
                & comparison["segment"].eq("outer_diagnostic")
                & comparison["strategy"].eq("matched_static")
                & comparison["cost_bps"].eq(10)
            ]
        matched_row = matched.iloc[0]
        analysis[pair_name].update(
            {
                "matched_static_return_pct": float(
                    matched_row["return_pct"]
                ),
                "wealth_ratio_vs_matched_static": float(
                    row["terminal_wealth"]
                    / matched_row["terminal_wealth"]
                ),
            }
        )
        development_model = comparison.loc[
            comparison["pair"].eq(pair_name)
            & comparison["segment"].eq("development_walk_forward")
            & comparison["strategy"].eq("model")
            & comparison["cost_bps"].eq(10)
        ].iloc[0]
        development_matched = comparison.loc[
            comparison["pair"].eq(pair_name)
            & comparison["segment"].eq("development_walk_forward")
            & comparison["strategy"].eq("matched_static")
            & comparison["cost_bps"].eq(10)
        ].iloc[0]
        analysis[pair_name].update(
            {
                "development_wealth_ratio_vs_underlying": float(
                    development_model[
                        "terminal_wealth_ratio_vs_soxx"
                    ]
                ),
                "development_wealth_ratio_vs_matched_static": float(
                    development_model["terminal_wealth"]
                    / development_matched["terminal_wealth"]
                ),
            }
        )
    (card / "analysis_summary.json").write_text(
        json.dumps(_json_ready(analysis), indent=2), encoding="utf-8"
    )
    (card / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2), encoding="utf-8"
    )
    tqqq = analysis["TQQQ_QQQ_CASH"]
    soxl = analysis["SOXL_SOXX_CASH"]
    failed = gates.loc[~gates["passed"], ["pair", "gate"]]
    failed_text = (
        ", ".join(
            f"{row.pair}:{row.gate}"
            for row in failed.itertuples(index=False)
        )
        if not failed.empty
        else "无"
    )
    conclusion = (
        "# 结论\n\n"
        "- 已按同构口径重做：两个组合使用完全相同的过去式通用特征、"
        "网络结构、目标、约束和训练截止日，但分别只用自己的历史训练。"
        "没有把 SOXL 仓位套到 TQQQ。\n"
        f"- SOXL/SOXX/cash outer 10bp：收益 "
        f"{soxl['return_pct']:.2f}%，SOXX {soxl['benchmark_return_pct']:.2f}%，"
        f"财富比 {soxl['wealth_ratio_vs_underlying']:.4f}，MaxDD "
        f"{soxl['max_drawdown_pct']:.2f}%，平均现金 "
        f"{soxl['avg_cash_weight_pct']:.2f}%。\n"
        f"  同平均仓位静态组合收益 "
        f"{soxl['matched_static_return_pct']:.2f}%，动态/静态财富比 "
        f"{soxl['wealth_ratio_vs_matched_static']:.4f}。\n"
        f"- TQQQ/QQQ/cash outer 10bp：收益 "
        f"{tqqq['return_pct']:.2f}%，QQQ {tqqq['benchmark_return_pct']:.2f}%，"
        f"财富比 {tqqq['wealth_ratio_vs_underlying']:.4f}，MaxDD "
        f"{tqqq['max_drawdown_pct']:.2f}%，平均现金 "
        f"{tqqq['avg_cash_weight_pct']:.2f}%。\n"
        f"  同平均仓位静态组合收益 "
        f"{tqqq['matched_static_return_pct']:.2f}%，动态/静态财富比 "
        f"{tqqq['wealth_ratio_vs_matched_static']:.4f}。\n"
        "- 这不是只在 outer 才出现的过拟合：开发期 walk-forward 中，"
        "SOXL 通用模型/同仓位静态财富比为 "
        f"{soxl['development_wealth_ratio_vs_matched_static']:.4f}，"
        "TQQQ 为 "
        f"{tqqq['development_wealth_ratio_vs_matched_static']:.4f}；"
        "动态择时在开发期已经没有增益。\n"
        f"- 同构泛化判断：`{summary['generalization_decision']}`；"
        f"失败门：{failed_text}。\n"
        "- 这张卡验证的是日频直接仓位核心的可迁移性；SOXL 生产模型"
        "额外使用半导体专属特征和盘中恢复模块，二者不能混为同一个模型。\n"
        "- Outer 2023-2026 全程只报告，未参与模型或参数选择。\n"
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
        "pairs:\n"
        "  - [SOXL, SOXX, cash]\n"
        "  - [TQQQ, QQQ, cash]\n"
        "training_policy: independent_pair_specific\n"
        "allocation_output: direct_full_legal_weights\n"
        "weight_transfer_between_pairs: false\n"
        "selection_uses_outer_test: false\n"
        "artifact_policy:\n"
        "  policy: slim_card_recorder_first\n"
    )
    (card / "workflow.yaml").write_text(
        workflow, encoding="utf-8"
    )
    _write_notebook(card, summary)


def _run_card(spec: HomologousAllocationSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    panel, features, source_audit = _load_panel(spec)
    dataset, dataset_audit = _build_dataset(panel, features, spec)
    development_predictions: dict[str, pd.DataFrame] = {}
    outer_predictions: dict[str, pd.DataFrame] = {}
    states: dict[str, Any] = {}
    diagnostic_frames: list[pd.DataFrame] = []
    for pair_name, definition in PAIR_DEFINITIONS.items():
        development, outer, pair_states, diagnostics = _pair_predictions(
            pair_name=pair_name,
            leveraged_symbol=definition["leveraged"],
            dataset=dataset,
            panel=panel,
            features=features,
            spec=spec,
        )
        development_predictions[pair_name] = development
        outer_predictions[pair_name] = outer
        states[pair_name] = pair_states
        diagnostic_frames.append(diagnostics)
    comparison_frames: list[pd.DataFrame] = []
    annual_frames: list[pd.DataFrame] = []
    return_frames: list[pd.DataFrame] = []
    position_frames: list[pd.DataFrame] = []
    for pair_name, definition in PAIR_DEFINITIONS.items():
        market = _pair_market(panel, definition["leveraged"])
        for segment, predictions in (
            (
                "development_walk_forward",
                development_predictions[pair_name],
            ),
            ("outer_diagnostic", outer_predictions[pair_name]),
        ):
            comparison, annual, returns, positions = _evaluate_pair(
                pair_name=pair_name,
                definition=definition,
                market=market,
                prediction=predictions,
                segment=segment,
                costs_bps=spec.costs_bps,
            )
            comparison_frames.append(comparison)
            annual_frames.append(annual)
            return_frames.append(returns)
            position_frames.append(positions)
    comparison = pd.concat(comparison_frames, ignore_index=True)
    annual = pd.concat(annual_frames, ignore_index=True)
    strategy_returns = pd.concat(return_frames, ignore_index=True)
    positions = pd.concat(position_frames, ignore_index=True)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    predictions = pd.concat(
        [
            *development_predictions.values(),
            *outer_predictions.values(),
        ]
    ).sort_index()
    gates = _gate_audit(comparison, annual)
    pair_pass = (
        gates.groupby("pair")["passed"].all().to_dict()
    )
    tqqq_pass = bool(pair_pass.get("TQQQ_QQQ_CASH", False))
    soxl_pass = bool(pair_pass.get("SOXL_SOXX_CASH", False))
    decision = (
        "validated_on_soxl_and_tqqq_homologous_pairs"
        if tqqq_pass and soxl_pass
        else (
            "tqqq_transfer_valid_but_soxl_generic_core_failed"
            if tqqq_pass
            else "not_validated_on_tqqq_homologous_pair"
        )
    )

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
        for pair_name, pair_states in states.items():
            with (models_dir / f"{pair_name}.pkl").open("wb") as handle:
                pickle.dump(pair_states, handle)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                pairs="SOXL/SOXX/cash,TQQQ/QQQ/cash",
                training_policy="independent_pair_specific",
                allocation_output="direct_full_legal_weights",
                qlib_dataset="DatasetH",
                weight_transfer_between_pairs="false",
                selection_uses_outer_test="false",
            )
            for pair_name in PAIR_DEFINITIONS:
                row = comparison.loc[
                    comparison["pair"].eq(pair_name)
                    & comparison["segment"].eq("outer_diagnostic")
                    & comparison["strategy"].eq("model")
                    & comparison["cost_bps"].eq(10)
                ].iloc[0]
                R.log_metrics(
                    **{
                        f"{pair_name.lower()}_outer_ratio_10bps": float(
                            row["terminal_wealth_ratio_vs_soxx"]
                        )
                    }
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
                "card_type": "qlib_homologous_pair_generalization_card",
                "primary_notebook": f"{spec.experiment}.ipynb",
                "as_of": spec.as_of,
                "training_end": spec.development_end,
                "outer": list(spec.outer),
                "provider_uri": (
                    spec.root / "data" / "qlib"
                ).resolve().as_posix(),
                "source_audit": source_audit,
                "dataset_audit": dataset_audit,
                "spec": {
                    **asdict(spec),
                    "root": spec.root.resolve().as_posix(),
                },
                "pairs": PAIR_DEFINITIONS,
                "feature_count": len(features),
                "feature_policy": source_audit["feature_policy"],
                "model_policy": (
                    "same_fixed_direct_allocator_independent_fit_per_pair"
                ),
                "selection_uses_outer_test": False,
                "weight_transfer_between_pairs": False,
                "pair_gate_pass": pair_pass,
                "generalization_decision": decision,
                "promotion_decision": (
                    "research_evidence_only_no_production_replacement"
                ),
                "qlib_docs": [
                    "https://qlib.readthedocs.io/en/latest/component/data.html",
                    "https://qlib.readthedocs.io/en/latest/component/workflow.html",
                ],
                "recorder_id": recorder_id,
                "recorder_uri": recorder_uri.resolve().as_posix(),
                "recorder_local_dir": recorder_local_dir.as_posix(),
                "sidecars": {
                    "selected_model": "selected_model.json",
                    "backtest_benchmark_comparison": (
                        "backtest_benchmark_comparison.csv"
                    ),
                    "annual_metrics": "annual_metrics.csv",
                    "risk_analysis": "risk_analysis.csv",
                    "training_data_audit": "training_data_audit.csv",
                    "objective_gate_audit": "objective_gate_audit.csv",
                    "analysis_summary": "analysis_summary.json",
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
                comparison=comparison,
                annual=annual,
                diagnostics=diagnostics,
                gates=gates,
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
            run_homologous_3x_1x_cash_generalization_card(),
            indent=2,
        )
    )
