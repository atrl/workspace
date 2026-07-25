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
from torch import nn

from src.workflow.auxiliary_3x_training_data_allocation import (
    _prepare_panel,
    _standardize,
    _training_end,
)
from src.workflow.homologous_3x_1x_cash_generalization import (
    PAIR_DEFINITIONS,
    HomologousAllocationSpec,
    _build_dataset,
    _evaluate_pair,
    _load_panel,
    _pair_market,
)
from src.workflow.multitask_classification_allocation import (
    TARGET_WEIGHT_COLUMNS,
)
from src.workflow.soxl_objective_feasibility_audit import _json_ready
from src.workflow.trainable_regime_policy_nested_wf import (
    _remove_qlib_code_snapshot_artifacts,
)


EXPERIMENT = "cross_pair_auxiliary_training_v1"
RECORDER_ONLY_ARTIFACTS = (
    f"artifacts/{EXPERIMENT}/models/",
    f"artifacts/{EXPERIMENT}/predictions.csv",
    f"artifacts/{EXPERIMENT}/positions.csv",
    f"artifacts/{EXPERIMENT}/strategy_returns.csv",
    f"artifacts/{EXPERIMENT}/training_diagnostics.csv",
)


@dataclass(frozen=True)
class CrossPairCandidate:
    name: str
    auxiliary_weight: float
    turnover_penalty: float


CANDIDATES = (
    CrossPairCandidate("own_only_mse", 0.0, 0.0),
    CrossPairCandidate("own_only_turnover", 0.0, 0.02),
    CrossPairCandidate("cross_aux25_turnover", 0.25, 0.02),
    CrossPairCandidate("cross_aux50_turnover", 0.50, 0.02),
    CrossPairCandidate("cross_equal_turnover", 1.00, 0.02),
)


@dataclass(frozen=True)
class CrossPairTrainingSpec:
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
    seed: int = 97
    as_of: str = "2026-07-24"

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def source_spec(self) -> HomologousAllocationSpec:
        return HomologousAllocationSpec(
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
            minimum_average_cash=self.minimum_cash_weight,
            costs_bps=self.costs_bps,
            seed=self.seed,
            as_of=self.as_of,
        )


class CashConstrainedAllocationNet(nn.Module):
    """Predict all three weights inside the legal simplex."""

    def __init__(self, feature_count: int, hidden_units: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, 2),
        )

    def forward(
        self,
        features: torch.Tensor,
        *,
        maximum_leveraged_weight: float,
        minimum_cash_weight: float,
    ) -> torch.Tensor:
        logits = self.network(features)
        leveraged = maximum_leveraged_weight * torch.sigmoid(
            logits[:, 0]
        )
        available_after_leverage = 1.0 - leveraged
        cash = minimum_cash_weight + (
            available_after_leverage - minimum_cash_weight
        ) * torch.sigmoid(logits[:, 1])
        underlying = 1.0 - leveraged - cash
        return torch.stack([leveraged, underlying, cash], dim=1)


def run_cross_pair_auxiliary_training_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    repo_root = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[2]
    )
    return _run_card(CrossPairTrainingSpec(root=repo_root))


def _other_leveraged_symbol(primary_symbol: str) -> str:
    symbols = {
        definition["leveraged"]
        for definition in PAIR_DEFINITIONS.values()
    }
    others = sorted(symbols - {primary_symbol})
    if len(others) != 1:
        raise ValueError("blocked: cross-pair universe must contain two pairs")
    return others[0]


def _candidate_asset_weights(
    candidate: CrossPairCandidate, primary_symbol: str
) -> dict[str, float]:
    weights = {primary_symbol: 1.0}
    if candidate.auxiliary_weight > 0.0:
        weights[_other_leveraged_symbol(primary_symbol)] = (
            candidate.auxiliary_weight
        )
    return weights


def _fit_candidate(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    primary_symbol: str,
    candidate: CrossPairCandidate,
    spec: CrossPairTrainingSpec,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, float]]:
    asset_weights = _candidate_asset_weights(candidate, primary_symbol)
    train = (
        train.loc[train["instrument"].isin(asset_weights)]
        .dropna(subset=[*features, *TARGET_WEIGHT_COLUMNS])
        .sort_values(["instrument", "datetime"])
        .copy()
    )
    test = (
        test.loc[test["instrument"].eq(primary_symbol)]
        .dropna(subset=features)
        .sort_values("datetime")
        .copy()
    )
    if train.empty or test.empty:
        raise ValueError(
            f"blocked: empty {candidate.name} {primary_symbol} train/test"
        )
    if not set(train["instrument"]).issubset(asset_weights):
        raise ValueError("blocked: unexpected asset entered candidate train")
    if set(test["instrument"].unique()) != {primary_symbol}:
        raise ValueError("blocked: auxiliary asset entered candidate test")

    train_x, test_x, scaler = _standardize(train, test, features)
    target = torch.tensor(
        train.loc[:, TARGET_WEIGHT_COLUMNS].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    sample_weight = torch.tensor(
        train["instrument"]
        .map(asset_weights)
        .to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    sample_weight = sample_weight / sample_weight.mean()
    same_asset = torch.tensor(
        train["instrument"]
        .shift(1)
        .eq(train["instrument"])
        .to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )[1:]

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CashConstrainedAllocationNet(
        feature_count=len(features), hidden_units=spec.hidden_units
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
            train_x,
            maximum_leveraged_weight=spec.maximum_leveraged_weight,
            minimum_cash_weight=spec.minimum_cash_weight,
        )
        row_mse = (allocation - target).pow(2).mean(dim=1)
        allocation_loss = (row_mse * sample_weight).mean()
        changes = 0.5 * (allocation[1:] - allocation[:-1]).abs().sum(
            dim=1
        )
        if float(same_asset.sum()) > 0.0:
            turnover_loss = (changes * same_asset).sum() / same_asset.sum()
        else:
            turnover_loss = torch.zeros((), dtype=torch.float32)
        loss = (
            allocation_loss
            + candidate.turnover_penalty * turnover_loss
        )
        if not torch.isfinite(loss):
            raise ValueError(
                f"blocked: non-finite {candidate.name} training loss"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        value = float(loss.detach())
        final_components = {
            "loss": value,
            "allocation_loss": float(allocation_loss.detach()),
            "turnover_loss": float(turnover_loss.detach()),
        }
        if value < best_loss:
            best_loss = value
            best_state = {
                key: tensor.detach().clone()
                for key, tensor in model.state_dict().items()
            }
    if best_state is None:
        raise ValueError(f"blocked: {candidate.name} state missing")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predicted = model(
            test_x,
            maximum_leveraged_weight=spec.maximum_leveraged_weight,
            minimum_cash_weight=spec.minimum_cash_weight,
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
        "candidate": asdict(candidate),
        "primary_symbol": primary_symbol,
        "asset_weights": asset_weights,
        "features": features,
    }
    auxiliary_symbol = _other_leveraged_symbol(primary_symbol)
    final_components.update(
        {
            "best_loss": best_loss,
            "train_rows": float(len(train)),
            "test_rows": float(len(test)),
            "auxiliary_train_rows": float(
                train["instrument"].eq(auxiliary_symbol).sum()
            ),
            "primary_train_rows": float(
                train["instrument"].eq(primary_symbol).sum()
            ),
        }
    )
    return output, state, final_components


def _candidate_predictions(
    *,
    pair_name: str,
    primary_symbol: str,
    candidate: CrossPairCandidate,
    dataset,
    panel: pd.DataFrame,
    features: list[str],
    spec: CrossPairTrainingSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    development_source = _prepare_panel(
        dataset, (spec.train_start, spec.development_end)
    )
    outer_source = _prepare_panel(dataset, "outer")
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
            primary_symbol=primary_symbol,
            candidate=candidate,
            spec=spec,
            seed=spec.seed + year,
        )
        prediction["pair"] = pair_name
        prediction["candidate"] = candidate.name
        prediction["segment"] = "development_walk_forward"
        prediction["fold_year"] = year
        development_predictions.append(prediction)
        states[f"fold_{year}"] = state
        diagnostics.append(
            {
                "pair": pair_name,
                "candidate": candidate.name,
                "segment": "development_walk_forward",
                "fold_year": year,
                "train_end": train_end.date().isoformat(),
                "primary_symbol": primary_symbol,
                "auxiliary_symbol": _other_leveraged_symbol(
                    primary_symbol
                ),
                "auxiliary_weight": candidate.auxiliary_weight,
                "turnover_penalty": candidate.turnover_penalty,
                "selection_uses_outer_test": False,
                **fit,
            }
        )
    outer_prediction, outer_state, outer_fit = _fit_candidate(
        train=development_source,
        test=outer_source,
        features=features,
        primary_symbol=primary_symbol,
        candidate=candidate,
        spec=spec,
        seed=spec.seed,
    )
    outer_prediction["pair"] = pair_name
    outer_prediction["candidate"] = candidate.name
    outer_prediction["segment"] = "outer_diagnostic"
    outer_prediction["fold_year"] = outer_prediction.index.year
    states["outer_refit"] = outer_state
    diagnostics.append(
        {
            "pair": pair_name,
            "candidate": candidate.name,
            "segment": "outer_diagnostic",
            "fold_year": "outer_refit",
            "train_end": spec.development_end,
            "primary_symbol": primary_symbol,
            "auxiliary_symbol": _other_leveraged_symbol(primary_symbol),
            "auxiliary_weight": candidate.auxiliary_weight,
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
    development_predictions: dict[tuple[str, str], pd.DataFrame],
    outer_predictions: dict[tuple[str, str], pd.DataFrame],
    spec: CrossPairTrainingSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparisons: list[pd.DataFrame] = []
    annual_frames: list[pd.DataFrame] = []
    return_frames: list[pd.DataFrame] = []
    position_frames: list[pd.DataFrame] = []
    for pair_name, definition in PAIR_DEFINITIONS.items():
        market = _pair_market(panel, definition["leveraged"])
        for candidate in CANDIDATES:
            key = (pair_name, candidate.name)
            for segment, predictions in (
                (
                    "development_walk_forward",
                    development_predictions[key],
                ),
                ("outer_diagnostic", outer_predictions[key]),
            ):
                comparison, annual, returns, positions = _evaluate_pair(
                    pair_name=pair_name,
                    definition=definition,
                    market=market,
                    prediction=predictions,
                    segment=segment,
                    costs_bps=spec.costs_bps,
                )
                for frame in (comparison, annual, returns, positions):
                    frame["candidate"] = candidate.name
                    frame["auxiliary_weight"] = (
                        candidate.auxiliary_weight
                    )
                    frame["turnover_penalty"] = (
                        candidate.turnover_penalty
                    )
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


def _select_candidates(
    comparison: pd.DataFrame, annual: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair_name in PAIR_DEFINITIONS:
        for candidate in CANDIDATES:
            selector = (
                comparison["pair"].eq(pair_name)
                & comparison["candidate"].eq(candidate.name)
                & comparison["segment"].eq(
                    "development_walk_forward"
                )
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
                annual["pair"].eq(pair_name)
                & annual["candidate"].eq(candidate.name)
                & annual["segment"].eq(
                    "development_walk_forward"
                )
                & annual["strategy"].eq("model")
                & annual["cost_bps"].eq(10)
            ]
            row = {
                "pair": pair_name,
                "candidate": candidate.name,
                "auxiliary_weight": candidate.auxiliary_weight,
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
                "return_pct": float(model10["return_pct"]),
                "max_drawdown_pct": float(
                    model10["max_drawdown_pct"]
                ),
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
                    annual10[
                        "terminal_wealth_ratio_vs_soxx"
                    ].gt(1.0).sum()
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
    gate_columns = [
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
    selection["gate_count"] = selection[gate_columns].sum(axis=1)
    selection["all_gate"] = selection[gate_columns].all(axis=1)
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
    selection["selected"] = False
    for pair_name in PAIR_DEFINITIONS:
        candidates = selection.loc[selection["pair"].eq(pair_name)]
        selected_index = candidates.sort_values(
            ["gate_count", "selection_score", "candidate"],
            ascending=[False, False, True],
        ).index[0]
        selection.loc[selected_index, "selected"] = True
    selection["selection_policy"] = (
        "development_only_gate_count_then_relative_wealth_score"
    )
    return selection


def _outer_gate_audit(
    *,
    comparison: pd.DataFrame,
    annual: pd.DataFrame,
    selection: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair_name in PAIR_DEFINITIONS:
        selected = selection.loc[
            selection["pair"].eq(pair_name) & selection["selected"]
        ].iloc[0]
        candidate_name = str(selected["candidate"])
        base_selector = (
            comparison["pair"].eq(pair_name)
            & comparison["segment"].eq("outer_diagnostic")
            & comparison["strategy"].eq("model")
            & comparison["cost_bps"].eq(10)
        )
        model10 = comparison.loc[
            base_selector & comparison["candidate"].eq(candidate_name)
        ].iloc[0]
        model25 = comparison.loc[
            comparison["pair"].eq(pair_name)
            & comparison["segment"].eq("outer_diagnostic")
            & comparison["strategy"].eq("model")
            & comparison["cost_bps"].eq(25)
            & comparison["candidate"].eq(candidate_name)
        ].iloc[0]
        matched10 = comparison.loc[
            comparison["pair"].eq(pair_name)
            & comparison["segment"].eq("outer_diagnostic")
            & comparison["strategy"].eq("matched_static")
            & comparison["cost_bps"].eq(10)
            & comparison["candidate"].eq(candidate_name)
        ].iloc[0]
        own10 = comparison.loc[
            base_selector
            & comparison["candidate"].eq("own_only_turnover")
        ].iloc[0]
        annual10 = annual.loc[
            annual["pair"].eq(pair_name)
            & annual["segment"].eq("outer_diagnostic")
            & annual["strategy"].eq("model")
            & annual["cost_bps"].eq(10)
            & annual["candidate"].eq(candidate_name)
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
            "selected_uses_auxiliary_data": (
                float(selected["auxiliary_weight"]) > 0.0
            ),
            "return_10bps_vs_underlying": (
                float(model10["terminal_wealth_ratio_vs_soxx"]) > 1.0
            ),
            "return_25bps_vs_underlying": (
                float(model25["terminal_wealth_ratio_vs_soxx"]) > 1.0
            ),
            "dynamic_vs_matched_static": (
                float(model10["terminal_wealth"])
                > float(matched10["terminal_wealth"])
            ),
            "selected_vs_own_only_turnover": (
                float(model10["terminal_wealth"])
                > float(own10["terminal_wealth"])
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
        for gate, passed in checks.items():
            rows.append(
                {
                    "pair": pair_name,
                    "selected_candidate": candidate_name,
                    "gate": gate,
                    "required": True,
                    "passed": bool(passed),
                    "outer_ratio_10bps": float(
                        model10["terminal_wealth_ratio_vs_soxx"]
                    ),
                    "outer_ratio_25bps": float(
                        model25["terminal_wealth_ratio_vs_soxx"]
                    ),
                    "dynamic_vs_matched_static": float(
                        model10["terminal_wealth"]
                        / matched10["terminal_wealth"]
                    ),
                    "selected_vs_own_only_turnover": float(
                        model10["terminal_wealth"]
                        / own10["terminal_wealth"]
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
            "# Cross-pair auxiliary training\n\n"
            "检验 TQQQ 数据能否改善 SOXL 训练，以及 SOXL 数据能否改善"
            " TQQQ 训练。Outer 不参与候选选择。"
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
            "from src.workflow.cross_pair_auxiliary_training "
            "import run_cross_pair_auxiliary_training_card\n"
            "summary = run_cross_pair_auxiliary_training_card("
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
    selected_names = {
        row.pair: row.candidate
        for row in selection.loc[selection["selected"]].itertuples()
    }
    selected_outer = comparison.loc[
        comparison["segment"].eq("outer_diagnostic")
        & comparison["strategy"].eq("model")
        & comparison["cost_bps"].isin([10, 25])
        & comparison.apply(
            lambda row: selected_names.get(row["pair"])
            == row["candidate"],
            axis=1,
        )
    ].copy()
    selected_outer.to_csv(card / "risk_analysis.csv", index=False)
    selected_payload = {
        "selection_policy": (
            "development_only_gate_count_then_relative_wealth_score"
        ),
        "selected_candidates": selected_names,
        "selection_uses_outer_test": False,
        "outer_role": "report_only",
        "diagnostic_only": not bool(summary["development_gate_pass"]),
    }
    (card / "selected_model.json").write_text(
        json.dumps(selected_payload, indent=2), encoding="utf-8"
    )
    analysis = {
        "selected_candidates": selected_names,
        "pair_gate_pass": summary["pair_gate_pass"],
        "promotion_decision": summary["promotion_decision"],
        "outer_selected_metrics": _json_ready(
            selected_outer.to_dict(orient="records")
        ),
    }
    (card / "analysis_summary.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
    )
    (card / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2), encoding="utf-8"
    )
    failed = audit.loc[~audit["passed"], ["pair", "gate"]]
    (card / "failure_modes.md").write_text(
        "# Failure modes\n\n"
        + (
            "\n".join(
                f"- {row.pair}: {row.gate}"
                for row in failed.itertuples(index=False)
            )
            if not failed.empty
            else "- none"
        )
        + "\n",
        encoding="utf-8",
    )
    conclusion_lines = [
        "# 结论",
        "",
        "- 已比较自身训练、换手正则，以及25%/50%/100%双向辅助样本；"
        "所有候选都由网络直接输出合法三资产权重，现金每日不低于15%。",
    ]
    for pair_name, definition in PAIR_DEFINITIONS.items():
        selected = selected_names[pair_name]
        row10 = selected_outer.loc[
            selected_outer["pair"].eq(pair_name)
            & selected_outer["cost_bps"].eq(10)
        ].iloc[0]
        row25 = selected_outer.loc[
            selected_outer["pair"].eq(pair_name)
            & selected_outer["cost_bps"].eq(25)
        ].iloc[0]
        pair_audit = audit.loc[audit["pair"].eq(pair_name)].iloc[0]
        conclusion_lines.append(
            f"- {definition['leveraged']}/{definition['underlying']}/cash "
            f"开发期选择 `{selected}`；outer 10bp收益 "
            f"{float(row10['return_pct']):.2f}%，"
            f"{definition['underlying']} "
            f"{float(row10['soxx_return_pct']):.2f}%，财富比 "
            f"{float(row10['terminal_wealth_ratio_vs_soxx']):.4f}，"
            f"25bp财富比 "
            f"{float(row25['terminal_wealth_ratio_vs_soxx']):.4f}，"
            f"MaxDD {float(row10['max_drawdown_pct']):.2f}%，换手 "
            f"{float(row10['total_turnover']):.2f}，平均现金 "
            f"{float(row10['avg_cash_weight_pct']):.2f}%；"
            f"相对自身训练财富比 "
            f"{float(pair_audit['selected_vs_own_only_turnover']):.4f}。"
        )
    conclusion_lines.extend(
        [
            "- 两组开发期所有候选都未通过完整门槛，因此这里的选择"
            "只是开发期最优诊断候选，不是可晋升模型。",
            f"- 晋升：`{summary['promotion_decision']}`。",
            "- 选择仅使用2016–2022 walk-forward；2023–2026 outer "
            "只报告，不参与候选或参数选择。",
            "- 下一步只保留开发期与outer均证明有增益的辅助方式；"
            "若动态模型仍输给同平均仓位静态组合，则停止增加网络复杂度。",
            f"- Qlib recorder：`{summary['recorder_id']}`。",
            "",
        ]
    )
    (card / "conclusion.md").write_text(
        "\n".join(conclusion_lines), encoding="utf-8"
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
        "training: independent_and_cross_pair_auxiliary\n"
        "model_selection: development_2016_2022_only\n"
        "outer_role: report_only\n"
        "allocation_output: direct_cash_constrained_weights\n"
        "artifact_policy:\n"
        "  policy: slim_card_recorder_first\n"
    )
    (card / "workflow.yaml").write_text(
        workflow, encoding="utf-8"
    )
    _write_notebook(card, summary)


def _run_card(spec: CrossPairTrainingSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    panel, features, source_audit = _load_panel(spec.source_spec)
    source_audit = {
        **source_audit,
        "training_policy": (
            "candidate_specific_own_only_or_bidirectional_cross_pair_rows"
        ),
        "weight_transfer_between_pairs": False,
        "auxiliary_training_rows_enabled": True,
    }
    dataset, dataset_audit = _build_dataset(
        panel, features, spec.source_spec
    )
    development_predictions: dict[
        tuple[str, str], pd.DataFrame
    ] = {}
    outer_predictions: dict[tuple[str, str], pd.DataFrame] = {}
    states: dict[str, Any] = {}
    diagnostic_frames: list[pd.DataFrame] = []
    for pair_name, definition in PAIR_DEFINITIONS.items():
        primary_symbol = definition["leveraged"]
        for candidate in CANDIDATES:
            development, outer, candidate_states, diagnostics = (
                _candidate_predictions(
                    pair_name=pair_name,
                    primary_symbol=primary_symbol,
                    candidate=candidate,
                    dataset=dataset,
                    panel=panel,
                    features=features,
                    spec=spec,
                )
            )
            key = (pair_name, candidate.name)
            development_predictions[key] = development
            outer_predictions[key] = outer
            states[f"{pair_name}__{candidate.name}"] = candidate_states
            diagnostic_frames.append(diagnostics)
    comparison, annual, strategy_returns, positions = (
        _evaluate_candidates(
            panel=panel,
            development_predictions=development_predictions,
            outer_predictions=outer_predictions,
            spec=spec,
        )
    )
    selection = _select_candidates(comparison, annual)
    audit = _outer_gate_audit(
        comparison=comparison,
        annual=annual,
        selection=selection,
    )
    pair_gate_pass = (
        audit.groupby("pair")["passed"].all().to_dict()
    )
    selected_development = selection.loc[selection["selected"]]
    development_pair_gate_pass = {
        str(row.pair): bool(row.all_gate)
        for row in selected_development.itertuples()
    }
    development_gate_pass = bool(
        all(development_pair_gate_pass.values())
    )
    objective_pass = bool(
        development_gate_pass and all(pair_gate_pass.values())
    )
    decision = (
        "validated_cross_pair_auxiliary_training"
        if objective_pass
        else "not_promoted_cross_pair_training_gates_failed"
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
        for name, candidate_states in states.items():
            with (models_dir / f"{name}.pkl").open("wb") as handle:
                pickle.dump(candidate_states, handle)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                pairs="SOXL/SOXX/cash,TQQQ/QQQ/cash",
                candidate_count=len(CANDIDATES),
                qlib_dataset="DatasetH",
                daily_cash_floor=spec.minimum_cash_weight,
                model_selection="development_2016_2022_only",
                outer_role="report_only",
            )
            R.log_metrics(objective_pass=float(objective_pass))
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
                "card_type": "qlib_cross_pair_auxiliary_training_card",
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
                "candidates": [
                    asdict(candidate) for candidate in CANDIDATES
                ],
                "feature_count": len(features),
                "model_policy": (
                    "direct_weights_daily_cash_floor_cross_pair_auxiliary"
                ),
                "selection_uses_outer_test": False,
                "outer_role": "report_only",
                "pair_gate_pass": pair_gate_pass,
                "development_pair_gate_pass": (
                    development_pair_gate_pass
                ),
                "development_gate_pass": development_gate_pass,
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
            run_cross_pair_auxiliary_training_card(),
            indent=2,
        )
    )
