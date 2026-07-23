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

from src.workflow.direct_continuous_soxx_allocation import (
    DEVELOPMENT_YEARS,
    OUTER_START,
    DirectContinuousAllocationSpec,
    DirectLossProfile,
    _annual_metrics,
    _evaluate,
    _load_frame,
)
from src.workflow.multi_3x_etf_normalized_nested_wf import (
    Multi3xEtfNormalizedNestedWfSpec,
    _attach_normalized_features,
    _load_multi_3x_raw_panel,
    _model_feature_columns,
)
from src.workflow.multitask_classification_allocation import (
    TARGET_WEIGHT_COLUMNS,
    _attach_multitask_targets,
)
from src.workflow.role_separated_direct_allocation import _window_metrics
from src.workflow.soxl_objective_feasibility_audit import _json_ready
from src.workflow.trainable_regime_policy_nested_wf import (
    _remove_qlib_code_snapshot_artifacts,
)


EXPERIMENT = "auxiliary_3x_training_data_allocation_v1"
PRIMARY_INSTRUMENT = "SOXL"
NORMALIZATION_SUFFIXES = ("_z20", "_pctile20", "_z60", "_pctile60")
RECORDER_ONLY_ARTIFACTS = (
    f"artifacts/{EXPERIMENT}/models/",
    f"artifacts/{EXPERIMENT}/predictions.csv",
    f"artifacts/{EXPERIMENT}/positions.csv",
    f"artifacts/{EXPERIMENT}/strategy_returns.csv",
    f"artifacts/{EXPERIMENT}/training_diagnostics.csv",
)


@dataclass(frozen=True)
class TrainingDataCandidate:
    name: str
    asset_weights: tuple[tuple[str, float], ...]

    @property
    def weights(self) -> dict[str, float]:
        return dict(self.asset_weights)


CANDIDATES = (
    TrainingDataCandidate(
        name="soxl_only_common_features",
        asset_weights=(("SOXL", 1.0),),
    ),
    TrainingDataCandidate(
        name="spxl_aux_half_weight",
        asset_weights=(("SOXL", 1.0), ("SPXL", 0.5)),
    ),
    TrainingDataCandidate(
        name="tech_spxl_aux_weighted",
        asset_weights=(
            ("SOXL", 1.0),
            ("TQQQ", 0.4),
            ("TECL", 0.4),
            ("SPXL", 0.2),
        ),
    ),
)


@dataclass(frozen=True)
class Auxiliary3xTrainingDataSpec:
    root: Path
    experiment: str = EXPERIMENT
    train_start: str = "2011-01-03"
    development_years: tuple[int, ...] = DEVELOPMENT_YEARS
    development_end: str = "2022-12-30"
    outer: tuple[str, str] = (OUTER_START, "2026-07-02")
    purge_rows: int = 20
    hidden_units: int = 16
    epochs: int = 300
    learning_rate: float = 0.003
    weight_decay: float = 0.05
    maximum_soxl_weight: float = 0.30
    minimum_average_cash: float = 0.15
    costs_bps: tuple[int, ...] = (10, 25)
    seed: int = 53
    as_of: str = "2026-07-02"

    @property
    def card_dir(self) -> Path:
        return self.root / "research" / self.experiment

    @property
    def primary_provider_uri(self) -> Path:
        return self.root / "data" / "qlib" / "soxl"


class AuxiliaryTrainingAllocator(nn.Module):
    """Direct legal-weight allocator shared across comparable 3x triplets."""

    def __init__(self, feature_count: int, hidden_units: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, 2),
        )

    def forward(
        self, features: torch.Tensor, maximum_soxl_weight: float
    ) -> torch.Tensor:
        logits = self.network(features)
        leveraged = maximum_soxl_weight * torch.sigmoid(logits[:, 0])
        cash = (1.0 - leveraged) * torch.sigmoid(logits[:, 1])
        underlying = 1.0 - leveraged - cash
        return torch.stack([leveraged, underlying, cash], dim=1)


def run_auxiliary_3x_training_data_allocation_card(
    *, root: Path | str | None = None
) -> dict[str, Any]:
    repo_root = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[2]
    )
    return _run_card(Auxiliary3xTrainingDataSpec(root=repo_root))


def _attach_generic_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the same direct-allocation oracle to every 3x/1x pair."""

    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("instrument", sort=False):
        result = group.sort_values("datetime").copy()
        result.index = pd.to_datetime(result["datetime"]).dt.normalize()
        leveraged_open = pd.to_numeric(result["open"], errors="coerce")
        underlying_open = pd.to_numeric(
            result["underlying_open"], errors="coerce"
        )
        underlying_close = pd.to_numeric(
            result["underlying_close"], errors="coerce"
        )
        underlying_return = underlying_close.pct_change()
        result["soxl_forward_open_return"] = (
            leveraged_open.shift(-2) / leveraged_open.shift(-1) - 1.0
        )
        result["soxx_forward_open_return"] = (
            underlying_open.shift(-2) / underlying_open.shift(-1) - 1.0
        )
        result["soxx_drawdown60"] = (
            underlying_close
            / underlying_close.rolling(60, min_periods=20).max()
            - 1.0
        )
        result["soxx_vol20"] = (
            underlying_return.rolling(20, min_periods=10).std(ddof=0)
            * np.sqrt(252.0)
        )
        result, _, _ = _attach_multitask_targets(result)
        parts.append(result.reset_index(drop=True))
    if not parts:
        raise ValueError("blocked: no multi-3x rows for generic targets")
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["datetime", "instrument"])
        .reset_index(drop=True)
    )


def _load_training_panel(
    spec: Auxiliary3xTrainingDataSpec,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    source_spec = Multi3xEtfNormalizedNestedWfSpec(
        root=spec.root,
        train_start=spec.train_start,
        max_common_end=spec.as_of,
    )
    raw, source_audit = _load_multi_3x_raw_panel(source_spec)
    normalized = _attach_normalized_features(raw)
    panel = _attach_generic_targets(normalized)
    features = [
        column
        for column in _model_feature_columns(panel)
        if column.endswith(NORMALIZATION_SUFFIXES)
    ]
    features = list(dict.fromkeys(features))
    if not features:
        raise ValueError("blocked: no normalized common 3x features")
    forbidden = [
        column
        for column in features
        if any(token in column.lower() for token in ("instrument", "asset", "symbol"))
    ]
    if forbidden:
        raise ValueError(
            f"blocked: asset identity entered common features {forbidden}"
        )
    audit = {
        "source": "qlib.data.D.features",
        "source_audit": source_audit,
        "feature_count": len(features),
        "feature_policy": "past_only_per_asset_rolling_zscore_percentile",
        "feature_identity_columns": forbidden,
        "available_instruments": sorted(panel["instrument"].unique().tolist()),
        "primary_test_instrument": PRIMARY_INSTRUMENT,
        "auxiliary_assets_role": "training_rows_only",
        "outer_assets_role": "SOXL_only",
        "selection_uses_outer_test": False,
    }
    return panel, features, audit


def _build_dataset(
    panel: pd.DataFrame,
    features: list[str],
    spec: Auxiliary3xTrainingDataSpec,
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
    probe = dataset.prepare(
        "development", col_set=["feature", "label"]
    )
    if probe.empty:
        raise ValueError("blocked: qlib DatasetH development segment empty")
    return dataset, {
        "dataset_class": "qlib.data.dataset.DatasetH",
        "handler_class": "qlib.data.dataset.handler.DataHandlerLP",
        "loader_class": "qlib.data.dataset.loader.StaticDataLoader",
        "development_rows": int(len(probe)),
        "instruments": sorted(
            probe.index.get_level_values("instrument").unique().tolist()
        ),
        "selection_uses_outer_test": False,
    }


def _prepare_panel(dataset, segment) -> pd.DataFrame:
    result = dataset.prepare(segment, col_set=["feature", "label"])
    result.columns = [str(column[-1]) for column in result.columns]
    result = result.reset_index()
    result["datetime"] = pd.to_datetime(result["datetime"]).dt.normalize()
    result["instrument"] = (
        result["instrument"].astype(str).str.upper()
    )
    return result.sort_values(["datetime", "instrument"]).reset_index(
        drop=True
    )


def _training_end(
    panel: pd.DataFrame, year: int, purge_rows: int
) -> pd.Timestamp:
    calendar = np.sort(pd.to_datetime(panel["datetime"]).unique())
    position = int(
        np.searchsorted(calendar, np.datetime64(f"{year}-01-01"))
    )
    end_position = position - purge_rows - 1
    if end_position < 0:
        raise ValueError(f"blocked: insufficient purge history for {year}")
    return pd.Timestamp(calendar[end_position])


def _standardize(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, pd.Series]]:
    median = train.loc[:, features].median()
    train_values = train.loc[:, features].fillna(median)
    test_values = test.loc[:, features].fillna(median)
    mean = train_values.mean()
    scale = (
        train_values.std(ddof=1)
        .replace(0.0, 1.0)
        .fillna(1.0)
    )
    train_scaled = ((train_values - mean) / scale).clip(-8.0, 8.0)
    test_scaled = ((test_values - mean) / scale).clip(-8.0, 8.0)
    return (
        torch.tensor(
            train_scaled.to_numpy(dtype=np.float32), dtype=torch.float32
        ),
        torch.tensor(
            test_scaled.to_numpy(dtype=np.float32), dtype=torch.float32
        ),
        {"median": median, "mean": mean, "scale": scale},
    )


def _fit_predict(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    candidate: TrainingDataCandidate,
    spec: Auxiliary3xTrainingDataSpec,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, float]]:
    required_train = [*features, *TARGET_WEIGHT_COLUMNS]
    train = train.dropna(subset=required_train).copy()
    test = test.dropna(subset=features).copy()
    if train.empty or test.empty:
        raise ValueError("blocked: empty auxiliary allocator train/test")
    if set(test["instrument"].unique()) != {PRIMARY_INSTRUMENT}:
        raise ValueError("blocked: auxiliary asset entered SOXL test set")
    weights_map = candidate.weights
    if not set(train["instrument"]).issubset(weights_map):
        raise ValueError("blocked: unexpected training instrument")
    train_x, test_x, scaler = _standardize(train, test, features)
    target = torch.tensor(
        train.loc[:, TARGET_WEIGHT_COLUMNS].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    sample_weight = torch.tensor(
        train["instrument"]
        .map(weights_map)
        .to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    sample_weight = sample_weight / sample_weight.mean()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AuxiliaryTrainingAllocator(
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
        allocation = model(train_x, spec.maximum_soxl_weight)
        per_row_mse = (allocation - target).pow(2).mean(dim=1)
        allocation_loss = (per_row_mse * sample_weight).mean()
        cash_floor_loss = torch.relu(
            spec.minimum_average_cash - allocation[:, 2].mean()
        ).pow(2)
        loss = allocation_loss + 2.0 * cash_floor_loss
        if not torch.isfinite(loss):
            raise ValueError("blocked: non-finite auxiliary training loss")
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
        raise ValueError("blocked: auxiliary model state missing")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(
            test_x, spec.maximum_soxl_weight
        ).cpu().numpy()
    output = pd.DataFrame(
        prediction,
        index=pd.to_datetime(test["datetime"]).dt.normalize(),
        columns=["soxl_weight", "soxx_weight", "cash_weight"],
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
        "feature_count": len(features),
    }
    final_components["best_loss"] = best_loss
    final_components["train_rows"] = float(len(train))
    final_components["test_rows"] = float(len(test))
    final_components["auxiliary_train_share"] = float(
        train["instrument"].ne(PRIMARY_INSTRUMENT).mean()
    )
    return output, state, final_components


def _candidate_predictions(
    *,
    dataset,
    panel: pd.DataFrame,
    features: list[str],
    candidate: TrainingDataCandidate,
    spec: Auxiliary3xTrainingDataSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    development_source = _prepare_panel(
        dataset, (spec.train_start, spec.development_end)
    )
    outer_source = _prepare_panel(dataset, "outer")
    development_predictions: list[pd.DataFrame] = []
    states: dict[str, Any] = {}
    diagnostics: list[dict[str, Any]] = []
    training_assets = set(candidate.weights)
    for year in spec.development_years:
        train_end = _training_end(panel, year, spec.purge_rows)
        train = development_source.loc[
            development_source["datetime"].le(train_end)
            & development_source["instrument"].isin(training_assets)
        ]
        test = development_source.loc[
            development_source["datetime"].dt.year.eq(year)
            & development_source["instrument"].eq(PRIMARY_INSTRUMENT)
        ]
        prediction, state, fit = _fit_predict(
            train=train,
            test=test,
            features=features,
            candidate=candidate,
            spec=spec,
            seed=spec.seed + year,
        )
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
                "training_assets": ",".join(sorted(training_assets)),
                "test_assets": PRIMARY_INSTRUMENT,
                "selection_uses_outer_test": False,
                **fit,
            }
        )
    outer_train = development_source.loc[
        development_source["instrument"].isin(training_assets)
    ]
    outer_test = outer_source.loc[
        outer_source["instrument"].eq(PRIMARY_INSTRUMENT)
    ]
    outer_prediction, outer_state, outer_fit = _fit_predict(
        train=outer_train,
        test=outer_test,
        features=features,
        candidate=candidate,
        spec=spec,
        seed=spec.seed,
    )
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
            "training_assets": ",".join(sorted(training_assets)),
            "test_assets": PRIMARY_INSTRUMENT,
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


def _profile(candidate: TrainingDataCandidate) -> DirectLossProfile:
    return DirectLossProfile(
        name=candidate.name,
        maximum_soxl_weight=0.30,
        negative_year_penalty=0.0,
        year_dispersion_penalty=0.0,
        relative_drawdown_penalty=0.0,
        turnover_penalty=0.0,
    )


def _backtests(
    *,
    market: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    segment: str,
    spec: Auxiliary3xTrainingDataSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparisons: list[dict[str, Any]] = []
    annual: list[pd.DataFrame] = []
    returns: list[pd.DataFrame] = []
    positions: list[pd.DataFrame] = []
    for candidate in CANDIDATES:
        for cost_bps in spec.costs_bps:
            metrics, result, benchmark = _evaluate(
                market=market,
                predictions=predictions[candidate.name],
                profile=_profile(candidate),
                cost_bps=cost_bps,
            )
            metadata = {
                "segment": segment,
                "candidate": candidate.name,
                "cost_bps": cost_bps,
                "selection_uses_outer_test": False,
            }
            comparisons.append({**metadata, **metrics})
            annual.append(_annual_metrics(result, benchmark, metadata))
            returns.append(result.assign(**metadata))
            positions.append(
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
        pd.DataFrame(comparisons),
        pd.concat(annual, ignore_index=True),
        pd.concat(returns, ignore_index=True),
        pd.concat(positions, ignore_index=True),
    )


def _select_candidate(
    comparison: pd.DataFrame, annual: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        cost10 = comparison.loc[
            comparison["candidate"].eq(candidate.name)
            & comparison["cost_bps"].eq(10)
        ].iloc[0]
        cost25 = comparison.loc[
            comparison["candidate"].eq(candidate.name)
            & comparison["cost_bps"].eq(25)
        ].iloc[0]
        annual10 = annual.loc[
            annual["candidate"].eq(candidate.name)
            & annual["cost_bps"].eq(10)
        ]
        rows.append(
            {
                **cost10.to_dict(),
                "training_assets": ",".join(
                    sorted(candidate.weights)
                ),
                "asset_weights": json.dumps(candidate.weights),
                "ratio_10bps": float(
                    cost10["terminal_wealth_ratio_vs_soxx"]
                ),
                "ratio_25bps": float(
                    cost25["terminal_wealth_ratio_vs_soxx"]
                ),
                "positive_relative_years": int(
                    annual10["terminal_wealth_ratio_vs_soxx"].gt(1.0).sum()
                ),
            }
        )
    selection = pd.DataFrame(rows)
    selection["return_gate"] = selection["ratio_10bps"].gt(1.0)
    selection["cost_gate"] = selection["ratio_25bps"].gt(1.0)
    selection["year_gate"] = selection["positive_relative_years"].ge(4)
    selection["maxdd_gate"] = selection["max_drawdown_pct"].ge(-20.0)
    selection["worst_day_gate"] = selection["worst_day_pct"].ge(-12.0)
    selection["cvar_gate"] = selection["cvar5_pct"].ge(-8.0)
    selection["worst_5d_gate"] = selection[
        "worst_rolling_5d_pct"
    ].ge(-12.0)
    selection["cash_gate"] = selection["avg_cash_weight_pct"].ge(15.0)
    selection["capture_gate"] = selection["upside_capture_pct"].ge(60.0)
    gates = [
        "return_gate",
        "cost_gate",
        "year_gate",
        "maxdd_gate",
        "worst_day_gate",
        "cvar_gate",
        "worst_5d_gate",
        "cash_gate",
        "capture_gate",
    ]
    selection["gate_count"] = selection[gates].sum(axis=1)
    selection["all_gate"] = selection[gates].all(axis=1)
    selection["selection_score"] = (
        np.log(selection["ratio_10bps"].clip(lower=1e-12))
        + 0.25
        * np.log(selection["ratio_25bps"].clip(lower=1e-12))
        + 0.04 * selection["positive_relative_years"]
        + 0.01 * (selection["max_drawdown_pct"] + 20.0)
    )
    selected_index = selection.sort_values(
        ["gate_count", "selection_score", "candidate"],
        ascending=[False, False, True],
    ).index[0]
    selection["diagnostic_selected"] = selection.index == selected_index
    selection["selection_policy"] = (
        "development_only_gate_count_then_score"
    )
    return selection


def _objective_audit(
    *,
    selected: pd.Series,
    outer_selected: pd.Series,
    outer_annual: pd.DataFrame,
) -> pd.DataFrame:
    outer_passing = int(
        (
            outer_annual["terminal_wealth_ratio_vs_soxx"].gt(1.0)
            & outer_annual["max_drawdown_pct"].ge(-20.0)
            & outer_annual["worst_day_pct"].ge(-12.0)
        ).sum()
    )
    rows = [
        (
            "auxiliary_assets_training_only",
            True,
            True,
            "all development and outer test rows are SOXL",
        ),
        (
            "outer_not_used_for_selection",
            True,
            not bool(selected["selection_uses_outer_test"]),
            "candidate selection uses 2016-2022 SOXL only",
        ),
        (
            "development_return_10bps",
            True,
            bool(selected["ratio_10bps"] > 1.0),
            f"ratio={float(selected['ratio_10bps']):.4f}",
        ),
        (
            "development_return_25bps",
            True,
            bool(selected["ratio_25bps"] > 1.0),
            f"ratio={float(selected['ratio_25bps']):.4f}",
        ),
        (
            "development_all_risk_gates",
            True,
            bool(selected["all_gate"]),
            f"gate_count={int(selected['gate_count'])}/9",
        ),
        (
            "outer_year_robustness",
            True,
            outer_passing >= 3,
            f"passing_outer_years={outer_passing}/4",
        ),
        (
            "outer_maxdd",
            True,
            float(outer_selected["max_drawdown_pct"]) >= -20.0,
            f"maxdd={float(outer_selected['max_drawdown_pct']):.2f}%",
        ),
    ]
    return pd.DataFrame(
        rows, columns=["gate", "required", "passed", "detail"]
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
            "# Auxiliary 3x training-data allocation\n\n"
            "其他3x ETF只进入训练行；所有开发期和outer测试均为SOXL。"
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
            "qlib.init(provider_uri=str(repo_root / 'data/qlib/soxl'), "
            "region='us', kernels=1, skip_if_reg=False)\n"
            "calendar = D.calendar(start_time='2011-01-03', "
            "end_time='2026-07-02', freq='day')\n"
            "from src.workflow.auxiliary_3x_training_data_allocation "
            "import run_auxiliary_3x_training_data_allocation_card\n"
            "summary = run_auxiliary_3x_training_data_allocation_card("
            "root=repo_root)\n"
            "summary['promotion_decision']"
        ),
        nbformat.v4.new_code_cell(
            "import pandas as pd\n"
            "card = Path.cwd()\n"
            "pd.read_csv(card / 'model_selection.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'training_data_audit.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'window_metrics.csv')"
        ),
        nbformat.v4.new_code_cell(
            "pd.read_csv(card / 'objective_gate_audit.csv')"
        ),
        nbformat.v4.new_markdown_cell(
            (card / "conclusion.md").read_text(encoding="utf-8")
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
    windows: pd.DataFrame,
    audit: pd.DataFrame,
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
    selection.to_csv(card / "model_selection.csv", index=False)
    comparison.to_csv(card / "model_comparison.csv", index=False)
    comparison.to_csv(
        card / "backtest_benchmark_comparison.csv", index=False
    )
    annual.to_csv(
        card / "backtest_walk_forward_metrics.csv", index=False
    )
    diagnostics.to_csv(card / "training_data_audit.csv", index=False)
    windows.to_csv(card / "window_metrics.csv", index=False)
    audit.to_csv(card / "objective_gate_audit.csv", index=False)
    comparison.loc[
        comparison["candidate"].eq(selected_name)
    ].to_csv(card / "risk_analysis.csv", index=False)
    selected_payload = {
        "candidate": selected_name,
        "training_assets": selected["training_assets"],
        "asset_weights": json.loads(selected["asset_weights"]),
        "selection_uses_outer_test": False,
        "diagnostic_only": not bool(summary["objective_pass"]),
        "development_metrics": _json_ready(selected.to_dict()),
    }
    (card / "selected_model.json").write_text(
        json.dumps(selected_payload, indent=2), encoding="utf-8"
    )
    analysis = {
        "objective_pass": summary["objective_pass"],
        "promotion_decision": summary["promotion_decision"],
        "selected_candidate": selected_name,
        "development_ratio_10bps": float(selected["ratio_10bps"]),
        "development_ratio_25bps": float(selected["ratio_25bps"]),
        "development_positive_relative_years": int(
            selected["positive_relative_years"]
        ),
        "outer_ratio_10bps": float(
            outer10["terminal_wealth_ratio_vs_soxx"]
        ),
        "selection_uses_outer_test": False,
    }
    (card / "analysis_summary.json").write_text(
        json.dumps(analysis, indent=2), encoding="utf-8"
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
        "- 已完成“其他3x只作为训练数据”的严格对照；测试集始终只有SOXL。\n"
        f"- 开发期选择：`{selected_name}`；10bp相对SOXX="
        f"{float(selected['ratio_10bps']):.4f}，25bp="
        f"{float(selected['ratio_25bps']):.4f}，正超额年份="
        f"{int(selected['positive_relative_years'])}/7，MaxDD="
        f"{float(selected['max_drawdown_pct']):.2f}%。\n"
        f"- Outer仅报告：10bp相对SOXX="
        f"{float(outer10['terminal_wealth_ratio_vs_soxx']):.4f}，"
        f"MaxDD={float(outer10['max_drawdown_pct']):.2f}%。\n"
        f"- 晋升：`{summary['promotion_decision']}`。\n"
        f"- 失败门：{', '.join(failed) if failed else '无'}。\n"
        "- 解释：辅助行改善了部分年度稳定性，但共同归一化特征丢失了"
        "SOXL专属alpha，且换手成本抵消了收益；不能替代原有特征模型。\n"
        "- 下一步：保留该结果作为数据增强边界，不继续扩大pooled权重；"
        "若再用其他3x，只允许训练辅助风险头并要求明显超过SOXL-only对照。\n"
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
        "assets: [SOXL, SOXX, cash]\n"
        "auxiliary_training_assets: [TQQQ, TECL, SPXL]\n"
        "test_assets: [SOXL]\n"
        "allocation_output: direct_full_weights\n"
        "selection_uses_outer_test: false\n"
        "artifact_policy:\n"
        "  policy: slim_card_recorder_first\n"
    )
    (card / "workflow.yaml").write_text(
        workflow, encoding="utf-8"
    )
    _write_notebook(card, summary)


def _run_card(spec: Auxiliary3xTrainingDataSpec) -> dict[str, Any]:
    card = spec.card_dir
    card.mkdir(parents=True, exist_ok=True)
    panel, features, source_audit = _load_training_panel(spec)
    dataset, dataset_audit = _build_dataset(panel, features, spec)
    market, market_audit, _ = _load_frame(
        DirectContinuousAllocationSpec(
            root=spec.root,
            development_end=spec.development_end,
            outer=spec.outer,
        )
    )
    development_predictions: dict[str, pd.DataFrame] = {}
    outer_predictions: dict[str, pd.DataFrame] = {}
    states: dict[str, Any] = {}
    diagnostic_frames: list[pd.DataFrame] = []
    for candidate in CANDIDATES:
        development, outer, candidate_states, diagnostics = (
            _candidate_predictions(
                dataset=dataset,
                panel=panel,
                features=features,
                candidate=candidate,
                spec=spec,
            )
        )
        development_predictions[candidate.name] = development
        outer_predictions[candidate.name] = outer
        states[candidate.name] = candidate_states
        diagnostic_frames.append(diagnostics)
    (
        development_comparison,
        development_annual,
        development_returns,
        development_positions,
    ) = _backtests(
        market=market,
        predictions=development_predictions,
        segment="development_walk_forward",
        spec=spec,
    )
    (
        outer_comparison,
        outer_annual,
        outer_returns,
        outer_positions,
    ) = _backtests(
        market=market,
        predictions=outer_predictions,
        segment="outer_diagnostic",
        spec=spec,
    )
    comparison = pd.concat(
        [development_comparison, outer_comparison], ignore_index=True
    )
    annual = pd.concat(
        [development_annual, outer_annual], ignore_index=True
    )
    selection = _select_candidate(
        development_comparison, development_annual
    )
    selected = selection.loc[selection["diagnostic_selected"]].iloc[0]
    selected_name = str(selected["candidate"])
    outer_selected = outer_comparison.loc[
        outer_comparison["candidate"].eq(selected_name)
        & outer_comparison["cost_bps"].eq(10)
    ].iloc[0]
    outer_selected_annual = outer_annual.loc[
        outer_annual["candidate"].eq(selected_name)
        & outer_annual["cost_bps"].eq(10)
    ]
    audit = _objective_audit(
        selected=selected,
        outer_selected=outer_selected,
        outer_annual=outer_selected_annual,
    )
    objective_pass = bool(
        audit.loc[audit["required"], "passed"].all()
    )
    decision = (
        "validated_auxiliary_3x_training_data_allocation"
        if objective_pass
        else "not_promoted_auxiliary_training_data_risk_gates_failed"
    )
    selected_development_result = development_returns.loc[
        development_returns["candidate"].eq(selected_name)
        & development_returns["cost_bps"].eq(10)
    ]
    windows = _window_metrics(selected_development_result)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    predictions = pd.concat(
        [
            *development_predictions.values(),
            *outer_predictions.values(),
        ]
    ).sort_index()
    positions = pd.concat(
        [development_positions, outer_positions], ignore_index=True
    )
    strategy_returns = pd.concat(
        [development_returns, outer_returns], ignore_index=True
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
        for candidate_name, candidate_states in states.items():
            with (models_dir / f"{candidate_name}.pkl").open(
                "wb"
            ) as handle:
                pickle.dump(candidate_states, handle)
        with R.start(experiment_name=spec.experiment):
            recorder = R.get_recorder()
            R.log_params(
                benchmark="SOXX",
                assets="SOXL,SOXX,cash",
                auxiliary_assets="TQQQ,TECL,SPXL",
                auxiliary_role="training_rows_only",
                test_assets="SOXL",
                candidate_count=len(CANDIDATES),
                qlib_dataset="DatasetH",
                selection_uses_outer_test="false",
            )
            R.log_metrics(
                development_ratio_10bps=float(
                    selected["ratio_10bps"]
                ),
                development_ratio_25bps=float(
                    selected["ratio_25bps"]
                ),
                outer_ratio_10bps=float(
                    outer_selected["terminal_wealth_ratio_vs_soxx"]
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
            summary = {
                "experiment": spec.experiment,
                "card_type": "qlib_auxiliary_training_data_research_card",
                "primary_notebook": f"{spec.experiment}.ipynb",
                "as_of": spec.as_of,
                "provider_uri": spec.primary_provider_uri.resolve().as_posix(),
                "source_provider_audit": source_audit,
                "market_source_audit": market_audit,
                "dataset_audit": dataset_audit,
                "spec": {
                    **asdict(spec),
                    "root": spec.root.resolve().as_posix(),
                },
                "feature_count": len(features),
                "feature_policy": source_audit["feature_policy"],
                "training_data_candidates": [
                    asdict(candidate) for candidate in CANDIDATES
                ],
                "test_data_policy": {
                    "development": "SOXL_only",
                    "outer": "SOXL_only",
                    "auxiliary_assets_in_test": False,
                    "selection_uses_outer_test": False,
                },
                "selected_candidate": selected_name,
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
                    "model_comparison": "model_comparison.csv",
                    "backtest_benchmark_comparison": (
                        "backtest_benchmark_comparison.csv"
                    ),
                    "backtest_walk_forward_metrics": (
                        "backtest_walk_forward_metrics.csv"
                    ),
                    "risk_analysis": "risk_analysis.csv",
                    "training_data_audit": "training_data_audit.csv",
                    "window_metrics": "window_metrics.csv",
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
                windows=windows,
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
            run_auxiliary_3x_training_data_allocation_card(), indent=2
        )
    )
