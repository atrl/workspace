from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    target_kind: str
    direction: str
    description: str


OBJECTIVE_SPECS: tuple[ObjectiveSpec, ...] = (
    ObjectiveSpec("classification", "binary_forward_return", "maximize", "预测未来收益方向。"),
    ObjectiveSpec("mse", "forward_return", "minimize", "最小化未来收益预测误差。"),
    ObjectiveSpec("ic", "forward_return_rank", "maximize", "最大化预测与未来收益的截面或时间序列相关。"),
    ObjectiveSpec("ranking", "relative_forward_return", "maximize", "优化候选信号或资产相对排序。"),
    ObjectiveSpec("utility", "portfolio_path", "maximize", "综合收益、回撤、换手和稳定性的晋升评分。"),
)
OBJECTIVE_NAMES: tuple[str, ...] = tuple(spec.name for spec in OBJECTIVE_SPECS)


def get_objective_spec(name: str) -> ObjectiveSpec:
    for spec in OBJECTIVE_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown objective spec: {name}")


def utility_score(*, return_pct: float, sharpe: float, max_drawdown_pct: float, turnover: float) -> float:
    return float(return_pct) + 10.0 * float(sharpe) - 1.25 * float(max_drawdown_pct) - 2.0 * float(turnover)
