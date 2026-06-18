from __future__ import annotations

from collections import Counter
from typing import Iterable

import pandas as pd

from .registry import QUANTBENCH_MODEL_NAMES, get_model_spec, iter_model_specs


PROXY_FIDELITIES = {"proxy_feature_adapter"}
NATIVE_FIDELITIES = {"native_graph", "native_library", "native_sequence", "native_torch"}


def _base_model_name(model_name: str) -> str:
    return str(model_name).split("__", 1)[0]


def model_capability_table(model_names: Iterable[str] = QUANTBENCH_MODEL_NAMES) -> pd.DataFrame:
    selected = tuple(dict.fromkeys(_base_model_name(name) for name in model_names))
    rows: list[dict[str, object]] = []
    for model_name in selected:
        spec = get_model_spec(model_name)
        rows.append(
            {
                "model": spec.name,
                "family": spec.family,
                "objective": spec.objective,
                "input_kind": spec.input_kind,
                "estimator": spec.estimator,
                "paper_fidelity": spec.paper_fidelity,
                "is_proxy": spec.paper_fidelity in PROXY_FIDELITIES,
                "implementation_notes": spec.implementation_notes,
                "promotion_blocker": spec.promotion_blocker,
                "trade_instrument": spec.trade_instrument,
                "support_universe": ",".join(spec.support_universe),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["is_proxy"] = result["is_proxy"].astype(object)
    return result


def model_fidelity_summary(model_names: Iterable[str] = QUANTBENCH_MODEL_NAMES) -> dict[str, object]:
    table = model_capability_table(model_names)
    counts = Counter(str(value) for value in table["paper_fidelity"].tolist()) if not table.empty else Counter()
    proxy_models = table.loc[table["is_proxy"], "model"].astype(str).tolist() if not table.empty else []
    native_models = (
        table.loc[table["paper_fidelity"].isin(NATIVE_FIDELITIES), "model"].astype(str).tolist()
        if not table.empty
        else []
    )
    return {
        "schema_version": "model.fidelity.v1",
        "model_count": int(len(table)),
        "fidelity_counts": dict(sorted(counts.items())),
        "proxy_models": sorted(proxy_models),
        "native_models": sorted(native_models),
        "promotion_blocker_count": int(table["promotion_blocker"].astype(bool).sum()) if not table.empty else 0,
    }


def all_model_capabilities() -> tuple[dict[str, object], ...]:
    return tuple(model_capability_table(spec.name for spec in iter_model_specs()).to_dict(orient="records"))
