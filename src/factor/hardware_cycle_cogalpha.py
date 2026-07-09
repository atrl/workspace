from __future__ import annotations

import numpy as np
import pandas as pd

from .cognitive_alpha import CognitiveAlphaCandidate


HARDWARE_COGALPHA_SOURCE_LAYER = "src.factor.hardware_cycle_cogalpha"
HARDWARE_COGALPHA_DATA_SEMANTICS = "daily_qlib_features_lagged_from_hardware_cycle_state"


def _col(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(frame[name], errors="coerce").astype("float64")


def _hardware_compute_scarcity_accelerator_vs_gpu_price(frame: pd.DataFrame) -> pd.Series:
    accelerator = _col(frame, "hardware_ai_accelerator_ret20_lag1")
    gpu_price = _col(frame, "hardware_cloud_gpu_modern_on_demand_ret21_lag1")
    available = _col(frame, "hardware_cloud_gpu_modern_on_demand_available_lag1").clip(0.0, 1.0)
    return (accelerator - gpu_price * available).rename("hardware_compute_scarcity_accelerator_vs_gpu_price")


def _hardware_capex_confirmed_accelerator_momentum(frame: pd.DataFrame) -> pd.Series:
    accelerator = _col(frame, "hardware_ai_accelerator_ret20_lag1")
    capex = _col(frame, "hardware_cloud_capex_ret60_lag1")
    return (accelerator * np.tanh(capex * 5.0)).rename("hardware_capex_confirmed_accelerator_momentum")


def _hardware_memory_bottleneck_spread(frame: pd.DataFrame) -> pd.Series:
    memory = _col(frame, "hardware_memory_storage_ret20_lag1")
    accelerator = _col(frame, "hardware_ai_accelerator_ret20_lag1")
    return (memory - accelerator).rename("hardware_memory_bottleneck_spread")


def _hardware_server_supply_chain_confirmation(frame: pd.DataFrame) -> pd.Series:
    server = _col(frame, "hardware_ai_server_ret20_lag1")
    accelerator = _col(frame, "hardware_ai_accelerator_ret20_lag1")
    carried = _col(frame, "hardware_cloud_gpu_h100_on_demand_is_carried_lag1").clip(0.0, 1.0)
    staleness = np.log1p(_col(frame, "hardware_cloud_gpu_h100_on_demand_staleness_days_lag1").clip(lower=0.0))
    return (server + 0.5 * accelerator - 0.05 * carried * staleness).rename("hardware_server_supply_chain_confirmation")


def _hardware_cpu_compute_breadth(frame: pd.DataFrame) -> pd.Series:
    cpu = _col(frame, "hardware_cpu_compute_ret20_lag1")
    accelerator = _col(frame, "hardware_ai_accelerator_ret20_lag1")
    return (0.5 * cpu + 0.5 * accelerator).rename("hardware_cpu_compute_breadth")


def _hardware_storage_cycle_confirmation(frame: pd.DataFrame) -> pd.Series:
    enterprise = _col(frame, "hardware_enterprise_storage_ret20_lag1")
    memory = _col(frame, "hardware_memory_storage_ret20_lag1")
    return (0.5 * enterprise + 0.5 * memory).rename("hardware_storage_cycle_confirmation")


def build_hardware_cycle_cogalpha_candidates() -> list[CognitiveAlphaCandidate]:
    """Return deterministic CognitiveAlpha candidates for lagged hardware-cycle features."""

    return [
        CognitiveAlphaCandidate(
            name="hardware_compute_scarcity_accelerator_vs_gpu_price",
            level="market_structure_cycle",
            agent="AgentMarketCycle",
            theme="accelerator equity strength versus cloud GPU price scarcity",
            required_columns=(
                "hardware_ai_accelerator_ret20_lag1",
                "hardware_cloud_gpu_modern_on_demand_ret21_lag1",
                "hardware_cloud_gpu_modern_on_demand_available_lag1",
            ),
            lookback=21,
            function=_hardware_compute_scarcity_accelerator_vs_gpu_price,
            economic_rationale="Rising accelerator equities with falling or stable cloud GPU rental prices can flag improving compute supply/demand economics.",
            code="ai_accelerator_ret20_lag1 - cloud_gpu_modern_on_demand_ret21_lag1 * availability_lag1",
            metadata={
                "history_role": "cloud_diagnostic_short_history",
                "data_semantics": HARDWARE_COGALPHA_DATA_SEMANTICS,
                "cloud_direct_pricing": True,
            },
        ),
        CognitiveAlphaCandidate(
            name="hardware_capex_confirmed_accelerator_momentum",
            level="geometric_fusion",
            agent="AgentComposite",
            theme="cloud capex confirmation of accelerator momentum",
            required_columns=("hardware_ai_accelerator_ret20_lag1", "hardware_cloud_capex_ret60_lag1"),
            lookback=60,
            function=_hardware_capex_confirmed_accelerator_momentum,
            economic_rationale="Accelerator momentum is more durable when cloud-capex proxies move in the same direction.",
            code="ai_accelerator_ret20_lag1 * tanh(5 * cloud_capex_ret60_lag1)",
            metadata={
                "history_role": "proxy_promotable",
                "data_semantics": HARDWARE_COGALPHA_DATA_SEMANTICS,
                "cloud_direct_pricing": False,
            },
        ),
        CognitiveAlphaCandidate(
            name="hardware_memory_bottleneck_spread",
            level="market_structure_cycle",
            agent="AgentMarketCycle",
            theme="memory and storage bottleneck relative to accelerators",
            required_columns=("hardware_memory_storage_ret20_lag1", "hardware_ai_accelerator_ret20_lag1"),
            lookback=20,
            function=_hardware_memory_bottleneck_spread,
            economic_rationale="Memory/storage relative strength can indicate a bottleneck phase that changes SOXL sensitivity.",
            code="memory_storage_ret20_lag1 - ai_accelerator_ret20_lag1",
            metadata={
                "history_role": "proxy_promotable",
                "data_semantics": HARDWARE_COGALPHA_DATA_SEMANTICS,
                "cloud_direct_pricing": False,
            },
        ),
        CognitiveAlphaCandidate(
            name="hardware_server_supply_chain_confirmation",
            level="stability_regime_gating",
            agent="AgentRegimeGating",
            theme="AI server confirmation with stale cloud GPU state penalty",
            required_columns=(
                "hardware_ai_server_ret20_lag1",
                "hardware_ai_accelerator_ret20_lag1",
                "hardware_cloud_gpu_h100_on_demand_is_carried_lag1",
                "hardware_cloud_gpu_h100_on_demand_staleness_days_lag1",
            ),
            lookback=31,
            function=_hardware_server_supply_chain_confirmation,
            economic_rationale="Server and accelerator strength is less informative when the direct cloud GPU state is stale and only carried forward.",
            code="ai_server_ret20_lag1 + 0.5 * ai_accelerator_ret20_lag1 - 0.05 * carried_lag1 * log1p(staleness_days_lag1)",
            metadata={
                "history_role": "cloud_diagnostic_short_history",
                "data_semantics": HARDWARE_COGALPHA_DATA_SEMANTICS,
                "cloud_direct_pricing": True,
            },
        ),
        CognitiveAlphaCandidate(
            name="hardware_cpu_compute_breadth",
            level="price_volume_dynamics",
            agent="AgentPriceVolumeCoherence",
            theme="CPU and accelerator compute breadth",
            required_columns=("hardware_cpu_compute_ret20_lag1", "hardware_ai_accelerator_ret20_lag1"),
            lookback=20,
            function=_hardware_cpu_compute_breadth,
            economic_rationale="Broader compute strength reduces the chance that accelerator-only momentum is a narrow overfit.",
            code="0.5 * cpu_compute_ret20_lag1 + 0.5 * ai_accelerator_ret20_lag1",
            metadata={
                "history_role": "proxy_promotable",
                "data_semantics": HARDWARE_COGALPHA_DATA_SEMANTICS,
                "cloud_direct_pricing": False,
            },
        ),
        CognitiveAlphaCandidate(
            name="hardware_storage_cycle_confirmation",
            level="geometric_fusion",
            agent="AgentComposite",
            theme="enterprise storage and memory cycle confirmation",
            required_columns=("hardware_enterprise_storage_ret20_lag1", "hardware_memory_storage_ret20_lag1"),
            lookback=20,
            function=_hardware_storage_cycle_confirmation,
            economic_rationale="Enterprise storage plus memory strength captures a broader data-center hardware cycle than memory alone.",
            code="0.5 * enterprise_storage_ret20_lag1 + 0.5 * memory_storage_ret20_lag1",
            metadata={
                "history_role": "proxy_promotable",
                "data_semantics": HARDWARE_COGALPHA_DATA_SEMANTICS,
                "cloud_direct_pricing": False,
            },
        ),
    ]
