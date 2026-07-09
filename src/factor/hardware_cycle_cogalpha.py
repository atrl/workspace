from __future__ import annotations

import numpy as np
import pandas as pd

from .cognitive_alpha import CognitiveAlphaCandidate


HARDWARE_COGALPHA_SOURCE_LAYER = "src.factor.hardware_cycle_cogalpha"
HARDWARE_COGALPHA_DATA_SEMANTICS = "daily_qlib_features_lagged_from_hardware_cycle_daily_proxy_surface"


def _col(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(frame[name], errors="coerce").astype("float64")


def _hardware_compute_scarcity_accelerator_vs_gpu_price(frame: pd.DataFrame) -> pd.Series:
    accelerator = _col(frame, "hardware_ai_accelerator_ret20_lag1")
    gpu_pressure = _col(frame, "hardware_cloud_gpu_daily_supply_demand_pressure_ret20_lag1")
    return (0.5 * accelerator + 0.5 * gpu_pressure).rename("hardware_compute_scarcity_accelerator_vs_gpu_price")


def _hardware_capex_confirmed_accelerator_momentum(frame: pd.DataFrame) -> pd.Series:
    accelerator = _col(frame, "hardware_ai_accelerator_ret20_lag1")
    capex = _col(frame, "hardware_cloud_capex_ret60_lag1")
    return (accelerator * np.tanh(capex * 5.0)).rename("hardware_capex_confirmed_accelerator_momentum")


def _hardware_memory_bottleneck_spread(frame: pd.DataFrame) -> pd.Series:
    memory = _col(frame, "hardware_memory_storage_ret20_lag1")
    accelerator = _col(frame, "hardware_ai_accelerator_ret20_lag1")
    return (memory - accelerator).rename("hardware_memory_bottleneck_spread")


def _hardware_server_supply_chain_confirmation(frame: pd.DataFrame) -> pd.Series:
    server = _col(frame, "hardware_cloud_gpu_daily_server_supply_proxy_ret20_lag1")
    accelerator = _col(frame, "hardware_ai_accelerator_ret20_lag1")
    tightness = _col(frame, "hardware_cloud_gpu_daily_server_tightness_ret20_lag1")
    return (server + 0.5 * accelerator + 0.5 * tightness).rename("hardware_server_supply_chain_confirmation")


def _hardware_free_h100_rental_price_pressure(frame: pd.DataFrame) -> pd.Series:
    h100_price = _col(frame, "hardware_free_cloud_gpu_h100_price_ret20_lag1")
    available = _col(frame, "hardware_free_cloud_gpu_h100_price_available_lag1").clip(lower=0.0, upper=1.0)
    return (-h100_price * available).rename("hardware_free_h100_rental_price_pressure")


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
                "hardware_cloud_gpu_daily_supply_demand_pressure_ret20_lag1",
            ),
            lookback=20,
            function=_hardware_compute_scarcity_accelerator_vs_gpu_price,
            economic_rationale="A daily supplier-versus-hyperscaler pressure proxy can flag tighter compute economics without relying on monthly rental price carry.",
            code="0.5 * ai_accelerator_ret20_lag1 + 0.5 * cloud_gpu_daily_supply_demand_pressure_ret20_lag1",
            metadata={
                "history_role": "daily_cloud_gpu_proxy_promotable",
                "data_semantics": HARDWARE_COGALPHA_DATA_SEMANTICS,
                "cloud_direct_pricing": False,
                "daily_proxy_pricing": True,
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
            theme="AI server confirmation with daily cloud GPU proxy tightness",
            required_columns=(
                "hardware_cloud_gpu_daily_server_supply_proxy_ret20_lag1",
                "hardware_ai_accelerator_ret20_lag1",
                "hardware_cloud_gpu_daily_server_tightness_ret20_lag1",
            ),
            lookback=20,
            function=_hardware_server_supply_chain_confirmation,
            economic_rationale="Daily AI server supply-chain strength and server tightness proxy can confirm GPU-cycle pressure without monthly price carry.",
            code="cloud_gpu_daily_server_supply_proxy_ret20_lag1 + 0.5 * ai_accelerator_ret20_lag1 + 0.5 * cloud_gpu_daily_server_tightness_ret20_lag1",
            metadata={
                "history_role": "daily_cloud_gpu_proxy_promotable",
                "data_semantics": HARDWARE_COGALPHA_DATA_SEMANTICS,
                "cloud_direct_pricing": False,
                "daily_proxy_pricing": True,
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
            name="hardware_free_h100_rental_price_pressure",
            level="market_structure_cycle",
            agent="AgentMarketCycle",
            theme="free no-auth daily H100 rental quote pressure",
            required_columns=(
                "hardware_free_cloud_gpu_h100_price_ret20_lag1",
                "hardware_free_cloud_gpu_h100_price_available_lag1",
            ),
            lookback=20,
            function=_hardware_free_h100_rental_price_pressure,
            economic_rationale=(
                "Observed daily H100 rental quote declines can indicate easing compute scarcity, but the free source has "
                "short history and is diagnostic unless it survives position and walk-forward gates."
            ),
            code="-free_cloud_gpu_h100_price_ret20_lag1 * free_cloud_gpu_h100_price_available_lag1",
            metadata={
                "history_role": "free_direct_gpu_price_short_history",
                "data_semantics": HARDWARE_COGALPHA_DATA_SEMANTICS,
                "cloud_direct_pricing": True,
                "free_daily_source": True,
                "no_forward_fill": True,
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
