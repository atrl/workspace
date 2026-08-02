from .agentic_alpha import (
    AgenticAlphaBacktestMetric,
    AgenticAlphaHypothesis,
    AgenticAlphaKnowledgeSource,
    AgenticAlphaSearchCandidate,
    AgenticAlphaSearchReport,
    AgenticAlphaWorkflowSpec,
    build_agentic_alpha_hypotheses,
    build_agentic_alpha_workflow_manifest,
    distill_agentic_alpha_search,
    load_agentic_alpha_backtest_metric,
)
from .qlib_native import QlibNativeExperimentResult, QlibNativeExperimentSpec, run_qlib_native_experiment

__all__ = [
    "AgenticAlphaBacktestMetric",
    "AgenticAlphaHypothesis",
    "AgenticAlphaKnowledgeSource",
    "AgenticAlphaSearchCandidate",
    "AgenticAlphaSearchReport",
    "AgenticAlphaWorkflowSpec",
    "CognitiveAlphaCodeCandidateProvider",
    "CognitiveAlphaEvolutionProfile",
    "CognitiveAlphaEvolutionResult",
    "CognitiveAlphaEvolutionSpec",
    "CognitiveAlphaExternalLLMCodeCandidateProvider",
    "CognitiveAlphaGoalGateSpec",
    "CognitiveAlphaPositionResult",
    "CognitiveAlphaPositionSpec",
    "CognitiveAlphaWeightedFactorSearchResult",
    "CognitiveAlphaWeightedFactorSearchSpec",
    "CognitiveAlphaGoalRunner",
    "CognitiveAlphaGoalRunnerSpec",
    "CognitiveAlphaSignalRefreshJob",
    "CognitiveAlphaSignalRefreshSpec",
    "CognitiveAlphaGoalCodexExecCardSpec",
    "CognitiveAlphaIntradayOverlaySpec",
    "CognitiveAlphaIntraday5MinRiskSleeveSpec",
    "IntradayExecutionLabelModelSpec",
    "IntradayExecutionSleeveV2Spec",
    "HybridIntraday15mOpen2OpenSpec",
    "HardwareCycleProbabilityBudgetOpen2OpenCardSpec",
    "MacroStateFactorAblationSpec",
    "MacroStateIncrementalAttributionSpec",
    "HardwareCycleCogAlphaMiningCardSpec",
    "HierarchicalFourAssetSpec",
    "HierarchicalSoxxExcessSpec",
    "HierarchicalSoxxExcessPriorOverlaySpec",
    "HierarchicalPolicyConfig",
    "MetaLabelRiskBudgetOpen2OpenSpec",
    "LeveragedEtfPaperReplaySpec",
    "Multi3xEtfNormalizedNestedWfSpec",
    "DynamicSoxxSoxlCashAllocationWfSpec",
    "DirectContinuousAllocationSpec",
    "Auxiliary3xTrainingDataSpec",
    "PitAuxiliaryAllocationSpec",
    "MultitaskClassificationAllocationSpec",
    "RoleSeparatedAllocationSpec",
    "TalebFragilityAllocationSpec",
    "ProbabilityBudgetOpen2OpenCardSpec",
    "QlibNativeExperimentResult",
    "QlibNativeExperimentSpec",
    "RiskHardenedOpen2OpenCardSpec",
    "RelativeWealthLfnnSoxxSpec",
    "SoxxConstituentAlphaAttributionSpec",
    "SmartLeverageSoxlWealthControlSpec",
    "SoxsDefensiveHurdleSpec",
    "TrainableRegimePolicyNestedWfSpec",
    "build_agentic_alpha_hypotheses",
    "build_agentic_alpha_workflow_manifest",
    "build_cognitive_alpha_code_candidate_provider",
    "build_cognitive_alpha_codex_exec_candidate_provider",
    "build_cognitive_alpha_command_llm_candidate_provider",
    "build_cognitive_alpha_external_llm_candidate_provider",
    "build_cognitive_alpha_evolution_profile",
    "codex_cli_readiness",
    "codex_exec_completion",
    "distill_agentic_alpha_search",
    "load_agentic_alpha_backtest_metric",
    "run_cognitive_alpha_goal_codex_exec_card",
    "run_cognitive_alpha_intraday_execution_overlay_card",
    "run_cognitive_alpha_intraday_external_context_overlay_card",
    "run_cognitive_alpha_intraday_delayed_observable_overlay_card",
    "run_cognitive_alpha_intraday_regime_overlay_diagnostic_card",
    "run_cognitive_alpha_intraday_5min_risk_sleeve_card",
    "run_intraday_execution_label_model_card",
    "run_intraday_execution_sleeve_v2_card",
    "run_hybrid_intraday15m_open2open_card",
    "run_hardware_cycle_probability_budget_open2open_card",
    "run_macro_state_factor_ablation_card",
    "run_macro_state_incremental_attribution_card",
    "run_hardware_cycle_cogalpha_mining_card",
    "run_hierarchical_four_asset_policy_card",
    "run_hierarchical_soxx_excess_walk_forward_card",
    "run_hierarchical_soxx_excess_prior_overlay_card",
    "run_dynamic_soxx_soxl_cash_allocation_wf_card",
    "run_direct_continuous_soxx_allocation_card",
    "run_auxiliary_3x_training_data_allocation_card",
    "run_multitask_classification_allocation_card",
    "run_role_separated_direct_allocation_card",
    "run_taleb_fragility_allocation_card",
    "run_meta_label_risk_budget_open2open_card",
    "run_leveraged_etf_paper_replay_card",
    "run_multi_3x_etf_normalized_nested_wf_card",
    "run_probability_budget_open2open_card",
    "run_cognitive_alpha_position_model",
    "run_cognitive_alpha_weighted_factor_search",
    "run_cognitive_alpha_evolution",
    "run_qlib_native_experiment",
    "run_risk_hardened_open2open_card",
    "run_drawdown_targeted_pit_allocation_card",
    "run_pit_constituent_auxiliary_training_allocation_card",
    "run_relative_wealth_lfnn_soxx_card",
    "run_soxx_constituent_alpha_attribution_card",
    "run_smart_leverage_soxl_wealth_control_card",
    "run_soxs_defensive_hurdle_card",
    "run_trainable_regime_policy_nested_wf_card",
]


_COGNITIVE_ALPHA_EXPORTS = {
    "CognitiveAlphaCodeCandidateProvider",
    "CognitiveAlphaEvolutionProfile",
    "CognitiveAlphaEvolutionResult",
    "CognitiveAlphaEvolutionSpec",
    "CognitiveAlphaExternalLLMCodeCandidateProvider",
    "build_cognitive_alpha_code_candidate_provider",
    "build_cognitive_alpha_command_llm_candidate_provider",
    "build_cognitive_alpha_external_llm_candidate_provider",
    "build_cognitive_alpha_evolution_profile",
    "run_cognitive_alpha_evolution",
}

_COGNITIVE_ALPHA_POSITION_EXPORTS = {
    "CognitiveAlphaPositionResult",
    "CognitiveAlphaPositionSpec",
    "CognitiveAlphaWeightedFactorSearchResult",
    "CognitiveAlphaWeightedFactorSearchSpec",
    "run_cognitive_alpha_position_model",
    "run_cognitive_alpha_weighted_factor_search",
}

_COGNITIVE_ALPHA_GOAL_EXPORTS = {
    "CognitiveAlphaGoalGateSpec",
    "CognitiveAlphaGoalRunner",
    "CognitiveAlphaGoalRunnerSpec",
    "CognitiveAlphaSignalRefreshJob",
    "CognitiveAlphaSignalRefreshSpec",
}

_COGNITIVE_ALPHA_CODEX_EXPORTS = {
    "build_cognitive_alpha_codex_exec_candidate_provider",
    "codex_cli_readiness",
    "codex_exec_completion",
}

_COGNITIVE_ALPHA_GOAL_CODEX_EXEC_EXPORTS = {
    "CognitiveAlphaGoalCodexExecCardSpec",
    "run_cognitive_alpha_goal_codex_exec_card",
}

_COGNITIVE_ALPHA_INTRADAY_OVERLAY_EXPORTS = {
    "CognitiveAlphaIntradayOverlaySpec",
    "run_cognitive_alpha_intraday_execution_overlay_card",
    "run_cognitive_alpha_intraday_external_context_overlay_card",
    "run_cognitive_alpha_intraday_delayed_observable_overlay_card",
    "run_cognitive_alpha_intraday_regime_overlay_diagnostic_card",
}

_COGNITIVE_ALPHA_INTRADAY_5MIN_RISK_EXPORTS = {
    "CognitiveAlphaIntraday5MinRiskSleeveSpec",
    "run_cognitive_alpha_intraday_5min_risk_sleeve_card",
}

_INTRADAY_EXECUTION_LABEL_EXPORTS = {
    "IntradayExecutionLabelModelSpec",
    "run_intraday_execution_label_model_card",
}

_INTRADAY_EXECUTION_SLEEVE_V2_EXPORTS = {
    "IntradayExecutionSleeveV2Spec",
    "run_intraday_execution_sleeve_v2_card",
}

_HYBRID_INTRADAY15M_OPEN2OPEN_EXPORTS = {
    "HybridIntraday15mOpen2OpenSpec",
    "run_hybrid_intraday15m_open2open_card",
}

_RISK_HARDENED_OPEN2OPEN_EXPORTS = {
    "RiskHardenedOpen2OpenCardSpec",
    "run_risk_hardened_open2open_card",
}

_META_LABEL_RISK_BUDGET_OPEN2OPEN_EXPORTS = {
    "MetaLabelRiskBudgetOpen2OpenSpec",
    "run_meta_label_risk_budget_open2open_card",
}

_MULTI_3X_ETF_NORMALIZED_NESTED_WF_EXPORTS = {
    "Multi3xEtfNormalizedNestedWfSpec",
    "run_multi_3x_etf_normalized_nested_wf_card",
}

_TRAINABLE_REGIME_POLICY_NESTED_WF_EXPORTS = {
    "TrainableRegimePolicyNestedWfSpec",
    "run_trainable_regime_policy_nested_wf_card",
}

_DYNAMIC_SOXX_SOXL_CASH_ALLOCATION_WF_EXPORTS = {
    "DynamicSoxxSoxlCashAllocationWfSpec",
    "run_dynamic_soxx_soxl_cash_allocation_wf_card",
}

_HIERARCHICAL_SOXX_EXCESS_EXPORTS = {
    "HierarchicalSoxxExcessSpec",
    "run_hierarchical_soxx_excess_walk_forward_card",
}

_HIERARCHICAL_SOXX_EXCESS_PRIOR_OVERLAY_EXPORTS = {
    "HierarchicalSoxxExcessPriorOverlaySpec",
    "run_hierarchical_soxx_excess_prior_overlay_card",
}

_DIRECT_CONTINUOUS_SOXX_ALLOCATION_EXPORTS = {
    "DirectContinuousAllocationSpec",
    "run_direct_continuous_soxx_allocation_card",
}

_AUXILIARY_3X_TRAINING_DATA_EXPORTS = {
    "Auxiliary3xTrainingDataSpec",
    "run_auxiliary_3x_training_data_allocation_card",
}

_PIT_CONSTITUENT_AUXILIARY_TRAINING_EXPORTS = {
    "PitAuxiliaryAllocationSpec",
    "run_drawdown_targeted_pit_allocation_card",
    "run_pit_constituent_auxiliary_training_allocation_card",
}

_MULTITASK_CLASSIFICATION_ALLOCATION_EXPORTS = {
    "MultitaskClassificationAllocationSpec",
    "run_multitask_classification_allocation_card",
}

_TALEB_FRAGILITY_ALLOCATION_EXPORTS = {
    "TalebFragilityAllocationSpec",
    "run_taleb_fragility_allocation_card",
}

_ROLE_SEPARATED_ALLOCATION_EXPORTS = {
    "RoleSeparatedAllocationSpec",
    "run_role_separated_direct_allocation_card",
}

_LEVERAGED_ETF_PAPER_REPLAY_EXPORTS = {
    "LeveragedEtfPaperReplaySpec",
    "run_leveraged_etf_paper_replay_card",
}

_PROBABILITY_BUDGET_OPEN2OPEN_EXPORTS = {
    "ProbabilityBudgetOpen2OpenCardSpec",
    "run_probability_budget_open2open_card",
}

_HARDWARE_CYCLE_PROBABILITY_BUDGET_EXPORTS = {
    "HardwareCycleProbabilityBudgetOpen2OpenCardSpec",
    "run_hardware_cycle_probability_budget_open2open_card",
}

_MACRO_STATE_FACTOR_ABLATION_EXPORTS = {
    "MacroStateFactorAblationSpec",
    "run_macro_state_factor_ablation_card",
}

_MACRO_STATE_INCREMENTAL_ATTRIBUTION_EXPORTS = {
    "MacroStateIncrementalAttributionSpec",
    "run_macro_state_incremental_attribution_card",
}

_HARDWARE_CYCLE_COGALPHA_MINING_EXPORTS = {
    "HardwareCycleCogAlphaMiningCardSpec",
    "run_hardware_cycle_cogalpha_mining_card",
}

_SOXX_CONSTITUENT_ATTRIBUTION_EXPORTS = {
    "SoxxConstituentAlphaAttributionSpec",
    "run_soxx_constituent_alpha_attribution_card",
}

_SMART_LEVERAGE_SOXL_WEALTH_CONTROL_EXPORTS = {
    "SmartLeverageSoxlWealthControlSpec",
    "run_smart_leverage_soxl_wealth_control_card",
}

_RELATIVE_WEALTH_LFNN_SOXX_EXPORTS = {
    "RelativeWealthLfnnSoxxSpec",
    "run_relative_wealth_lfnn_soxx_card",
}

_SOXS_DEFENSIVE_HURDLE_EXPORTS = {
    "SoxsDefensiveHurdleSpec",
    "run_soxs_defensive_hurdle_card",
}

_HIERARCHICAL_FOUR_ASSET_EXPORTS = {
    "HierarchicalFourAssetSpec",
    "HierarchicalPolicyConfig",
    "run_hierarchical_four_asset_policy_card",
}


def __getattr__(name: str):
    if name in _COGNITIVE_ALPHA_EXPORTS:
        from . import cognitive_alpha_evolution

        return getattr(cognitive_alpha_evolution, name)
    if name in _COGNITIVE_ALPHA_POSITION_EXPORTS:
        if name in {"CognitiveAlphaWeightedFactorSearchResult", "CognitiveAlphaWeightedFactorSearchSpec", "run_cognitive_alpha_weighted_factor_search"}:
            from . import cognitive_alpha_weighted

            return getattr(cognitive_alpha_weighted, name)
        from . import cognitive_alpha_position

        return getattr(cognitive_alpha_position, name)
    if name in _COGNITIVE_ALPHA_GOAL_EXPORTS:
        from . import cognitive_alpha_goal

        return getattr(cognitive_alpha_goal, name)
    if name in _COGNITIVE_ALPHA_CODEX_EXPORTS:
        from . import cognitive_alpha_codex_provider

        return getattr(cognitive_alpha_codex_provider, name)
    if name in _COGNITIVE_ALPHA_GOAL_CODEX_EXEC_EXPORTS:
        from . import cognitive_alpha_goal_codex_exec

        return getattr(cognitive_alpha_goal_codex_exec, name)
    if name in _COGNITIVE_ALPHA_INTRADAY_OVERLAY_EXPORTS:
        from . import cognitive_alpha_intraday_overlay

        return getattr(cognitive_alpha_intraday_overlay, name)
    if name in _COGNITIVE_ALPHA_INTRADAY_5MIN_RISK_EXPORTS:
        from . import cognitive_alpha_intraday_5min_risk_sleeve

        return getattr(cognitive_alpha_intraday_5min_risk_sleeve, name)
    if name in _INTRADAY_EXECUTION_LABEL_EXPORTS:
        from . import intraday_execution_label_model

        return getattr(intraday_execution_label_model, name)
    if name in _INTRADAY_EXECUTION_SLEEVE_V2_EXPORTS:
        from . import intraday_execution_sleeve_v2

        return getattr(intraday_execution_sleeve_v2, name)
    if name in _HYBRID_INTRADAY15M_OPEN2OPEN_EXPORTS:
        from . import hybrid_intraday15m_open2open

        return getattr(hybrid_intraday15m_open2open, name)
    if name in _RISK_HARDENED_OPEN2OPEN_EXPORTS:
        from . import risk_hardened_open2open

        return getattr(risk_hardened_open2open, name)
    if name in _META_LABEL_RISK_BUDGET_OPEN2OPEN_EXPORTS:
        from . import meta_label_risk_budget_open2open

        return getattr(meta_label_risk_budget_open2open, name)
    if name in _MULTI_3X_ETF_NORMALIZED_NESTED_WF_EXPORTS:
        from . import multi_3x_etf_normalized_nested_wf

        return getattr(multi_3x_etf_normalized_nested_wf, name)
    if name in _TRAINABLE_REGIME_POLICY_NESTED_WF_EXPORTS:
        from . import trainable_regime_policy_nested_wf

        return getattr(trainable_regime_policy_nested_wf, name)
    if name in _DYNAMIC_SOXX_SOXL_CASH_ALLOCATION_WF_EXPORTS:
        from . import dynamic_soxx_soxl_cash_allocation_wf

        return getattr(dynamic_soxx_soxl_cash_allocation_wf, name)
    if name in _HIERARCHICAL_SOXX_EXCESS_EXPORTS:
        from . import hierarchical_soxx_excess_walk_forward

        return getattr(hierarchical_soxx_excess_walk_forward, name)
    if name in _HIERARCHICAL_SOXX_EXCESS_PRIOR_OVERLAY_EXPORTS:
        from . import hierarchical_soxx_excess_prior_overlay

        return getattr(hierarchical_soxx_excess_prior_overlay, name)
    if name in _DIRECT_CONTINUOUS_SOXX_ALLOCATION_EXPORTS:
        from . import direct_continuous_soxx_allocation

        return getattr(direct_continuous_soxx_allocation, name)
    if name in _AUXILIARY_3X_TRAINING_DATA_EXPORTS:
        from . import auxiliary_3x_training_data_allocation

        return getattr(auxiliary_3x_training_data_allocation, name)
    if name in _PIT_CONSTITUENT_AUXILIARY_TRAINING_EXPORTS:
        from . import pit_constituent_auxiliary_training_allocation

        return getattr(
            pit_constituent_auxiliary_training_allocation, name
        )
    if name in _MULTITASK_CLASSIFICATION_ALLOCATION_EXPORTS:
        from . import multitask_classification_allocation

        return getattr(multitask_classification_allocation, name)
    if name in _TALEB_FRAGILITY_ALLOCATION_EXPORTS:
        from . import taleb_fragility_allocation

        return getattr(taleb_fragility_allocation, name)
    if name in _ROLE_SEPARATED_ALLOCATION_EXPORTS:
        from . import role_separated_direct_allocation

        return getattr(role_separated_direct_allocation, name)
    if name in _LEVERAGED_ETF_PAPER_REPLAY_EXPORTS:
        from . import leveraged_etf_paper_replay

        return getattr(leveraged_etf_paper_replay, name)
    if name in _PROBABILITY_BUDGET_OPEN2OPEN_EXPORTS:
        from . import probability_budget_open2open

        return getattr(probability_budget_open2open, name)
    if name in _HARDWARE_CYCLE_PROBABILITY_BUDGET_EXPORTS:
        from . import hardware_cycle_probability_budget_open2open

        return getattr(hardware_cycle_probability_budget_open2open, name)
    if name in _MACRO_STATE_FACTOR_ABLATION_EXPORTS:
        from . import macro_state_factor_ablation

        return getattr(macro_state_factor_ablation, name)
    if name in _MACRO_STATE_INCREMENTAL_ATTRIBUTION_EXPORTS:
        from . import macro_state_incremental_attribution

        return getattr(macro_state_incremental_attribution, name)
    if name in _HARDWARE_CYCLE_COGALPHA_MINING_EXPORTS:
        from . import hardware_cycle_cogalpha_mining

        return getattr(hardware_cycle_cogalpha_mining, name)
    if name in _SOXX_CONSTITUENT_ATTRIBUTION_EXPORTS:
        from . import soxx_constituent_alpha_attribution

        return getattr(soxx_constituent_alpha_attribution, name)
    if name in _SMART_LEVERAGE_SOXL_WEALTH_CONTROL_EXPORTS:
        from . import smart_leverage_soxl_wealth_control

        return getattr(smart_leverage_soxl_wealth_control, name)
    if name in _RELATIVE_WEALTH_LFNN_SOXX_EXPORTS:
        from . import relative_wealth_lfnn_soxx

        return getattr(relative_wealth_lfnn_soxx, name)
    if name in _SOXS_DEFENSIVE_HURDLE_EXPORTS:
        from . import soxs_defensive_hurdle

        return getattr(soxs_defensive_hurdle, name)
    if name in _HIERARCHICAL_FOUR_ASSET_EXPORTS:
        from . import hierarchical_four_asset_policy

        return getattr(hierarchical_four_asset_policy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
