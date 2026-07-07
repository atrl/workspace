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
    "MetaLabelRiskBudgetOpen2OpenSpec",
    "Multi3xEtfNormalizedNestedWfSpec",
    "DynamicSoxxSoxlCashAllocationWfSpec",
    "ProbabilityBudgetOpen2OpenCardSpec",
    "QlibNativeExperimentResult",
    "QlibNativeExperimentSpec",
    "RiskHardenedOpen2OpenCardSpec",
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
    "run_dynamic_soxx_soxl_cash_allocation_wf_card",
    "run_meta_label_risk_budget_open2open_card",
    "run_multi_3x_etf_normalized_nested_wf_card",
    "run_probability_budget_open2open_card",
    "run_cognitive_alpha_position_model",
    "run_cognitive_alpha_weighted_factor_search",
    "run_cognitive_alpha_evolution",
    "run_qlib_native_experiment",
    "run_risk_hardened_open2open_card",
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

_PROBABILITY_BUDGET_OPEN2OPEN_EXPORTS = {
    "ProbabilityBudgetOpen2OpenCardSpec",
    "run_probability_budget_open2open_card",
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
    if name in _PROBABILITY_BUDGET_OPEN2OPEN_EXPORTS:
        from . import probability_budget_open2open

        return getattr(probability_budget_open2open, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
