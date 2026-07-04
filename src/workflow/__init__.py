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
    "IntradayExecutionLabelModelSpec",
    "QlibNativeExperimentResult",
    "QlibNativeExperimentSpec",
    "RiskHardenedOpen2OpenCardSpec",
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
    "run_intraday_execution_label_model_card",
    "run_cognitive_alpha_position_model",
    "run_cognitive_alpha_weighted_factor_search",
    "run_cognitive_alpha_evolution",
    "run_qlib_native_experiment",
    "run_risk_hardened_open2open_card",
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

_INTRADAY_EXECUTION_LABEL_EXPORTS = {
    "IntradayExecutionLabelModelSpec",
    "run_intraday_execution_label_model_card",
}

_RISK_HARDENED_OPEN2OPEN_EXPORTS = {
    "RiskHardenedOpen2OpenCardSpec",
    "run_risk_hardened_open2open_card",
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
    if name in _INTRADAY_EXECUTION_LABEL_EXPORTS:
        from . import intraday_execution_label_model

        return getattr(intraday_execution_label_model, name)
    if name in _RISK_HARDENED_OPEN2OPEN_EXPORTS:
        from . import risk_hardened_open2open

        return getattr(risk_hardened_open2open, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
