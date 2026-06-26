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
    "QlibNativeExperimentResult",
    "QlibNativeExperimentSpec",
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
    "run_cognitive_alpha_position_model",
    "run_cognitive_alpha_weighted_factor_search",
    "run_cognitive_alpha_evolution",
    "run_qlib_native_experiment",
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
