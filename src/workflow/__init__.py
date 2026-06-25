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
    "CognitiveAlphaEvolutionProfile",
    "CognitiveAlphaEvolutionResult",
    "CognitiveAlphaEvolutionSpec",
    "QlibNativeExperimentResult",
    "QlibNativeExperimentSpec",
    "build_agentic_alpha_hypotheses",
    "build_agentic_alpha_workflow_manifest",
    "build_cognitive_alpha_evolution_profile",
    "distill_agentic_alpha_search",
    "load_agentic_alpha_backtest_metric",
    "run_cognitive_alpha_evolution",
    "run_qlib_native_experiment",
]


_COGNITIVE_ALPHA_EXPORTS = {
    "CognitiveAlphaEvolutionProfile",
    "CognitiveAlphaEvolutionResult",
    "CognitiveAlphaEvolutionSpec",
    "build_cognitive_alpha_evolution_profile",
    "run_cognitive_alpha_evolution",
}


def __getattr__(name: str):
    if name in _COGNITIVE_ALPHA_EXPORTS:
        from . import cognitive_alpha_evolution

        return getattr(cognitive_alpha_evolution, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
