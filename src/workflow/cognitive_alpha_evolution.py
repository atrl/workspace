from __future__ import annotations

import ast
import json
import re
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from src.factor import (
    COGALPHA_AGENT_LEVELS,
    CognitiveAlphaCandidate,
    CognitiveAlphaReport,
    build_cognitive_alpha_seed_candidates,
    build_column_alpha_candidates,
    evaluate_cognitive_alpha_candidates,
)


COGNITIVE_ALPHA_PAPER_REFERENCE = "arxiv:2511.18850"
COGNITIVE_ALPHA_EVOLUTION_OPS = ("mutation", "crossover", "crossover_mutation")
COGNITIVE_ALPHA_GUIDANCE_STYLES = (
    "trend_continuation",
    "mean_reversion",
    "volatility_regime",
    "liquidity_pressure",
    "fragility_tail",
)


@dataclass(frozen=True)
class CognitiveAlphaEvolutionProfile:
    profile: str
    agent_count: int
    candidates_per_agent: int
    generations: int
    initial_pool_size: int
    parent_pool_size: int
    children_per_generation: int


@dataclass(frozen=True)
class CognitiveAlphaEvolutionSpec:
    run_id: str
    profile: str
    runtime_dir: Path
    generations: int
    parent_pool_size: int
    children_per_generation: int
    initial_pool_size: int | None = None
    qualified_percentile: float = 0.65
    elite_percentile: float = 0.80
    max_abs_correlation: float = 0.95
    min_observations: int = 60
    max_missing_ratio: float = 0.45
    min_distinct_values: int = 5
    fallback_top_n: int = 3
    resume: bool = False


class CognitiveAlphaCandidateProvider(Protocol):
    provider_id: str
    provider_kind: str

    def generate_candidates(
        self,
        *,
        profile: CognitiveAlphaEvolutionProfile,
        frame: pd.DataFrame,
        alpha158_columns: list[str] | None,
        target_initial_pool_size: int,
    ) -> list[CognitiveAlphaCandidate]:
        ...


@dataclass(frozen=True)
class CognitiveAlphaEvolutionResult:
    run_dir: Path
    summary: dict[str, Any]
    generation_metrics: pd.DataFrame
    selected_feature_names: list[str]
    selected_candidates: list[CognitiveAlphaCandidate]
    final_report: CognitiveAlphaReport
    candidate_map: dict[str, CognitiveAlphaCandidate]


@dataclass(frozen=True)
class CognitiveAlphaCodeCandidateProvider:
    provider_id: str = "deterministic_code_agent_pool"
    provider_kind: str = "repo_owned_code"
    guidance_styles: tuple[str, ...] = COGNITIVE_ALPHA_GUIDANCE_STYLES

    def generate_candidates(
        self,
        *,
        profile: CognitiveAlphaEvolutionProfile,
        frame: pd.DataFrame,
        alpha158_columns: list[str] | None,
        target_initial_pool_size: int,
    ) -> list[CognitiveAlphaCandidate]:
        _ = frame, alpha158_columns
        agent_specs = _cogalpha_agent_specs()
        if profile.agent_count > len(agent_specs):
            raise ValueError(
                f"profile {profile.profile} requested {profile.agent_count} agents but only {len(agent_specs)} exist"
            )
        selected_agents = agent_specs[: profile.agent_count]
        candidates = []
        for index in range(target_initial_pool_size):
            level, agent, theme = selected_agents[index % len(selected_agents)]
            guidance_style = self.guidance_styles[index % len(self.guidance_styles)]
            candidates.append(
                _code_provider_candidate(
                    index=index,
                    level=level,
                    agent=agent,
                    theme=theme,
                    guidance_style=guidance_style,
                    provider_id=self.provider_id,
                    provider_kind=self.provider_kind,
                )
            )
        return candidates

    def provider_summary(self) -> dict[str, Any]:
        return {
            "codex_direct_compatible": True,
            "codex_execution_mode": "repo_owned_codex_generated_code",
            "openai_api_key_required": False,
        }


def build_cognitive_alpha_code_candidate_provider() -> CognitiveAlphaCodeCandidateProvider:
    return CognitiveAlphaCodeCandidateProvider()


@dataclass
class CognitiveAlphaExternalLLMCodeCandidateProvider:
    completion: Callable[[dict[str, Any]], str | dict[str, Any]]
    provider_id: str = "external_llm_code_provider"
    provider_kind: str = "external_llm_code"
    model: str = "unspecified"
    openai_api_key_required: bool | None = None
    codex_direct_compatible: bool = False
    codex_execution_mode: str | None = None
    request_context: dict[str, Any] | None = None
    guidance_styles: tuple[str, ...] = COGNITIVE_ALPHA_GUIDANCE_STYLES
    max_candidates_per_request: int = 1
    request_log: list[dict[str, Any]] = field(default_factory=list, init=False)
    response_log: list[dict[str, Any]] = field(default_factory=list, init=False)
    failure_count: int = field(default=0, init=False)
    invalid_candidate_count: int = field(default=0, init=False)
    invalid_candidate_log: list[dict[str, Any]] = field(default_factory=list, init=False)

    def generate_candidates(
        self,
        *,
        profile: CognitiveAlphaEvolutionProfile,
        frame: pd.DataFrame,
        alpha158_columns: list[str] | None,
        target_initial_pool_size: int,
    ) -> list[CognitiveAlphaCandidate]:
        agent_specs = _cogalpha_agent_specs()
        if profile.agent_count > len(agent_specs):
            raise ValueError(
                f"profile {profile.profile} requested {profile.agent_count} agents but only {len(agent_specs)} exist"
            )
        selected_agents = agent_specs[: profile.agent_count]
        candidates: list[CognitiveAlphaCandidate] = []
        request_index = 0
        max_requests = max(target_initial_pool_size * 4, target_initial_pool_size + 8)
        while len(candidates) < target_initial_pool_size and request_index < max_requests:
            level, agent, theme = selected_agents[request_index % len(selected_agents)]
            guidance_style = self.guidance_styles[request_index % len(self.guidance_styles)]
            request = _external_llm_request_payload(
                profile=profile,
                frame=frame,
                alpha158_columns=alpha158_columns,
                request_index=request_index,
                level=level,
                agent=agent,
                theme=theme,
                guidance_style=guidance_style,
                model=self.model,
                target_candidate_count=min(self.max_candidates_per_request, target_initial_pool_size - len(candidates)),
                request_context=self.request_context,
            )
            self.request_log.append(request)
            prompt_hash = _stable_payload_hash(request)
            try:
                response = _parse_external_llm_response(self.completion(request))
            except Exception as exc:
                self.failure_count += 1
                raise ValueError(f"external LLM candidate provider failed at request {request_index}: {exc}") from exc
            response_hash = _stable_payload_hash(response)
            valid_candidate_count = 0
            invalid_candidate_count = 0
            response_candidates: list[CognitiveAlphaCandidate] = []
            for response_index, payload in enumerate(response["candidates"]):
                try:
                    response_candidates.append(
                        _external_llm_candidate(
                            payload,
                            request=request,
                            prompt_hash=prompt_hash,
                            response_hash=response_hash,
                            response_index=response_index,
                            provider_id=self.provider_id,
                            provider_kind=self.provider_kind,
                            model=self.model,
                        )
                    )
                    valid_candidate_count += 1
                except Exception as exc:
                    invalid_candidate_count += 1
                    self.invalid_candidate_count += 1
                    self.invalid_candidate_log.append(
                        {
                            "request_index": int(request_index),
                            "response_index": int(response_index),
                            "prompt_hash": prompt_hash,
                            "response_hash": response_hash,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
            self.response_log.append(
                {
                    "request_index": request_index,
                    "prompt_hash": prompt_hash,
                    "response_hash": response_hash,
                    "candidate_count": len(response["candidates"]),
                    "valid_candidate_count": valid_candidate_count,
                    "invalid_candidate_count": invalid_candidate_count,
                }
            )
            for candidate in response_candidates:
                candidates.append(candidate)
                if len(candidates) >= target_initial_pool_size:
                    break
            request_index += 1
        if len(candidates) < target_initial_pool_size:
            raise ValueError(
                "external LLM candidate provider returned too few valid candidates: "
                f"{len(candidates)} valid for target {target_initial_pool_size} after {request_index} requests"
            )
        return candidates

    def provider_summary(self) -> dict[str, Any]:
        summary = {
            "llm_model": self.model,
            "prompt_count": int(len(self.request_log)),
            "response_count": int(len(self.response_log)),
            "provider_failure_count": int(self.failure_count),
            "invalid_candidate_count": int(self.invalid_candidate_count),
            "feedback_context_included": self.request_context is not None,
            "external_llm_configured": True,
        }
        if self.request_context is not None:
            summary["feedback_context_hash"] = _stable_payload_hash(_jsonable_payload(self.request_context))
        if self.openai_api_key_required is not None:
            summary["openai_api_key_required"] = bool(self.openai_api_key_required)
        if self.codex_direct_compatible:
            summary["codex_direct_compatible"] = True
        if self.codex_execution_mode:
            summary["codex_execution_mode"] = self.codex_execution_mode
        return summary


def build_cognitive_alpha_external_llm_candidate_provider(
    *,
    completion: Callable[[dict[str, Any]], str | dict[str, Any]],
    provider_id: str = "external_llm_code_provider",
    model: str = "unspecified",
    request_context: dict[str, Any] | None = None,
) -> CognitiveAlphaExternalLLMCodeCandidateProvider:
    return CognitiveAlphaExternalLLMCodeCandidateProvider(
        completion=completion,
        provider_id=provider_id,
        model=model,
        request_context=request_context,
    )


def build_cognitive_alpha_command_llm_candidate_provider(
    *,
    command: tuple[str, ...] | list[str],
    provider_id: str = "external_command_llm_code_provider",
    model: str = "external-command",
    timeout_seconds: float = 60.0,
    openai_api_key_required: bool | None = None,
    codex_direct_compatible: bool = False,
    codex_execution_mode: str | None = None,
    request_context: dict[str, Any] | None = None,
) -> CognitiveAlphaExternalLLMCodeCandidateProvider:
    command_tuple = tuple(str(part) for part in command)
    if not command_tuple:
        raise ValueError("external command LLM provider requires a non-empty command")

    def _completion(request: dict[str, Any]) -> str:
        completed = subprocess.run(
            command_tuple,
            input=json.dumps(request, sort_keys=True),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            raise ValueError(
                f"external command LLM provider exited {completed.returncode}"
                + (f": {stderr}" if stderr else "")
            )
        return completed.stdout

    return CognitiveAlphaExternalLLMCodeCandidateProvider(
        completion=_completion,
        provider_id=provider_id,
        model=model,
        openai_api_key_required=openai_api_key_required,
        codex_direct_compatible=codex_direct_compatible,
        codex_execution_mode=codex_execution_mode,
        request_context=request_context,
    )


def _cogalpha_agent_specs() -> list[tuple[str, str, str]]:
    return [
        (str(level["level"]), str(agent), str(level["theme"]))
        for level in COGALPHA_AGENT_LEVELS
        for agent in level["agents"]
    ]


def _code_provider_candidate(
    *,
    index: int,
    level: str,
    agent: str,
    theme: str,
    guidance_style: str,
    provider_id: str,
    provider_kind: str,
) -> CognitiveAlphaCandidate:
    window = 5 + (index % 8) * 5
    name = f"llm_code_{_candidate_slug(agent, prefix_chars=20, digest_chars=6)}_{guidance_style}_{index:03d}"

    def _func(frame: pd.DataFrame, *, style: str = guidance_style, lookback: int = window) -> pd.Series:
        close = pd.to_numeric(frame["close"], errors="coerce").astype("float64")
        open_ = pd.to_numeric(frame["open"], errors="coerce").astype("float64")
        high = pd.to_numeric(frame["high"], errors="coerce").astype("float64")
        low = pd.to_numeric(frame["low"], errors="coerce").astype("float64")
        volume = pd.to_numeric(frame["volume"], errors="coerce").astype("float64")
        returns = close.pct_change()
        if style == "trend_continuation":
            values = returns.rolling(lookback, min_periods=max(3, lookback // 2)).mean().shift(1)
        elif style == "mean_reversion":
            anchor = close.rolling(lookback, min_periods=max(3, lookback // 2)).mean()
            values = (-(close / (anchor + 1e-12) - 1.0)).shift(1)
        elif style == "volatility_regime":
            short = returns.rolling(max(3, lookback // 2), min_periods=3).std(ddof=0)
            long = returns.rolling(max(10, lookback * 2), min_periods=max(5, lookback)).std(ddof=0)
            values = (short / (long + 1e-12) - 1.0).shift(1)
        elif style == "liquidity_pressure":
            close_location = (close - open_) / (high - low + 1e-12)
            volume_z = _rolling_zscore(np.log1p(volume), max(10, lookback)).shift(1)
            values = close_location.shift(1) * volume_z
        elif style == "fragility_tail":
            rolling_max = close.rolling(max(10, lookback * 2), min_periods=max(5, lookback)).max()
            drawdown = close / (rolling_max + 1e-12) - 1.0
            downside = returns.where(returns < 0.0, 0.0).rolling(max(5, lookback), min_periods=3).std(ddof=0)
            values = (drawdown.abs() * downside).shift(1)
        else:
            raise ValueError(f"unsupported cognitive alpha guidance style: {style}")
        return values.rename(name)

    return CognitiveAlphaCandidate(
        name=name,
        level=level,
        agent=agent,
        theme=f"{theme}; guidance={guidance_style}",
        required_columns=("open", "high", "low", "close", "volume"),
        lookback=max(window * 2, 10),
        function=_func,
        economic_rationale=(
            f"{agent} explores {guidance_style.replace('_', ' ')} within {theme} using lagged OHLCV evidence."
        ),
        code=f"provider:{provider_id}:{guidance_style}:window={window}:lagged",
        declared_lookahead=0,
        metadata={
            "provider_id": provider_id,
            "provider_kind": provider_kind,
            "candidate_source": "llm_code_candidate_provider_boundary",
            "agent_layer": level,
            "guidance_style": guidance_style,
            "paper_agent_count": 21,
            "paper_layer_count": 7,
        },
    )


def _external_llm_request_payload(
    *,
    profile: CognitiveAlphaEvolutionProfile,
    frame: pd.DataFrame,
    alpha158_columns: list[str] | None,
    request_index: int,
    level: str,
    agent: str,
    theme: str,
    guidance_style: str,
    model: str,
    target_candidate_count: int,
    request_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "paper_reference": COGNITIVE_ALPHA_PAPER_REFERENCE,
        "profile": profile.profile,
        "model": model,
        "request_index": int(request_index),
        "target_candidate_count": int(target_candidate_count),
        "level": level,
        "agent": agent,
        "theme": theme,
        "guidance_style": guidance_style,
        "available_columns": [str(column) for column in frame.columns],
        "alpha158_columns": list(alpha158_columns or []),
        "contract": {
            "response_format": "json_object",
            "top_level_key": "candidates",
            "required_candidate_fields": [
                "name",
                "level",
                "agent",
                "theme",
                "required_columns",
                "lookback",
                "code",
                "economic_rationale",
            ],
            "code_contract": (
                "Return one pandas Series expression using variables open, high, low, close, volume, "
                "and optional alpha158 column names. Use lagged operators only; future shifts are rejected."
            ),
        },
    }
    if request_context is not None:
        payload["feedback_context"] = _jsonable_payload(request_context)
    return payload


def _parse_external_llm_response(raw_response: str | dict[str, Any]) -> dict[str, Any]:
    response = json.loads(raw_response) if isinstance(raw_response, str) else raw_response
    if not isinstance(response, dict):
        raise ValueError("external LLM response must be a JSON object")
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("external LLM response requires non-empty candidates list")
    for item in candidates:
        if not isinstance(item, dict):
            raise ValueError("external LLM candidate payload must be an object")
    return {"candidates": candidates}


def _external_llm_candidate(
    payload: dict[str, Any],
    *,
    request: dict[str, Any],
    prompt_hash: str,
    response_hash: str,
    response_index: int,
    provider_id: str,
    provider_kind: str,
    model: str,
) -> CognitiveAlphaCandidate:
    validated_fields = _required_external_candidate_fields(payload)
    name = str(payload["name"])
    required_columns = tuple(str(column) for column in payload["required_columns"])
    code = str(payload["code"])
    return CognitiveAlphaCandidate(
        name=name,
        level=str(payload.get("level") or request["level"]),
        agent=str(payload.get("agent") or request["agent"]),
        theme=str(payload.get("theme") or request["theme"]),
        required_columns=required_columns,
        lookback=int(payload["lookback"]),
        function=_compile_external_llm_expression(name=name, code=code, required_columns=required_columns),
        economic_rationale=str(payload["economic_rationale"]),
        code=code,
        declared_lookahead=int(payload.get("declared_lookahead", 0) or 0),
        metadata={
            "provider_id": provider_id,
            "provider_kind": provider_kind,
            "candidate_source": provider_kind,
            "llm_model": model,
            "paper_reference": COGNITIVE_ALPHA_PAPER_REFERENCE,
            "prompt_hash": prompt_hash,
            "response_hash": response_hash,
            "request_index": int(request["request_index"]),
            "response_index": int(response_index),
            "guidance_style": str(request["guidance_style"]),
            "agent_layer": str(request["level"]),
            "validated_payload_fields": validated_fields,
        },
    )


def _required_external_candidate_fields(payload: dict[str, Any]) -> list[str]:
    required = [
        "name",
        "level",
        "agent",
        "theme",
        "required_columns",
        "lookback",
        "code",
        "economic_rationale",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"external LLM candidate missing required fields: {missing}")
    if not isinstance(payload["required_columns"], list) or not payload["required_columns"]:
        raise ValueError("external LLM candidate requires non-empty required_columns list")
    return required


_ALLOWED_LLM_NAMES = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "np",
    "abs",
    "log1p",
    "sqrt",
    "tanh",
    "maximum",
    "minimum",
    "where",
    "zscore",
}
_ALLOWED_LLM_METHODS = {
    "abs",
    "clip",
    "corr",
    "diff",
    "ewm",
    "fillna",
    "max",
    "mean",
    "median",
    "min",
    "pct_change",
    "rank",
    "rolling",
    "shift",
    "std",
}
_ALLOWED_NUMPY_ATTRS = {
    "log1p",
    "sqrt",
    "tanh",
    "maximum",
    "minimum",
    "where",
}
_ALLOWED_LLM_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Attribute,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.keyword,
)


def _compile_external_llm_expression(
    *,
    name: str,
    code: str,
    required_columns: tuple[str, ...],
) -> Callable[[pd.DataFrame], pd.Series]:
    expression = _normalize_external_llm_expression(code)
    tree = ast.parse(expression, mode="eval")
    allowed_names = _ALLOWED_LLM_NAMES | {column for column in required_columns if column.isidentifier()}
    _validate_external_llm_ast(tree, allowed_names=allowed_names)
    compiled = compile(tree, filename=f"<external-llm-alpha:{name}>", mode="eval")

    def _func(frame: pd.DataFrame) -> pd.Series:
        locals_map = {
            "open": pd.to_numeric(frame["open"], errors="coerce").astype("float64") if "open" in frame else None,
            "high": pd.to_numeric(frame["high"], errors="coerce").astype("float64") if "high" in frame else None,
            "low": pd.to_numeric(frame["low"], errors="coerce").astype("float64") if "low" in frame else None,
            "close": pd.to_numeric(frame["close"], errors="coerce").astype("float64") if "close" in frame else None,
            "volume": pd.to_numeric(frame["volume"], errors="coerce").astype("float64") if "volume" in frame else None,
        }
        for column in required_columns:
            if column not in locals_map and column in frame.columns and column.isidentifier():
                locals_map[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
        value = eval(  # noqa: S307 - the AST and globals are restricted above.
            compiled,
            {
                "__builtins__": {},
                "np": np,
                "abs": abs,
                "log1p": np.log1p,
                "sqrt": np.sqrt,
                "tanh": np.tanh,
                "maximum": np.maximum,
                "minimum": np.minimum,
                "where": np.where,
                "zscore": _rolling_zscore,
            },
            locals_map,
        )
        if not isinstance(value, pd.Series):
            raise ValueError("external LLM alpha expression did not return a pandas Series")
        return pd.to_numeric(value.reindex(frame.index), errors="coerce").replace([np.inf, -np.inf], np.nan).rename(name)

    return _func


def _normalize_external_llm_expression(code: str) -> str:
    expression = str(code).strip()
    if expression.startswith("return "):
        expression = expression[len("return ") :].strip()
    if "\n" in expression:
        lines = [line.strip() for line in expression.splitlines() if line.strip()]
        returns = [line[len("return ") :].strip() for line in lines if line.startswith("return ")]
        if len(lines) == 1:
            expression = lines[0]
        elif returns:
            expression = returns[-1]
        else:
            raise ValueError("external LLM code must be a single Series expression or return expression")
    return expression


def _validate_external_llm_ast(tree: ast.AST, *, allowed_names: set[str]) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_LLM_NODES):
            raise ValueError(f"external LLM expression uses unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"external LLM expression uses unsupported name: {node.id}")
        if isinstance(node, ast.Attribute):
            is_numpy_attr = isinstance(node.value, ast.Name) and node.value.id == "np" and node.attr in _ALLOWED_NUMPY_ATTRS
            if node.attr.startswith("_") or (node.attr not in _ALLOWED_LLM_METHODS and not is_numpy_attr):
                raise ValueError(f"external LLM expression uses unsupported method: {node.attr}")


def _stable_payload_hash(payload: Any) -> str:
    return sha1(json.dumps(_jsonable_payload(payload), sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _jsonable_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(key): _jsonable_payload(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_jsonable_payload(value) for value in payload]
    if isinstance(payload, Path):
        return payload.as_posix()
    return payload


def _candidate_provider_summary(
    candidate_provider: CognitiveAlphaCandidateProvider,
    *,
    profile: CognitiveAlphaEvolutionProfile,
    requested_initial_pool_size: int,
    generated_candidate_count: int,
) -> dict[str, Any]:
    guidance_styles = getattr(candidate_provider, "guidance_styles", ())
    summary = {
        "provider_id": str(getattr(candidate_provider, "provider_id", "unknown_candidate_provider")),
        "provider_kind": str(getattr(candidate_provider, "provider_kind", "unknown")),
        "profile_agent_count": int(profile.agent_count),
        "guidance_style_count": int(len(guidance_styles)),
        "requested_initial_pool_size": int(requested_initial_pool_size),
        "generated_candidate_count": int(generated_candidate_count),
    }
    if hasattr(candidate_provider, "provider_summary"):
        summary.update(getattr(candidate_provider, "provider_summary")())
    return summary


def build_cognitive_alpha_evolution_profile(profile: str) -> CognitiveAlphaEvolutionProfile:
    normalized = profile.lower()
    if normalized == "soxl_smoke":
        return CognitiveAlphaEvolutionProfile(
            profile="soxl_smoke",
            agent_count=3,
            candidates_per_agent=3,
            generations=2,
            initial_pool_size=9,
            parent_pool_size=4,
            children_per_generation=6,
        )
    if normalized == "soxl_full":
        return CognitiveAlphaEvolutionProfile(
            profile="soxl_full",
            agent_count=21,
            candidates_per_agent=4,
            generations=24,
            initial_pool_size=80,
            parent_pool_size=32,
            children_per_generation=96,
        )
    raise ValueError("profile must be one of: soxl_smoke, soxl_full")


def run_cognitive_alpha_evolution(
    frame: pd.DataFrame,
    *,
    spec: CognitiveAlphaEvolutionSpec,
    label_col: str,
    initial_candidates: list[CognitiveAlphaCandidate] | None = None,
    alpha158_columns: list[str] | None = None,
    candidate_provider: CognitiveAlphaCandidateProvider | None = None,
) -> CognitiveAlphaEvolutionResult:
    if frame.empty:
        raise ValueError("cognitive alpha evolution frame is empty")
    if spec.generations <= 0:
        raise ValueError("generations must be positive")
    if spec.parent_pool_size <= 0:
        raise ValueError("parent_pool_size must be positive")
    if spec.children_per_generation <= 0:
        raise ValueError("children_per_generation must be positive")

    run_dir = Path(spec.runtime_dir) / spec.run_id
    _ensure_run_dirs(run_dir)
    profile_contract = build_cognitive_alpha_evolution_profile(spec.profile)
    initial_pool, candidate_provider_summary = _initial_candidate_pool(
        initial_candidates,
        alpha158_columns,
        candidate_provider=candidate_provider,
        frame=frame,
        profile=profile_contract,
        target_initial_pool_size=spec.initial_pool_size or profile_contract.initial_pool_size,
    )
    initial_candidate_count = len(initial_pool)
    candidate_map = _candidate_map(initial_pool)
    generation_metrics: list[pd.DataFrame] = []
    start_generation, previous_selected, resumed = _resume_state(run_dir, spec)
    if resumed:
        replayed_selected = _replay_candidate_map(
            run_dir=run_dir,
            completed_generation=start_generation - 1,
            candidate_map=candidate_map,
            parent_pool_size=spec.parent_pool_size,
            children_per_generation=spec.children_per_generation,
        )
        if replayed_selected:
            previous_selected = replayed_selected

    if start_generation == 0:
        generation_candidates = list(candidate_map.values())
    else:
        generation_candidates = _build_next_generation(
            generation=start_generation,
            parent_names=previous_selected,
            candidate_map=candidate_map,
            children_per_generation=spec.children_per_generation,
        )
        candidate_map.update(_candidate_map(generation_candidates))

    final_report: CognitiveAlphaReport | None = None
    selected_names: list[str] = previous_selected
    for generation in range(start_generation, spec.generations):
        if generation > start_generation:
            generation_candidates = _build_next_generation(
                generation=generation,
                parent_names=selected_names,
                candidate_map=candidate_map,
                children_per_generation=spec.children_per_generation,
            )
            candidate_map.update(_candidate_map(generation_candidates))

        prepared_candidates = _with_repair_candidates(generation_candidates)
        candidate_map.update(_candidate_map(prepared_candidates))
        _write_candidate_pool(run_dir, generation, prepared_candidates)
        final_report = evaluate_cognitive_alpha_candidates(
            frame,
            candidates=prepared_candidates,
            label_col=label_col,
            generation=generation,
            min_observations=spec.min_observations,
            max_missing_ratio=spec.max_missing_ratio,
            min_distinct_values=spec.min_distinct_values,
            qualified_percentile=spec.qualified_percentile,
            elite_percentile=spec.elite_percentile,
            max_abs_correlation=spec.max_abs_correlation,
            fallback_top_n=spec.fallback_top_n,
        )
        selected_names = _parent_names(final_report.metrics, limit=spec.parent_pool_size)
        _write_generation_outputs(run_dir, generation, final_report)
        _write_state(
            run_dir,
            spec=spec,
            completed_generation=generation,
            selected_names=selected_names,
            resumed=resumed,
            resume_start_generation=start_generation,
            reports=generation_metrics + [final_report.metrics],
            initial_candidate_count=initial_candidate_count,
            candidate_provider_summary=candidate_provider_summary,
        )
        generation_metrics.append(final_report.metrics.assign(evolution_generation=generation))

    if final_report is None:
        final_generation = spec.generations - 1
        final_report = _load_final_report(run_dir, final_generation)
        generation_metrics.append(final_report.metrics.assign(evolution_generation=final_generation))
        selected_names = final_report.selected_feature_names

    combined_metrics = pd.concat(generation_metrics, ignore_index=True) if generation_metrics else pd.DataFrame()
    summary = _summary(
        run_dir,
        spec=spec,
        selected_names=selected_names,
        generation_metrics=combined_metrics,
        resumed=resumed,
        resume_start_generation=start_generation,
        initial_candidate_count=initial_candidate_count,
        candidate_provider_summary=candidate_provider_summary,
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return CognitiveAlphaEvolutionResult(
        run_dir=run_dir,
        summary=summary,
        generation_metrics=combined_metrics,
        selected_feature_names=selected_names,
        selected_candidates=[candidate_map[name] for name in selected_names if name in candidate_map],
        final_report=final_report,
        candidate_map=dict(candidate_map),
    )


def _ensure_run_dirs(run_dir: Path) -> None:
    for child in ["candidate_pool", "quality", "fitness", "rejected", "lineage"]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)


def _initial_candidate_pool(
    initial_candidates: list[CognitiveAlphaCandidate] | None,
    alpha158_columns: list[str] | None,
    *,
    candidate_provider: CognitiveAlphaCandidateProvider | None,
    frame: pd.DataFrame,
    profile: CognitiveAlphaEvolutionProfile,
    target_initial_pool_size: int,
) -> tuple[list[CognitiveAlphaCandidate], dict[str, Any]]:
    if candidate_provider is not None:
        explicit_candidates = list(initial_candidates) if initial_candidates is not None else []
        remaining_pool_size = max(int(target_initial_pool_size) - len(explicit_candidates), 0)
        provider_candidates = (
            candidate_provider.generate_candidates(
                profile=profile,
                frame=frame,
                alpha158_columns=alpha158_columns,
                target_initial_pool_size=remaining_pool_size,
            )
            if remaining_pool_size > 0
            else []
        )
        if remaining_pool_size > 0 and not provider_candidates:
            raise ValueError("candidate provider returned no candidates")
        candidates = [*explicit_candidates, *provider_candidates]
        candidates = _dedupe_candidates(candidates)[:target_initial_pool_size]
        if not candidates:
            raise ValueError("candidate provider returned no candidates")
        return candidates, _candidate_provider_summary(
            candidate_provider,
            profile=profile,
            requested_initial_pool_size=target_initial_pool_size,
            generated_candidate_count=len(candidates),
        )

    candidates = list(initial_candidates) if initial_candidates is not None else build_cognitive_alpha_seed_candidates()
    if alpha158_columns:
        candidates.extend(
            build_column_alpha_candidates(
                alpha158_columns,
                level="alpha158_baseline",
                agent="Alpha158",
            )
        )
    if not candidates:
        raise ValueError("at least one initial cognitive alpha candidate is required")
    candidates = _dedupe_candidates(candidates)
    if target_initial_pool_size > 0:
        candidates = candidates[:target_initial_pool_size]
    return candidates, {
        "provider_id": "static_seed_alpha158_pool",
        "provider_kind": "repo_owned_static",
        "profile_agent_count": 0,
        "guidance_style_count": 0,
        "requested_initial_pool_size": int(target_initial_pool_size),
        "generated_candidate_count": int(len(candidates)),
    }


def _dedupe_candidates(candidates: Iterable[CognitiveAlphaCandidate]) -> list[CognitiveAlphaCandidate]:
    out: dict[str, CognitiveAlphaCandidate] = {}
    for candidate in candidates:
        out.setdefault(candidate.name, candidate)
    return list(out.values())


def _candidate_map(candidates: Iterable[CognitiveAlphaCandidate]) -> dict[str, CognitiveAlphaCandidate]:
    return {candidate.name: candidate for candidate in candidates}


def _resume_state(run_dir: Path, spec: CognitiveAlphaEvolutionSpec) -> tuple[int, list[str], bool]:
    state_path = run_dir / "state.json"
    if not spec.resume or not state_path.exists():
        return 0, [], False
    state = json.loads(state_path.read_text(encoding="utf-8"))
    completed = int(state.get("completed_generation", -1))
    start_generation = min(completed + 1, spec.generations)
    selected = [str(name) for name in state.get("selected_feature_names", [])]
    return start_generation, selected, True


def _load_generation_metrics(run_dir: Path, generation: int) -> pd.DataFrame:
    path = run_dir / "fitness" / f"generation_{generation:03d}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _replay_candidate_map(
    *,
    run_dir: Path,
    completed_generation: int,
    candidate_map: dict[str, CognitiveAlphaCandidate],
    parent_pool_size: int,
    children_per_generation: int,
) -> list[str]:
    candidate_map.update(_candidate_map(_with_repair_candidates(list(candidate_map.values()))))
    selected: list[str] = []
    for generation in range(0, completed_generation + 1):
        if generation > 0:
            children = _build_next_generation(
                generation=generation,
                parent_names=selected,
                candidate_map=candidate_map,
                children_per_generation=children_per_generation,
            )
            candidate_map.update(_candidate_map(_with_repair_candidates(children)))
        metrics = _load_generation_metrics(run_dir, generation)
        if not metrics.empty:
            selected = _parent_names(metrics, limit=parent_pool_size)
    return selected


def _with_repair_candidates(candidates: list[CognitiveAlphaCandidate]) -> list[CognitiveAlphaCandidate]:
    repaired: list[CognitiveAlphaCandidate] = []
    for candidate in candidates:
        repaired.append(candidate)
        repair = _repair_candidate(candidate)
        if repair is not None:
            repaired.append(repair)
    return _dedupe_candidates(repaired)


def _repair_candidate(candidate: CognitiveAlphaCandidate) -> CognitiveAlphaCandidate | None:
    code = candidate.code or ""
    has_future_pattern = ".shift(-" in code or "Ref(" in code and ", -" in code
    if candidate.declared_lookahead <= 0 and not has_future_pattern:
        return None
    if not candidate.required_columns:
        return None
    source_col = candidate.required_columns[0]
    repaired_name = f"{candidate.name}_repaired_lag1"
    repair_reason = "future_lookahead_declared" if candidate.declared_lookahead > 0 else "future_lookahead_code_pattern"

    def _safe_lagged_change(frame: pd.DataFrame, col: str = source_col) -> pd.Series:
        values = pd.to_numeric(frame[col], errors="coerce").astype("float64")
        return values.pct_change().shift(1).rolling(3, min_periods=1).mean().rename(repaired_name)

    return CognitiveAlphaCandidate(
        name=repaired_name,
        level=candidate.level,
        agent=f"{candidate.agent}Repair",
        theme=f"repaired {candidate.theme}",
        required_columns=(source_col,),
        lookback=max(int(candidate.lookback), 3),
        function=_safe_lagged_change,
        economic_rationale=f"Repair of {candidate.name} using only lagged changes in {source_col}.",
        code=f"safe_lagged_pct_change({source_col}).rolling(3).mean()",
        declared_lookahead=0,
        metadata={
            **dict(candidate.metadata),
            "candidate_source": "repair_loop",
            "repair_source_candidate": candidate.name,
            "repair_source_provider_id": candidate.metadata.get("provider_id", ""),
            "repair_source_provider_kind": candidate.metadata.get("provider_kind", ""),
            "repair_reason": repair_reason,
        },
    )


def _rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    mean = values.rolling(window, min_periods=max(3, min(window, 10))).mean()
    std = values.rolling(window, min_periods=max(3, min(window, 10))).std(ddof=0).replace(0.0, np.nan)
    return (values - mean) / std


def _factor_values(candidate: CognitiveAlphaCandidate, frame: pd.DataFrame) -> pd.Series:
    cache = frame.attrs.setdefault("_cognitive_alpha_factor_cache", {})
    if candidate.name in cache:
        return cache[candidate.name]
    values = pd.to_numeric(candidate.function(frame), errors="coerce").replace([np.inf, -np.inf], np.nan)
    cache[candidate.name] = values
    return values


def _build_next_generation(
    *,
    generation: int,
    parent_names: list[str],
    candidate_map: dict[str, CognitiveAlphaCandidate],
    children_per_generation: int,
) -> list[CognitiveAlphaCandidate]:
    parents = [candidate_map[name] for name in parent_names if name in candidate_map]
    if not parents:
        parents = list(candidate_map.values())[: max(1, min(len(candidate_map), children_per_generation))]
    if not parents:
        raise ValueError("cannot evolve without parent candidates")

    children: list[CognitiveAlphaCandidate] = []
    for child_index in range(children_per_generation):
        operation = COGNITIVE_ALPHA_EVOLUTION_OPS[child_index % len(COGNITIVE_ALPHA_EVOLUTION_OPS)]
        left = parents[child_index % len(parents)]
        right = parents[(child_index + 1) % len(parents)]
        if operation == "mutation":
            children.append(_mutation_candidate(left, generation=generation, child_index=child_index))
        elif operation == "crossover":
            children.append(_crossover_candidate(left, right, generation=generation, child_index=child_index))
        else:
            children.append(
                _crossover_mutation_candidate(left, right, generation=generation, child_index=child_index)
            )
    return children


def _mutation_candidate(
    parent: CognitiveAlphaCandidate,
    *,
    generation: int,
    child_index: int,
) -> CognitiveAlphaCandidate:
    name = f"g{generation:03d}_{child_index:03d}_mutation_{_candidate_slug(parent.name)}"

    def _func(frame: pd.DataFrame, candidate: CognitiveAlphaCandidate = parent) -> pd.Series:
        return _rolling_zscore(_factor_values(candidate, frame), 10).shift(1).rename(name)

    return CognitiveAlphaCandidate(
        name=name,
        level=parent.level,
        agent=parent.agent,
        theme=f"mutation of {_candidate_slug(parent.name)}",
        required_columns=parent.required_columns,
        lookback=max(int(parent.lookback), 10),
        function=_func,
        economic_rationale=f"Smooths and lags parent {_candidate_slug(parent.name)} to reduce overreaction.",
        code=f"mutation:zscore({parent.name},10).shift(1)",
    )


def _crossover_candidate(
    left: CognitiveAlphaCandidate,
    right: CognitiveAlphaCandidate,
    *,
    generation: int,
    child_index: int,
) -> CognitiveAlphaCandidate:
    name = (
        f"g{generation:03d}_{child_index:03d}_crossover_"
        f"{_candidate_slug(left.name)}__{_candidate_slug(right.name)}"
    )

    def _func(
        frame: pd.DataFrame,
        left_candidate: CognitiveAlphaCandidate = left,
        right_candidate: CognitiveAlphaCandidate = right,
    ) -> pd.Series:
        left_values = _rolling_zscore(_factor_values(left_candidate, frame), 20)
        right_values = _rolling_zscore(_factor_values(right_candidate, frame), 20)
        return ((left_values + right_values) / 2.0).rename(name)

    return CognitiveAlphaCandidate(
        name=name,
        level="geometric_fusion",
        agent="AgentComposite",
        theme=f"crossover of {_candidate_slug(left.name)} and {_candidate_slug(right.name)}",
        required_columns=tuple(dict.fromkeys((*left.required_columns, *right.required_columns))),
        lookback=max(int(left.lookback), int(right.lookback), 20),
        function=_func,
        economic_rationale=(
            f"Combines parents {_candidate_slug(left.name)} and {_candidate_slug(right.name)} after scale normalization."
        ),
        code=f"crossover:mean(zscore({left.name},20), zscore({right.name},20))",
    )


def _crossover_mutation_candidate(
    left: CognitiveAlphaCandidate,
    right: CognitiveAlphaCandidate,
    *,
    generation: int,
    child_index: int,
) -> CognitiveAlphaCandidate:
    name = (
        f"g{generation:03d}_{child_index:03d}_crossover_mutation_"
        f"{_candidate_slug(left.name)}__{_candidate_slug(right.name)}"
    )

    def _func(
        frame: pd.DataFrame,
        left_candidate: CognitiveAlphaCandidate = left,
        right_candidate: CognitiveAlphaCandidate = right,
    ) -> pd.Series:
        spread = _rolling_zscore(_factor_values(left_candidate, frame), 10) - _rolling_zscore(
            _factor_values(right_candidate, frame),
            10,
        )
        return spread.ewm(span=5, adjust=False).mean().shift(1).rename(name)

    return CognitiveAlphaCandidate(
        name=name,
        level="geometric_fusion",
        agent="AgentCreative",
        theme=f"mutated crossover of {_candidate_slug(left.name)} and {_candidate_slug(right.name)}",
        required_columns=tuple(dict.fromkeys((*left.required_columns, *right.required_columns))),
        lookback=max(int(left.lookback), int(right.lookback), 10),
        function=_func,
        economic_rationale=(
            f"Uses lagged spread between parents {_candidate_slug(left.name)} and {_candidate_slug(right.name)}."
        ),
        code=f"crossover_mutation:ewm(zscore({left.name},10)-zscore({right.name},10),5).shift(1)",
    )


def _candidate_slug(name: str, *, prefix_chars: int = 18, digest_chars: int = 10) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_") or "candidate"
    if len(slug) <= prefix_chars + digest_chars + 1:
        return slug
    digest = sha1(str(name).encode("utf-8")).hexdigest()[:digest_chars]
    return f"{slug[:prefix_chars]}_{digest}"


def _parent_names(metrics: pd.DataFrame, *, limit: int) -> list[str]:
    selectable = metrics.loc[metrics["redundancy_status"].isin(["selected", "selected_fallback"])].copy()
    if selectable.empty:
        selectable = metrics.loc[metrics["quality_status"] == "pass"].copy()
    if selectable.empty:
        return []
    selectable = selectable.sort_values(["selection_score", "min_metric_percentile"], ascending=[False, False])
    return selectable["factor"].astype(str).head(limit).tolist()


def _candidate_manifest(candidate: CognitiveAlphaCandidate) -> dict[str, Any]:
    operation = _operation_from_name(candidate.name)
    return {
        "name": candidate.name,
        "level": candidate.level,
        "agent": candidate.agent,
        "theme": candidate.theme,
        "required_columns": list(candidate.required_columns),
        "lookback": int(candidate.lookback),
        "economic_rationale": candidate.economic_rationale,
        "code": candidate.code,
        "declared_lookahead": int(candidate.declared_lookahead),
        "metadata": dict(candidate.metadata),
        "operation": operation,
        "parents": _parents_from_code(candidate.code),
    }


def _operation_from_name(name: str) -> str:
    if "_crossover_mutation_" in name:
        return "crossover_mutation"
    if "_crossover_" in name:
        return "crossover"
    if "_mutation_" in name:
        return "mutation"
    if name.endswith("_repaired_lag1"):
        return "repair"
    return "initial"


def _parents_from_code(code: str) -> list[str]:
    if ":" not in code:
        return []
    payload = code.split(":", 1)[1]
    tokens = []
    for marker in ["zscore(", "mean("]:
        payload = payload.replace(marker, "(")
    for part in payload.replace(")", "(").split("("):
        if part and "," not in part and " " not in part and part not in {"mutation", "crossover_mutation"}:
            tokens.append(part)
    return list(dict.fromkeys(tokens))


def _write_candidate_pool(run_dir: Path, generation: int, candidates: list[CognitiveAlphaCandidate]) -> None:
    path = run_dir / "candidate_pool" / f"generation_{generation:03d}.jsonl"
    rows = [_candidate_manifest(candidate) | {"generation": generation} for candidate in candidates]
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    lineage_path = run_dir / "lineage" / f"generation_{generation:03d}.jsonl"
    lineage_path.write_text(
        "".join(
            json.dumps(
                {
                    "generation": generation,
                    "candidate": row["name"],
                    "operation": row["operation"],
                    "parents": row["parents"],
                },
                sort_keys=True,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_generation_outputs(run_dir: Path, generation: int, report: CognitiveAlphaReport) -> None:
    metrics = report.metrics.copy()
    metrics.to_csv(run_dir / "fitness" / f"generation_{generation:03d}.csv", index=False)
    metrics[
        [
            "factor",
            "generation",
            "level",
            "agent",
            "quality_status",
            "quality_reason",
            "missing_ratio",
            "distinct_values",
            "selection_tier",
            "redundancy_status",
        ]
    ].to_csv(run_dir / "quality" / f"generation_{generation:03d}.csv", index=False)
    rejected = metrics.loc[metrics["quality_status"] == "reject"].to_dict(orient="records")
    (run_dir / "rejected" / f"generation_{generation:03d}.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rejected),
        encoding="utf-8",
    )


def _write_state(
    run_dir: Path,
    *,
    spec: CognitiveAlphaEvolutionSpec,
    completed_generation: int,
    selected_names: list[str],
    resumed: bool,
    resume_start_generation: int,
    reports: list[pd.DataFrame],
    initial_candidate_count: int,
    candidate_provider_summary: dict[str, Any],
) -> None:
    combined = pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()
    state = _summary(
        run_dir,
        spec=spec,
        selected_names=selected_names,
        generation_metrics=combined,
        resumed=resumed,
        resume_start_generation=resume_start_generation,
        initial_candidate_count=initial_candidate_count,
        candidate_provider_summary=candidate_provider_summary,
    ) | {
        "completed_generation": int(completed_generation),
        "status": "running" if completed_generation < spec.generations - 1 else "complete",
    }
    (run_dir / "state.json").write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _summary(
    run_dir: Path,
    *,
    spec: CognitiveAlphaEvolutionSpec,
    selected_names: list[str],
    generation_metrics: pd.DataFrame,
    resumed: bool,
    resume_start_generation: int,
    initial_candidate_count: int,
    candidate_provider_summary: dict[str, Any],
) -> dict[str, Any]:
    repair_attempt_count = 0
    if not generation_metrics.empty:
        repair_attempt_count = int(generation_metrics["factor"].astype(str).str.endswith("_repaired_lag1").sum())
    return {
        "workflow": "cognitive_alpha_evolution",
        "paper_reference": COGNITIVE_ALPHA_PAPER_REFERENCE,
        "run_id": spec.run_id,
        "profile": spec.profile,
        "run_dir": run_dir.as_posix(),
        "completed_generations": int(spec.generations),
        "parent_pool_size": int(spec.parent_pool_size),
        "children_per_generation": int(spec.children_per_generation),
        "initial_candidate_count": int(initial_candidate_count),
        "candidate_provider": candidate_provider_summary,
        "selected_feature_names": selected_names,
        "selected_count": int(len(selected_names)),
        "candidate_metric_rows": int(len(generation_metrics)),
        "metric_names": ["ic", "icir", "rank_ic", "rank_icir", "mutual_information"],
        "repair_attempt_count": repair_attempt_count,
        "evolution_ops": list(COGNITIVE_ALPHA_EVOLUTION_OPS),
        "resumed": bool(resumed),
        "resume_start_generation": int(resume_start_generation),
        "thresholds": {
            "qualified_percentile": float(spec.qualified_percentile),
            "elite_percentile": float(spec.elite_percentile),
            "max_abs_correlation": float(spec.max_abs_correlation),
            "min_observations": int(spec.min_observations),
            "max_missing_ratio": float(spec.max_missing_ratio),
            "min_distinct_values": int(spec.min_distinct_values),
            "fallback_top_n": int(spec.fallback_top_n),
        },
        "checkpoints": {
            "candidate_pool": "candidate_pool/generation_*.jsonl",
            "quality": "quality/generation_*.csv",
            "fitness": "fitness/generation_*.csv",
            "rejected": "rejected/generation_*.jsonl",
            "lineage": "lineage/generation_*.jsonl",
            "state": "state.json",
        },
        "spec": {
            **asdict(spec),
            "runtime_dir": Path(spec.runtime_dir).as_posix(),
        },
    }


def _load_final_report(run_dir: Path, generation: int) -> CognitiveAlphaReport:
    metrics = pd.read_csv(run_dir / "fitness" / f"generation_{generation:03d}.csv")
    selected = _parent_names(metrics, limit=len(metrics))
    return CognitiveAlphaReport(
        metrics=metrics,
        correlation=pd.DataFrame(),
        feature_frame=pd.DataFrame(),
        selected_feature_names=selected,
        summary={
            "workflow": "cognitive_alpha_evolution",
            "paper_reference": COGNITIVE_ALPHA_PAPER_REFERENCE,
            "generation": int(generation),
        },
    )
