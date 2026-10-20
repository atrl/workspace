"""Bounded, data-free validation helpers for pre-registered research candidates."""
from __future__ import annotations

import ast, hashlib, json, math, re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    mechanism: str
    receiver: str
    expression: str
    allowed_fields: tuple[str, ...]
    fields_identity: str
    available_at_rule: str
    target_rule: str
    maturity_rule: str
    fixed_parameters: Mapping[str, Any]
    matched_control: str
    rejection_rule: str
    source_urls: tuple[str, ...] = ()
    prior_route: str = ""
    new_evidence_explanation: str = ""
    parent_route_closed: bool = False
    complete_economic_experiment: bool = False
    complexity_bound: int = 32
    expression_kind: str = "qlib_formula"
    code_review_ref: str = ""
    independently_verified_review_identity: str = ""


_FIELD = re.compile(r"\$([A-Za-z_]\w*)")
_CODE_ID = re.compile(r"([A-Za-z0-9_./-]+)#([0-9a-f]{64})\Z")
_RECEIVERS = {"allocation", "risk", "open", "recovery"}
_FUNCTIONS = {"Ref", "Mean", "Sum", "Std", "Var", "Max", "Min", "Med", "Mad",
              "Rank", "Quantile", "Count", "Slope", "Rsquare", "Resi", "WMA", "EMA",
              "Corr", "Cov", "Abs", "Log", "Sign", "If"}
_WINDOWS = _FUNCTIONS - {"Ref", "Corr", "Cov", "Abs", "Log", "Sign", "If", "Quantile"}
_ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant, ast.Add,
                  ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.UAdd, ast.USub, ast.Compare, ast.Lt,
                  ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)


def _literal_integer(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant): return node.value if type(node.value) is int else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _literal_integer(node.operand); return None if value is None else value * (-1 if isinstance(node.op, ast.USub) else 1)
    return None


def _literal_number(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = float(node.value)
        return value if math.isfinite(value) else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _literal_number(node.operand)
        return None if value is None else value * (-1 if isinstance(node.op, ast.USub) else 1)
    return None


def expression_digest(spec: CandidateSpec) -> str:
    """Parse a safe Qlib-like formula or a pinned code identity; never evaluates it."""
    if spec.expression_kind == "code_identity":
        match = _CODE_ID.fullmatch(spec.expression)
        if not match or ".." in match.group(1).split("/") or match.group(1).startswith("/"):
            raise ValueError("code_identity must be repo-relative path#sha256")
        if not spec.code_review_ref:
            raise ValueError("code_identity requires a review or evidence reference")
        if spec.allowed_fields:
            raise ValueError("code_identity candidates must not declare formula fields")
        return hashlib.sha256((match.group(1) + "#" + match.group(2)).encode()).hexdigest()
    if spec.expression_kind != "qlib_formula":
        raise ValueError("unknown expression_kind")
    fields = set(spec.allowed_fields)
    replaced = _FIELD.sub(lambda m: "__field_" + m.group(1), spec.expression.strip())
    names = set(re.findall(r"\$([A-Za-z_]\w*)", spec.expression))
    placeholders = {"__field_" + name for name in names}
    if not names or not names <= fields:
        raise ValueError("formula fields are empty or outside allowed_fields")
    try:
        tree = ast.parse(replaced, mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid expression syntax") from exc
    if any(not isinstance(node, _ALLOWED_NODES) for node in ast.walk(tree)):
        raise ValueError("expression contains a disallowed syntax node")
    if any(isinstance(n, ast.Constant) and not (isinstance(n.value, (int, bool)) or
           isinstance(n.value, float) and math.isfinite(n.value))
           for n in ast.walk(tree)):
        raise ValueError("only finite numeric and boolean constants are allowed")
    operations = sum(isinstance(n, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Compare)) for n in ast.walk(tree))
    if not 1 <= spec.complexity_bound <= 32 or operations > spec.complexity_bound:
        raise ValueError("expression exceeds the supplied complexity bound (maximum 32)")
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id.startswith("__field_"):
                if node.id not in placeholders: raise ValueError("field placeholder must use $field syntax")
                continue
            parent = parents.get(node)
            if node.id not in _FUNCTIONS or not (isinstance(parent, ast.Call) and parent.func is node):
                raise ValueError(f"unknown name: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS or node.keywords:
                raise ValueError("call is outside the operator whitelist")
            name, args = node.func.id, node.args
            if name == "Ref":
                offset = _literal_integer(args[1]) if len(args) == 2 else None
                if offset is None or offset < 0:
                    raise ValueError("Ref requires a known nonnegative offset; negative offsets are future-looking")
            elif name in _WINDOWS:
                window = _literal_integer(args[1]) if len(args) == 2 else None
                if window is None or window <= 0:
                    raise ValueError(f"{name} requires a known positive integer window")
            elif name in {"Corr", "Cov"}:
                window = _literal_integer(args[2]) if len(args) == 3 else None
                if window is None or window <= 0:
                    raise ValueError(f"{name} requires a known positive integer window")
            elif name == "Quantile":
                window = _literal_integer(args[1]) if len(args) == 3 else None
                quantile = _literal_number(args[2]) if len(args) == 3 else None
                if window is None or window <= 0 or quantile is None or not 0 <= quantile <= 1:
                    raise ValueError("Quantile requires a positive window and q in [0, 1]")
            elif name in {"Abs", "Log", "Sign"} and len(args) != 1:
                raise ValueError(f"{name} takes one argument")
            elif name == "If" and len(args) != 3:
                raise ValueError("If takes three arguments")
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def _binding(spec: CandidateSpec, digest: str) -> str:
    value = asdict(spec)
    value.pop("candidate_id"); value["expression_digest"] = digest
    value["allowed_fields"] = sorted(spec.allowed_fields)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_batch(candidates: Sequence[CandidateSpec]) -> dict[str, dict[str, str]]:
    if not candidates or len(candidates) > 8:
        raise ValueError("candidate batch must contain 1..8 candidates")
    if len({c.mechanism for c in candidates}) > 2:
        raise ValueError("candidate batch allows at most two mechanisms")
    if sum(c.complete_economic_experiment for c in candidates) > 1:
        raise ValueError("candidate batch allows at most one complete economic experiment")
    if len({c.candidate_id for c in candidates}) != len(candidates):
        raise ValueError("candidate IDs must be unique")
    result, seen_ast = {}, set()
    for spec in candidates:
        if not spec.candidate_id or not spec.mechanism or spec.receiver not in _RECEIVERS:
            raise ValueError("candidate ID, mechanism, or receiver is invalid")
        if not spec.fields_identity or not spec.available_at_rule or not spec.target_rule or not spec.maturity_rule:
            raise ValueError("field identity, availability, target, and maturity rules are required")
        if not spec.matched_control or not spec.rejection_rule or not isinstance(spec.fixed_parameters, Mapping):
            raise ValueError("fixed parameters, matched control, and rejection rule are required")
        if spec.source_urls and any(not re.match(r"^https://[^/\s]+", u) for u in spec.source_urls):
            raise ValueError("source references must use HTTPS URLs")
        if spec.parent_route_closed and (not spec.prior_route or not spec.new_evidence_explanation):
            raise ValueError("closed-route candidates require prior route and new-evidence explanation")
        if not spec.source_urls and not (spec.prior_route or spec.new_evidence_explanation):
            raise ValueError("candidate needs source URLs or explicit prior-route/new-evidence context")
        digest = expression_digest(spec)
        binding = _binding(spec, digest)
        if digest in seen_ast:
            raise ValueError("duplicate canonical expression/binding in candidate batch")
        seen_ast.add(digest)
        result[spec.candidate_id] = {"expression_sha256": digest, "binding_sha256": binding,
            "code_review_status": ("not_applicable" if spec.expression_kind == "qlib_formula" else
                "review_identity_supplied" if spec.independently_verified_review_identity else "needs_code_review")}
    return result


def memo_key(spec: CandidateSpec, *, data_identity: str, code_identity: str, split_identity: str) -> str:
    digest = expression_digest(spec)
    value = {"expression_sha256": digest, "allowed_fields": sorted(spec.allowed_fields),
             "fields_identity": spec.fields_identity, "receiver": spec.receiver,
             "available_at_rule": spec.available_at_rule, "target_rule": spec.target_rule,
             "maturity_rule": spec.maturity_rule, "fixed_parameters": spec.fixed_parameters,
             "data_identity": data_identity, "code_identity": code_identity,
             "split_identity": split_identity, "review_identity": spec.independently_verified_review_identity}
    if not all(isinstance(v, str) and v for v in (data_identity, code_identity, split_identity)):
        raise ValueError("data, code, and split identities are required")
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def failure_record(spec: CandidateSpec, *, failed_gate: str, reason: str) -> dict[str, Any]:
    return {"schema_version": 1, "candidate_id": spec.candidate_id, "mechanism": spec.mechanism,
            "failed_gate": failed_gate, "reason": reason,
            "route_state": "closed",
            "parent_route_state": "closed" if spec.parent_route_closed else "open",
            "reopening_rule": "requires new_evidence_ref or reviewed_mechanism_change; parameter/name rescue is insufficient"}


def reopening_assessment(failure: Mapping[str, Any], *, new_evidence_ref: str = "", reviewed_mechanism_change: str = "",
                         changed_parameters: Sequence[str] = ()) -> dict[str, Any]:
    closed = failure.get("route_state") == "closed" or failure.get("parent_route_state") == "closed"
    basis = bool(new_evidence_ref or reviewed_mechanism_change)
    return {"eligible_for_independent_review": not closed or basis, "route_reopened": False,
            "new_evidence_ref": new_evidence_ref or None, "reviewed_mechanism_change": reviewed_mechanism_change or None,
            "parameter_only_rescue_rejected": bool(changed_parameters) and not basis, "review_required": bool(closed and basis)}


def correlation_report(series: Mapping[str, Any], *, scope: str) -> dict[str, Any]:
    """Report aligned development-scope Pearson correlations; never selects or transforms."""
    if scope != "development":
        raise ValueError("correlation reports are limited to the development scope")
    if len(series) < 2:
        raise ValueError("at least two candidate series are required")
    names = list(series)
    first = series[names[0]]
    index = getattr(first, "index", None)
    if callable(index): index = None
    values: dict[str, list[float]] = {}
    for name in names:
        item = series[name]
        item_index = getattr(item, "index", None)
        if callable(item_index):
            item_index = None
        if (index is None) != (item_index is None) or (index is not None and not index.equals(item_index)):
            raise ValueError("candidate series must have aligned indexes")
        raw = item.tolist() if hasattr(item, "tolist") else list(item)
        values[name] = [float(v) for v in raw]
        if any(not math.isfinite(v) for v in values[name]):
            raise ValueError("candidate series must contain finite numeric values")
    lengths = {len(v) for v in values.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 2:
        raise ValueError("candidate series must have equal lengths of at least two")
    matrix = {name: {} for name in names}
    for left in names:
        for right in names:
            x, y = values[left], values[right]
            mx, my = sum(x) / len(x), sum(y) / len(y)
            xx = sum((v - mx) ** 2 for v in x)
            yy = sum((v - my) ** 2 for v in y)
            matrix[left][right] = (sum((a - mx) * (b - my) for a, b in zip(x, y)) /
                                   math.sqrt(xx * yy)) if xx and yy else None
    return {"scope": scope, "n": next(iter(lengths)), "correlation": matrix,
            "orthogonalized": False, "selected_candidate": None,
            "use": "redundancy_diagnostic_only"}
