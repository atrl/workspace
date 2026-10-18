"""One matched S/Z supervision head; this module grants no execution permission.

The fit loop narrowly copies the original MinutePitAllocator preparation and
L_original operation order. Only after that expression is the fixed half-MSE
added. Sources, train-only residency, source-to-PG index alignment, approvals
and the six-model evaluation barrier belong to the workflow.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.model.minute_pit_allocator import MinutePitAllocator, TARGETS, FLAGS, _flat_index, _scaled
from src.workflow.pit_constituent_auxiliary_training_allocation import (
    COMMON_FEATURES, _project_legal_weights, _standardize_pair,
)

ORIGINAL_MODULE_SHA256 = "d06962ac2739633c0af34e3843365f1ad4c0a9bbc9dff0ba38e1636db6a175c8"
ORIGINAL_FIT_SOURCE_SHA256 = "0e499b0d2abc7782623370badb66e2fe12a8278cac8e4127456c9ab4342153e2"
SUPERVISION_NAMES = ("S", "Z")


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _index_hash(index):
    if isinstance(index, pd.MultiIndex):
        if "datetime" not in index.names:
            raise ValueError("supervision index requires datetime")
        dates = index.get_level_values("datetime")
        dtypes = [str(index.get_level_values(i).dtype) for i in range(index.nlevels)]
    elif isinstance(index, pd.DatetimeIndex):
        dates, dtypes = index, [str(index.dtype)]
    else:
        raise ValueError("supervision requires a native date index")
    if (not isinstance(dates, pd.DatetimeIndex) or dates.hasnans or dates.tz is not None
            or not dates.is_unique or not dates.is_monotonic_increasing
            or not index.is_unique or not dates.equals(dates.normalize())):
        raise ValueError("supervision requires unique ordered naive session dates")
    schema = {"class": type(index).__name__, "names": list(index.names), "dtypes": dtypes}
    digest = hashlib.sha256(_json_hash(schema).encode())
    digest.update(pd.util.hash_pandas_object(index, index=True).to_numpy(dtype="<u8").tobytes())
    return digest.hexdigest()


def _array_hash(values, dtype):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


def _target_values(series):
    if (not isinstance(series, pd.Series) or series.name not in SUPERVISION_NAMES
            or not pd.api.types.is_numeric_dtype(series.dtype)
            or pd.api.types.is_bool_dtype(series.dtype)
            or pd.api.types.is_complex_dtype(series.dtype) or len(series) < 2):
        raise ValueError("invalid S/Z supervision series")
    values = series.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("supervision target must be finite")
    return values


def _normalize(values, mean, std):
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("supervision scale must be positive and finite")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = ((values - np.float64(mean)) / np.float64(std)).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("normalized supervision target must be finite")
    return normalized


def _validate_scaler(stats):
    if stats.get("schema_version") != 1 or stats.get("name") not in SUPERVISION_NAMES:
        raise ValueError("invalid frozen supervision scaler")
    if stats.get("ddof") != 1 or stats.get("normalized_dtype") != "float32":
        raise ValueError("frozen supervision scale contract differs")
    if not np.isfinite(stats["mean"]) or not np.isfinite(stats["std"]) or stats["std"] <= 0:
        raise ValueError("supervision scale must be positive and finite")
    body = {k: v for k, v in stats.items() if k != "stats_sha256"}
    if stats.get("stats_sha256") != _json_hash(body):
        raise ValueError("frozen supervision scaler hash mismatch")


def fit_supervision_scalers(raw_S_Z: pd.DataFrame) -> dict:
    """Fit once in the authorized train-only preflight, never inside model.fit.

    The workflow passes S/Z already aligned onto the original PG full index.
    The returned stats and their package identity are frozen for all six fits.
    """
    if not isinstance(raw_S_Z, pd.DataFrame) or tuple(raw_S_Z.columns) != SUPERVISION_NAMES:
        raise ValueError("expected exactly ordered S,Z supervision columns")
    index_sha = _index_hash(raw_S_Z.index)
    stats, normalized = {}, pd.DataFrame(index=raw_S_Z.index.copy())
    for name in SUPERVISION_NAMES:
        values = _target_values(raw_S_Z[name])
        with np.errstate(over="ignore", invalid="ignore"):
            mean, std = float(values.mean(dtype=np.float64)), float(values.std(ddof=1, dtype=np.float64))
        transformed = _normalize(values, mean, std)
        item = {"schema_version": 1, "name": name, "rows": len(values), "ddof": 1,
                "mean": mean, "std": std, "normalized_dtype": "float32",
                "index_sha256": index_sha, "raw_sha256": _array_hash(values, "<f8"),
                "normalized_sha256": _array_hash(transformed, "<f4")}
        item["stats_sha256"] = _json_hash(item)
        stats[name] = item
        normalized[name] = transformed
    return {"stats": stats, "normalized_targets": normalized, "index_sha256": index_sha,
            "package_sha256": _json_hash(stats)}


def transform_supervision_target(series: pd.Series, target_scaler: dict) -> pd.Series:
    """Verify frozen identities and transform; do not estimate any statistic."""
    _validate_scaler(target_scaler)
    values = _target_values(series)
    if (series.name != target_scaler["name"] or len(series) != target_scaler["rows"]
            or _index_hash(series.index) != target_scaler["index_sha256"]
            or _array_hash(values, "<f8") != target_scaler["raw_sha256"]):
        raise ValueError("frozen supervision target/index identity mismatch")
    normalized = _normalize(values, target_scaler["mean"], target_scaler["std"])
    if _array_hash(normalized, "<f4") != target_scaler["normalized_sha256"]:
        raise ValueError("frozen normalized supervision target hash mismatch")
    return pd.Series(normalized, index=series.index, name=series.name)


def primary_forward_with_state(network, primary_x, common_x):
    """Capture the actual original primary state once, then remove the hook.

    Nothing is stored on the network/model. Subsequent auxiliary encoder calls
    cannot replace this local state, and exceptions also remove the hook.
    """
    captured = []
    handle = network.tail_encoder.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output))
    try:
        outputs = network(primary_x, common_x)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("expected one original primary encoder forward")
    return outputs, captured.pop()


def append_supervision_loss(original_loss, prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 1:
        raise ValueError("supervision prediction/target shape mismatch")
    half_mse = .5 * ((prediction - target) ** 2).mean()
    return original_loss + .02 * half_mse, half_mse


def clip_once(parameters):
    return nn.utils.clip_grad_norm_(parameters, max_norm=2.)


class DayNightSupervisionAllocator(MinutePitAllocator):
    """Same paired 982 parameters and original inference, with S or Z loss."""

    def __init__(self, *, target_name, target_scaler, **original_args):
        super().__init__(**original_args)
        _validate_scaler(target_scaler)
        if target_name not in SUPERVISION_NAMES or target_scaler["name"] != target_name:
            raise ValueError("target name and frozen scale differ")
        if (self.extra_features or len(self.base_features) != 39
                or len(set(self.base_features)) != 39 or len(COMMON_FEATURES) != 16
                or set(self.base_features) & set(SUPERVISION_NAMES)
                or self.spec.hidden_units != 16 or self.spec.tail_hidden_units != 8):
            raise ValueError("original39/16 inputs and16/8 network required")
        self.target_name, self.target_scaler = target_name, deepcopy(target_scaler)
        self.training_trace = []
        self.completed_step_count = 0
        self.optimizer_state_dict = None

    def _initialize(self):
        network = super()._initialize()
        # Creating nn.Linear normally draws random values. Restore that state
        # before returning, and overwrite every new parameter with exact zero.
        rng = torch.get_rng_state().clone()
        try:
            head = nn.Linear(8, 1)
            with torch.no_grad():
                head.weight.zero_()
                head.bias.zero_()
        finally:
            torch.set_rng_state(rng)
        network.supervision_head = head
        if any(p.dtype != torch.float32 or p.device.type != "cpu" for p in network.parameters()):
            raise ValueError("original CPU float32 network required")
        return network

    def fit(self, dataset, *, auxiliary_dataset, **kwargs):
        prepared = dataset.prepare("train", col_set=["feature", "label", "common", "meta", "supervision"])
        source = _flat_index(prepared)
        maturity = pd.to_datetime(source[("meta", "label_available_at")])
        eligible = maturity.notna() & maturity.lt(self.cutoff) & (source.index <= self.train_end)
        eligible &= np.isfinite(source["label"].loc[:, list(TARGETS)].to_numpy(float)).all(axis=1)
        train = source.loc[eligible].copy()
        if len(train) < 100:
            raise ValueError("needs_data: insufficient matured allocation training rows")
        # Full-index identity is checked before the original flattening. No
        # new-target eligibility/filtering, index repair, filling or scale fit.
        if not eligible.all() or tuple(prepared["supervision"].columns) != SUPERVISION_NAMES:
            raise ValueError("supervision requires the complete original mature cohort")
        transformed = transform_supervision_target(prepared["supervision"][self.target_name], self.target_scaler)
        supervision_target = torch.tensor(transformed.to_numpy(), dtype=torch.float32)
        features, labels, common = train["feature"], train["label"], train["common"]
        primary_x, _, primary_scaler = _standardize_pair(features, features, self.base_features)
        common_x, _, common_scaler = _standardize_pair(common, common, COMMON_FEATURES)
        if not torch.isfinite(primary_x).all() or not torch.isfinite(common_x).all():
            raise ValueError("needs_data: an original model input has no finite training support")
        self.primary_scaler = {key: value.to_dict() for key, value in primary_scaler.items()}
        self.common_scaler = {key: value.to_dict() for key, value in common_scaler.items()}
        self._fit_extra_scaler(features)
        primary_x = torch.cat([primary_x, self._extra_tensor(features)], dim=1)

        aux = auxiliary_dataset.prepare("train", col_set=["feature", "label", "meta"])
        if len(aux) and pd.to_datetime(aux[("meta", "observations_end")]).max() > self.train_end:
            raise ValueError("blocked: auxiliary observations exceed the training cutoff")
        aux_features = aux["feature"]
        aux_labels = aux["label"]
        auxiliary_x = _scaled(aux_features, list(COMMON_FEATURES), self.common_scaler)
        aux_tail = torch.tensor(aux_labels["synthetic_2x_crash_label"].to_numpy(np.float32))
        aux_alpha = torch.tensor(aux_labels["synthetic_2x_upside_label"].to_numpy(np.float32))
        aux_weight = torch.tensor(aux_labels["training_weight"].to_numpy(np.float32))
        if len(aux_weight):
            aux_weight /= aux_weight.mean()
        returns = torch.tensor(labels.loc[:, list(TARGETS[:2])].to_numpy(np.float32))
        tail = torch.tensor(labels.tail_event_5d.to_numpy(np.float32))
        alpha = torch.tensor(labels.relative_alpha_5d.gt(0).to_numpy(np.float32))
        cash = torch.tensor(labels.tail_cash_target_5d.to_numpy(np.float32))
        years = torch.tensor(train.index.year.to_numpy(), dtype=torch.int64)
        if ((len(train), len(aux)) != (3014, 20741) or self.spec.epochs != 300
                or self.seed not in (71, 172, 273) or self.spec.auxiliary_batch_size != 2048
                or self.spec.primary_tail_loss_weight != .02 or self.candidate.auxiliary_weight != .05
                or self.spec.training_cost_bps != 10 or self.spec.turnover_loss_weight != .006):
            raise ValueError("fixed original cohort/epoch/sampling/loss contract required")
        self.network = self._initialize()
        self.prepare_epoch_observation(dataset)
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.spec.learning_rate,
                                     weight_decay=self.spec.weight_decay)
        self.training_trace, self.completed_step_count = [], 0
        primary_state = prediction = logits = tail_logits = alpha_logits = None
        state = aux_tail_logits = aux_alpha_logits = None
        allocation = turnover = annual = primary_loss = auxiliary_loss = original_loss = loss = half_mse = None
        try:
            for epoch in range(self.spec.epochs):
                optimizer.zero_grad()
                (logits, tail_logits, alpha_logits), primary_state = primary_forward_with_state(self.network, primary_x, common_x)
                weights = _project_legal_weights(logits, maximum_soxl_weight=self.spec.maximum_soxl_weight,
                                                minimum_cash_weight=self.spec.minimum_cash_weight)
                allocation, turnover, annual, _ = self.allocation_objective(weights=weights, returns=returns,
                    years=years, cash_target=cash, tail_event=tail, candidate=self.candidate,
                    drawdown_target_pct=-30., worst_5d_target_pct=self.spec.research_worst_5d_target_pct,
                    minimum_cash_weight=self.spec.minimum_cash_weight, cost_bps=self.spec.training_cost_bps,
                    turnover_weight=self.spec.turnover_loss_weight)
                primary_loss = nn.functional.binary_cross_entropy_with_logits(tail_logits, tail)
                primary_loss += nn.functional.binary_cross_entropy_with_logits(alpha_logits, alpha)
                auxiliary_loss = torch.zeros((), dtype=allocation.dtype)
                sampled = None
                if len(auxiliary_x):
                    generator = torch.Generator().manual_seed(self.seed * 1000 + epoch)
                    sampled = torch.randint(len(auxiliary_x), (min(self.spec.auxiliary_batch_size, len(auxiliary_x)),),
                                            generator=generator)
                    state = self.network.tail_encoder(auxiliary_x[sampled])
                    aux_tail_logits, aux_alpha_logits = self.auxiliary_logits(state)
                    auxiliary_loss = ((nn.functional.binary_cross_entropy_with_logits(aux_tail_logits,
                        aux_tail[sampled], reduction="none") + nn.functional.binary_cross_entropy_with_logits(
                        aux_alpha_logits, aux_alpha[sampled], reduction="none")) * aux_weight[sampled]).mean()
                original_loss = allocation + self.spec.primary_tail_loss_weight * primary_loss + self.candidate.auxiliary_weight * auxiliary_loss
                prediction = self.network.supervision_head(primary_state).squeeze(1)
                loss, half_mse = append_supervision_loss(original_loss, prediction, supervision_target)
                if not torch.isfinite(loss):
                    raise ValueError("blocked: allocation training loss is nonfinite")
                loss.backward()
                norm = clip_once(self.network.parameters())
                optimizer.step()
                self.completed_step_count += 1
                self.training_trace.append({"epoch": epoch + 1, "target_name": self.target_name,
                    "allocation_loss": float(allocation.detach()), "primary_bce": float(primary_loss.detach()),
                    "auxiliary_bce": float(auxiliary_loss.detach()), "original_loss": float(original_loss.detach()),
                    "supervision_half_mse": float(half_mse.detach()), "loss": float(loss.detach()),
                    "global_preclip_norm": float(norm), "completed_step_count": self.completed_step_count,
                    "sampled_indices_sha256": hashlib.sha256(sampled.numpy().tobytes()).hexdigest()})
                self.observe_epoch(epoch + 1)
            self.network.eval()
            self.audit = {"training_rows": len(train), "first_training_signal": str(train.index.min().date()),
                "last_training_signal": str(train.index.max().date()), "last_mature_label_at": str(maturity.loc[eligible].max()),
                "cutoff": str(self.cutoff), "auxiliary_rows": len(aux), "epochs": self.spec.epochs,
                "checkpoint_policy": "fixed_final_epoch", "loss_before_last_update": float(loss.detach()),
                "allocation_loss_before_last_update": float(allocation.detach()),
                "observed_minute_training_rows": {flag: int(features[flag].eq(1).sum()) for flag in FLAGS},
                "extra_scaler": self.extra_scaler, "feature_count": len(self.base_features) + len(self.extra_features),
                "native_dataset_class": type(dataset).__name__, "auxiliary_dataset_class": type(auxiliary_dataset).__name__,
                "supervision_target": self.target_name, "target_scaler": deepcopy(self.target_scaler),
                "original_loss_before_last_update": float(original_loss.detach()),
                "supervision_half_mse_before_last_update": float(half_mse.detach()),
                "completed_step_count": self.completed_step_count}
        finally:
            # These are the actual new optimizer's state and completed steps,
            # including failures. Never reconstruct or inherit old AdamW state.
            self.optimizer_state_dict = deepcopy(optimizer.state_dict())
            optimizer.zero_grad(set_to_none=True)
            primary_state = prediction = logits = tail_logits = alpha_logits = None
            state = aux_tail_logits = aux_alpha_logits = weights = None
            allocation = turnover = annual = primary_loss = auxiliary_loss = original_loss = loss = half_mse = None
            _ = None  # allocation component dictionary may contain graph tensors
        return self

    def state_metadata(self):
        result = super().state_metadata()
        result.update(target_name=self.target_name, target_scaler=deepcopy(self.target_scaler),
                      completed_step_count=self.completed_step_count)
        return result


def initialization_contract(original, B, C):
    """Toy-only initial tensors/RNG/original outputs; never real input data."""
    if B.target_name != "S" or C.target_name != "Z":
        raise ValueError("expected paired B=S,C=Z")
    networks, rngs = [], []
    for model in (original, B, C):
        networks.append(model._initialize())
        rngs.append(torch.get_rng_state().clone())
    base, left, right = networks
    if not all(torch.equal(rngs[0], state) for state in rngs[1:]):
        raise ValueError("supervision head changed original RNG")
    if [sum(p.numel() for p in n.parameters()) for n in networks] != [973, 982, 982]:
        raise ValueError("original/paired parameter budget differs")
    if list(left.state_dict()) != list(base.state_dict()) + ["supervision_head.weight", "supervision_head.bias"]:
        raise ValueError("original parameter registration order changed")
    expected_names = [name for name, _ in base.named_parameters()] + ["supervision_head.weight", "supervision_head.bias"]
    if any([name for name, _ in net.named_parameters()] != expected_names for net in (left, right)):
        raise ValueError("new parameters must follow all original parameters")
    for name, tensor in base.state_dict().items():
        if not torch.equal(tensor, left.state_dict()[name]) or not torch.equal(tensor, right.state_dict()[name]):
            raise ValueError("original initial tensor changed")
    for name, tensor in left.state_dict().items():
        if not torch.equal(tensor, right.state_dict()[name]):
            raise ValueError("paired initial tensors differ")
    if any(torch.count_nonzero(p) for p in left.supervision_head.parameters()):
        raise ValueError("new head is not zero")
    primary = torch.linspace(-1., 1., 3 * 39).reshape(3, 39)
    common = torch.linspace(-.5, .5, 3 * 16).reshape(3, 16)
    with torch.no_grad():
        expected = base(primary, common)
        for network in (left, right):
            if not all(torch.equal(a, b) for a, b in zip(expected, network(primary, common))):
                raise ValueError("original initial output differs")
    return {"original_parameters": 973, "paired_parameters": 982, "original_tensors_identical": True,
            "paired_tensors_identical": True, "original_outputs_identical": True, "rng_identical": True,
            "new_head_zero": True, "original_parameter_order_preserved": True,
            "input_scope": "fixed synthetic39/16 only"}
