"""Original Stage1 allocator with one deterministic auxiliary-gradient intervention.

Control B remains MinutePitAllocator. This module does not authorize training.
The candidate fit body copies the original preparation, forward, loss, sampling,
optimizer and prediction contract; only its backward/step block is replaced.
Future workflows must freeze approvals and set Dataset.config(dump_all=True,
recursive=True) before saving; this model never reads research cards or sources.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.model.minute_pit_allocator import MinutePitAllocator, TARGETS, FLAGS, _flat_index, _scaled
from src.workflow.pit_constituent_auxiliary_training_allocation import (
    COMMON_FEATURES, _project_legal_weights, _standardize_pair,
)
from src.workflow.aux_projection_preflight import (
    SHARED, PRIVATE, _flat, _norm, _dot_screen, vector_audit,
)

ORIGINAL_FIT_SOURCE_SHA256 = "0e499b0d2abc7782623370badb66e2fe12a8278cac8e4127456c9ab4342153e2"
PREVIOUS_MECHANICAL_MODULE_SHA256 = "d56464ba1dfa9d98f308dad39812b24c5dcf140a2056a66f79b111c9e3417868"


class ProjectionNumericalError(ValueError):
    """A fixed arithmetic gate failed; callers retain the audit without retry."""
    def __init__(self, audit):
        super().__init__("auxiliary projection full/shared parity or private-gradient contract failed")
        self.audit = audit


def assemble_projected_gradient(gp, ga, combined, *, shared_count=154):
    """Pure saved-array operation; combined backward is the control authority.

    No-conflict and zero-primary branches return the exact combined tensor, not
    a re-summed copy. On strict conflict, private coordinates still come from
    combined. The reviewed engineering bounds only audit, never gate projection.
    """
    if gp.ndim != 1 or gp.shape != ga.shape or gp.shape != combined.shape:
        raise ValueError("gradient vectors must share one flat shape")
    if not 0 < shared_count <= gp.numel():
        raise ValueError("invalid shared parameter boundary")
    audit, _, old_candidate = vector_audit(gp, ga, combined, shared_count=shared_count)
    private_aux_zero = bool(torch.count_nonzero(ga[shared_count:]) == 0)
    audit['private_auxiliary_gradient_zero'] = private_aux_zero
    if not (audit['full_parity_pass'] and audit['shared_parity_pass'] and private_aux_zero):
        raise ProjectionNumericalError(audit)
    if audit['strict_projection_triggered']:
        result = torch.cat([old_candidate[:shared_count], combined[shared_count:].clone()])
    else:
        result = combined
    audit['private_original_combined_unchanged'] = torch.equal(result[shared_count:], combined[shared_count:])
    audit['no_conflict_original_combined_identity'] = (
        None if audit['strict_projection_triggered'] else result is combined and torch.equal(result, combined))
    return result, audit


def _parameter_vector(parameters):
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


def clip_once(parameters):
    """The original one global norm clip, exposed for a synthetic contract."""
    return nn.utils.clip_grad_norm_(parameters, max_norm=2.)


class AuxProjectionAllocator(MinutePitAllocator):
    """Same 973 parameters and outputs; conditional projection only on shared154."""

    def _validate_projection_contract(self, primary_rows, auxiliary_rows):
        named = dict(self.network.named_parameters())
        if tuple(named) != SHARED + PRIVATE:
            raise ValueError("original parameter names/order required")
        if sum(named[n].numel() for n in SHARED) != 154 or sum(named[n].numel() for n in PRIVATE) != 819:
            raise ValueError("original154 shared and819 private parameters required")
        if any(p.dtype != torch.float32 or p.device.type != 'cpu' for p in named.values()):
            raise ValueError("original CPU float32 computation required")
        if (primary_rows, auxiliary_rows) != (3014, 20741):
            raise ValueError("approved primary/auxiliary training cohort required")
        if self.extra_features or len(self.base_features) != 39 or self.seed not in (71, 172, 273):
            raise ValueError("original39 inputs and fixed seed required")
        if (self.spec.epochs != 300 or self.spec.auxiliary_batch_size != 2048
                or self.spec.primary_tail_loss_weight != .02 or self.candidate.auxiliary_weight != .05
                or self.spec.training_cost_bps != 10 or self.spec.turnover_loss_weight != .006):
            raise ValueError("fixed original loss/epoch/sampling contract required")

    def _projected_step(self, allocation, primary_loss, auxiliary_loss, loss, optimizer, epoch, sampled):
        named = dict(self.network.named_parameters())
        parameters = [named[name] for name in SHARED + PRIVATE]
        primary = allocation + self.spec.primary_tail_loss_weight * primary_loss
        auxiliary = self.candidate.auxiliary_weight * auxiliary_loss
        pg = torch.autograd.grad(primary, parameters, retain_graph=True, allow_unused=True)
        ag = torch.autograd.grad(auxiliary, parameters, retain_graph=True, allow_unused=True)
        lg = torch.autograd.grad(allocation, parameters, retain_graph=True, allow_unused=True)
        if any(g is None for g in pg[:6] + ag[:6]):
            raise ValueError("shared task gradient unexpectedly missing")
        gp, ga, gl = [_flat(grads, parameters) for grads in (pg, ag, lg)]
        # This is the original combined backward, not a sum of separate gradients.
        loss.backward()
        combined = _flat([p.grad for p in parameters], parameters)
        original_grad_ids = [id(p.grad) for p in parameters]
        try:
            selected, audit = assemble_projected_gradient(gp, ga, combined)
        except ProjectionNumericalError as error:
            self.training_trace.append({'epoch': epoch + 1, 'status': 'failed_before_step', **error.audit})
            self.gradient_trace.append({'epoch': epoch + 1, 'gP': gp.numpy(), 'gA': ga.numpy(),
                'g_allocation': gl.numpy(), 'combined_backward': combined.numpy()})
            raise
        if audit['strict_projection_triggered']:
            offset = 0
            for parameter in parameters[:6]:
                parameter.grad = selected[offset:offset + parameter.numel()].reshape_as(parameter).clone()
                offset += parameter.numel()
        # No-conflict executes no gradient assignment. Private tensors are never assigned.
        private_original = all(id(p.grad) == old_id for p, old_id in zip(parameters[6:], original_grad_ids[6:]))
        if not private_original or not torch.equal(_flat([p.grad for p in parameters], parameters)[154:], combined[154:]):
            raise ValueError("private original combined gradients were changed")
        if not audit['strict_projection_triggered'] and [id(p.grad) for p in parameters] != original_grad_ids:
            raise ValueError("no-conflict path replaced original gradient tensors")
        before = _parameter_vector(parameters).double()
        # Exactly the original single global clip followed by the original AdamW step.
        total_norm = clip_once(self.network.parameters())
        clipped = _flat([p.grad for p in parameters], parameters)
        audit.update(epoch=epoch + 1, status='completed',
            allocation_loss=float(allocation.detach()), primary_loss=float(primary.detach()),
            auxiliary_loss=float(auxiliary.detach()), total_loss=float(loss.detach()),
            actual_global_preclip_norm=float(total_norm),
            actual_global_clip_coefficient=float(torch.clamp(2. / (total_norm + 1e-6), max=1.)),
            original_combined_preclip_norm_audit64=_norm(combined),
            sampled_indices_sha256=hashlib.sha256(sampled.numpy().tobytes()).hexdigest())
        for objective, reference in [('P', gp), ('allocation', gl)]:
            for scope, sl in [('full', slice(None)), ('shared', slice(0, 154))]:
                for stage, gradient in [('original_preclip', combined), ('candidate_preclip', selected), ('candidate_postclip', clipped)]:
                    value, tolerance = _dot_screen(reference[sl], gradient[sl], audit['r'])
                    audit[f'{objective}_{scope}_{stage}_dot'] = value
                    audit[f'{objective}_{scope}_{stage}_tau_dot'] = tolerance
        optimizer.step()
        delta = _parameter_vector(parameters).double() - before
        for objective, reference in [('P', gp), ('allocation', gl)]:
            for scope, sl in [('full', slice(None)), ('shared', slice(0, 154))]:
                audit[f'{objective}_{scope}_actual_delta_linear_loss_change'] = float(torch.dot(reference[sl].double(), delta[sl]))
        audit['actual_delta_sign_convention'] = 'negative is a first-order loss-decrease estimate, not a finite-step guarantee'
        self.training_trace.append(audit)
        self.gradient_trace.append({'epoch': epoch + 1, 'gP': gp.numpy(), 'gA': ga.numpy(),
            'g_allocation': gl.numpy(), 'combined_backward': combined.numpy(),
            'candidate_preclip': selected.numpy(), 'candidate_postclip': clipped.numpy(),
            'actual_parameter_delta': delta.numpy(), 'sampled_indices': sampled.numpy()})

    def fit(self, dataset, *, auxiliary_dataset, **kwargs):
        source = _flat_index(dataset.prepare("train", col_set=["feature", "label", "common", "meta"]))
        maturity = pd.to_datetime(source[("meta", "label_available_at")])
        eligible = maturity.notna() & maturity.lt(self.cutoff) & (source.index <= self.train_end)
        eligible &= np.isfinite(source["label"].loc[:, list(TARGETS)].to_numpy(float)).all(axis=1)
        train = source.loc[eligible].copy()
        if len(train) < 100:
            raise ValueError("needs_data: insufficient matured allocation training rows")
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
        self.network = self._initialize()
        self.prepare_epoch_observation(dataset)
        self._validate_projection_contract(len(train), len(aux))
        self.training_trace = []
        self.gradient_trace = []
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.spec.learning_rate,
                                     weight_decay=self.spec.weight_decay)
        for epoch in range(self.spec.epochs):
            optimizer.zero_grad()
            logits, tail_logits, alpha_logits = self.network(primary_x, common_x)
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
            if len(auxiliary_x):
                generator = torch.Generator().manual_seed(self.seed * 1000 + epoch)
                sampled = torch.randint(len(auxiliary_x), (min(self.spec.auxiliary_batch_size, len(auxiliary_x)),),
                                        generator=generator)
                state = self.network.tail_encoder(auxiliary_x[sampled])
                aux_tail_logits, aux_alpha_logits = self.auxiliary_logits(state)
                auxiliary_loss = ((nn.functional.binary_cross_entropy_with_logits(aux_tail_logits,
                    aux_tail[sampled], reduction="none") + nn.functional.binary_cross_entropy_with_logits(
                    aux_alpha_logits, aux_alpha[sampled], reduction="none")) * aux_weight[sampled]).mean()
            loss = allocation + self.spec.primary_tail_loss_weight * primary_loss + self.candidate.auxiliary_weight * auxiliary_loss
            if not torch.isfinite(loss):
                raise ValueError("blocked: allocation training loss is nonfinite")
            self._projected_step(allocation, primary_loss, auxiliary_loss, loss,
                                 optimizer, epoch, sampled)
            self.observe_epoch(epoch + 1)
        self.network.eval()
        self.audit = {"training_rows": len(train), "first_training_signal": str(train.index.min().date()),
            "last_training_signal": str(train.index.max().date()), "last_mature_label_at": str(maturity.loc[eligible].max()),
            "cutoff": str(self.cutoff), "auxiliary_rows": len(aux), "epochs": self.spec.epochs,
            "checkpoint_policy": "fixed_final_epoch", "loss_before_last_update": float(loss.detach()),
            "allocation_loss_before_last_update": float(allocation.detach()),
            "observed_minute_training_rows": {flag: int(features[flag].eq(1).sum()) for flag in FLAGS},
            "extra_scaler": self.extra_scaler, "feature_count": len(self.base_features) + len(self.extra_features),
            "native_dataset_class": type(dataset).__name__, "auxiliary_dataset_class": type(auxiliary_dataset).__name__}
        self.optimizer_state_dict = deepcopy(optimizer.state_dict())
        self.audit.update(gradient_method="primary_anchored_auxiliary_projection",
            projected_steps=sum(row["strict_projection_triggered"] for row in self.training_trace),
            completed_optimizer_steps=len(self.training_trace),
            original_fit_source_sha256=ORIGINAL_FIT_SOURCE_SHA256,
            saved_source_optimizer_reconstructed=False)
        return self
