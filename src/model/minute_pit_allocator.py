"""Qlib model extending the current PIT auxiliary allocation network's inputs."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from qlib.model.base import Model
from torch import nn

from src.workflow.pit_constituent_auxiliary_training_allocation import (
    COMMON_FEATURES, AuxiliaryTrainingCandidate, PitAuxiliaryAllocationNet,
    PitAuxiliaryAllocationSpec, _allocation_loss, _project_legal_weights,
    _standardize_pair, _transform_with_scaler,
)


TARGETS = ("soxl_forward_open_return", "soxx_forward_open_return", "tail_event_5d",
           "tail_cash_target_5d", "relative_alpha_5d")
FLAGS = ("minute_core_ready", "minute_peer_ready")


def _flat_index(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if isinstance(result.index, pd.MultiIndex):
        result.index = result.index.get_level_values("datetime")
    return result


def _scaled(frame: pd.DataFrame, columns: list[str], scaler: dict) -> torch.Tensor:
    converted = {key: pd.Series(value) for key, value in scaler.items()}
    return _transform_with_scaler(frame, tuple(columns), converted)


class MinutePitAllocator(Model):
    """Keep the existing network, legal simplex and auxiliary objective.

    The added input weights start at zero, with the original layer/encoder
    initialization shared across paired candidates. Unknown minute channels
    are zero in latent standardized space, while canonical observations stay
    NaN. No minute observation is synthesized before source coverage starts.
    """

    def __init__(self, *, base_features: list[str], extra_features: list[str],
                 feature_masks: dict[str, str], spec: dict, seed: int,
                 cutoff: str, train_end: str, minimum_observations: int):
        self.base_features = list(base_features)
        self.extra_features = list(extra_features)
        self.feature_masks = dict(feature_masks)
        self.spec = PitAuxiliaryAllocationSpec(root=Path(spec.get("root", ".")),
            **{key: value for key, value in spec.items() if key != "root"})
        self.seed, self.cutoff = int(seed), pd.Timestamp(cutoff)
        self.train_end, self.minimum_observations = pd.Timestamp(train_end), int(minimum_observations)
        self.candidate = AuxiliaryTrainingCandidate(name="pit_only_aux_005", auxiliary_weight=.05,
                                                    include_union_backfill=False)

    def _fit_extra_scaler(self, features: pd.DataFrame) -> None:
        self.extra_scaler = {}
        for name in self.extra_features:
            if name in FLAGS:
                continue
            mask = features[self.feature_masks[name]].eq(1) & np.isfinite(features[name])
            observed = features.loc[mask, name].astype(float)
            active = len(observed) >= self.minimum_observations
            self.extra_scaler[name] = {"mean": float(observed.mean()) if active else 0.,
                "scale": float(observed.std(ddof=1)) if active and observed.std(ddof=1) > 0 else 1.,
                "observations": len(observed), "active": active}

    def _extra_tensor(self, features: pd.DataFrame) -> torch.Tensor:
        values = np.zeros((len(features), len(self.extra_features)), dtype=np.float32)
        for number, name in enumerate(self.extra_features):
            if name in FLAGS:
                flag = features[name].fillna(0.)
                if not flag.isin([0., 1.]).all():
                    raise ValueError("blocked: minute availability flag is not binary")
                values[:, number] = flag.to_numpy(np.float32)
                continue
            scaler = self.extra_scaler[name]
            if not scaler["active"]:
                continue
            known = features[self.feature_masks[name]].eq(1) & np.isfinite(features[name])
            values[known.to_numpy(), number] = ((features.loc[known, name] - scaler["mean"])
                                               / scaler["scale"]).clip(-8., 8.).to_numpy(np.float32)
        return torch.tensor(values)

    def _initialize(self):
        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        original = PitAuxiliaryAllocationNet(primary_feature_count=len(self.base_features),
            common_feature_count=len(COMMON_FEATURES), hidden_units=self.spec.hidden_units,
            tail_hidden_units=self.spec.tail_hidden_units)
        if not self.extra_features:
            return original
        extended = PitAuxiliaryAllocationNet(primary_feature_count=len(self.base_features) + len(self.extra_features),
            common_feature_count=len(COMMON_FEATURES), hidden_units=self.spec.hidden_units,
            tail_hidden_units=self.spec.tail_hidden_units)
        with torch.no_grad():
            for name, tensor in extended.state_dict().items():
                if name != "allocation_head.0.weight":
                    tensor.copy_(original.state_dict()[name])
            weights = extended.allocation_head[0].weight
            weights.zero_()
            base = len(self.base_features)
            extra = len(self.extra_features)
            weights[:, :base].copy_(original.allocation_head[0].weight[:, :base])
            weights[:, base + extra:].copy_(original.allocation_head[0].weight[:, base:])
        return extended

    def allocation_objective(self, **kwargs):
        return _allocation_loss(**kwargs)

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
                aux_tail_logits = self.network.tail_head(state).squeeze(1)
                aux_alpha_logits = self.network.alpha_head(state).squeeze(1)
                auxiliary_loss = ((nn.functional.binary_cross_entropy_with_logits(aux_tail_logits,
                    aux_tail[sampled], reduction="none") + nn.functional.binary_cross_entropy_with_logits(
                    aux_alpha_logits, aux_alpha[sampled], reduction="none")) * aux_weight[sampled]).mean()
            loss = allocation + self.spec.primary_tail_loss_weight * primary_loss + self.candidate.auxiliary_weight * auxiliary_loss
            if not torch.isfinite(loss):
                raise ValueError("blocked: allocation training loss is nonfinite")
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=2.)
            optimizer.step()
        self.network.eval()
        self.audit = {"training_rows": len(train), "first_training_signal": str(train.index.min().date()),
            "last_training_signal": str(train.index.max().date()), "last_mature_label_at": str(maturity.loc[eligible].max()),
            "cutoff": str(self.cutoff), "auxiliary_rows": len(aux), "epochs": self.spec.epochs,
            "checkpoint_policy": "fixed_final_epoch", "loss_before_last_update": float(loss.detach()),
            "allocation_loss_before_last_update": float(allocation.detach()),
            "observed_minute_training_rows": {flag: int(features[flag].eq(1).sum()) for flag in FLAGS},
            "extra_scaler": self.extra_scaler, "feature_count": len(self.base_features) + len(self.extra_features),
            "native_dataset_class": type(dataset).__name__, "auxiliary_dataset_class": type(auxiliary_dataset).__name__}
        return self

    def predict(self, dataset, segment="predict", **kwargs):
        prepared = dataset.prepare(segment, col_set=["feature", "common"])
        primary_x = _scaled(prepared["feature"], self.base_features, self.primary_scaler)
        primary_x = torch.cat([primary_x, self._extra_tensor(prepared["feature"])], dim=1)
        common_x = _scaled(prepared["common"], list(COMMON_FEATURES), self.common_scaler)
        if not torch.isfinite(primary_x).all() or not torch.isfinite(common_x).all():
            raise ValueError("needs_data: invalid inference input")
        with torch.no_grad():
            logits, tail, alpha = self.network(primary_x, common_x)
            weights = _project_legal_weights(logits, maximum_soxl_weight=self.spec.maximum_soxl_weight,
                                            minimum_cash_weight=self.spec.minimum_cash_weight).numpy().astype(float)
        weights /= weights.sum(axis=1, keepdims=True)
        result = pd.DataFrame(weights, index=prepared.index, columns=["soxl_weight", "soxx_weight", "cash_weight"])
        result["tail_probability"] = torch.sigmoid(tail).numpy()
        result["alpha_probability"] = torch.sigmoid(alpha).numpy()
        return result

    def state_metadata(self) -> dict:
        return {"spec": {**asdict(self.spec), "root": str(self.spec.root)}, "seed": self.seed,
                "base_features": self.base_features, "extra_features": self.extra_features,
                "feature_masks": self.feature_masks, "audit": self.audit}
