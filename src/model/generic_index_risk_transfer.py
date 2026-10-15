"""Index-only equal-family tail supervision; original primary allocation objective."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.model.tangent_allocator import TangentHeadAllocator
from src.model.minute_pit_allocator import _flat_index,_scaled,TARGETS,FLAGS
from src.workflow.pit_constituent_auxiliary_training_allocation import COMMON_FEATURES,_standardize_pair,_project_legal_weights

def maturity_known_before(series,cutoff):
    stamps=pd.to_datetime(series)
    if stamps.dt.tz is None:stamps=stamps.dt.tz_localize("America/New_York")
    return stamps.notna() & stamps.lt(pd.Timestamp(cutoff))

def external_tail_loss(network,groups):
    if set(groups)!={'QQQ','SPY','XLK'}:raise ValueError('fixed three index families required')
    return torch.stack([nn.functional.binary_cross_entropy_with_logits(network.tail_head(network.tail_encoder(x)).squeeze(1),y) for x,y in groups.values()]).mean()

class IndexTailAllocator(TangentHeadAllocator):
    def __init__(self,*,external_tail_enabled,**kwargs):
        super().__init__(**kwargs);self.external_tail_enabled=bool(external_tail_enabled)
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

        aux = auxiliary_dataset
        auxiliary_groups={}
        for index,group in aux.items():
            if not maturity_known_before(group.mature_at,self.cutoff).all():
                raise ValueError('immature external label')
            x=_scaled(group,list(COMMON_FEATURES),self.common_scaler)
            y=torch.tensor(group.tail_label.to_numpy(np.float32))
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('nonfinite external risk input')
            auxiliary_groups[index]=(x,y)
        returns = torch.tensor(labels.loc[:, list(TARGETS[:2])].to_numpy(np.float32))
        tail = torch.tensor(labels.tail_event_5d.to_numpy(np.float32))
        alpha = torch.tensor(labels.relative_alpha_5d.gt(0).to_numpy(np.float32))
        cash = torch.tensor(labels.tail_cash_target_5d.to_numpy(np.float32))
        years = torch.tensor(train.index.year.to_numpy(), dtype=torch.int64)
        self.network = self._initialize()
        self.prepare_epoch_observation(dataset)
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
            if self.external_tail_enabled:
                auxiliary_loss=external_tail_loss(self.network,auxiliary_groups)
            loss = allocation + self.spec.primary_tail_loss_weight * primary_loss + self.candidate.auxiliary_weight * auxiliary_loss
            if not torch.isfinite(loss):
                raise ValueError("blocked: allocation training loss is nonfinite")
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=2.)
            optimizer.step()
            self.observe_epoch(epoch + 1)
        self.network.eval()
        self.audit = {"training_rows": len(train), "first_training_signal": str(train.index.min().date()),
            "last_training_signal": str(train.index.max().date()), "last_mature_label_at": str(maturity.loc[eligible].max()),
            "cutoff": str(self.cutoff), "auxiliary_rows": sum(len(g) for g in aux.values()) if self.external_tail_enabled else 0, "epochs": self.spec.epochs,
            "checkpoint_policy": "fixed_final_epoch", "loss_before_last_update": float(loss.detach()),
            "allocation_loss_before_last_update": float(allocation.detach()),
            "observed_minute_training_rows": {flag: int(features[flag].eq(1).sum()) for flag in FLAGS},
            "extra_scaler": self.extra_scaler, "feature_count": len(self.base_features) + len(self.extra_features),
            "native_dataset_class": type(dataset).__name__, "auxiliary_dataset_class": type(auxiliary_dataset).__name__}
        return self

