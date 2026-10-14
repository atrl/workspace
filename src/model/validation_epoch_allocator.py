"""Observe a fixed validation segment without changing the training trajectory."""
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
from src.model.minute_pit_allocator import MinutePitAllocator, TARGETS, _flat_index, _scaled
from src.workflow.pit_constituent_auxiliary_training_allocation import COMMON_FEATURES, _project_legal_weights


class ValidationEpochAllocator(MinutePitAllocator):
    def prepare_epoch_observation(self, dataset):
        source=_flat_index(dataset.prepare('valid',col_set=['feature','common','label','meta']))
        if source.index.min() <= self.train_end:
            raise ValueError('validation overlaps training')
        maturity=pd.to_datetime(source[('meta','label_available_at')])
        valid=maturity.notna() & maturity.lt(pd.Timestamp(self.validation_cutoff))
        valid &= np.isfinite(source['label'][list(TARGETS)]).all(axis=1)
        source=source.loc[valid]
        if len(source)<100:raise ValueError('insufficient matured validation rows')
        self.validation_audit={'rows':len(source),'first_signal':str(source.index.min().date()),'last_signal':str(source.index.max().date()),'last_mature_label_at':str(maturity.loc[valid].max()),'cutoff':self.validation_cutoff}
        primary=_scaled(source['feature'],self.base_features,self.primary_scaler)
        self.validation_primary=torch.cat([primary,self._extra_tensor(source['feature'])],dim=1)
        self.validation_common=_scaled(source['common'],list(COMMON_FEATURES),self.common_scaler)
        self.validation_returns=torch.tensor(source['label'][list(TARGETS[:2])].to_numpy(np.float32))
        self.validation_tail=torch.tensor(source[('label','tail_event_5d')].to_numpy(np.float32))
        self.validation_alpha=torch.tensor(source[('label','relative_alpha_5d')].gt(0).to_numpy(np.float32))
        self.validation_cash=torch.tensor(source[('label','tail_cash_target_5d')].to_numpy(np.float32))
        self.validation_years=torch.tensor(source.index.year.to_numpy(),dtype=torch.int64)
        if not torch.isfinite(self.validation_primary).all() or not torch.isfinite(self.validation_common).all():raise ValueError('invalid validation inputs')
        self.validation_history=[];self.best_validation_loss=float('inf');self.selected_epoch=None

    def observe_epoch(self, epoch):
        with torch.no_grad():
            logits,tail,alpha=self.network(self.validation_primary,self.validation_common)
            weights=_project_legal_weights(logits,maximum_soxl_weight=self.spec.maximum_soxl_weight,minimum_cash_weight=self.spec.minimum_cash_weight)
            allocation,_,_,_=self.allocation_objective(weights=weights,returns=self.validation_returns,years=self.validation_years,cash_target=self.validation_cash,tail_event=self.validation_tail,candidate=self.candidate,drawdown_target_pct=-30.,worst_5d_target_pct=self.spec.research_worst_5d_target_pct,minimum_cash_weight=self.spec.minimum_cash_weight,cost_bps=self.spec.training_cost_bps,turnover_weight=self.spec.turnover_loss_weight)
            classification=torch.nn.functional.binary_cross_entropy_with_logits(tail,self.validation_tail)+torch.nn.functional.binary_cross_entropy_with_logits(alpha,self.validation_alpha)
            value=float(allocation+self.spec.primary_tail_loss_weight*classification)
        if not np.isfinite(value):raise ValueError('nonfinite validation loss')
        self.validation_history.append({'epoch':epoch,'loss':value})
        # Strict improvement retains earliest epoch on exact ties.
        if value<self.best_validation_loss:
            self.best_validation_loss=value;self.selected_epoch=epoch;self.selected_state=deepcopy(self.network.state_dict())
