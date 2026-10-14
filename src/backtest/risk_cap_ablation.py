"""Diagnostic removal of dynamic risk caps; never a production risk policy."""
import numpy as np
from src.workflow.stage1_open_action_value import build_open_action_features,decide_open_execution
from src.workflow.intraday_recovery_direct_allocation import project_legal_weights


class LegalCapOpenPolicy:
    def __init__(self,*,open_model,action_spec,maximum_soxl_weight,minimum_cash_weight):
        self.open_model=open_model;self.action_spec=action_spec
        self.maximum_soxl_weight=maximum_soxl_weight;self.minimum_cash_weight=minimum_cash_weight
        self.cap=1-minimum_cash_weight+2*maximum_soxl_weight
        self.previous_known=None;self.history=[];self.audit=[]

    def target(self,row,quantities,cash,known,peak):
        nav=float(np.asarray(quantities)@np.asarray(known)+cash)
        if self.previous_known is not None:self.history.append(nav/self.previous_known-1)
        self.previous_known=nav
        pretrade=np.r_[np.asarray(quantities)*np.asarray(known),cash]/nav
        drawdown=max(0.,1-nav/peak)
        target=project_legal_weights(float(row.soxl_weight),float(row.soxx_weight),self.cap,minimum_cash_weight=self.minimum_cash_weight,maximum_soxl_weight=self.maximum_soxl_weight)
        features=build_open_action_features(row,target=target,pretrade=pretrade,drawdown=drawdown,cap=self.cap,state=0,history=self.history)
        decision=decide_open_execution(features,self.open_model,execute_probability_threshold=self.action_spec.execute_probability_threshold,maximum_skippable_turnover=self.action_spec.maximum_skippable_turnover,mandatory_derisk_beta_delta=self.action_spec.mandatory_derisk_beta_delta)
        legal=bool(np.isfinite(pretrade).all() and pretrade.min()>=-1e-12 and pretrade[0]<=self.maximum_soxl_weight+1e-12 and pretrade[2]>=self.minimum_cash_weight-1e-12 and 3*pretrade[0]+pretrade[1]<=self.cap+1e-12)
        execute=bool(decision['execute_open_target'] or not legal)
        self.audit.append({'trade_date':row.trade_date,'known_drawdown':drawdown,'legal_beta_cap':self.cap,'execute':execute,'mandatory_position_repair':not legal,'execute_probability':decision['execute_probability']})
        return target if execute else None
