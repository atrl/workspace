"""Causal common decisions from distinct fixed-cost hypothetical ledgers."""
import numpy as np
import pandas as pd
from src.backtest.ibkr_tiered import TieredFees,settle_tiered_event
from src.workflow.stage1_open_action_value import prepare_on_policy_open,prepare_on_policy_recovery
from src.workflow.intraday_recovery_direct_allocation import project_legal_weights
from src.model.nominal_beta_composition import cancel_leverage_cash


def replay_cost_scenarios(frame,*,plan,controller,recovery_spec,action_spec,selection,scenarios=None):
    costs=list(plan['internal_cost_scenarios'] if scenarios is None else scenarios)
    if not costs or len(set(costs))!=len(costs):raise ValueError('distinct nonempty fixed scenarios required')
    ledgers={c:dict(q=np.zeros(2),cash=float(plan['initial_cash']),peak=float(plan['initial_cash']),previous_known=None,
        history=[],state=0,clock=0,fees=TieredFees(plan['tariff'],slippage_bps=c),records=[],orders=[],targets=[],prior_fees=0.) for c in costs}
    trace=[]
    for _,row in frame.iterrows():
        known=row[['known_soxl','known_soxx']].to_numpy(float);executable=row[['soxl_open','soxx_open']].to_numpy(float)
        snapshots={}
        # Finish every pre-fill snapshot and decision before settling any scenario.
        for cost in costs:
            ledger=ledgers[cost];nav=float(ledger['q']@known+ledger['cash']);ledger['peak']=max(ledger['peak'],nav)
            if ledger['previous_known'] is not None:ledger['history'].append(nav/ledger['previous_known']-1)
            ledger['previous_known']=nav
            opened=prepare_on_policy_open(row,pretrade_weights=np.r_[ledger['q']*known,ledger['cash']]/nav,
                equity=nav,peak=ledger['peak'],risk_state=ledger['state'],risk_clock=ledger['clock'],history=ledger['history'],
                recovery_spec=recovery_spec,model=None,action_spec=action_spec)
            ledger['state'],ledger['clock']=opened['risk_state'],opened['risk_clock']
            snapshots[cost]={'opened':opened,'known_nav':nav,'prior_fees':ledger['prior_fees'],
                'pre_fill_nav':float(ledger['q']@executable+ledger['cash']), 'local_cap':float(opened['current_beta_cap'])}
        common_cap=min(x['local_cap'] for x in snapshots.values())
        before=project_legal_weights(float(row.soxl_weight),float(row.soxx_weight),common_cap,
            minimum_cash_weight=recovery_spec.minimum_cash_weight,maximum_soxl_weight=recovery_spec.maximum_soxl_weight)
        common=cancel_leverage_cash(before)
        sessions={}
        for cost in costs:
            ledger=ledgers[cost];snap=snapshots[cost];opened=snap['opened']
            # Recovery receives the original pre-fill state, with only actual shared cap/target replaced.
            snap['recovery_open_state']={**opened,'current_beta_cap':common_cap,'desired_open_weights':before.copy(),'open_weights':common.copy()}
            ledger['targets'].append({'signal_date':row.signal_date,'trade_date':row.trade_date,'decision_at':row.trade_date+pd.Timedelta(hours=9,minutes=46),
                'known_equity':snap['known_nav'],'known_drawdown':opened['drawdown'],**dict(zip(['soxl_weight','soxx_weight','cash_weight'],common))})
            fill=settle_tiered_event(ledger['q'],ledger['cash'],desired=common,adjusted_prices=executable,
                raw_prices=row[['raw_soxl_open','raw_soxx_open']],stamp=row.trade_date+pd.Timedelta(hours=9,minutes=47),fees=ledger['fees'])
            ledger['q'],ledger['cash']=fill['quantities'],fill['cash'];sessions[cost]=[{**order,'slot':'open','trade_date':row.trade_date} for order in fill['orders']]
            after_nav=float(ledger['q']@executable+ledger['cash']);snap['actual_after_open_beta']=float((3*ledger['q'][0]*executable[0]+ledger['q'][1]*executable[1])/after_nav)
        # All votes precede every recovery fill. Classifier input is pre-fill snapshot + common cap.
        votes={c:prepare_on_policy_recovery(row,open_state=snapshots[c]['recovery_open_state'],recovery_spec=recovery_spec,
            model=controller['frozen_recovery_model'],selected_model=selection) for c in costs}
        recover=all(bool(v['recovery_action']) for v in votes.values());recovery_before=None;recovery_target=None
        if recover:
            recovery_before=votes[costs[0]]['target_weights']
            for vote in votes.values():np.testing.assert_allclose(vote['target_weights'],recovery_before,rtol=0,atol=1e-12)
            recovery_target=cancel_leverage_cash(recovery_before)
        for cost in costs:
            ledger=ledgers[cost];snap=snapshots[cost];opened=snap['opened'];vote=votes[cost]
            if recover:
                fill=settle_tiered_event(ledger['q'],ledger['cash'],desired=recovery_target,
                    adjusted_prices=row[['soxl_recovery','soxx_recovery']],raw_prices=row[['raw_soxl_recovery','raw_soxx_recovery']],
                    stamp=row.trade_date+pd.Timedelta(hours=10,minutes=32),fees=ledger['fees'])
                ledger['q'],ledger['cash']=fill['quantities'],fill['cash'];sessions[cost]+=[{**order,'slot':'recovery','trade_date':row.trade_date} for order in fill['orders']]
            closing=row[['close_soxl','close_soxx']].to_numpy(float);ending=row[['soxl_exit_open','soxx_exit_open']].to_numpy(float)
            close_nav=float(ledger['q']@closing+ledger['cash']);ledger['peak']=max(ledger['peak'],close_nav)
            monitored=max(opened['drawdown'],max(0.,1-close_nav/ledger['peak']))
            ending_nav=float(ledger['q']@ending+ledger['cash']);weights=np.r_[ledger['q']*ending,ledger['cash']]/ending_nav
            session=sessions[cost];fees=sum(x['total'] for x in session);slippage=sum(x['slippage'] for x in session)
            ledger['orders']+=session;ledger['prior_fees']+=fees
            ledger['records'].append({'signal_date':row.signal_date,'trade_date':row.trade_date,'exit_date':row.exit_date,
                'strategy_return':ending_nav/snap['pre_fill_nav']-1,'equity':ending_nav/plan['initial_cash'],'native_expected_nav':ending_nav,
                'turnover':sum(x['notional'] for x in session)/snap['pre_fill_nav'],'fees':fees/snap['pre_fill_nav'],
                'broker_and_external_fees_usd':fees-slippage,'slippage_reserve_usd':slippage,
                'portfolio_drawdown':opened['drawdown'],'monitored_drawdown':monitored,'risk_state':ledger['state'],'risk_clock':ledger['clock'],
                'current_beta_cap':common_cap,'execute_open_target':True,'recovery_action':recover,
                'ending_soxl_quantity':ledger['q'][0],'ending_soxx_quantity':ledger['q'][1],'ending_cash':ledger['cash'],
                **dict(zip(['ending_soxl_weight','ending_soxx_weight','ending_cash_weight'],weights))})
            trace.append({'trade_date':row.trade_date,'slippage_bps':cost,'known_at':row.trade_date+pd.Timedelta(hours=9,minutes=45),
                'known_nav':snap['known_nav'],'known_drawdown':opened['drawdown'],'risk_state':ledger['state'],'risk_clock':ledger['clock'],
                'prior_cumulative_fees':snap['prior_fees'],'trailing4_known_return':float(np.prod(1+np.asarray(ledger['history'][-4:]))-1) if len(ledger['history'])>=4 else np.nan,
                'local_cap':snap['local_cap'],'common_cap':common_cap,'binding_scenario':snap['local_cap']==common_cap,
                **{f'open_before_T_{a}':w for a,w in zip(['soxl','soxx','cash'],before)},
                **{f'open_after_T_{a}':w for a,w in zip(['soxl','soxx','cash'],common)},
                'actual_after_open_nominal_beta':snap['actual_after_open_beta'],
                'recovery_eligible':bool(pd.notna(vote['recovery_probability'])),'recovery_probability':vote['recovery_probability'],
                'recovery_local_approval':bool(vote['recovery_action']),'recovery_consensus':recover,
                **{f'recovery_before_T_{a}':w for a,w in zip(['soxl','soxx','cash'],recovery_before if recover else [np.nan]*3)},
                **{f'recovery_after_T_{a}':w for a,w in zip(['soxl','soxx','cash'],recovery_target if recover else [np.nan]*3)}})
    return {c:{'replay':pd.DataFrame(l['records']),'orders':pd.DataFrame(l['orders']),'targets':pd.DataFrame(l['targets']),'monthly_fee_shares':l['fees'].monthly_shares} for c,l in ledgers.items()},pd.DataFrame(trace)
