"""Native Qlib raw-share validation with independently owned corporate-action state."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.backtest.qlib_continuous_position import QlibContinuousPosition
from src.backtest.raw_corporate_policy import IncomeLedger,SYMBOLS
from src.backtest.ibkr_tiered import IBKRTieredExchange

class RawCashPosition(QlibContinuousPosition):
    def __init__(self,*args,**kwargs):
        self.income=IncomeLedger();super().__init__(*args,**kwargs)
    def calculate_value(self):return super().calculate_value()+self.income.value
    def _buy_stock(self,stock_id,trade_val,cost,trade_price):
        super()._buy_stock(stock_id,trade_val,cost,trade_price)
        self.position[stock_id]['amount']=np.float64(self.position[stock_id]['amount'])
    def apply_corporate(self,day,schedule):
        before=np.array([self.get_stock_amount(s) if self.check_stock(s) else 0. for s in SYMBOLS])
        after,cash,receipts=self.income.apply(day,before,self.get_cash(),schedule)
        for i,s in enumerate(SYMBOLS):
            if self.check_stock(s):
                if before[i]>0 and after[i]!=before[i]:self.position[s]['price']*=before[i]/after[i] # carry-value normalization, not a new observed mark
                self.position[s]['amount']=np.float64(after[i])
        self.position['cash']=cash
        self.income.receipts=[]  # receipts are retained once by native wrapper, not copied at every bar
        return receipts


def native_raw_validate(path,*,provider_uri,point_prices,schedule,tariff,cost,initial_cash):
    import qlib
    from qlib.backtest import backtest
    from qlib.backtest.decision import Order,TradeDecisionWO
    from src.backtest.strategy import DirectTargetWeightStrategy
    qlib.init(provider_uri=str(provider_uri),region='us',kernels=1)
    orders=path['orders'];scheduled=dict(tuple(orders.groupby('stamp',sort=False))) if len(orders) else {}
    endpoints={};ca_receipts=[]
    class PrescribedRawOrders(DirectTargetWeightStrategy):
        def __init__(self):super().__init__(signal=pd.Series(dtype=float,index=pd.MultiIndex.from_arrays([[],[]],names=['datetime','instrument'])),risk_degree=1.)
        def generate_trade_decision(self,execute_result=None):
            start,end=self.trade_calendar.get_step_time(self.trade_calendar.get_trade_step());stamp=pd.Timestamp(start);position=self.trade_position
            if stamp.hour==9 and stamp.minute==45:ca_receipts.extend(position.apply_corporate(stamp.normalize(),schedule))
            if stamp.hour==9 and stamp.minute==47:
                q=np.array([position.get_stock_amount(s) if position.check_stock(s) else 0. for s in SYMBOLS]);prices=point_prices.loc[stamp,list(SYMBOLS)].to_numpy(float)
                endpoints[stamp]=float(q@prices+position.get_cash()+position.income.value)
            records=scheduled.get(stamp)
            submitted=[] if records is None else [Order(str(row.symbol),float(row.raw_shares),Order.SELL if row.sell else Order.BUY,start,end) for row in records.itertuples()]
            return TradeDecisionWO(submitted,self)
    strategy=PrescribedRawOrders();first,last=point_prices.index[0],point_prices.index[-1]
    exchange=IBKRTieredExchange(tariff=tariff,slippage_bps=cost,freq='1min',start_time=first,end_time=last,codes=list(SYMBOLS),deal_price='execution_price',limit_threshold=None)
    outputs,indicators=backtest(start_time=first,end_time=last,strategy=strategy,executor={'class':'SimulatorExecutor','module_path':'qlib.backtest.executor','kwargs':{'time_per_step':'1min','generate_portfolio_metrics':True}},account=initial_cash,benchmark=pd.Series(0.,index=point_prices.index),pos_type=RawCashPosition,exchange_kwargs={'exchange':exchange})
    report,positions=outputs['1min'];expected=path['points'].set_index('stamp');rows=[]
    np.testing.assert_allclose(np.cumprod(1+report['return']-report['cost'])*initial_cash,report['account'],rtol=1e-9,atol=1e-7)
    if set(positions)!=set(expected.index):raise ValueError('native point-grid differs')
    for stamp,position in positions.items():
        row=expected.loc[stamp];q=np.array([position.get_stock_amount(s) if position.check_stock(s) else 0. for s in SYMBOLS]);cash=position.get_cash();receivable=position.income.value;nav=position.calculate_value();scale=max(1.,row.nav)
        err=max(abs(cash-row.cash),abs(receivable-row.receivable),abs(nav-row.nav),float(np.max(abs(q-row[['q_soxl','q_soxx']].to_numpy(float))*point_prices.loc[stamp].to_numpy(float))))/scale
        rows.append({'stamp':stamp,'nav':nav,'cash':cash,'receivable':receivable,'q_soxl':q[0],'q_soxx':q[1],'relative_error':err})
    check=pd.DataFrame(rows)
    np.testing.assert_allclose(report['account'].reindex(pd.DatetimeIndex(check.stamp)).to_numpy(float),check.nav.to_numpy(float),rtol=1e-10,atol=1e-7)
    if check.relative_error.max()>1e-7:raise ValueError('native corporate raw point mismatch '+str(check.relative_error.max()))
    native_orders=pd.DataFrame(exchange.executed_fees)
    if len(native_orders)!=len(orders) or (len(orders) and not np.allclose(native_orders.total,orders.total,rtol=0,atol=1e-7)):raise ValueError('native raw fees differ')
    authority=path['replay'].copy();values=[];returns=[]
    for row in authority.itertuples():
        beginning=endpoints[pd.Timestamp(row.trade_date)+pd.Timedelta('9h47min')];ending=endpoints[pd.Timestamp(row.exit_date)+pd.Timedelta('9h47min')]
        if not np.isclose(ending,row.native_expected_nav,rtol=1e-7,atol=1e-7):raise ValueError('native post-CA pretrade endpoint differs')
        values.append(ending/initial_cash);returns.append(ending/beginning-1)
    authority['equity']=values;authority['strategy_return']=returns
    np.testing.assert_allclose(np.cumprod(1+authority.strategy_return),authority.equity,rtol=1e-10,atol=1e-10)
    return {'authority':authority,'orders':orders,'native_orders':native_orders,'native_report':report,'native_positions':positions,'native_indicators':indicators,'points':check,'independent_points':path['points'],'corporate_receipts':path['corporate_receipts'],'native_corporate_receipts':pd.DataFrame(ca_receipts),'targets':path['targets'],'reconciliation':check,'audit':{'status':'passed','sessions':len(authority),'points':len(check),'orders':len(orders),'max_relative_error':float(check.relative_error.max()),'native_CA_from_own_positions':True,'receivable_in_NAV_not_spendable':True,'currency_assumption':'USD gross pretax paper','historical_PIT_qualified':False}}
