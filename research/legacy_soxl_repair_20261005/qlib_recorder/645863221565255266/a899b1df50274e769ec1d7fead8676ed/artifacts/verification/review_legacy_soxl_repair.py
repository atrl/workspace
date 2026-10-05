from pathlib import Path
import json,hashlib,numpy as np,pandas as pd
import qlib
from qlib.workflow import R
from soxl_backtest.config.backtest_config import BacktestConfig
from soxl_backtest.position.position_manager import PositionManager
from soxl_backtest.strategy.buy_strategy import BuyStrategy
r=Path.cwd();c=r/'research/legacy_soxl_repair_20261005';s=json.loads((c/'summary.json').read_text());qlib.init(provider_uri=str(r/s['provider_uri']),region='us')
with R.uri_context(s['recorder_uri']):
 rec=R.get_recorder(experiment_name=s['experiment'],recorder_id=s['recorder_id']);assert rec.client.get_run(rec.id).info.status=='FINISHED';ar=Path(rec.get_local_dir())/'artifacts'
 for p,h in s['recorder_artifact_sha256'].items():assert hashlib.sha256((ar/p).read_bytes()).hexdigest()==h
 paths=rec.load_object('portfolio_analysis/native_reports.pkl');orders=rec.load_object('analysis/orders.pkl');market=rec.load_object('inputs/market.pkl').set_index('date');native=paths[('continuous',25,'legacy_repaired')]['daily'];t=orders[('continuous',25)].copy();lookup={pd.Timestamp(x.date):x for x in t.itertuples()};cash=100000.;shares=0.;errs=[]
 for day,row in native.iterrows():
  order=lookup.get(day)
  if order is not None:
   price=float(market.loc[day,'open']);quantity=float(order.amount);assert float(order.price)==price;assert pd.Timestamp(order.signal_date)<day;cash-=quantity*price+abs(quantity)*price*.0025;shares+=quantity
  nav=cash+shares*float(market.loc[day,'close']);errs.append(max(abs(nav-row.account),abs(cash-row.cash),abs(shares-row.soxl_shares)));assert cash>=-1e-8 and shares>=0
 assert max(errs)<1e-5
 baseline=paths[('continuous',25,'soxx_buy_hold')]['daily'];dev=native.loc['2016':'2022'];soxx=baseline.loc['2016':'2022'];ex=np.log1p(dev.net_return)-np.log1p(soxx.net_return);wins=int((ex.groupby(ex.index.year).sum()>0).sum());assert wins==4
 recent=paths[('recent_cash_start',25,'legacy_repaired')]['daily'];assert len(recent)==39 and orders[('recent_cash_start',25)].empty and recent.net_return.eq(0).all()
 namespace={'__package__':'soxl_backtest.strategy'};oldfile=Path('/tmp/legacy-soxl-repair-20261005/buy_strategy.py.original');assert hashlib.sha256(oldfile.read_bytes()).hexdigest()=='4393972303343ccc78ac9067a8cc204244de3da8e36b0d1cddff2ddad75f3068';exec(compile(oldfile.read_text(),str(oldfile),'exec'),namespace);before=[];after=[]
 for last in [50.,200.]:
  frame=pd.DataFrame({'date':pd.bdate_range('2020-01-01',periods=2),'price':[100.,last],'VOL_COEF':1.,'panic_score':-100.,'MA20':1000.});cfg=BacktestConfig({'VOLATILITY_MEDIAN':0});before.append(bool(namespace['BuyStrategy'](cfg,PositionManager(cfg)).should_buy(frame.iloc[0],frame,0)));after.append(bool(BuyStrategy(cfg,PositionManager(cfg)).should_buy(frame.iloc[0],frame,0)))
 assert before==[False,True] and after==[False,False]
 out={'passed':True,'scope':'One bounded independent audit: all16frozenartifact hashes, fullcontinuous25bp cash/share/closeNAV reconstruction fromsavedorders, annualwin count,recentzero-trade path, syntheticoriginal-vs-fixedfirstrowcounterexample. No newmarketpolicyrun.','recorder_id':rec.id,'book_rows':len(native),'reconstructed_order_count':len(t),'max_cash_share_nav_error':max(errs),'development_positive_years':wins,'recent_cash_days':len(recent),'original_firstrow_buy_with_future_last_prices50_200':before,'repaired_firstrow_buy':after,'historical_stable_excess_pass':False,'objective_achieved':False}
 (c/'independent_review.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
