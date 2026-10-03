import json,csv,hashlib
from pathlib import Path
root=Path('/Users/rziiiii/dev/workspace-worktrees/feasible-parent-sparse-search-20261003')
card=root/'research/feasible_parent_sparse_search_20261003'
a=json.load(open('/tmp/phase9_economic_audit.json'))
market=json.loads((card/'market_reference.json').read_text())
assert market['complete'] is True and market['recorder_id']==a['recorder_id']
refs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (card/'market_reference.json',card/'preflight_reference.json')}
s=json.loads((card/'summary.json').read_text())
assert s['historical_qualification_pass'] is False and s['objective_achieved'] is False
# All diagnostics below derive solely from the existing saved-book audit.
def table(name,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (card/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for row in rows: writer.writerow({k:json.dumps(v,ensure_ascii=False,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in row.items()})
table('startup_training_contributions.csv',a['holding_only_saved_training_startup'])
table('saved_book_startup_contributions.csv',a['saved_book_contribution_diagnostics'])
table('annual_excess_magnitude.csv',a['annual_magnitude_annotations'])
note=Path('/tmp/phase9_economic_review.md').read_text()
note=note.replace('可执行只读复核：[phase9_economic_audit.ipynb](/tmp/phase9_economic_audit.ipynb)。完整指标、来源SHA与已有账本的起始贡献在`/tmp/phase9_economic_audit.json`。','可执行只读复核及完整来源SHA将归档在市场Recorder的`analysis/diagnostics/economic_review/phase9_economic_audit.ipynb`与同名JSON。瘦表为[startup_training_contributions.csv](startup_training_contributions.csv)、[saved_book_startup_contributions.csv](saved_book_startup_contributions.csv)、[annual_excess_magnitude.csv](annual_excess_magnitude.csv)；原费用、年度、窗口和父模型比较表保持不变。')
note+='\n下一方向尚未登记、实现或运行。优先核实已有输入是否具有能跨时间复用的信息，并设计诚实的训练内部时间验证，使固定现金起点的收益与持续决策能力分开；不据本轮结果追加阈值、搜索预算、模型起点或事后候选。所有未来新实验仍须先登记再计算。\n'
(card/'economic_review.md').write_text(note)
conclusion='''# 可行父模型上的稀疏搜索：三个组均未通过历史资格

本轮完成登记的固定父编码器、33坐标COBYLA搜索。合成预检通过，实际市场开发期搜索、两种外层诊断与原生Qlib对账均完成，但**三个组全部未通过原定历史门槛，不晋升；稳定净超额目标仍未完成**。`holdings_only`只是失败候选中的诊断排序第一，不能称为合格策略。

开发期为2016–2022信号年，1,762个区间，首交易2016-01-05，最后于2023-01-04开盘结算。冻结行情截至2026-09-25，来自原生Qlib provider `data/qlib/published_futu_affine_20260925_v1`，源Recorder为`bd754e6362b241f194351a1e7aebff6f`。2023以后是已查看历史诊断，不能当作新留出样本。

|组|相对SOXX终值10/25bp|正超额年份10/25bp|MaxDD 25bp|CVaR5 25bp|
|---|---:|---:|---:|---:|
|持仓状态 `holdings_only`|1.014755 / 1.014742|1/7、1/7|−47.373%|−4.416%|
|价格 `price_holdings`|1.385531 / 1.036266|3/7、3/7|−49.432%|−5.771%|
|价格与波动预测 `price_vol_holdings`|1.556598 / 1.114310|3/7、3/7|−64.570%|−6.605%|

同窗SOXX25bp的MaxDD为−47.373%，CVaR5为−4.418%。两个条件组在两档成本下均失败于年度稳定性、回撤、尾部及bootstrap门。价格与波动组的10bp删一年门通过，但25bp失败。持仓状态组虽保住风险门，其余六年25bp超额精确为0；10bp微小负残差未虚增正年份。三个组的bootstrap下界均未通过，原门槛与排序未被修改。

持仓状态组25bp整段净对数超额0.0146348，第一天贡献却为+0.0156723，其余全部观察合计−0.00103754：开始只持56.40%SOXX，第三天达到纯SOXX，之后1,759天逐日收益与基准相同。它不是持续的择时信号。11个已保存集合训练账本也都从2011-01-04现金起步，相对父模型的改进只发生在开头2–10行，之后逐日相同。三种子结果相同，不能作为三个独立市场证据。2023以后诊断的微小正增量同样由首次入场贡献。

两个条件组25bp相对父模型终值分别改善2.49%和7.89%，说明本轮确实搜索出了新训练可行状态；但交易天数从父模型1,762天下降至1,455/1,541天的同时，总换手从54.61/59.23倍上升至194.58/223.76倍，为父模型约3.56/3.78倍。保存账本的年化对数费用前贡献为7.433%/9.516%，费用项为−6.924%/−7.968%，净贡献仅0.509%/1.548%。这些是同一持仓路径的恒等分解，不是零成本新策略，也不证明未来毛收益优势。25bp最差单日恶化至−21.59%/−27.35%；平均现金多仍未保护危机日。两组HAC alpha区间跨零，未形成可靠信息优势证据。

三个集合开发路径都没有单独HOLD专家触发的日期。精确零订单日分别为1,759、307、188，不能把已经持有资产顶点时的零订单归因于学会HOLD。价格与波动组另有33天微小订单低于原报告阈值，仍不属于精确零订单。

用户“8月3日以来”按实际交易日定义为2026-08-03至09-25开盘，共38区间，首信号07-31。仅对开发期选出的持仓状态组作下列已登记诊断：

|起点/模式|10bp策略 / SOXX|25bp策略 / SOXX|交易天数|
|---|---:|---:|---:|
|持续持仓，年更或冻结|15.843% / 15.843%|15.843% / 15.843%|0|
|8/3现金起步，年更|12.541% / 15.727%|12.373% / 15.554%|7|
|8/3现金起步，冻结|12.936% / 15.727%|12.769% / 15.554%|8|

持续持仓已是纯SOXX，没有超额；现金起步落后约2.8–3.2个百分点。各窗口MaxDD约−10.984%，与SOXX相同。这不是用户真实账户，也不能与截至09-30收盘的旧窗口混比。

本轮实际工作量为75个新市场搜索、25个集合审计、109,070次唯一双成本评估，复用63个开发父模型，另按原协议重建12个缺失外层父模型（1,440步旧Adam）。合成预检另有3个父模型重建及3个搜索。COBYLA评估次数和旧Adam步数不可作为等计算预算。原生Qlib两档成本对账均通过，最大逐期误差约5.95e−8；工程正确不等于金融资格通过。

未解决的是跨时间泛化、费用和下一年尾部。下一方向尚未登记或运行：先核实已有输入的可复用信息，再设计诚实的训练内部时间验证，防止固定现金起点冒充持续能力；不再把优化器可找到训练改进等同于存在稳定超额。既有有限搜索不保证全局最优，父模型保留原梯度/截断近似；数据含16条次来源观察、复权差异，历史开盘连续份额亦不代表真实点差、冲击或部分成交。

没有新前向合格证据，没有生产/SIM切换，没有实盘订单。完整经济解释见[economic_review.md](economic_review.md)，年度、费用、父模型增量和起点贡献见对应小表。市场Recorder：`8696cd2152df4ffbbb5ac6692b139ef9`；合成预检Recorder：`a0cbd66785b44ccc8ab2080b36026195`，其合成F约0.650不能读作市场收益。
'''
(card/'conclusion.md').write_text(conclusion)
nbpath=card/'feasible_parent_sparse_search_20261003.ipynb';nb=json.loads(nbpath.read_text())
nb['cells'][-1]['source']=conclusion.splitlines(keepends=True)
nb['cells'][0]['source']=('# 可行父模型上的33坐标稀疏持仓搜索\n\n本Notebook已完成登记的合成增益预检、开发期搜索、已查看历史外层诊断及原生Qlib对账。三个信息组均未通过历史资格，不晋升，稳定净超额目标未完成。完整定义见[selection_plan.json](selection_plan.json)和[design.md](design.md)；最终经济解释见末尾中文结论及[economic_review.md](economic_review.md)。重跑默认只核对已封存证据，不启动新的拟合活动。\n').splitlines(keepends=True)
nbpath.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
counts={'market_model_fits':75,'market_search_fits':75,'market_ensemble_audits':25,'cached_development_parents':63,'outer_parent_reproductions':12,'synthetic_parent_reproductions':3,'synthetic_incremental_searches':3}
s.update(counts)
s['market_model_fits_scope']='75 new market COBYLA seed searches only; excludes 12 original-protocol outer parent reproductions, 25 ensemble audits and the separately recorded 3+3 synthetic preflight fits/searches.'
s['economic_interpretation']={'scope':'Read-only additive attribution of already saved books; no new policy replay, fit, threshold or qualification change. Card enrichment does not alter Recorder analysis/final_summary.pkl or bound references.','all_registered_families_failed':True,'selected_is_diagnostic_only':True,'state_only_gain_is_startup_concentrated':True,'development_state_only_first_observation_exceeds_total_log_excess':True,'state_only_training_differences_confined_to_first_2_to_10_rows':True,'conditional_fewer_trade_days_but_higher_notional_turnover':True,'next_direction_registered':False,'next_direction_run':False,'next_direction':'Verify reusable information and honest internal temporal validation, separating fixed cash-entry transient from continuing decision value.','review':'economic_review.md','tables':['startup_training_contributions.csv','saved_book_startup_contributions.csv','annual_excess_magnitude.csv'],'recorder_artifacts':['analysis/diagnostics/economic_review/phase9_economic_audit.ipynb','analysis/diagnostics/economic_review/phase9_economic_audit.json','analysis/diagnostics/economic_review/phase9_economic_audit.py','analysis/diagnostics/economic_review/phase9_startup_attribution_code.txt'],'source_market_reference_sha256':refs['market_reference.json']}
for f in ['economic_review.md','startup_training_contributions.csv','saved_book_startup_contributions.csv','annual_excess_magnitude.csv']:
    if f not in s['sidecars']:s['sidecars'].append(f)
(card/'summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
assert refs=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (card/'market_reference.json',card/'preflight_reference.json')}
assert ''.join(json.loads(nbpath.read_text())['cells'][-1]['source'])==(card/'conclusion.md').read_text()
print(json.dumps({'files':[str(card/p) for p in ['conclusion.md',nbpath.name,'summary.json','economic_review.md','startup_training_contributions.csv','saved_book_startup_contributions.csv','annual_excess_magnitude.csv']], 'references_unchanged':True,'notebook_conclusion_equal':True,'counts':counts},indent=2))
