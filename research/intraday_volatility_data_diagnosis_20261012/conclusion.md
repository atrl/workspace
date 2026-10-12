# 中文结论

本次执行原生 Qlib 1 分钟 store 构建、D.calendar/D.instruments/D.features 读回、分钟资格审计和分时波动统计；没有拟合模型、回测策略或改变现役运行。

- 源数据：1,518,062 行，2018-12-31 至 2026-10-09；逐日显式预期 1955 个交易日/标的。完整常规且两标的均通过价量基本检查的配对统计为 1937 日；另有 15 个正常半日市。
- 缺口：2026-07-27 每标的缺 329 分钟，2026-07-29 每标的整日缺 390 分钟；SOXX 2020-03-16 首分钟 OHLC 不一致，排除该配对日。
- 分时：开盘后 30 分钟平均贡献 SOXL 日内实现方差 28.38%、SOXX 28.56%；该时段仅占常规交易日 7.69%。这是风险集中度，不是可交易收益。
- 年度/窗口差异：SOXL 2020 年日内 RV 中位数 4.08%，2026 年 5.46%；更多年份和最近 20 个完整交易日见 regime_metrics.csv/window_metrics.csv，全部是复用诊断。
- 数据风险：成交额/成交量 VWAP 偏离 OHLC 的比例约 85.56%（按全部记录计；容差 0.1%）。不能直接将它作为复权价 VWAP 因子。补采共 1560 行已在 Recorder 独立保存，但重叠价格和成交量变化尚未解释，未合入 canonical。
- 当前模型：绑定 allocation f696f656b3d44e868588e09a5d97c905、action b472a458fc9042b4905094fddeaebd33；action development gate 为 false。可决策样本历史 AUC 为 0.5168，区间包含 0.5；分钟风险增量尚未验证。
- 晋升：本卡为 completed_descriptive_diagnosis，promotion_eligible=false、objective_achieved=false。风险持续性相关或数据补齐都不证明稳定跑赢 SOXX。
- 下一步：把显式缺口审计接入独立采集管道，核验复权/成交量来源，保存 received_at/finalized_at；再按 plan.json 的最多 3 个对照检验首小时 RV、下行半方差、趋势效率。09:31 仅用昨日分钟，10:31 仅用 10:30 前已收到并结束的分钟；10/25bp 各自生成完整账户和成熟净执行优势标签，development 选择，独立前向决定晋升。
- Qlib Recorder：70ec0bf081df4ec9b6bc47dd2e621ba9；大表和独立补采快照仅在 Recorder，根目录保持小摘要、主 notebook、图和结论。
