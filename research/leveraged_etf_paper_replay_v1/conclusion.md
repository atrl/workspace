# 结论

## 跑了什么

使用 qlib 日线数据（截至 2026-07-20）按 close(t) 生成信号、在下一交易日开盘执行，复刻或适配了 200 日均线杠杆切换、滞后一月方差缩放、3x LETF CPPI，以及 1x 动态复制基准。所有策略均报告 0/10/25bp，outer 2023–2026 完全不参与参数选择。

## 关键结果

10bp 下通过全部正式门槛的策略数为 0。当前最佳诊断项为 `moreira_muir_lagged_inverse_variance`：CAGR 22.04%，MaxDD -20.79%，单日最差 -5.20%，CVaR5 -2.41%，真实滚动五日最差 -7.35%，SOXL 平均仓位 15.14%，上涨捕获 14.19%。

## 年份与成本差异

逐年 outer 结果见 `annual_outer_metrics.csv`，25bp 压力结果见 `strategy_comparison.csv`。2026 为截至数据日的未完整年度，不参与任何选择。

## 晋升决定

`research_reject_no_paper_strategy_passed_all_gates`。本卡是论文机制复盘，不晋升未通过全部风险、参与度、上涨捕获、双基准 Calmar 与 3/4 年度门槛的方案。

## 风险

CPPI 现金腿暂按 0 收益，缺少论文使用的 PIT 一年期国债收益；SOXX/SOXL 是半导体 ETF 映射，不等同论文的宽基指数；实际 SOXL 存在跳空、交易摩擦和杠杆衰减。`smart leverage` 的债券输入和 bootstrap 神经控制未在本卡伪复刻。

## 下一步

若继续，只应补齐 PIT T-bill/T-bond 序列后复刻 smart-leverage 数值框架，或把通过研究宽松门但未过正式门的论文规则作为模型的风险上限；不应再把底仓硬塞进不满足尾部门槛的模型。

不构成买卖建议。
