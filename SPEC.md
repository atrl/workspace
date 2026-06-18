# SPEC.md

## Project

维护一个 SOXL 全平台研究、验证与 paper 运行工作区。

项目成功标准：未来的单因子、多因子、模型、objective、回测、analysis、
dashboard export 和 paper adapter 工作，都能通过稳定 `src/` 接口复用，并用
`research/` 或 `production/` notebook 留下中文可审阅证据。

## In Scope

- SOXL 为交易标的，SOXX/VIX/QQQ/SPY/半导体或科技 ETF、3x ETF 作为支持宇宙。
- canonical data、qlib calendar/store、数据 snapshot 和 schema hash。
- `src/` 下的稳定框架能力。
- `research/<experiment>/<experiment>.ipynb` 的研究实验。
- `production/<strategy>/<strategy>.ipynb` 的晋升方案。
- QuantBench 风格模型矩阵、训练、预测、回测和横向分析。
- qlib analyzer 扩展、regime 分层、区间对比、模型差异解释。
- paper/sim API adapter、schedule 编排、dashboard manifest export。

## Out of Scope

- v1 真实 broker 下单。
- 聊天记录作为持久证据。
- 把单次实验参数硬编码进 `src/` 稳定模块。
- 绕过 manifest 直接让 dashboard 读取散落实验内部文件。

## Layering

保持以下层次分离：

1. data loading、provider、qlib calendar alignment、snapshot。
2. factor construction、factor diagnostics、factor combination。
3. objective 和 label/score 定义。
4. model registry、fit/predict、artifact persistence。
5. backtest accounting、cost、slippage、walk-forward。
6. analysis：regime、window、model comparison、failure mode。
7. dashboard export、schedule、paper API。
8. research notebook 和 production notebook。

`src/` 拥有稳定接口。`research/` 和 `production/` notebook 拥有单次流程、
参数、结论和 sidecar artifact 链接。
