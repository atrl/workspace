# 机制依据与适用边界

- [Measuring downside risk — realised semivariance, Barndorff-Nielsen/Kinnebrock/Shephard](https://scholar.harvard.edu/files/downside200808.pdf)：用高频负收益分解已实现风险。本实验检验上一交易日的下行方差是否帮助既有分配器；不据论文宣称收益预测成立。
- [Market intraday momentum, Gao/Han/Li/Zhou](https://profiles.wustl.edu/en/publications/market-intraday-momentum/)：其 SPY 同日首段/尾段关系只提供分段路径的研究动机，不证明半导体 ETF 的次日或五日可预测性；这些迁移假设需本卡验证。
- [Qlib Data](https://qlib.readthedocs.io/en/latest/component/data.html)、[Model](https://qlib.readthedocs.io/en/latest/component/model.html)、[Recorder](https://qlib.readthedocs.io/en/latest/component/recorder.html)、[Strategy](https://qlib.readthedocs.io/en/latest/component/strategy.html)：数据、DatasetH、模型、原生账户和证据的 API 权威。

本轮不重开此前固定失败的 Ridge 日内路线、VXN/VIX、财报、夜盘或手工限仓路线。新增机制仅为 raw NONE 同一交易时段内的全天路径与同期价格联动，进入现役同结构的三资产神经分配器。
