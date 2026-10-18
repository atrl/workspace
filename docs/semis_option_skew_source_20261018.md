# 半导体期权偏斜：来源核验与恢复条件

2026-10-18，用户授权继续解决期权偏斜研究。当前结论为
`needs_data_historical_option_quotes_entitlement`，不是策略无效，也不是研究目标完成。
本次仅核验来源和权限：0拟合、0回测、未改生产。机器证据见
[接口核验](semis_option_skew_access_20261018.json)。

## 已完成的实际核验

- 当前环境存在 Polygon API 密钥；未输出或持久化密钥。
- SOXX、SMH 的历史合约列表均返回 HTTP 200；各取一个合约，查询
  2026-09-01 的历史报价，均返回 HTTP 403 / NOT_AUTHORIZED，明确要求升级权限。
  合约列表可访问不代表历史报价可访问；这次小样本仅证明接口权限缺口，未验证历史覆盖。
- 本地两份 SOXL 期权缓存分别有3838、5087行，覆盖2026-01-02至2026-07-07。
  字段为成交OHLC、量、VWAP等；没有同步bid/ask、IV、Delta，也不是SOXX/SMH。
  不以日线最后成交价和不同时间的标的价反推一条可交易偏斜历史。
- 当前环境未提供 ORATS_API_KEY、ORATS_TOKEN 或 ThetaData 用户名/密码。
  这不等于检查过所有外部账户、NAS归档或用户订阅。

## 两条具体数据路径

1. **现有 Massive/Polygon 历史报价权限**：官方 Quotes API 提供带时间戳的
   bid/ask和报价量，历史起于2022-03-07。现有密钥实测无权访问。
   取得权限后仍需同步标的原始价、利率、分红、合约调整和美式期权定价处理；
   该接口本身并不交付历史IV/Delta。不能用当前snapshot Greeks回填过去。
2. **Cboe Volatility Surfaces 或 Option EOD Summary + Calcs**：前者提供固定
   到期日/固定期限的Delta曲面，产品级历史自2011年8月起；后者提供15:45快照的
   IV与Greeks附加数据。尚未确认SOXX/SMH逐日覆盖、具体授权、价格和修订历史。
   产品级起始日期不能冒充某个标的的完整历史。

官方依据：
[Massive Quotes](https://massive.com/docs/rest/options/trades-quotes/quotes)、
[Cboe Volatility Surfaces](https://datashop.cboe.com/volatility-surfaces)、
[Cboe Option EOD Summary](https://datashop.cboe.com/option-eod-summary)。

## 数据验收与待冻结实验

优先检验SOXX；SMH仅作为来源覆盖备选，在读取收益标签前决定，不按回测成绩选标的。
机制假设：半导体自身看跌保护相对看涨的定价，可能在ATM波动率、已实现波动和
趋势之外，提供未来5个交易日下行风险信息。大盘SKEW既有失败卡保持封存。

候选定义为30日固定期限的 `IV(put delta=-0.25) - IV(call delta=+0.25)`，
IV单位采用小数。固定同期15:45美东快照，最早下一交易日消费；真实可用时间晚于
决策时刻则继续延迟或记缺失。必须确认供应商delta符号、期限插值、ATM定义、
美式期权处理和公司行动口径，再冻结具体实现。半日市没有该快照则保留缺失，
不得静默换时间。原始链必须具备同步有效双边报价、标的价格及时间；现成曲面
必须有方法版本、质量标记和来源时间。首次发布/修订资格单独记录。

最小交付字段：symbol、quote_datetime（含时区）、available_at或明确研究假设、
expiry或constant_maturity_days、put/call、delta及其定义、IV及单位、ATM IV、
bid/ask与标的报价（若原始链）、合约调整标记、source/method_version。
先检查两标的覆盖清单及少量完整链样本，再下载多年历史。覆盖不足不删除压力期。

取得合格来源后，在 `research/semis_option_skew_information_<date>/` 建立可执行
notebook。先冻结数据、共同成熟样本、训练/诊断年份、标签定义、费用、拟合预算与门槛，
再通过原生Qlib store、D.calendar/D.features、Dataset和Recorder运行匹配消融：
控制组含ATM IV、历史波动、趋势和可用性标志；候选仅增加该偏斜。
先检验下行风险预测增量及年度稳定性，通过后才进入完整三资产成本账户。
模型选择和统计合同沿用三资产研究合同，复用历史一律为开发/诊断，不冒充独立验证。
本文件是来源准备方案，不声称已完成实验冻结、独立评审或研究运行。

## 当前边界和下一动作

最小外部条件是：现有账户获得历史期权报价授权，或提供有使用权限的SOXX/SMH
历史曲面/期权链导出。新授权需先通过小样本字段与时点验收，不能只凭订阅名称启动训练。
`docs/three_asset_research_contract.md` 明确“不购买数据”，本次未购买或升级订阅。
权限未变时不重复请求同一个403接口，不生成伪IV，不重开已失败的大盘SKEW调参路线。
