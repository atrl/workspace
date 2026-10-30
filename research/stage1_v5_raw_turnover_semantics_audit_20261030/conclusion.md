# 成交额隐含价越界不是简单前后一分钟错位

前卡冻结的同bar、1bp容忍、≥99.9%来源门保持**失败**；本卡零模型/账户，只读原Futu字节与Qlib已对齐原价OHLC，按同一交易日比较`turnover/volume`和前一、同一、后一bar高低区间。整体同bar落入率**90.77%**，SOXL约91.95%、SOXX约89.59%；邻bar落入率SOXL约47.31%/51.13%、SOXX约45.04%/47.81%，均更差，不支持简单分钟错位。首例2018-10-15 SOXL 09:37高低同为115.12，126股、成交额14543.68暗示115.426，越界约26.6bp；具体样本和每年越界幅度见[`audit.json`](audit.json)、[`shifted_range_summary.csv`](shifted_range_summary.csv)。

第一次执行已算完Qlib结果但最后显示把元组键写成JSON失败；原计划/收据保留，仅修显示并如实记录第二次Qlib读取，未改容忍、移位集合或样本。官方[富途历史K线字段说明](https://openapi.futunn.com/futu-api-doc/quote/request-history-kline.html)称`turnover`为成交额，但不解释这里与OHLC的成交过滤差异；具体原因`unknown`。不得事后移时间戳或提高容忍使本固定VWAP版本入场，历史首收和真实交易口径仍待数据证明。
