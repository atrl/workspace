# OCC公开 SOXX 2019 认购／认沽量：两年回溯限制

OCC官方批处理说明允许按期权标的、日频、认购／认沽字段查询；其网页同时提示成交量查询仅提供过去24个月。按冻结计划只对 **2019-05-31、SOXX标的**各发一次认购和认沽只读GET，没有改参数或重试。两个请求均返回HTTP200，但正文都是相同的41字节：`Report date cannot be prior to  2 years`，不是CSV，也没有合格SOXX合约行。精确URL、UTC请求／响应时间、头、原始字节和SHA-256见 `source_receipt.json` 及 `data/snapshots/occ_soxx_put_call_source_probe_20261024/`。

来源门失败，决定 `fixed_old_date_OCC_source_query_failed_two_year_public_age_gate`。这个公开OCC Volume Query批处理地址不能按本次精确查询提供2019历史日，更不能证明2011–2022 SOXX认购／认沽序列、历史首次送达或修订。它不否定其他尚未测试的合法公开或授权归档，也不是对SOXX期权25-Delta隐含波动率的检验。用户当前不联系供应商；本卡没有询价、购买、订阅或读取账户。

两份原始响应均逐字节保留，Qlib Recorder归档NAS并对原始body做限定恢复回读，远端SHA已核对；大型Recorder仅在本地及NAS保留，未将自动生成的无关工作区代码快照提交到Git。0市场结果、0拟合、0资金账户、0独立前向日。该固定URL版本关闭；SOXX目标仍未达到。
