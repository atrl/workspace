# SOXX发行人份额数：完整回溯数值来源，但历史时点未认证

[发行人 SOXX 页面](https://www.ishares.com/us/products/239705/fund)的 Data Download 原文件含 2001-07-10 至 2026-10-21 的 8937 个日历行。原生 Qlib 日历上的2011–2022全部3020个训练交易日和已封存的944个资金信号日都有份额数；两个原文件缺值是2024-03-29和03-31，不在上述交易日。两会话保守延迟的五会话份额变化也无缺口。

2016、2018、2024三个原始发行人网页存档的份额数与当前下载表精确呈3倍，符合[发行人披露的2024年3拆1](https://www.ishares.com/us/literature/tax-information/form-8937-ishares-semiconductor-etf-3-6-2024.pdf)单位重述；2022页面未提取出份额字段。该解释仅适用于抽样核对，不认证整个历史日序列的首次公布时点或修订。2026文件是回溯版本，网页存档只覆盖少数日。

训练期假设性份额变化与原Stage1 39项输入的最大绝对Spearman为0.12975（`soxx_ret10`），说明线性秩冗余较低；相关性不是未来收益信息或可交易超额。一级市场份额变化不同于已失败的二级成交量参与度。

原文件与四份网页存档的精确字节均已归档NAS，下载恢复后SHA逐项一致；回执见 `nas_source_archive_receipt.json` 与 `nas_vintage_archive_receipt.json`。本卡0拟合、0账户、0评价收益读取。决定 `candidate_source_only_needs_historical_first_public_and_revision_evidence_before_Stage1_receiver`。下一步先补发行人历史公布时点或足够完整的旧版本证据，再另行冻结匹配信息检验；在时点认证前不拟合、不进0bp经济账户或宣称超额SOXX。
