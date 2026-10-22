# HYG/LQD 历史信用 ETF 价格来源

按冻结的 query2 Yahoo 图表 URL 分别取得 HYG 与 LQD 的原始 JSON、响应头及请求时刻。两只基金各有 **4,227 个**正且有限的日收盘价，日期一致，从 **2010-01-04 至 2026-10-22**，覆盖冻结 Stage1 2011–2022 训练期和复用评价期所需外部观察范围；两份原始字节 SHA 及检查见 `source_result.json`。来源归档至 Yanxuan NAS，SHA-256 `499a2880df04cc22826f11a5ce3cf2da1d0ebd27d7d46fc7363b99ed75090ee7`，远端哈希和限定恢复通过。

同一响应含完整长度的 `adjclose` 数组，但这只是**当前回溯快照**，不能把今日的历史调整价当作当时已发布版本。固定 `events=history` 请求返回的公司行动事件集为空；月度分红和可能拆股须另行取得并验证，不能直接把原始 close 或追溯 adjclose 当作合格的两 ETF 相对总收益。进一步还需按交易日次日可用时点接入 Qlib、对照 Stage1 原 39 输入及原生费用账户口径。

本卡 **2 次外部 GET、0 SOXX 结果读取、0 拟合、0 原生账户**。决定：`HYG_LQD_current_chart_raw_full_span_source_received_corporate_actions_and_PIT_pending`。它解决了此前 Futu 零行和 IBKR 1,000 柱上限下的**历史覆盖可行性**，尚未证明信用风险偏好因子有效。
