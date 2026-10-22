# Cboe VIX/VIX3M 期限结构：来源完整，候选拟合前关闭

两份 Cboe 官方日线原始 CSV 分别有 9,299 与 4,300 行。限定修复后的原生 Qlib 日历把上一实际交易日的 `log(VIX close / VIX3M close)` 映射到原始 Stage1 的全部 3,966 个信号日，训练 3,020/3,020、复用评价 946/946 有数值；训练期对原 39 项输入最大逐项相关绝对值 0.6317。v1 首日漏值只是日历左边界错误，原始 2010-12-31 两文件均有值；v1 失败和 Recorder 保留。

**有数值不等于历史可用。** Cboe 方法文件给 VIX3M 的首次值为 2009 年 9 月、推出时间为 2013 年 10 月。原训练段有 691 日使用确定早于 2013 年 10 月的回算值，另有 23 个推出月份信号日缺精确首发日；即使其余日有当前 CSV 数值，也未证明原版在下一次 09:46 决策前公开或未被修订。两份文件在本次收取时更新不同步，不能反推历史同日可用。

独立去重审查发现，已关闭的 `cboe_h5_information_20261009` 用上一会话的 `log(VIX9D/VIX3M)` 做过匹配 H5 期限结构信息检验并在开发/复用历史上失败。本候选将短端改为约 30 日 VIX、仍以三个月为分母且保留 H5 接收任务，主要是失败后的期限选择。因此决定 `source_map_verified_candidate_closed_before_fit_duplicate_term_horizon_and_prelaunch_backfill`：**0 收益读取、0 拟合、0 新资金账户**，不以原 Stage1 39 项接收器重包装同一经济机制。

原始 CSV 与响应元数据在 Yanxuan NAS 的 SHA-256 为 `fd9562a34727d30617d4c103f513056769c4ec9e5deed1f9cfee8fc72f27e5ec`；v2 原生 Recorder `306545c51d9549a4ab68e37a33a04dde` 的 NAS SHA-256 为 `bd8842b2f34363bc42cf7e1ee8876da18c8b21b16b194f1ebaf9284ae817cdff`。两组都经远端哈希及限定恢复，Recorder 已定向读回。整体 Stage1 0bp/SOXX 目标未达成；下一步必须寻找实质不同的决策时机制。

[Cboe VIX 历史文件](https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv) · [Cboe VIX3M 历史文件](https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv) · [Cboe 首次值与推出年月](https://cdn.cboe.com/api/global/us_indices/governance/Volatility_Index_Methodology_Selected_SPX_Target_Expected_Volatility_Term_Indices.pdf)
