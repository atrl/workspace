# 美联储原始G.17发布：旧半导体行存在，定义桥接未完成

已固定并读取[2011-02-16](https://www.federalreserve.gov/releases/g17/20110216/g17.txt)、[2014-06-16](https://www.federalreserve.gov/releases/g17/20140616/g17.txt)、[2015-07-15](https://www.federalreserve.gov/releases/g17/20150715/g17.txt)三份美联储原始G.17 ASCII发布。每份都显示09:15东部发布标识，并在工业生产表保留“Semiconductors and related electronic components”、行业代码 `334412-9`；这证明ALFRED 2015年前缺版本不等于原始发布不存在。

现行[FRED IPG3344S](https://fred.stlouisfed.org/series/IPG3344S)是 `NAICS 3344` 的“Semiconductor and Other Electronic Component”，不能把旧 `334412-9` 数值未经定义桥接拼进去。2010–2014每月原版清单（含Stage1初期特征预热）、旧新口径对照及可用于Stage1决策的公告时点仍未认证；09:15公告标记也不等于当时已在账户09:29决策前收到。

本卡来源限定，0因子值、0市场结果、0拟合、0账户。三份原件已NAS归档并远端哈希、冷恢复一致。决定 `three_original_2011_2014_2015_Fed_release_rows_verified_definition_bridge_missing`。仓库已有 `fed_semiconductor_capacity_source_20261013` 的192份原始G.17补充表（2015-08以前59份旧口径排除），无需重复盘点；当前另有单独冻结的Table1 NAICS3344行审核计划。先读其结果，若2015前目标口径缺失，不能拼接旧334412-9或缩短Stage1训练。当前不能进行Stage1信息或经济检验。
