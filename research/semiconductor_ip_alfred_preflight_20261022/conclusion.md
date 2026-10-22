# 半导体工业生产 ALFRED 版本：缺 Stage1 早期历史

[IPG3344S](https://fred.stlouisfed.org/series/IPG3344S)测量美国半导体及其他电子元件工业生产量，与已失败的PPI价格路线不同；[IPG33441A1S](https://fred.stlouisfed.org/series/IPG33441A1S)为2015年停用、定义不同的旧系列。只读核对两张ALFRED官方原始下载表单：前者有150个可选日期版本，后者6个，最早都为2015-02-18。原Stage1训练从2011-01-03开始，2011–2014同期数值版本缺失；不能把后续修订值提前填充，也不能直接拼接定义不同的旧系列。

决定 `needs_pre2015_original_vintages_no_numeric_download_no_fit`。没有下载数值ZIP、读取SOXX结果、拟合模型或开经济账户。两份原始表单已归档NAS，远端SHA256和冷恢复一致。若取得2011–2014的原始美联储G.17发布档案及可辩护的定义桥接，另立冻结计划；此来源缺口不阻断其他机制。
