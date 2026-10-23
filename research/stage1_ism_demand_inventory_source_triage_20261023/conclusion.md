# Stage1 新订单与客户库存：来源预检

2026-10-23，来源预检；0 拟合、0 回测、0 SOXX 收益读取。完整 Stage1 A0 仍是候选母本，首个经济门仍是 0bp 附加滑点、IBKR Tiered 费用后的完整 Qlib 账户。本卡不产生因子或策略结论。

## 从失败区间提出的机制

原 Stage1 对 SOXX 全程对数超额 -0.376868，其中现金机会损失 -0.600323、SOXL 相对选择 +0.225931。最差 126 日（2023-08-01 至 2024-01-31）同时有现金 -0.127238 和 SOXL 选择 -0.121381；2026Q3 则以 SOXL 选择 -0.143901 为主。需要一个事前能区分需求扩张与库存消化的输入，并分别检验现金优势和 SOXL 相对 SOXX 的标签；不能根据已知亏损季度设开关。

候选仅为 **美国 ISM 制造业新订单指数与客户库存指数** 的事前组合。ISM 对客户库存的解释是受访者判断客户库存过高/过低，低于 50 表示偏低；它并非制造商自身原材料库存。为避免混淆，不能把报告中的 `Inventories` 行代替 `Customers' Inventories` 行。预注册前要决定唯一数值定义、固定符号、发布时钟、版本和缺失处理；这次不作选择。

## 已核查的原始来源与缺口

- [ISM 2026 年 9 月制造业报告](https://www.ismworld.org/supply-management-news-and-reports/reports/ism-pmi-reports/pmi/september/) 的官方表同时列出 New Orders、Inventories 和 Customers' Inventories，并写明季调字段。此为当前页面存在性的证明，不是历史原始版本或完整时间序列。
- [ISM 2024 年 8 月原始 PDF](https://www.ismworld.org/globalassets/pub/research-and-surveys/rob/pmi/rob202408pmi.pdf) 和 [2025 年 5 月原始 PDF](https://www.ismworld.org/globalassets/pub/research-and-surveys/rob/pmi/d4mn202505pmi.pdf) 证明跨年 PDF 报告有月度表；页面搜到的 PDF 创建/索引日期不能冒充首次公开时刻。
- [ISM 对库存子指数的说明](https://www.ismworld.org/supply-management-news-and-reports/news-publications/inside-supply-management-magazine/blog/2019-01/the-monthly-metric-rob-inventories-subindexes/) 区分客户库存与本企业原材料库存，也显示 2018 年旧文章存在；它没有提供完整 2018–2022 的逐月原始表。
- [ISM 报告入口](https://www.ismworld.org/supply-management-news-and-reports/reports/ism-pmi-reports/) 说明制造业月报按月发布。仍需逐月原始 PDF/HTML、首次发布及后来修订/季调重算的版本链；当前搜索仅找到散点，**未认证** 2018–2022 训练期和 2023–2026 诊断期的连续覆盖。

现阶段 `needs_source`：不从当前可见的晚期 PDF 或事后重算序列倒填历史，不拟合，不开 0bp 账户，也不把来源存在性写成预测能力。若获得每月原始版本和发布时间，先用 Qlib 日历把发布映射到严格晚于实际发布的 Stage1 决策时点；再在 2018–2022 冻结受控 H5 信息任务，与原 39 输入及训练均值对照，分别评估现金优势和 SOXL 相对 SOXX，完整年份/半段/区块下界均通过才准许一条完整 0bp 自身状态账户。复用 2023–2026 仅作开发诊断，独立前向仍须另取。

## 旧留存因子去重

`src/workflow/soxl_rolling_3x_retention_factor.py` 的 `actual_3x_residual_log_21` 为 `log(1+SOXL 21日收益)-3log(1+SOXX 21日收益)`，`retention_z252_21` 只是其历史标准化。原 Stage1 已含 SOXL/SOXX 3/5/10/20/60 日收益、`levered_decay_20`、`soxx_lead_10` 和 5 日跟踪误差的标准化/分位数。21 日留存值仍可能有非线性差别，但没有新来源或不同决策机制；旧模块还以未来 21/63 日残差为目标，不对应当前 Stage1 H5 现金/选择接受门。本轮不把旧模块直接运行或称作新的 Stage1 特征成功。
