# 前日尾盘有符号杠杆残差：条件性来源与数值去重通过

固定定义为当日收盘前30根已完成 Qlib 一分钟 raw NONE bar 的 `Σlog(SOXL close/open)−3×Σlog(SOXX close/open)`，只供**下一交易日09:29**的 f696 配置器考虑。实际早收盘按封存日程的收盘时间后推30分钟；缺 bar、无效价或源日期保持未知，不填零。杠杆ETF再平衡流可能影响尾盘价格并在次日回转的研究动机见[原研究论文](https://abarbon.com/assets/Liquidity_Provision_to_Rebalancing_Flows_from_Leveraged_ETFs_and_Equity_Options.pdf)；本式是ETF价格残差，**没有测量基金申赎或再平衡流**。

从封存原生Qlib一分钟来源读入 **1,558,560** 条双ETF记录，原分钟卡2008个日期中 **2005个**有双标的有效尾盘坐标；2018–2022训练段 **1060/1061**，2023–2026复用来源段 **945/946**，精确f696前944个信号日 **944/944**。缺值在2018、2020、2026各1日；原39项与四个已有分钟坐标的训练期最大绝对Spearman **0.16781**、最大绝对Pearson **0.29177**，最高相关是旧跟踪误差坐标，非单列复制。43项完整对照见 `result.json`。来源卡Recorder及NAS回读、core_1min Qlib manifest SHA已验证；零未来标签、拟合、账户或外部GET。

初次实现错误要求每个2008日期都存在两个Qlib分钟组，而封存来源本就允许缺失日；失败发生在未来标签与拟合之前。`attempt1_preflight_failure.json`与`source_group_missing_repair.json`保留第一遍本地Qlib读取及第二遍机械修复预算；公式、年份和去重门没有改变。

决定 `conditional_source_novelty_pass_first_receipt_unqualified`。当前历史分钟bar没有逐日首次直播接收或最终版本认证，低相关也不证明预测力。下一步只能另冻一个与09:29→09:47执行钟匹配的单期经济标签信息门；通过后仍需精确f696全链0bp Tiered自身资金账户、2/5bp和独立前向。SOXX稳定净超额尚未证明。
