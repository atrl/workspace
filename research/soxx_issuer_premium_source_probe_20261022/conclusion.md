# SOXX 发行人折溢价：当前页面不覆盖 Stage1 训练期

[iShares SOXX官网](https://www.ishares.com/us/products/239705/fund)当前 HTML 内嵌逐日 `premiumDiscountChartData`。默认页面及一次指定 `asOfDate=20161230` 的只读页面，各有完全相同的452条记录，日期仅从2025-01-02至2026-10-21。发行人说明该百分比是收盘价相对NAV；这是一项独立测量线索，但本次页面没有2011–2022完整训练历史，也没有原始逐日发布时间。不能把现有只有NAV的历史表与另一个调整后close随意相除。

两个原始HTML、精确URL/字节哈希和结果已归档NAS，远端SHA256 `3d31b8780744556780b84023f9418331da345019feb6872962f4f336e8ecfe50`核对一致。网页参数未改变数组，并不排除另一个官方历史接口；其字段、全期覆盖、份额口径和公布时钟仍需证据。

决定：`needs_original_full_history_and_publication_clock_no_factor`。0预测/拟合/账户，不能算SOXX超额。若有新的发行人原始历史来源，再与原Stage1输入做匹配信息检验；不缩短训练窗口或根据已复用收益选择年份。
