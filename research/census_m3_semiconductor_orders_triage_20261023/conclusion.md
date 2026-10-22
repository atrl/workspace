# Census M3 半导体订单候选的来源语义排除

为 Stage1 寻找区别于已失败的 G.17 产量信号的需求端信息时，检查了美国人口普查局 2011、2018、2026 年的原始 M3 月报。三份报告的 Table 2/3 注脚一致：**半导体行业的新订单和未完成订单数据不可得，并被相关估计排除**。出货和库存只进入更大的汇总行业，不单列半导体。[2011 年报告](https://www.census.gov/manufacturing/m3/historical_data/pressreleases/prel/2011/mar11prel.pdf)、[2018 年报告](https://www.census.gov/manufacturing/m3/historical_data/pressreleases/prel/2018/apr18prel.pdf)、[2026 年报告](https://www.census.gov/manufacturing/m3/historical_data/pressreleases/prel/2026/jan26prel.pdf)及[历史序列说明](https://www.census.gov/manufacturing/m3/historical/timeseries.html)。

决定：关闭“M3 电子元件订单/积压订单就是半导体订单/积压订单”的来源假设，不构造 Stage1 因子。更宽泛的非半导体电子元件需求是另一命题，须重新冻结来源与接收任务。本次仅核查官方页面和 PDF 提取文本，没有接收原始 PDF 字节、建立 PIT 序列、拟合模型或运行账户；SOXX 目标仍未实现。
