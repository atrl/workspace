# 冻结分析优化结论

完成既有现金预测的 Qlib Recorder 分析重放：开发期 2016–2022（1762 行）、复用期 2023–2026-09-23（934 行），各 20/60 日块、2000 次抽样。没有模型拟合或市场回测。

四组共 72,000 个统计值、36 行区间、全部抽样索引 hash、NaN 位置及区间符号通过等价核验；最大绝对差 8.53e-14，声明浮点容差 atol=rtol=1e-12。全部小表与归档 Recorder 结果一致。

完整 analyze_information 中位耗时从 3.4609s 降至 0.5930s，约 5.84 倍；各窗口见 timing_summary.csv，重复样本仅存 Recorder。该结果仅适用于当前主机上的冻结分析，不代表训练或整个平台加速。逐年/窗口科学结论保持原值；未新增事后 regime 筛选。

不晋升。信号有效性与历史时点数据不足的瓶颈仍在：广度路线已失败，财报预测误差未改善且缺少历史版本，旧规则未通过风险门。运行更快不会产生新的稳定超额证据。

原 Recorder、源码归档与执行 seal 保持不变；当前优化源码须由新实验另行绑定，不能重签旧结论。下一步应先取得可信历史时点数据或新的连续样本；工程上仅在新 profile 指向其他热点时继续优化。

Qlib Recorder：a200119a467d4cb197b5dded98a3c28a。输入源：ef3da36b5e6a4f4297c2bf76a6a19afd；provider：data/qlib/published_futu_affine_20260925_v1，as_of=2026-09-25。

归档说明：本次在隔离 worktree 执行，旧输入从完整主仓的 Qlib Recorder 只读加载并校验 hash。执行后将本结论同步到 notebook，补存最终 notebook、结论、测试源码及精确文件清单。仅新 Recorder 的 artifact URI 元数据映射至合并后的主仓位置，详情见 `relocation.json`；归档文件的用途、来源、日期范围及 SHA-256 见 `verification_manifest.json`。
