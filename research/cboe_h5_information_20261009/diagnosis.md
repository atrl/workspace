# CBOE H5 失败归因与下一独立机制

状态：ongoing；全部结果为复用历史的事后诊断。来源是 Qlib Recorder `0d98dcb2d9e64360b8cf126c7aa3b32b` 的 matched_frame、annual_fits、matched_predictions、paths 和按 hash 读取的压缩 solver_details；没有新拟合、优化求解或回测。完整小表见本目录 CSV。

## 训练改善没有稳定外推

两模型的每年训练日期与保存审计逐条核对，首折476行，最后2990行。使用保存的标准化参数与系数计算训练预测；后续年度直接使用保存预测。CBOE模型22个年度/资产训练 MSE 全部优于 matched BASE，但后续只有2020和2025两个年份、两个资产同时改善，其余9/11年均恶化。

2016年 SOXL：BASE训练/次年 MSE=0.007195/0.008930，full=0.006222/0.012339；full预测标准差从训练0.0363升至后续0.0526，后续校准斜率−0.1483。2017、2022分别出现明显负/正预测bias，说明统一放大或缩小单一系数不能保证跨年修复。

开发期SOXL BASE/full/prior MSE=0.016033/0.016408/0.014652，SOXX=0.001756/0.001796/0.001619；两个学习模型均未超过共同过去均值。full pooled校准斜率SOXL−0.0657、SOXX0.00225，仅为事后描述，不能据此翻号或设新缩放。

令delta=full−base、residual=actual−base，MSE改善恒等为 2E[delta*residual]−E[delta²]。开发SOXL前项+0.000347、后项0.000722，合计−0.000375；新增变化的幅度惩罚超过其残差信息收益。后段前项已为−0.000096，不能解释为仅“信号正确但尺度稍大”。full−base包含旧价格系数重拟合变化，不是单个外部字段的因果效应。

开发误差恶化主要来自2016/2017（SOXL SSE增加0.8592/0.3706），2020抵消0.8085；2022又恶化0.1589。不能删除这些年份或只保留2020。后段2023/2024/2026恶化，仅2025改善。

## 组合失败不是主要由费用造成

25bp full开发期相对SOXX财富0.6982；净log超额−0.3592=固定实际持仓的毛log超额−0.3367+增量费用−0.02253。后段财富0.6509；−0.4294=−0.4021−0.02722。毛路径保留原实际持仓，不是重新优化的无费用策略；分解最大日记账误差3.23e−12、窗口log误差1.87e−11。

开发主动持仓集中2016–2019，full25bp平均现金权重分别49.2%、79.4%、62.8%、62.0%；2021–22完整退回SOXX。后段主要2023与2026，2025退回SOXX。开发211次成熟主动调仓，平均预测毛H5主动优势+20.33bp、实际−16.49bp；后段76次，+39.49bp对−55.89bp。事件只作状态归因，不取代完整组合结果。

高成本改变了仓位路径，不能把10/25bp差异当作同一持仓简单多扣费。保存的首个solver状态显示从现金出发，两模型25bp均选择约33.05%SOXL+66.95%cash，在当期协方差下几乎用满SOXX风险预算；初次买入费约8.26bp，明显少于买入全SOXX约24.94bp。这解释了费用与初始路径依赖的渠道，未证明它解释所有后续亏损。solver记录future_transactions_assumed=False；HOLD5只匹配首个五会话持有，不代表模型规划了未来切换成本。

## 同类历史研究去重

当前src/卡片、隐藏卡文件名及相关git对象的定向检索没有找到“同一25价格字段、真实open-to-open H20/H63标签、相同H20/H63持有、严格成熟年度训练、原生Qlib组合对账”的完整既有实验。此结论是未找到直接证据，不是穷尽全部历史后证明不存在。

- `4d0ec3ef:src/data/convex_allocation.py` 明确HORIZON_SESSIONS=5、25字段、Ridge alpha10；不是H20/H63。
- `c1a77947:research/oos_uncertainty_horizon_20261009/plan.json` 三方案均用H5，仅改变daily/HOLD5及OOS误差下限。
- `74584f35:research/soxl_3x_state_factor_lab_v1/summary.json` 是21日close-to-close目标、4基线字段加候选、三大fold和固定Ridge10；六候选均未过稳定增量门，不是当前25字段/匹配执行实验。
- `0c62c782:research/soxl_long_horizon_excess_drawdown_v1/summary.json` 明确descriptive_path_dependence_confirmed_not_entry_signal，无模型拟合。已有21/63期描述不能当作H20/H63策略验收。
- `b5bebdc6:research/stage1_regime_deployment_20261001/selection_plan.json` 有Ridge10/100，但预测的是一会话execute-versus-hold动作优势，按整个开发期组合门选候选；不是每个外层年度内做purged时间验证选择当前25特征收益头alpha。
- `src/workflow/stage1_controller_aware_distillation.py` 也有alpha10/100与shrinkage/阈值联合候选，同样不是该独立正则化协议。当前src未发现RidgeCV/alpha_grid的同类实现。全仓历史RidgeCV字符串搜索因耗时已停止，不能声称该搜索完整。

## 下一问题应分别登记

直接证据支持先检验“严格过去的时间验证能否拒绝训练改善但外推失败的固定Ridge10”，作为独立正则化选择机制。固定少量alpha与显式过去均值/null，scaler内层fit、按标签成熟purge，外层年度只评价，保持H5/特征/allocator不变；须记录全部内层候选和失败，并区分预测prior与投资组合精确SOXX，二者不是同一个null。当前alpha对应SSE正则，alpha/n会随扩展样本变化，内层协议应明确惩罚单位。

另一项尚未找到直接检验的机制是单独改变预测/执行至较长匹配期限，但不能同时做H20/H63、alpha、CBOE特征的组合网格。更长持有可能摊薄交易成本，也可能增加重叠标签、估计误差与中途风险漂移；不能依据当前亏损就宣称长期限会产生alpha。本报告不选择候选、不翻号、不提供晋升或可达标承诺。
