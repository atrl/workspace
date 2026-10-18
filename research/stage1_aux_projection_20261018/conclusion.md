# Stage1辅助梯度投影：固定机械预检

完成原3个seed、初始化/已保存终态、辅助抽样序号[0,99,199,299]的全部24个probe。**1个可辨投影、23个恒等、0数值或机械失败**；全973参数和共享154参数拆分梯度误差均为0，所有参数前后哈希相同。决定：**resolvable_intervention_requires_separate_training_design**，不自动启动训练。

唯一可辨项是seed273的已保存终态、抽样序号0：共享P/A余弦约−0.07102，投影后原dtype位移约2.47824e−4，高于预登记筛查界约2.27e−8。其余23项保持恒等。12个初始化锚点全部恒等；12个终态锚点只有上述1项可辨。这不是训练轨迹或冲突频率估计，不证明旧的全辅助集合正余弦诊断错误，也不证明负迁移。

P的原始方向在全部24项中均为正，对照和候选相同；配置损失单项的全参数方向有4项为负，共享方向有21项为负，对照与候选的符号计数相同。唯一可辨项的配置单项方向仍为负，不能把合成P的方向保护解释为配置收益优化。全部probe的裁剪系数均为1，本组未覆盖实际触发裁剪的情形。

只使用原3014条成熟主训练样本和20741条辅助样本，2011–2022训练域；完整源对象哈希校验后立即按日期裁剪，成熟切片进入原生Qlib DatasetH。未加载outer_predictions，未使用2023+评价收益或标签。**0优化器步、0拟合、0账户、0新市场HTTP**。未创建、推断或重建AdamW状态；实际更新步保护未知。

首次notebook启动因MLflow文件后端需要MLFLOW_ALLOW_FILE_STORE=true而在首个源对象load前退出，0probe。保留preprobe_failure_1.json和environment_resume_receipt.json；仅补环境后续行同一冻结计划，未改源码、数学、数值界、抽样或来源，也未重跑任何probe。

负责人完成一次NAS冷恢复：96个artifact哈希及24行结果全部吻合。归档的两个Dataset直接prepare失败（DataHandlerLP缺少_infer缓存），**不能称归档Dataset可直接读回**。负责人用固定原来源和冻结代码只读重建原生DatasetH，得到3014/20741行；全部重建tensor与本卡归档tensor逐元素及SHA一致。现场原生Dataset运行有效与归档pickle可独立prepare是不同资格。详见readback_verification.json；原Recorder及NAS归档未重封，0新增probe/fit/account。

这些工程精度筛查不构成真实梯度误差界、泛化、交易或独立前向证据。现役release未改，不晋升，稳定费用后SOXX超额目标未实现。下一步只交独立结果评审；如继续，必须另行冻结完整训练和账户设计。ROD仍为主线。

Recorder：`7f2db7d7b3d54b9db0460489f8d2bc9f`。小表为probe_results.csv、direction_metrics.csv；训练切片、源码、模型、全部梯度及抽样索引仅存Recorder。NAS身份、来源和交付限制见summary.json、execution_identity.json、readback_verification.json。

独立结果评审：`ACCEPT_MECHANICAL_RESULT_ONLY`。24格机械证据获接受；正式训练与账户设计仍须另行冻结和审查，未授权拟合或晋升。见mechanical_result_review.json。
