# SOXX 成分篮子领先信息筛查

状态：实现可执行，等待 root 对精确代码、依赖、notebook、设计评审及实现评审签发 execution_authorization.json；尚未读取本卡实际数值来源、拟合或创建实验 Recorder。

notebook 调用 `src.workflow.basket_etf_lead_information.run(root=ROOT)`，以独占 run_state.json 限制一次尝试。2024 原生 Qlib 来源、全 30 成分 PIT 权重及未知行掩码先于新 Recorder；B/C 两次 OLS 的系数和标准化参数逐个立即保存，关闭并读回冻结回执后才允许读取 2025。评价只执行一次固定 stationary bootstrap，不新增账户。

主门要求 gamma>0、2025 配对损失改善均值>0、固定区块置信下界>0。相关性与入场前滞后相关只作诊断。2025 是复用历史；即使信息筛查通过，也不构成独立验证、费用后收益或生产晋升。实际结果、年度行数和 Recorder 身份将在唯一一次授权执行后同步。
