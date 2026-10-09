# Stage1 费用一致动作训练（增量 2，中文结论）
- 做了什么：冻结 own-state 面板训练两本书（10bp 书=自身 state+label10bp；25bp 书=自身 state+label25bp），与冻结参照 active/no_hold/own_state_frozen/rules_only 分组比较。plan sha `4bc91c1a4ef7`。
- 控制 parity（新 wrapper label25/eval10 全路径复现源 control，含 returns/endweights/decisions）：passed。
- 结果按 **segment × cost** 分组，不合并；真实门与年度/财富下限门按原阈值计算，失败如实列出。
- 本执行**不依赖价格一致性接口**；增量 1（价格口径）留待原生接口就绪后单独跑并单独报告。
- 门阈值（复用原卡）：{'annual_wealth_floor': 0.985, 'open_trade_days_reduction': 0.92}。
- 候选 not_promoted：缺完整 native intraday 对账，历史门不自动授权晋升。
- 子 Recorder：a100fa4745f4421c9dface98c2921ae9；保存 models/scaler/training/maturity/predictions/replays，源只读不改。
