# 日夜监督配对实验：设计待审

本卡仅完成完整设计草案，frozen=false、implementation=false、fit=false。没有读取实际价格、标签或预测对象，没有生成特征、训练或账户。

唯一机制已在 S/Z 构造前由 owner 采纳：原 8 维共享状态增加同一个 9 参数回归头，B 学训练域标准化 S，C 学标准化 Z；0.02×half-MSE、原一次 globalclip(2) 与 AdamW 不变。新增头在推理时弃用。零头第一步对 encoder 的直接新梯度为零，但其 head 梯度可经全局裁剪影响 allocation 私有参数更新；完整对照评估整个监督包。

来源 C 的具体 Recorder/归档/标签对象及独评身份已绑定，独评接受为条件性训练标签来源。标签是 distribution-adjusted research-price plane，日内共同因子抵消、隔夜包含跨日因子；不是实际费用账户或原 27/63/10 alpha。来源资格不授予模型资格。

预算固定为 6 fits、6 个新等权三 seed 原生账户与 15 个精确复用参考，共 21 账户、15 组配对区间。原 39/16 输入、3014/20741 训练行、H1 allocation 和 H5 分类/共同成熟、三 seeds 与 300 epochs 保持。全部六个终态模型及 SHA 封存后，才允许在另行授权范围内读取评价 Dataset、标签、收益、预测或参考账户对象；不生成或读取评价 S/Z 标签及新价格。

本阶段没有模型或经济指标。未来报告全部成本、完整 2024/2025 与部分 2023/2026、全部规定 126 日窗口；C 必须同时通过 SOXX、新 matched B、旧 B_shared、A、strong9097 的固定门，additional27baa 始终只报告。训练 MSE、分类分数或现金下降不决定是否运行账户，也不能宣称成功。历史复用、PIT=false、前向与完整分钟风险未资格限制保留。

下一步由 owner 交完整设计独评；之后仍需分别批准实现、训练域无拟合预检及实际运行。没有晋升、真实下单、现役 release 修改或参数救援授权。权威为 [design_draft.json](design_draft.json) 与 [三资产研究合同](../../docs/three_asset_research_contract.md)。
