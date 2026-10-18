# SEC uncertainty 来源设计：待审，未执行

本卡只登记一个机制的来源资格设计：固定金融不确定性词表下，原始年度 10-K 措辞占比的同比变化；最终只拟作为当时已披露 SOXX 篮子代理。当前 frozen=false，未下载词表或申报原文，未解析实际语料、计数、生成特征、拟合或回放。

找到一个可固定的候选镜像：2018 命名的词表位于 2020-11-10 commit c74b5019ff75979c5163839dca6537a96ba40777，Git blob e349eaa4c5c10fb50a37dd2072da2a2ccbb9f60f。只读了 metadata，原字节 SHA/词表语义尚未资格；作者当前条款与本用途是否相容仍待明确，public mirror 不等于无限制授权。不用当前 2026 版本按年份过滤替代，也不要求字典早于每份 2011 训练文本，只要求固定训练组件不含 2023+ 信息。

A 最多 2 次请求核词表/使用条件，许可仍 unresolved，不推定 academic。B0 是可独立审批的离线步骤：只复用已封 NPORT/CIK metadata 与原生 calendar，锚定固定 corpus 中 2019-09-30 报告／2019-11-25 披露的篮子，strict next session 为 11-26，再按唯一 CIK 排序封存两家候选和身份缺口。B1 当前 needs_contact_config，真实 SEC User-Agent 配置与单独 GO 后才可用最多 10 次 metadata/header 请求；保持 anchor 后首个 original10-K 加紧前 distinct annual 的配对规则，并以连续日期覆盖证明不是 first found。C 另获 GO 后至多 4 份原文，只核解析，不计算词频。各阶段不自动扩成全语料。布局 table 正文保留；隐藏 XBRL、附件、正文与表格分层留证，不盲删所有 table。

09:29 是信息截止，09:47 是同一 trade session 的执行（相对 signal_date 为次日），不能因此偷看中间新披露。日期级来源采用事前固定的保守 next-decision 约定并标 conditional，不称 first-receipt。原件、10-K/A、PAC 和当前抓取日期分别保留。

2011–2018 未有当时披露篮子证据的范围、外国非 10-K 发行人与缺失原文仍是 gap，不用当前成分回填、不重归一化冒充全篮子、不删原 3014 行。四文档 pilot 即使通过，也不认证全行业覆盖、可交易输入或经济成功；本阶段没有模型/收益指标、无晋升资格。

下一步由 root 与独评审 [source_design_draft.json](source_design_draft.json)，逐阶段决定是否 GO。A 许可尚未决不遮蔽 B0；B1/C 的网络许可和真实配置也不借 B0 自动升级。完整费用后稳定胜 SOXX、风险、年度/126 滚动、CI 和充分独立前向门保持原合同。
