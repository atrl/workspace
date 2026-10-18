# 09:29 条件日历接收设计草案

本卡只完成设计，frozen=false；没有生成日历特征、读取原收益、拟合或回放账户。repair_1来源身份已登记，来源独评已接受，接收设计批准仍待绑定，不继承source执行许可。

唯一窗口固定为{−1,+1,+2,+3}，研究日历事前选NYSE。09:29只使用d之前的SOXL/SOXX共同session快照前缀和当时已知的NYSE普通计划，不要求d已经开市，也不把共同session称为官方NYSE开市记录。三态OR保留：任一已知真即真，两支已知假才是假；缺公告、2022修订缺口、冲突及prefix缺口不被抹去。不会使用source的qualification_status或current_date_in_observed_calendar作新mask。

拟保持原39/16输入、共享主辅模型、损失、300epochs、3seeds及3014/20741成熟训练cohort。B/C同shape、三availability mask和初始化；C仅增加一个固定centered日历数值，B该槽中性。unknown在规范源中仍为NaN，模型latent0只是encoding。预计6fits、B/C各一个等权3seed集合在三成本下共6个新增独立费用账户（单seed仅预测/分类诊断，不建账户），精确复用15参考账户，保留946预测行与944账户区间。

目前没有经济指标或分窗结果；不得把来源覆盖或现金下降当成稳定超额。完整继承0/2/5bp、SOXX及强参考、MaxDD/CVaR、完整年度和126滚动70%、配对CI等门。PIT与独立前向资格仍false，不晋升，不改变生产。

下一步：负责人审查并冻结接收定义和完整实验合同后，才另行授权最小预检及经济执行。

独立前向完整条件见[持续研究合同](../../docs/three_asset_research_contract.md)：冻结全决策/来源/成本与观察时点后，至少60完整配对日、各预登记市场状态15日及3次转换仅是观察下限；年度与126日资格仍需真实足量的新观测，不因达到60日自动通过。

设计预检补充：同一三mask组合内必须在原成熟训练域同时出现已知0和1，否则calendar值只是mask的确定函数，按不可辨识停止。B已从可用性模式获得部分日历信息，C只检验其外的数值增量。训练必须使用已核验train-only对象，不能先反序列化含outer收益的大对象再声称未读；6个模型身份封存后才访问评估对象。异常非空时钟视为来源完整性失败。

独立设计评审已通过，root仅批准实现与聚焦合成验证。真实日历值、训练对象预检、6次拟合及账户回放仍须最终代码/来源绑定与分阶段GO。见[design_draft_review.json](design_draft_review.json)及[implementation_authorization.json](implementation_authorization.json)。
