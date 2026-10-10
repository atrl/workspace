# TSMC 月营收 PIT 信息数据面（源资格，model=0）
- source_gate=`pass`，decision=`source_plane_qualified_information_only`；0 model / 0 returns / 0 IC；predictive_not_tested=True（信息预测效果本轮未测，不claim alpha）。
- 因子：`reported_yoy_acceleration_pct_points`；clock=max(current,prior) 原文可用性→严格下一 US session；revision 契约=`first_as_filed_only`。
- coverage：expected 130，present 130，missing []，not_admitted ['2015-10']，numeric_gate_failures []，complete=True。
- 维度：scope_extra_not_admitted / precision / clock / model_gate / production_collector(needs_data)。
- 边界：非交易价格信息 instrument；发布日期不假定每月10日；不使用 consensus/beat 口径；不扩窗口、不为过门缩范围。
- 下一动作：先独立评估该信息因子的前向预测/经济价值（predictive_not_tested），不得据此直接进入 production。
- 子 Recorder：4d1dc18e75c7489d8251af1e737c1a0f。
