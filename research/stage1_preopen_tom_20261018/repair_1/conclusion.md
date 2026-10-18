# 09:29条件NYSE代理：repair_1无拟合预检结果

获批唯一repair预检74423 exit0。公开CS_RAW保留原分组和列顺序后，原主Dataset3014×85、辅助20741×20全部驻留handler/loader帧均在训练范围；未调用setup_data或反序列化全期评价标签再crop。全部11原训练tensor逐值及SHA一致、原39/16 scaler与三个源模型完全一致，只在主Dataset追加4槽至89列。

三seed初始化核验：保留原973参数对应tensor，新网络1037参数（增加64个零权重）；B/C初始tensor、RNG和合成初始输出一致，原网络与扩展网络输出最大差均0。B/C接收同一canonical centered nullable数据，只由B模型编码把最后一槽置0；unknown原始proxy始终NaN。

|first3 known|last known|proxy known|训练行数|proxy=0|proxy=1|unknown|
|---|---|---|---:|---:|---:|---:|
|False|False|False|999|0|0|999|
|True|False|False|35|0|0|35|
|True|False|True|6|0|6|0|
|True|True|True|1974|1597|377|0|

**固定可辨识性门通过**：相同(1,1,1)mask组内1974行，0/1分别1597/377；另(1,0,1)的6行全true，说明mask本身确含部分日历信息。已知合计1980、unknown1034，全3014行保留；没有按覆盖删年、改mask或搜索新定义。该检查只证明输入不完全冗余，不证明预测或经济增量。

canonical共3966行；946个outer signal、944个account日期和两个非账户tail均保留。2026-10-08使用原2026-10-09 09:29，2026-10-09保持NaT，未补造执行日期。预检记录中的timestamp_only_mapping子对象单独不验3014标签资格；其后的本轮train-only tensor/scaler与共同成熟期检查已经核验该cohort。

已知覆盖按signal_year及decision_year分别列于preflight_coverage.csv。按决策年，2011–2013和2022仍全unknown；2014–2019各年全部所含行known，2020为237/253，2021为233/252，2023为250/250、2024为252/252、2025为236/250、2026为194/194；另1个未知决策NaT行保留。不能拿来源卡2925行conditional计数替代本次新定义known mask。

本次0optimizer step、0fit、0account、0评价标签/收益对象反序列化、0新sourceHTTP。18个原生artifact已独立保存并NAS归档，包含训练/辅助/canonical三套自包含Dataset，保存后直接prepare一致。父卡15060失败的plan/GO/attempt/receipt身份全部未变。

当前结果为eligible_for_formal_design_review，尚未批准六次正式拟合或六个ensemble账户。来源仍conditional/PITfalse，原配置H1和辅助H5共同成熟期不变，没有收益、年度超额或晋升结论。下一步独评合并预检结果与正式设计，并由owner GO同时绑定本prefit manifest/summary后才启动未来6fits+6accounts。

Preflight Recorder：`f0f68379c58b4f859e0fccb981eca00c`；NAS：`55c29a488ad2149ad29bf62c7e4f4ce45ee5c2a020fe5f0fb72be0419301a25c`。原生证据见[preflight/summary.json](preflight/summary.json)及[manifest](preflight/verification_manifest.json)，未修改封存预检summary。
