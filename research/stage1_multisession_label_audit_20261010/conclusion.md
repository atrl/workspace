# multi-session label 审计（diagnostic-only，中文结论）
- 0 新模型；不训练候选、不晋升；控制模型从已保存 Recorder 读回（全部年度 hash 绑定），本轮不调用任何 fit。
- plan sha `9541eddd593f`；control recorder `e4d888a42c914bf69660ab78a35ba6e7`；label fee=25bp；父账户成本 10bp 与 25bp 各自独立。
- observed-parent 精确 parity（rtol=0）：{10: {'development_walk_forward': {'status': 'exact', 'rtol': 0, 'atol': 1e-12, 'sessions_equal': True, 'trade_dates_equal': True, 'return': True, 'end_soxl': True, 'end_soxx': True, 'end_cash': True}, 'outer_reused_diagnostic': {'status': 'exact', 'rtol': 0, 'atol': 1e-12, 'sessions_equal': True, 'trade_dates_equal': True, 'return': True, 'end_soxl': True, 'end_soxx': True, 'end_cash': True}}, 25: {'development_walk_forward': {'status': 'exact', 'rtol': 0, 'atol': 1e-12, 'sessions_equal': True, 'trade_dates_equal': True, 'return': True, 'end_soxl': True, 'end_soxx': True, 'end_cash': True}, 'outer_reused_diagnostic': {'status': 'exact', 'rtol': 0, 'atol': 1e-12, 'sessions_equal': True, 'trade_dates_equal': True, 'return': True, 'end_soxl': True, 'end_soxx': True, 'end_cash': True}}}；10bp training-row label parity values/available/status：True/True/True（rows=2690, finite=2688, boundary_unknown=['2022-12-21', '2026-09-23']）。
## cost 10bp / segment development_walk_forward（独立 bootstrap，不跨段拼接）
- eligible available 1529；by year {2016: 201, 2017: 245, 2018: 246, 2019: 248, 2020: 244, 2021: 198, 2022: 147}。
- sign disagreement 0.2492；orig mean 2.54bp vs 5d mean 5.77bp；std 141.6 vs 266.5；Spearman 0.4561747695664339。
- branch risk_state/clock/recovery 分歧率 0.0255 / 0.0255 / 0.0046。
- 严格因果 OOS（20-session block）：{'point': 0.06594034820500863, 'ci_low': -0.02382479194515184, 'ci_high': 0.15102788176638682, 'blocks': 1737, 'status': 'ok'}；by year=2016:None, 2017:0.14692490350798457, 2018:0.13054009991475604, 2019:0.16854803306211655, 2020:0.020598986406948936, 2021:0.03519752276224995, 2022:-0.069373399118996。
## cost 10bp / segment outer_reused_diagnostic（独立 bootstrap，不跨段拼接）
- eligible available 748；by year {2023: 209, 2024: 226, 2025: 219, 2026: 94}。
- sign disagreement 0.1818；orig mean 0.88bp vs 5d mean -9.60bp；std 75.6 vs 166.6；Spearman 0.5760181010914566。
- branch risk_state/clock/recovery 分歧率 0.0134 / 0.0134 / 0.0000。
- 严格因果 OOS（20-session block）：{'point': 0.06673944516132398, 'ci_low': -0.06507386301796853, 'ci_high': 0.19591268946114151, 'blocks': 915, 'status': 'ok'}；by year=2023:0.2190138808559861, 2024:-0.18292048393175056, 2025:0.13499491585453635, 2026:0.03935267131452516。
## cost 25bp / segment development_walk_forward（独立 bootstrap，不跨段拼接）
- eligible available 1513；by year {2016: 201, 2017: 245, 2018: 246, 2019: 248, 2020: 243, 2021: 189, 2022: 141}。
- sign disagreement 0.2479；orig mean 2.92bp vs 5d mean 5.47bp；std 140.8 vs 264.3；Spearman 0.4580496772549183。
- branch risk_state/clock/recovery 分歧率 0.0264 / 0.0264 / 0.0046。
- 严格因果 OOS（20-session block）：{'point': 0.07290839931662547, 'ci_low': -0.01602686308928399, 'ci_high': 0.15834184459274145, 'blocks': 1737, 'status': 'ok'}；by year=2016:None, 2017:0.14686451949832313, 2018:0.1304490118354046, 2019:0.16845755753849928, 2020:0.03153241305253227, 2021:-0.06796779181997548, 2022:-0.20789845455713002。
## cost 25bp / segment outer_reused_diagnostic（独立 bootstrap，不跨段拼接）
- eligible available 736；by year {2023: 209, 2024: 217, 2025: 215, 2026: 95}。
- sign disagreement 0.1889；orig mean 2.72bp vs 5d mean -5.95bp；std 74.1 vs 158.6；Spearman 0.5756907966419414。
- branch risk_state/clock/recovery 分歧率 0.0217 / 0.0217 / 0.0000。
- 严格因果 OOS（20-session block）：{'point': 0.06408862760489718, 'ci_low': -0.06367681783251708, 'ci_high': 0.18942211407033593, 'blocks': 915, 'status': 'ok'}；by year=2023:0.19783505967716494, 2024:-0.19398878225454136, 2025:0.15137530488541137, 2026:0.05393337066069429。
- native：needs_data（no native cash/quantity reconciliation executed this diagnostic round）。
- 结论边界：本卡只诊断标签口径差异与因果 OOS 信息，未证明新 alpha；outer 为 reused_diagnostic（非 fresh）。
- 下一动作：read-only diagnostic complete; do NOT train a 5-session candidate without a separate pre-registration; outer is reused_diagnostic, never a fresh out-of-sample pass。
- 子 Recorder：5adde25b815042a78396492ea71169d1；parents/labels/trajectories 按 cost 分别保存。
