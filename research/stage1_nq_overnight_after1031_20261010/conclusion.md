# NQ overnight 18:00→10:00 → 10:31 之后收益（0 模型诊断）
- 0 新模型；固定决策 10:31 ET；固定单一信号 NQ overnight 18:00→10:00（由 60min Qlib store 读取，09:00 bar close=10:00，早于决策）。
- 官方交易日 receipt digest 校验通过（sha `d7f5577b2730`）；session_close 由 receipt 的 WHOLE/MORNING 决定（16:00/13:00）。
- 覆盖：official_days_in_range=492，sessions=469，needs_data=23，morning_days=5（有效 0），missing_nq_signal_days=22。
- 执行口径：entry mark is the predeclared historical 10:30 close; decision is 10:31 ET; NOT verified real 10:31 fills；the NQ sign is a directional description; economics are long-only, no short portfolio。
### 主预测目标：SOXL−SOXX 价差 IC（20-session 移动块 CI）
- spread IC +0.0386 CI [-0.0449, +0.1139]
### long-only 日内经济净优势（bps/日，cash→持仓→cash，两腿费用）
- rule_NQpos_SOXL_else_SOXX @10bp：+0.85 CI [-29.07, +32.66]
- rule_NQpos_SOXL_else_SOXX @25bp：-29.11 CI [-58.94, +2.60]
- soxl_only @10bp：-12.09 CI [-46.68, +27.84]
- soxl_only @25bp：-42.01 CI [-76.50, -2.20]
- soxx_only @10bp：-17.31 CI [-29.10, -3.82]
- soxx_only @25bp：-47.21 CI [-58.97, -33.76]
### SOXX 真正 buyhold（跨夜连续账户，独立于 10:30 入场）
- @10bp：147.78%（490 sessions）
- @25bp：147.41%（490 sessions）
### 相对描述：rule − same-window SOXX（同费用完整 roundtrip，非 signed short spread）
- @10bp：+18.16 bps CI [-1.14, +38.29]（仅描述，相对正不能掩盖绝对亏损）
- @25bp：+18.10 bps CI [-1.14, +38.18]（仅描述，相对正不能掩盖绝对亏损）
### 三账户 absolute/net 风险与财富
- rule_NQpos_SOXL_else_SOXX @10bp：cum -24.71% / MaxDD -56.89% / worst -23.05% / CVaR5 -7.84%
- soxl_only @10bp：cum -66.36% / MaxDD -76.69% / worst -23.05% / CVaR5 -11.70%
- soxx_only @10bp：cum -58.23% / MaxDD -60.44% / worst -8.54% / CVaR5 -4.02%
- rule_NQpos_SOXL_else_SOXX @25bp：cum -81.56% / MaxDD -84.56% / worst -23.28% / CVaR5 -8.11%
- soxl_only @25bp：cum -91.76% / MaxDD -93.01% / worst -23.28% / CVaR5 -11.97%
- soxx_only @25bp：cum -89.77% / MaxDD -90.12% / worst -8.81% / CVaR5 -4.31%
- **决定：`end_route_no_further_search`** — no long-only NQ>0->SOXL rule edge with a positive CI at the 25bp roundtrip stress。
- native：full_strategy_native=`needs_data`（root helper covers opening/recovery slots; the 10:30 entry->session_close two-leg roundtrip is not directly adaptable）；已记录 real settle_target_session fee/cash/weight paths per day (normalized account)；不借用之前 control native PASS。
- 本固定机制未获得足够证据（预测 IC 跨 0、25bp 绝对为负）；不反号/改 horizon/阈值，仅停止本路，不宣称“信息不存在”或“模型永无救”。
- NQ=F 为 Yahoo 前月连续序列，roll/调整不构成独立证据。
- 子 Recorder：846542c84829474d9f8c409f670b3829；保存 panel/audit/IC/economic。
