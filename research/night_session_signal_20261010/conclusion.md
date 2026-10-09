# 夜盘信号稳定超额研究（中文结论）

- 新数据：`data/raw/external/night_session_20261010_v1`，Yahoo Finance 1 小时含盘前盘后（2024-10-09 至 2026-10-09），含 SOXL/SOXX/SMH/NVDA/TSM 盘前及 ES/NQ 股指期货夜盘。
- 因果对齐：期货隔夜收益取前一日 18:00 ET 至当日 09:30 ET；盘前取 04:00–09:30 ET；两者都在开盘前已知；交易标的为当日 09:30–16:00 regular session，基准为 SOXX 同一窗口。
- 信息系数（对 SOXL 与对 SOXL−SOXX 超额）：nq_overnight(SOXL +0.220, 超额 +0.224)；es_overnight(SOXL +0.138, 超额 +0.144)；soxl_premarket(SOXL -0.029, 超额 -0.025)；soxx_premarket(SOXL +0.001, 超额 +0.004)。
- development 稳定超额信号：['es_overnight__cash', 'nq_overnight__cash']；选中 `nq_overnight__cash`。
- 选中信号 development 10bp：收益 157.25%（SOXX -0.20%），Sharpe 差 +1.27，回撤改善 -3.4pp，年换手 235。
- 诊断区间 25bp：收益 168.89%（SOXX 11.85%），Sharpe 差 +1.62。
- qlib 原生：数据经 `data/qlib/night_session_20261010_v1` 由 `qlib.data.D.features` 读取，策略由 `qlib.backtest.backtest`+SimulatorExecutor 执行；原生对账状态 passed，最大逐日误差 4.4e-16。
- 季度一致性（高成本）：8/9 季度跑赢 SOXX。
- 是否晋升：True；决定 `stable_overnight_excess_candidate`。
- 风险：样本仅约两年，诊断段短；期货夜盘价为连续合约收盘，不保证可成交；换手高（年约 120-240 次），更宽价差会侵蚀边缘；隔夜价已部分反映上一美股收盘。
- qlib Recorder：6719efdbc2734e999feb282bcf7cd42e；qlib docs：['https://qlib.readthedocs.io/en/latest/component/data.html', 'https://qlib.readthedocs.io/en/latest/component/recorder.html']。
