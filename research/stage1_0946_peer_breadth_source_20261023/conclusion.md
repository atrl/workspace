# Stage1 09:46 同业早盘广度来源预检

冻结 AMD、NVDA、SMH 09:35、09:40、09:45 三根已完成的 5 分钟 bar；每只标的取 09:45 收盘相对 09:35 开盘的收益是否为正，再取三只的同涨比例。Qlib 来源核验通过，2018–2022 原 Stage1 训练期 1061/1061 日、2023–2026 复用历史期 944/944 日已知。

原始 Stage1 09:46 配置模型接收的是另一组 39 项输入；本卡只建立新坐标的时钟和覆盖，不读取未来收益、不拟合、不跑账户。09:45 完成 bar 在 09:46 决策前可用仅是封存 bar 时钟假设，历史首收与最终版本未获核证，后续信息检验只能是条件性开发证据。

原生 Qlib Recorder `e7312100d7274be7aca700754c900f0a` 已读回；Yanxuan NAS 归档 SHA-256 `3947e13dfd1decd685992063a2d13cad3144cef2b9ec831df6c9cdcc5e373120` 已远端回读。结论：`conditional_0946_peer_breadth_source_gate_passed_first_receipt_unqualified_no_fit`。下一步另行冻结原 39 项对照加一个坐标的 SOXL−SOXX H5 信息检验；即使通过，仍须来源资格、完整 Stage1 0bp Tiered 独立资金账户及 v2 其他门槛。
