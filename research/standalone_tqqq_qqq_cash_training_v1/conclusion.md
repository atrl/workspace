# 结论

- 本实验只从 `data/qlib/tqqq` 读取 TQQQ 与 QQQ；训练、开发和outer数据中 SOXL/SOXX 行数均为0。
- 开发期选择 `tqqq_own_turnover_0p1`；outer 10bp收益 162.75%，QQQ 163.60%，财富比 0.9967；25bp财富比 0.9869。
- MaxDD -25.33%，最差五日 -11.85%，平均现金 18.22%，换手 7.60；动态/同平均仓位静态财富比 1.0032。
- 开发期完整门：未通过；晋升：`not_promoted_standalone_tqqq_gates_failed`。
- Outer仅报告，未参与候选选择；任何outer更好的未选候选都不能事后替换开发期选择。
- Qlib recorder：`1a90171034d14ba4a7e5259ae2cd29bc`。
