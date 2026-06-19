# License Review

## 中文结论

- GitHub 来源：https://github.com/microsoft/qlib，仓库 license 复核为 MIT。
- Recorder 文档来源：https://qlib.readthedocs.io/en/latest/component/recorder.html，只保存 URL 和自有摘要。
- 未复制 Qlib 文档正文，未保存未授权正文，未 vendoring Qlib 或第三方实现代码。
- 本 notebook 只验证本仓库已有 `src.analysis.qlib_recorder` 的 artifact 标准化、regime/window 分析和 manifest export 契约。
- 后续如要直接依赖 Qlib runtime 或扩展 recorder 对象，需要单独复核依赖版本、license 和真实 recorder artifact 兼容性。
