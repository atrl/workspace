 在当前 repo 内重建 SOXL 全平台，不再考虑强化学习。核心目标是把可复用能力沉到 src/，把研究和生产方案放到 notebook 中维护；skill 只保留“方法论/发散思考/研究流程抽象”，不承载具体业务功能。

  新增 src/analysis：专门负责 qlib 扩展分析、不同时间区间和 regime 下的模型效果差异、模型归因、稳定性诊断和横向对比。

  ## 核心架构

  - src/core: 稳定契约、注册表、schema version、manifest、hash、配置、校验。
  - src/data: 数据源、数据更新、qlib/calendar 对齐、canonical snapshot，最终写入 data/。
  - src/factor: 单因子、多因子、IC、相关性、分层回测、因子组合诊断。
  - src/model: QuantBench 模型注册、训练、预测、保存、加载、模型能力说明。
  - src/objective: 分类、MSE、IC、ranking、收益风险 utility、晋升评分；替代原 reward 概念。
  - src/backtest: 信号转仓位、成本、滑点、walk-forward、压力窗口、基准对比。
  - src/analysis: 模型/因子/回测分析层，支持 qlib analyzer 扩展、regime 分层、区间对比、模型差异解释。
  - src/dashboard: 只读取统一 manifest/export，不直接读散落实验文件。
  - src/schedule: 数据更新、模型刷新、dashboard export、通知、锁、重试。
  - src/api: broker/data adapter contract，v1 只实现 paper/sim。
  - research/<experiment>/: 研究中实验，每个实验一个主 .ipynb。
  - production/<strategy>/: 晋升后的可运行方案，每个方案一个主 .ipynb。

  ## src/analysis 设计

  - qlib 集成：
      - 封装 qlib recorder/analyzer 读取、portfolio metrics、risk analysis、report 数据。
      - 提供统一 AnalysisReport，让 dashboard、notebook、promotion 都读同一套结果。

  - Regime 分析：
      - 支持按 VIX、趋势、回撤、波动、成交量冲击、牛熊/震荡等 regime 切片。
      - 每个模型输出全区间表现 + 分 regime 表现 + regime 间稳定性差异。

  - 区间对比：
      - 固定窗口：YTD、近 1/3/6/12 个月、2020、2022、2023-2024、用户自定义窗口。
      - Walk-forward：每个 fold 的收益、回撤、Sharpe、turnover、暴露、相对 SOXL。

  - 模型差异：
      - 对比不同模型、不同 objective、不同因子集、不同训练窗口。
      - 输出 winner/loser 不是只看收益，还要看回撤、换手、压力期暴露、稳定性。

  - 解释层：
      - tabular 模型支持 feature importance / permutation / SHAP 可选。
      - deep/sequence/graph 模型先支持输入敏感度、窗口贡献、regime 表现差异。

  - 产物：
      - analysis_summary.json
      - regime_metrics.csv
      - window_metrics.csv
      - model_comparison.csv
      - failure_modes.md 或中文 notebook 结论 cell。

  ## Skill 边界

  - skill 不实现业务逻辑，不保存项目状态，不作为功能入口。
  - skill 用途：
      - 抽象研究方法。
      - 提醒如何提出假设、拆因子、设计 regime、解释模型差异。
      - 帮模型发散思考候选方向。

  - 具体可运行功能必须放在：
      - src/ 的稳定模块；
      - research/<experiment>/*.ipynb 的实验流程；
      - production/<strategy>/*.ipynb 的晋升方案。

  - 如果 skill 里出现具体路径、模型结果、固定参数或项目状态，应迁回 notebook 或 src/core/manifest。

  ## Notebook 规则

  - research/<experiment>/<experiment>.ipynb: 研究入口，包含假设、数据、训练、回测、analysis、结论。
  - production/<strategy>/<strategy>.ipynb: 生产方案，包含固定参数、回放、运行输入输出、风险说明。
  - 每个 notebook 最后必须有中文结论 cell：做了什么、核心指标、regime 差异、是否晋升、风险、下一步。
  - CSV/JSON 是 sidecar，不能替代 notebook 结论。

  ## 测试计划

  - 架构测试：允许新 src/, research/, production/，替换当前 retired surface 测试。
  - 契约测试：schema version、manifest、hash、序列化稳定性。
  - 数据测试：calendar 对齐、snapshot、防未来函数。
  - 因子测试：单因子 IC、多因子相关性、组合回测。
  - Objective 测试：classification/MSE/IC/ranking/utility score 输出一致。
  - 模型测试：QuantBench registry 完整性，每个模型 family 有最小 fit/predict smoke test。
  - Backtest 测试：固定窗口、walk-forward、成本滑点、benchmark 对比。
  - Analysis 测试：regime 切片、窗口对比、模型差异表、qlib report 解析。
  - Dashboard 测试：只读 manifest/export。
  - Schedule/API 测试：job lock、retry、paper broker target-position flow。

  ## 假设

  - 不再做强化学习，不保留 RL reward。
  - reward 改为 objective。
  - 中文是默认文档和 notebook 结论语言，代码命名仍用英文。
  - v1 不做真实 broker 下单，只实现 adapter contract 和 paper/sim。
  - src/analysis 是正式框架能力；skill 只做研究思路抽象。
