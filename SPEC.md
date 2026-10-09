# SPEC.md

在当前 repo 内重建 SOXL 全平台，不再考虑强化学习。核心目标是把可复用能力沉到 `src/`，把研究和生产方案放到 notebook 中维护；skill 只保留方法论、发散思考、研究流程抽象，不承载具体业务功能。

本 SPEC 的最高优先级约束是：**必须使用 qlib 原生工作流**。只做 qlib-like 文件布局、只用 pandas 读取本地 CSV、只手写 calendar 对齐、只自定义 backtest/recorder，都不算满足本项目重建要求。

## Qlib 强制合同

所有数据、训练、回测、分析、晋升和生产 paper/sim 工作都必须以 qlib 为系统入口。

- 每个 research/production notebook 必须显式执行 `qlib.init(...)`，并在 `summary.json` 或 notebook 结论中记录 `provider_uri`、`market/region`、`as_of`、数据区间和 qlib recorder 信息。
- 数据读取必须通过 qlib Data API 或 Dataset API：
  - 日历、标的、特征读取优先使用 `qlib.data.D.calendar`、`D.instruments`、`D.features`。
  - 模型训练/验证必须使用 qlib `DatasetH`、`DataHandlerLP` 或项目内对这些 qlib Dataset/Handler 的薄封装。
  - 本地 CSV、raw parquet、broker export 只能作为构建 qlib store 的输入，不能作为研究/训练/回测的最终数据源。
- qlib store 构建必须有明确门禁：
  - raw CSV/DataFrame/parquet 只能进入 `src/data` 的 qlib store builder。
  - builder 必须写出 qlib provider 结构、calendar、instrument、feature bin 和 manifest。
  - builder 完成后必须通过 `qlib.init(...)` + `qlib.data.D.calendar`/`D.instruments`/`D.features` 读回审计。
  - manifest 必须记录 source、provider_uri、market/region/freq、as_of、字段、标的、数据区间、feature 文件数、hash 和使用的 qlib API。
  - raw 数据日期超过 `as_of` 时必须 fail closed，不得截断后继续。
- 实验证据必须使用 qlib Workflow/Recorder：
  - 使用 `from qlib.workflow import R` 和 `with R.start(...)` 创建实验记录。
  - 参数、指标、模型、预测、portfolio report、risk analysis、sidecar manifest 必须通过 qlib recorder 写入或登记。
  - 每个 notebook 的 `summary.json` 必须包含 experiment name、recorder id、artifact uri/path、qlib docs reference。
- 回测和分析必须使用 qlib Record/Report 输出作为主证据：
  - 信号分析使用 qlib `SigAnaRecord` 或等价 qlib analyzer 输出。
  - 投组回测使用 qlib `SignalRecord`、`PortAnaRecord` 或 qlib backtest/report 输出。
  - `src/analysis` 可以把 qlib recorder/report/risk_analysis 包装成稳定项目报告，但不得绕过 qlib 直接把散落 CSV 当作主分析来源。
- `src/` 只能封装 qlib，不得替代 qlib：
  - `src/data` 负责 qlib store 构建、provider_uri 校验、calendar/instrument/feature 审计、snapshot manifest。
  - `src/model` 负责 qlib Dataset 到模型训练/预测/保存/加载的合同。
  - `src/backtest` 负责 qlib backtest/report 结果的稳定接口和项目侧压力窗口扩展。
  - `src/dashboard`、`src/schedule`、`src/api` 只能读取 qlib recorder 或由 qlib recorder 登记的 manifest/export。
- 如果 qlib 不可用、provider_uri 不存在、recorder 写入失败、Dataset 无法构建，任务必须 fail closed，并明确写出 `blocked: qlib unavailable` 或具体缺失项；不得降级为 pandas-only 或 fake qlib。

## 产物合同

qlib research/production card 必须使用瘦根目录、recorder-first 的产物结构。

- 每个 card 根目录必须保留：
  - 主 notebook：`<experiment>.ipynb` 或 `<strategy>.ipynb`。
  - `workflow.yaml`：记录 provider_uri、recorder_uri、market/region、segments、model_selection 和 artifact_policy。
  - `summary.json`：机器入口，必须包含 recorder id/path、qlib artifact 列表、sidecars、benchmark、model_selection 和 artifact_policy。
  - `conclusion.md` 和 notebook 中文结论 cell。
  - 小型 sidecar：`selected_model.json`、`model_selection.csv`、`risk_analysis.csv`、`backtest_benchmark_comparison.csv`、`backtest_window_metrics.csv`、`backtest_walk_forward_metrics.csv`、`analysis_summary.json`、`regime_metrics.csv`、`window_metrics.csv`、`model_comparison.csv`、`explainability.csv`、`failure_modes.md`。
- `summary.json.artifact_policy.policy` 必须是 `slim_card_recorder_first`。
- `summary.json.artifact_policy.recorder_only_artifacts` 必须明确指向 qlib recorder 中的：
  - `artifacts/pred.pkl`
  - `artifacts/label.pkl`
  - `artifacts/model_selection/all_model_predictions.csv`
  - `artifacts/portfolio_analysis/report_normal_1day.pkl`
  - `artifacts/portfolio_analysis/positions_normal_1day.pkl`
  - `artifacts/portfolio_analysis/port_analysis_1day.pkl`
- 以下文件不得作为 card 根目录持久产物提交，除非任务明确要求临时导出且提交前删除：
  - `predictions.csv`
  - `signals.csv`
  - `all_model_predictions.csv`
  - `portfolio_report.csv`
  - `qlib_portfolio_report.csv`
  - `qlib_positions.csv`
- `model_results.csv` 不是默认产物。模型比较应统一写入 `model_selection.csv`、`model_comparison.csv` 和 qlib recorder；只有外部系统明确要求单文件聚合时才生成，并必须在 `summary.json` 说明用途。

## 晋升语义

- `research/qlib_native_reference_v1` 类 card 是 qlib 原生链路验收，不承担生产晋升。
- search card 可以是诊断卡：允许记录没有跑赢 benchmark 或 legacy SOXL rulebase 的候选，但结论必须写清“不晋升”和失败原因。
- validated card 是晋升前研究门：必须固定评估窗口、直接 benchmark、成本/滑点、风险说明，并记录是否超过直接比较基准。
- production paper/sim card 是生产门：必须复用 validated 参数或明确记录固定参数来源，直接从 qlib recorder 读取预测/组合证据，不读取根目录大 CSV。

## 禁止的伪 qlib 实现

以下实现一律不算完成：

- 只创建 `data/qlib/...` 文件夹但没有调用 qlib API。
- 只解析 `calendars/day.txt` 或 `.bin` 文件但没有通过 `qlib.data.D` 读取。
- 只用 pandas DataFrame 完成训练、回测、分析，之后把结果命名为 qlib artifact。
- 自己实现 recorder/experiment id，而不是使用 qlib `R` 或 qlib recorder。
- notebook 只写 “qlib calendar” 字样，但没有 `qlib.init`、Dataset、Recorder、Record/Report 的可执行证据。
- skill 中出现具体路径、模型结果、固定参数或项目状态。

## 核心架构

### SOXX 稳定超额研究入口

新一轮基准相对研究使用 `research/benchmark_relative_architecture_20261009/`
作为可执行参考，方法与外部机制依据分别保存在卡片的 `architecture.md`、
`research_sources.md`；参数和候选预算以执行前冻结的 `plan.json` 为准。
这是一张历史诊断卡，不构成生产策略或稳定超额已实现的声明。

- `src/data/benchmark_relative_data.py` 通过 Qlib Data API 和 DatasetH 提供
  独立的信号、成交、结算与反馈可用时点；不以缺失数据截短共同样本。
- `src/model/online_expert_allocation.py` 只负责有明确可用时点的在线专家状态，
  不读取研究目录、未来标签或评估汇总；所有配置由卡片传入。
- `src/backtest/benchmark_relative_replay.py` 为各费用情景及主组合单独结算，
  随后必须经原生 Qlib 回放对账；不得把专家收益加权当作主账户收益。
- `src/analysis/benchmark_relative_protocol.py` 统一核验完整日历、同窗同成本的
  SOXX 比较、年度与滚动超额、MaxDD/CVaR 及区块区间。历史门不签发前向资格，
  调用者给出一个“forward”字符串也不能获得晋升。
- `src/workflow/benchmark_relative_research.py` 显式接收源码根、只读证据根、
  输出卡和 Recorder 路径，核验实际导入源码身份；不硬编码主 checkout。
  原生回放在活跃 Recorder 外执行，恢复时必须核验原 ID，成功归档后才发布摘要。

既有封存实验及生产入口保留原身份，不因新合同重新解释旧结果。后续机制一次只
登记一个增量、一个有限对照集，保留失败和所有成本场景。新的可晋升方案必须先
冻结，再取得连续、独立、决策早于市场结果的前向证据；不能把复用历史改名为留出期。

后续实验已将以下边界独立出来，新的组合研究应复用这些合同，而不是在 notebook
中重新拼接预测、费用与评价：

- `src/data/oos_horizon_inputs.py` 绑定原生价格、标签、成熟时点及日风险协方差；
  五日收益预测与每日风险的单位分别保存，终端未成熟标签保留为未知。
- `src/model/annual_ridge_selection.py` 在当前年之前的完整成熟验证年选择预测模型，
  再以当时可得数据重拟合；历史均值预测是显式模型，不等于全仓 SOXX 策略。
- `src/backtest/horizon_matched_allocation.py` 管理固定调仓节奏、精确 HOLD 与真实
  费用结算。调仓日风险约束和持有期间的风险漂移分别报告。
- `src/model/benchmark_terminal_value.py` 只转换规划中的期末计价和均值误差协方差；
  不改变每日市场风险、不生成未来订单，也不把假设的未来费用扣入实际净值。
- `src/core/recorder_sources.py` 通过卡片 manifest、显式对象哈希及 Qlib Recorder
  读取已经完成的父证据。读取不启动或恢复父 run，不把重新拟合冒充冻结预测复用。
  旧 `artifact_manifest.json` 只允许通过显式文件名及固定 manifest、对象哈希读取，
  不自动回退、迁移或改写已封存卡片。
- `src/model/benchmark_core_risk.py` 将给定风险矩阵与预算映射成两种持仓表示，
  不读取收益均值或未来结果。`src/backtest/target_schedule_replay.py` 独立管理
  逐日目标的真实持仓与费用；目标的数学风险证书、实际结算及 Qlib 持仓风险分层验证。
- `src/workflow/benchmark_evidence.py` 统一完整策略和费用清单及共同日历上的比较；
  `src/analysis/excess_resolution.py` 在保留原收益、估计值和置信区间的前提下，
  排除数值不可辨识的正超额，原始分类同时保留。

预测误差改善、配置目标修正、原生账本对账通过和稳定超额是不同结论。前两项可以
改善系统，但不能替代最后一项的历史与独立前向证据。执行参数、有限候选预算、
费用、数值分辨下限及失败结果保留在各自冻结卡片，不写入技能或生产默认值。

单一标的的长历史不再通过双标的共同面截短：
`src/data/soxx_extended_history.py` 分离显式只读采集与离线资格 notebook，以新的
SOXX-only provider 保存真实较早历史，并核验与原 provider 的重叠值不变。更长
SOXX 历史不能补造 SOXL 成立前的价格、特征或标签。价格、事件来源、真实分红
现金流和历史时点认证是不同资格；复权值相同不能代替后两项。

- `src/core`: 稳定契约、注册表、schema version、manifest、hash、配置、校验。
- `src/data`: qlib store 构建、qlib provider 审计、数据更新、calendar/instrument/feature 对齐、canonical snapshot，最终写入 `data/`。
- `src/factor`: 基于 qlib Dataset 的单因子、多因子、IC、相关性、分层回测、因子组合诊断。
- `src/model`: QuantBench 模型注册、qlib Dataset 训练、预测、保存、加载、模型能力说明。
- `src/objective`: 分类、MSE、IC、ranking、收益风险 utility、晋升评分；替代原 reward 概念。
- `src/backtest`: qlib 信号记录、portfolio report、成本、滑点、walk-forward、压力窗口、基准对比。
- `src/analysis`: qlib recorder/report/risk_analysis 读取、regime 分层、区间对比、模型差异解释、稳定性诊断。
- `src/dashboard`: 只读取 qlib recorder 登记的 manifest/export，不直接读散落实验文件。
- `src/schedule`: qlib 数据更新、模型刷新、dashboard export、通知、锁、重试。
- `src/api`: broker/data adapter contract，v1 只实现 paper/sim，不做真实 broker 下单。
- `research/<experiment>/`: qlib research notebooks；每个实验一个主 `.ipynb`。
- `production/<strategy>/`: qlib-backed promoted paper/sim notebooks；每个方案一个主 `.ipynb`。

## src/analysis 设计

- qlib 集成：
  - 读取 qlib recorder、portfolio metrics、risk analysis、report 数据。
  - 提供统一 `AnalysisReport`，让 dashboard、notebook、promotion 都读同一套 qlib-derived 结果。

- Regime 分析：
  - 支持按 VIX、趋势、回撤、波动、成交量冲击、牛熊/震荡等 regime 切片。
  - 每个模型输出全区间表现、分 regime 表现、regime 间稳定性差异。

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
  - `analysis_summary.json`
  - `regime_metrics.csv`
  - `window_metrics.csv`
  - `model_comparison.csv`
  - `failure_modes.md`
  - qlib recorder artifact links
  - 中文 notebook 结论 cell

## Skill 边界

- skill 不实现业务逻辑，不保存项目状态，不作为功能入口。
- skill 用途：
  - 抽象研究方法。
  - 提醒如何提出假设、拆因子、设计 regime、解释模型差异。
  - 帮模型发散思考候选方向。
- 具体可运行功能必须放在：
  - `src/` 的稳定模块；
  - `research/<experiment>/*.ipynb` 的 qlib 实验流程；
  - `production/<strategy>/*.ipynb` 的 qlib paper/sim 晋升方案；
  - `data/` 的 qlib store、snapshot、manifest。
- 如果 skill 里出现具体路径、模型结果、固定参数或项目状态，应迁回 notebook 或 `src/core` manifest。

## Notebook 规则

- `research/<experiment>/<experiment>.ipynb`: 研究入口，必须包含 qlib init、qlib data/Dataset、训练/预测、qlib recorder、qlib record/report、analysis、sidecar、结论。
- `production/<strategy>/<strategy>.ipynb`: 生产 paper/sim 入口，必须包含固定参数、qlib data/Dataset、qlib recorder、运行输入输出、风险说明。
- 每个 notebook 最后必须有中文结论 cell：做了什么、核心指标、regime 差异、是否晋升、风险、下一步、qlib recorder id。
- CSV/JSON 是 sidecar，不能替代 notebook 结论，也不能替代 qlib recorder。
- 如果 notebook 需要读取预测或组合明细，优先从 `summary.json.recorder_local_dir` 下的 qlib recorder artifact 读取，不从 card 根目录大 CSV 读取。

## 测试计划

- 架构测试：允许新 `src/`、`research/`、`production/`，替换 retired surface 测试。
- qlib 强制测试：
  - notebook 必须包含 `import qlib`、`qlib.init`、`qlib.data.D` 或 qlib Dataset、`qlib.workflow.R`。
  - 至少一个 research notebook 必须执行 qlib Dataset + Recorder + Record/Report 路径。
  - qlib provider_uri 缺失时必须 fail closed，不允许 pandas-only fallback。
- 契约测试：schema version、manifest、hash、序列化稳定性。
- 数据测试：qlib `D.calendar`/`D.features` 对齐、snapshot、防未来函数。
- 因子测试：基于 qlib Dataset 的单因子 IC、多因子相关性、组合回测。
- Objective 测试：classification/MSE/IC/ranking/utility score 输出一致。
- 模型测试：QuantBench registry 完整性，每个 model family 有 qlib Dataset fit/predict smoke test。
- Backtest 测试：qlib SignalRecord/PortAnaRecord 输出、固定窗口、walk-forward、成本滑点、benchmark 对比。
- Analysis 测试：qlib recorder/report/risk_analysis 解析、regime 切片、窗口对比、模型差异表。
- Dashboard 测试：只读 qlib recorder 登记的 manifest/export。
- Schedule/API 测试：qlib data refresh、job lock、retry、paper broker target-position flow。

## 验收标准

一个任务只有同时满足以下条件才算完成：

1. 代码或 notebook 直接调用 qlib 原生 API。
2. 产物能追溯到 qlib provider_uri、Dataset、Recorder、Record/Report。
3. focused tests 覆盖 qlib 强制路径和 fail-closed 路径。
4. notebook 已执行，中文结论和 sidecars 与 qlib recorder 结果一致。
5. 没有把 skill、pandas-only 脚本、散落 CSV、临时 cache 当作主工作流。
6. card 根目录没有持久化 recorder-only 大 CSV，`summary.json.artifact_policy` 能说明所有保留与 recorder-only 产物。

## 假设

- 不再做强化学习，不保留 RL reward。
- reward 改为 objective。
- 中文是默认文档和 notebook 结论语言，代码命名仍用英文。
- v1 不做真实 broker 下单，只实现 adapter contract 和 paper/sim。
- `src/analysis` 是正式框架能力；skill 只做研究思路抽象。
