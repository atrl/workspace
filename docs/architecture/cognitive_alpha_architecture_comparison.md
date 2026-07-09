# Cognitive Alpha 架构对比

本文档说明 Cognitive Alpha 的目标架构，与当前 SOXL qlib-native 项目中已经实现的方案之间的关系。它不是新的实验产物；当前状态以 `research/cognitive_alpha_goal_codex_exec_v1/summary.json`、`workflow.yaml`、`conclusion.md` 和对应 qlib recorder 为准。

## 依据

- 本仓库合同：`AGENTS.md`、`SPEC.md` 要求 notebook-first、recorder-first、slim-card，并要求数据、训练、回测和分析以 qlib 为系统入口。
- qlib 官方文档：[Data Layer](https://qlib.readthedocs.io/en/latest/component/data.html) 说明 qlib format、DataHandlerLP 和 DatasetH；[Workflow](https://qlib.readthedocs.io/en/latest/component/workflow.html) 说明 Data、Model、Evaluation 和 execution tracking；[Model](https://qlib.readthedocs.io/en/latest/component/model.html) 文档示例使用 `qlib.init`、`DatasetH`、`R.start`、`SignalRecord`；[API Reference](https://qlib.readthedocs.io/en/latest/reference/api.html) 列出 `SignalRecord`、`SigAnaRecord`、`PortAnaRecord` 等 recorder template。
- 当前实现入口：`src/factor/cognitive_alpha.py`、`src/factor/cognitive_alpha_mining.py`、`src/workflow/cognitive_alpha_evolution.py`、`src/workflow/cognitive_alpha_goal.py`、`src/workflow/cognitive_alpha_goal_codex_exec.py`、`src/workflow/cognitive_alpha_position.py`、`src/workflow/cognitive_alpha_weighted.py`。

## Cognitive Alpha 目标架构

```mermaid
flowchart TB
    Objective["研究目标<br/>生成可解释 alpha<br/>并转化为可验证仓位"]
    Hierarchy["CogAlpha hierarchy<br/>7 个 level / 21 个 agent"]
    Candidate["Candidate contract<br/>name / level / agent<br/>required_columns / lookback<br/>function / rationale / code"]
    Quality["Quality checker<br/>禁止未来函数<br/>校验 required columns<br/>missing / distinct gates"]
    Fitness["Fitness evaluation<br/>IC / RankIC<br/>ICIR / RankICIR<br/>Mutual Information"]
    Evolution["Evolution loop<br/>repair<br/>mutation / crossover<br/>crossover_mutation"]
    Library["Factor library<br/>非冗余候选<br/>保留 lineage 与指标"]
    Position["Position layer<br/>benchmark comparison<br/>risk gates<br/>walk-forward"]
    Evidence["Evidence<br/>qlib recorder<br/>slim card sidecars<br/>notebook conclusion"]

    Objective --> Hierarchy --> Candidate --> Quality --> Fitness --> Evolution
    Evolution --> Candidate
    Fitness --> Library --> Position --> Evidence

    classDef target fill:#eef6ff,stroke:#2f6f9f,color:#111827
    class Objective,Hierarchy,Candidate,Quality,Fitness,Evolution,Library,Position,Evidence target
```

目标形态不是让 LLM 直接给交易指令，而是把多 agent 的研究假设压成可执行、可审计的 alpha candidate，再经过质量、统计、相关性、仓位和 walk-forward 风险门禁。能不能晋升取决于 recorder-backed 证据，不取决于候选生成看起来是否合理。

## 当前项目实现

```mermaid
flowchart TB
    subgraph Qlib["qlib system of record"]
        Store["data/qlib/soxl<br/>data/qlib/soxx_constituents<br/>data/qlib/soxl_intraday_15min"]
        DFeatures["qlib.data.D.features"]
        Dataset["Alpha158<br/>DatasetH / DataHandlerLP"]
        Recorder["qlib.workflow.R.start<br/>research card recorder"]
        Store --> DFeatures
        Store --> Dataset
    end

    subgraph Provider["Candidate provider"]
        Codex["codex exec provider<br/>read-only sandbox<br/>output schema validated"]
        Seeds["static seed + Alpha158 columns"]
        Canary["future-leak canary<br/>quality checker proof"]
    end

    subgraph Mining["Factor mining and evolution"]
        CandidatePool["initial candidate pool"]
        QC["quality checker<br/>reject / repair"]
        Metrics["fitness metrics<br/>IC / RankIC / ICIR / MI"]
        Ops["mutation / crossover<br/>crossover_mutation"]
        Selected["selected feature library"]
        CandidatePool --> QC --> Metrics --> Ops
        Ops --> CandidatePool
        Metrics --> Selected
    end

    subgraph Positioning["Position and promotion chain"]
        Direct["direct weighted-factor search<br/>prior only"]
        DirectFeatures["direct_factor_base_position<br/>direct_factor_raw_target"]
        Model["position model<br/>LGBM / XGB candidates<br/>base_gate / probability / throttle hooks"]
        Gates["benchmark + participation<br/>risk + walk-forward gates"]
        Signal["latest signal<br/>paper/sim only"]
        Direct --> DirectFeatures --> Model --> Gates --> Signal
    end

    subgraph Card["Slim card"]
        Sidecars["small sidecars<br/>summary / comparison / risk"]
        BigArtifacts["large artifacts<br/>positions / candidate pools<br/>recorder-only"]
        Conclusion["Chinese conclusion<br/>promotion decision"]
    end

    DFeatures --> CandidatePool
    Dataset --> CandidatePool
    Codex --> CandidatePool
    Seeds --> CandidatePool
    Canary --> QC
    Selected --> Direct
    Selected --> Model
    Recorder --> Sidecars
    Gates --> Recorder
    Signal --> Recorder
    Recorder --> BigArtifacts
    Sidecars --> Conclusion

    classDef qlib fill:#ecfdf5,stroke:#047857,color:#111827
    classDef provider fill:#fff7ed,stroke:#c2410c,color:#111827
    classDef mining fill:#f5f3ff,stroke:#6d28d9,color:#111827
    classDef position fill:#fef2f2,stroke:#b91c1c,color:#111827
    classDef card fill:#f8fafc,stroke:#475569,color:#111827
    class Store,DFeatures,Dataset,Recorder qlib
    class Codex,Seeds,Canary provider
    class CandidatePool,QC,Metrics,Ops,Selected mining
    class Direct,DirectFeatures,Model,Gates,Signal position
    class Sidecars,BigArtifacts,Conclusion card
```

当前实现已经把 Cognitive Alpha 的候选生成、质量检查、演化、因子库、direct-factor prior、position model、walk-forward gates 和 recorder/card 证据链串起来。当前问题不是“没有架构”，而是最新 goal card 仍未通过晋升门。

## 对比表

| 维度 | Cognitive Alpha 目标 | 当前项目实现 | 当前判断 |
| --- | --- | --- | --- |
| 数据系统 | 统一、可复现、无未来函数的数据入口 | qlib provider + `D.features` + Alpha158 `DatasetH` / `DataHandlerLP` | aligned |
| 候选生成 | 多 agent 按经济含义生成 alpha idea | `COGALPHA_AGENT_LEVELS` 定义 7 个 level / 21 个 agent；Codex exec provider 用 schema 生成候选 | partial |
| 候选合同 | 候选必须可执行、可解释、可审计 | `CognitiveAlphaCandidate` 记录 `required_columns`、`lookback`、`economic_rationale`、`code` | aligned |
| 质量门 | 禁止未来函数、坏数据和不可执行表达式 | `quality_checker` 拒绝 `shift(-n)` / future `Ref`，并支持 repair loop 与 canary | aligned |
| 统计评价 | 不只看收益，要先看信号质量 | IC、RankIC、ICIR、RankICIR、Mutual Information、相关性去冗余 | aligned |
| 自主迭代 | 能基于失败反馈继续尝试 | Goal runner 支持 attempt、resume、state、feedback context 和 `needs_iteration` | partial |
| direct-factor | 作为可解释先验，而不是最终晋升捷径 | `cognitive_alpha_weighted_factor_search` 输出 base position 和 raw target，并接入 position frame | aligned |
| 仓位模型 | alpha 必须转化为仓位并通过风险门 | LGBM/XGB position candidates、benchmark comparison、participation gates、walk-forward gates | partial |
| 证据形态 | recorder-first，根目录只保留小 sidecar | `summary.json` 记录 `slim_card_recorder_first`，positions/candidate pools 等大产物在 recorder | aligned |
| 生产边界 | 未晋升前不得当作实盘策略 | 只生成 paper/sim signal；真实 broker 下单不在 v1 范围内 | blocked |

## 最新证据状态

```mermaid
flowchart LR
    Run["cognitive_alpha_goal_codex_exec_v1<br/>attempt 2 / 2"]
    Status["status = needs_iteration<br/>terminal_reason = max_attempts_exhausted_needs_iteration"]
    Position["position model<br/>return = 403.5389%<br/>beats listed benchmarks"]
    WF["walk-forward gates<br/>all_folds_pass_risk_gates = false<br/>worst drawdown = -28.2626%"]
    Direct["direct-factor search<br/>return = 73.3041%<br/>selection_eligible = false"]
    Decision["promotion_decision<br/>not_promoted_failed_walk_forward_risk_gates"]
    Signal["latest signal exists<br/>BUY / target_position = 0.2324<br/>paper/sim only"]
    Next["next_action<br/>continue_codex_exec_factor_generation<br/>requires budget refresh"]

    Run --> Status --> Position --> WF --> Decision
    Run --> Direct --> Decision
    Decision --> Signal
    Decision --> Next

    classDef ok fill:#ecfdf5,stroke:#047857,color:#111827
    classDef warn fill:#fff7ed,stroke:#c2410c,color:#111827
    classDef block fill:#fef2f2,stroke:#b91c1c,color:#111827
    class Run,Position,Signal ok
    class Status,Next warn
    class WF,Direct,Decision block
```

当前结论：

- 架构实现状态：`partial`，核心链路已实现并有 recorder/card 证据。
- 晋升状态：`blocked`，因为 latest goal card 为 `needs_iteration`，最终 `promotion_decision=not_promoted_failed_walk_forward_risk_gates`。
- direct-factor 状态：`prior_only_not_final_promotion`，不能把 direct-factor 结果当作生产晋升依据。
- 下一步应继续扩展 Codex exec 候选和稳定性筛选，重点修复 walk-forward 风险门，而不是重写 qlib 证据链或绕过 recorder。
