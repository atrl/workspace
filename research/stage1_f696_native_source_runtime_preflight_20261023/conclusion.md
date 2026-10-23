# f696 Stage1 原生来源加载：运行条件与父本逐值校验

仓库 `qlib_venv` 的 Qlib0.9.7/Scikit-learn1.9.1 在导入旧MLflow文件Recorder前须设置 `MLFLOW_ALLOW_FILE_STORE=true`。此前的 `No valid experiment has been found` 是缺少此运行条件，不是已封存实验ID或来源缺失。设置后，f696 Recorder按名称和ID均能取得，helper载入944行，原Stage1 A0和SOXX原生审计通过。

helper原27baa五输出与f696原A0提议 944/944 日期不一致；逐行替换五项`frozen_current`后，对封存A0的最大绝对误差 0.0。所有信号日均在钉住的Qlib日历且SOXX收盘价可读。本卡0拟合、0账户、0新Recorder；已停止的静态组成重测并未重开。下一真正候选仍须从f696失败区间得出独立决策时机制，并运行完整0bp Tiered自身状态账户。
