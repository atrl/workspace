# SOXL 强化学习项目 - 执行清单

## 📅 项目时间线
- **Week 1**: 数据重建与验证
- **Week 2**: 基线模型训练
- **Week 3**: 模型优化与调参
- **Week 4**: 部署准备与测试

---

## ✅ Phase 0: 环境准备（Day 1）

### 0.1 创建 Worktree
- [ ] 创建数据重建 worktree
  ```bash
  cd ~/workspace/soxl-rl-qlib
  git worktree add ../soxl-rl-data -b feature/data-rebuild
  ```
- [ ] 创建训练实验 worktree
  ```bash
  git worktree add ../soxl-rl-training -b feature/training-experiments
  ```
- [ ] 验证 worktree 创建成功
  ```bash
  git worktree list
  ```

### 0.2 设置虚拟环境
- [ ] 创建 Python 虚拟环境
  ```bash
  cd ~/workspace/soxl-rl-qlib
  python3.10 -m venv .venv
  source .venv/bin/activate
  ```
- [ ] 安装依赖
  ```bash
  pip install -r requirements.txt
  ```
- [ ] 验证安装
  ```bash
  python -c "import stable_baselines3; import gymnasium; print('✅ OK')"
  ```
- [ ] 安装 ta-lib（如果需要）
  ```bash
  brew install ta-lib  # macOS
  pip install ta-lib
  ```

### 0.3 创建项目骨架
- [ ] 在 soxl-rl-data 中创建目录结构
  ```bash
  cd ../soxl-rl-data
  mkdir -p dataset/{raw,features,processed}
  mkdir -p logs
  mkdir -p notebooks
  mkdir -p tests
  ```
- [ ] 在 soxl-rl-training 中创建目录结构
  ```bash
  cd ../soxl-rl-training
  mkdir -p experiments/{baseline,optimized,ensemble}
  mkdir -p checkpoints
  mkdir -p logs/tensorboard
  mkdir -p results/{backtest,reports}
  mkdir -p notebooks
  ```

---

## ✅ Phase 1: 数据重建（Day 2-4）

### 1.1 数据获取（Day 2）
- [ ] 下载 SOXL 分钟数据
  ```bash
  cd ~/workspace/soxl-rl-data
  python -m data.downloader --symbols SOXL --interval 1m --lookback-days 30 --output-dir ./dataset/raw
  ```
- [ ] 下载 SOXS 分钟数据
  ```bash
  python -m data.downloader --symbols SOXS --interval 1m --lookback-days 30 --output-dir ./dataset/raw
  ```
- [ ] 下载 TQQQ 分钟数据
  ```bash
  python -m data.downloader --symbols TQQQ --interval 1m --lookback-days 30 --output-dir ./dataset/raw
  ```
- [ ] 下载 SQQQ 分钟数据
  ```bash
  python -m data.downloader --symbols SQQQ --interval 1m --lookback-days 30 --output-dir ./dataset/raw
  ```
- [ ] 下载 5 分钟数据（获取更多历史）
  ```bash
  python -m data.downloader --symbols SOXL SOXS TQQQ SQQQ --interval 5m --lookback-days 60 --output-dir ./dataset/raw
  ```
- [ ] 验证数据下载完成
  ```bash
  ls -lh ./dataset/raw/
  ```

### 1.2 数据质量检查（Day 2）
- [ ] 创建数据质量检查脚本
  ```bash
  cp ../soxl-rl-qlib/notebooks/data_quality_check.ipynb ./notebooks/
  ```
- [ ] 运行 SOXL 数据质量检查
  ```bash
  python scripts/check_data_quality.py ./dataset/raw/SOXL_1m.csv
  ```
- [ ] 检查数据完整性
  - [ ] 无缺失值或缺失值 < 1%
  - [ ] 无异常价格（high >= low）
  - [ ] 时间连续性检查
  - [ ] 成交量合理性检查
- [ ] 生成数据质量报告
  ```bash
  python scripts/generate_data_report.py ./dataset/raw/ --output ./logs/data_report.html
  ```

### 1.3 特征工程（Day 3）
- [ ] 创建高级特征工程脚本
  ```bash
  cp ../soxl-rl-qlib/data/advanced_features.py ./data/
  ```
- [ ] 为 SOXL 生成特征
  ```bash
  python -m data.features --input ./dataset/raw/SOXL_1m.csv --output ./dataset/features/SOXL_1m_features.csv
  ```
- [ ] 为 SOXS 生成特征
  ```bash
  python -m data.features --input ./dataset/raw/SOXS_1m.csv --output ./dataset/features/SOXS_1m_features.csv
  ```
- [ ] 为 TQQQ 生成特征
  ```bash
  python -m data.features --input ./dataset/raw/TQQQ_1m.csv --output ./dataset/features/TQQQ_1m_features.csv
  ```
- [ ] 为 SQQQ 生成特征
  ```bash
  python -m data.features --input ./dataset/raw/SQQQ_1m.csv --output ./dataset/features/SQQQ_1m_features.csv
  ```
- [ ] 验证特征生成
  ```bash
  head -n 1 ./dataset/features/SOXL_1m_features.csv
  wc -l ./dataset/features/*.csv
  ```

### 1.4 数据预处理（Day 3）
- [ ] 特征标准化
  ```bash
  python scripts/normalize_features.py ./dataset/features/ --output ./dataset/processed/
  ```
- [ ] 划分训练/验证/测试集
  ```bash
  python scripts/split_data.py ./dataset/processed/SOXL_1m_features.csv \
      --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 \
      --output ./dataset/processed/
  ```
- [ ] 保存数据划分信息
  ```bash
  python scripts/save_data_splits.py --output ./dataset/processed/splits_info.json
  ```

### 1.5 数据验证（Day 4）
- [ ] 运行完整数据验证
  ```bash
  python scripts/validate_dataset.py ./dataset/processed/
  ```
- [ ] 检查特征相关性
  ```bash
  python scripts/check_feature_correlation.py ./dataset/processed/SOXL_1m_train.csv
  ```
- [ ] 生成特征重要性分析
  ```bash
  python scripts/analyze_features.py ./dataset/processed/ --output ./logs/feature_analysis.html
  ```
- [ ] 创建数据文档
  ```bash
  python scripts/create_data_documentation.py --output ./docs/data_documentation.md
  ```

---

## ✅ Phase 2: 环境测试（Day 5）

### 2.1 环境单元测试
- [ ] 运行环境初始化测试
  ```bash
  cd ~/workspace/soxl-rl-qlib
  python tests/test_trading_env.py::test_env_initialization
  ```
- [ ] 运行环境重置测试
  ```bash
  python tests/test_trading_env.py::test_env_reset
  ```
- [ ] 运行环境步进测试
  ```bash
  python tests/test_trading_env.py::test_env_step
  ```
- [ ] 运行完整回合测试
  ```bash
  python tests/test_trading_env.py::test_full_episode
  ```
- [ ] 运行交易成本测试
  ```bash
  python tests/test_trading_env.py::test_transaction_cost
  ```
- [ ] 运行所有测试
  ```bash
  pytest tests/test_trading_env.py -v
  ```

### 2.2 环境可视化调试
- [ ] 运行环境调试脚本
  ```bash
  jupyter notebook notebooks/debug_environment.ipynb
  ```
- [ ] 验证随机策略表现
- [ ] 检查奖励分布
- [ ] 确认环境逻辑正确

### 2.3 环境压力测试
- [ ] 测试极端价格波动
- [ ] 测试零成交量情况
- [ ] 测试连续涨跌停
- [ ] 测试数据边界情况

---

## ✅ Phase 3: 基线模型训练（Day 6-7）

### 3.1 基线训练（Day 6）
- [ ] 创建训练配置
  ```bash
  cd ~/workspace/soxl-rl-training
  cp ../soxl-rl-qlib/configs/qlib.yaml ./configs/baseline.yaml
  ```
- [ ] 开始基线训练
  ```bash
  python scripts/train.py \
      --data ../soxl-rl-data/dataset/processed/SOXL_1m_train.csv \
      --config ./configs/baseline.yaml \
      --output ./experiments/baseline \
      --total-timesteps 50000
  ```
- [ ] 启动 TensorBoard 监控
  ```bash
  tensorboard --logdir=./logs/tensorboard --port=6006
  ```
- [ ] 训练过程中检查
  - [ ] 奖励是否逐渐上升
  - [ ] Loss 是否收敛
  - [ ] 无异常值或 NaN

### 3.2 基线评估（Day 7）
- [ ] 在验证集上评估
  ```bash
  python scripts/backtest.py \
      --model ./experiments/baseline/final_model.zip \
      --data ../soxl-rl-data/dataset/processed/SOXL_1m_val.csv \
      --output ./results/backtest/baseline_val
  ```
- [ ] 在测试集上评估
  ```bash
  python scripts/backtest.py \
      --model ./experiments/baseline/final_model.zip \
      --data ../soxl-rl-data/dataset/processed/SOXL_1m_test.csv \
      --output ./results/backtest/baseline_test
  ```
- [ ] 记录基线指标
  - [ ] Total Return: ____%
  - [ ] Sharpe Ratio: ____
  - [ ] Max Drawdown: ____%
  - [ ] Win Rate: ____%

### 3.3 Walk-Forward 验证（Day 7）
- [ ] 运行 5-fold Walk-Forward
  ```bash
  python scripts/train.py \
      --data ../soxl-rl-data/dataset/processed/SOXL_1m_features.csv \
      --config ./configs/baseline.yaml \
      --output ./experiments/walk_forward \
      --walk-forward \
      --n-splits 5
  ```
- [ ] 分析各 fold 表现
- [ ] 计算平均指标和标准差
- [ ] 检查过拟合迹象

---

## ✅ Phase 4: 模型优化（Day 8-10）

### 4.1 超参数调优（Day 8）
- [ ] 设置 Optuna 调优
  ```bash
  python scripts/hyperparameter_tuning.py --n-trials 50
  ```
- [ ] 运行调优实验
- [ ] 分析最佳超参数
- [ ] 保存调优结果
  ```bash
  cp ./experiments/optuna/best_params.json ./configs/optimized_params.json
  ```

### 4.2 模型改进（Day 9）
- [ ] 使用最佳超参数训练
  ```bash
  python scripts/train.py \
      --data ../soxl-rl-data/dataset/processed/SOXL_1m_train.csv \
      --config ./configs/optimized.yaml \
      --output ./experiments/optimized \
      --total-timesteps 100000
  ```
- [ ] 尝试不同算法（SAC, TD3）
- [ ] 实验 Ensemble 方法
- [ ] 对比各模型表现

### 4.3 特征工程优化（Day 10）
- [ ] 特征选择
  ```bash
  python scripts/feature_selection.py --method importance
  ```
- [ ] 移除冗余特征
- [ ] 添加新特征（如果需要）
- [ ] 重新训练并对比

---

## ✅ Phase 5: 综合评估（Day 11-12）

### 5.1 完整回测（Day 11）
- [ ] 在所有测试集上回测
  ```bash
  python scripts/comprehensive_backtest.py \
      --model ./experiments/optimized/final_model.zip \
      --data-dir ../soxl-rl-data/dataset/processed/ \
      --output ./results/reports/final_backtest
  ```
- [ ] 生成完整报告
- [ ] 与基准对比（Buy & Hold）
- [ ] 可视化结果

### 5.2 压力测试（Day 12）
- [ ] 2020年3月熔断场景测试
- [ ] 2022年熊市场景测试
- [ ] 2023年牛市场景测试
- [ ] 震荡市场场景测试
- [ ] 记录各场景表现

### 5.3 风险分析（Day 12）
- [ ] 计算 VaR 和 CVaR
- [ ] 分析最大回撤时段
- [ ] 评估尾部风险
- [ ] 检查极端情况

---

## ✅ Phase 6: 部署准备（Day 13-14）

### 6.1 模型持久化（Day 13）
- [ ] 保存最终模型
  ```bash
  python scripts/save_final_model.py \
      --model ./experiments/optimized/final_model.zip \
      --output ./models/production/soxl_ppo_v1.zip
  ```
- [ ] 保存配置文件
- [ ] 保存特征工程 Pipeline
- [ ] 创建模型版本文档

### 6.2 部署脚本（Day 13）
- [ ] 创建实时推理脚本
  ```bash
  python scripts/create_inference_script.py --output ./scripts/inference.py
  ```
- [ ] 创建风控模块
  ```bash
  python scripts/create_risk_manager.py --output ./scripts/risk_manager.py
  ```
- [ ] 创建监控脚本
  ```bash
  python scripts/create_monitor.py --output ./scripts/monitor.py
  ```

### 6.3 文档完善（Day 14）
- [ ] 更新 README.md
- [ ] 创建 API 文档
- [ ] 创建部署指南
- [ ] 创建运维手册

### 6.4 最终验证（Day 14）
- [ ] 端到端测试
- [ ] 性能基准测试
- [ ] 安全检查
- [ ] 准备上线

---

## 📊 成功标准检查表

### 基线模型（Week 2 结束）
- [ ] Sharpe Ratio > 1.0
- [ ] Max Drawdown < 30%
- [ ] Win Rate > 50%
- [ ] Profit Factor > 1.0
- [ ] 与 Buy & Hold 对比

### 优化模型（Week 3 结束）
- [ ] Sharpe Ratio > 1.5
- [ ] Max Drawdown < 20%
- [ ] Win Rate > 55%
- [ ] Profit Factor > 1.2
- [ ] 跑赢 Buy & Hold 5%+

### 可上线模型（Week 4 结束）
- [ ] Sharpe Ratio > 2.0
- [ ] Max Drawdown < 15%
- [ ] Win Rate > 60%
- [ ] Profit Factor > 1.5
- [ ] 稳定跑赢 Buy & Hold 10%+
- [ ] 通过所有压力测试
- [ ] 完整文档就绪

---

## 🐛 问题跟踪

### 数据问题
- [ ] 问题：__________________
  - 状态：待解决 / 进行中 / 已解决
  - 解决方案：__________________

### 模型问题
- [ ] 问题：__________________
  - 状态：待解决 / 进行中 / 已解决
  - 解决方案：__________________

### 环境问题
- [ ] 问题：__________________
  - 状态：待解决 / 进行中 / 已解决
  - 解决方案：__________________

---

## 📝 每日进度日志

### Day 1 (YYYY-MM-DD)
- 完成任务：
- 遇到问题：
- 明天计划：

### Day 2 (YYYY-MM-DD)
- 完成任务：
- 遇到问题：
- 明天计划：

（继续添加每天的记录）

---

## 📞 需要帮助时

**遇到问题时的步骤**:
1. 查看技术指导文档的"常见问题"部分
2. 检查本项目清单的"问题跟踪"部分
3. 在 Discord #soxl 频道提问
4. @clawQuant 寻求技术指导

**协作流程**:
- 每日更新进度到 #soxl
- 每周技术评审
- 重大问题及时沟通

---

**项目开始日期**: ___________
**预计完成日期**: ___________
**当前阶段**: ___________
**整体进度**: ____%
