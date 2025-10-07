## SWIRL 实验设计与数据流程指南

本指南详细说明 SWIRL 的实验架构、训练集与测试集的数据输入与处理流程，并给出关键代码入口片段，帮助你快速理解与复现实验。

### 目录
- 架构总览
- 实验配置文件（experiments/*.json）
- 实验生命周期与核心模块
- 训练/验证/测试数据流
- 推荐索引标签的生成与导出
- 运行入口与示例命令
- 常见路径与产物说明

---

## 架构总览

SWIRL 实验由以下关键部分组成：

- 配置层：`experiments/*.json` 定义实验参数、工作负载来源、RL 算法等。
- 调度入口：`swirl/__main__.py` 或项目根的 `main.py` 作为训练与评估的主入口。
- 实验对象：`swirl/experiment.py` 封装实验生命周期（prepare → learn → evaluate → finish）。
- 环境构建：`Experiment.make_env(...)` 基于配置构造 Gym 环境（训练/验证/测试）。
- 表示与观测：`workload_embedder`、`observation_manager` 决定状态特征；`action_manager` 决定动作空间（索引候选）。
- 奖励函数：`reward_calculator` 定义单步/回合目标（如成本下降相对存储）。
- 基线对比：Extend 与 DB2Advis（可选），用于与 SWIRL 索引选择效果对比。
- 结果输出：模型、评估报告、工作负载标签与索引选择过程 JSON 等。

## 实验配置文件（experiments/*.json）

以 `experiments/localtest.json` 为例，核心字段：

```json
"id": "dsblabel",
"description": "...",
"gym_version": 1,
"timesteps": 50000,
"random_seed": 60,
"parallel_environments": 16,
"action_manager": "MultiColumnIndexActionManager",
"observation_manager": "SingleColumnIndexObservationManager",
"reward_calculator": "RelativeDifferenceRelativeToStorageReward",
"max_steps_per_episode": 200,
"validation_frequency": 5000,
"max_index_width": 3,
"reenable_indexes": true,
"result_path": "experiment_results",
"workload_embedder": { "type": "PlanEmbedderLSIBOW", "representation_size": 50 },
"rl_algorithm": {
  "algorithm": "PPO2",
  "stable_baselines_version": 2,
  "gamma": 0.5,
  "policy": "MlpPolicy",
  "model_architecture": { "net_arch": [{"vf": [256,256], "pi": [256,256]}] },
  "args": { "n_steps": 64 }
},
"workload": {
  "benchmark": "MIX3",
  "scale_factor": 10,
  "size": 10,
  "training_instances": 10,
  "validation_testing": { "number_of_workloads": 1, "unknown_query_probabilities": [0.0] },
  "similar_workloads": false,
  "unknown_queries": 0
},
"comparison_algorithms": [],
"budgets": { "training": false, "validation_and_testing": [500] },
"column_filters": { "TableNumRowsFilter": 10000 },
"ExternalWorkload": true,
"WorkloadPath": "/home/baiyutao/SWIRL/baiyutao/dsb250k.labeled.json",
"TestExternalWorkload": false,
"TestWorkloadPath": null
```

- 训练/验证/测试集规模与抽样策略在 `workload` 字段中控制。
- `ExternalWorkload`/`WorkloadPath` 支持从外部已标注数据加载训练集；`TestExternalWorkload`/`TestWorkloadPath` 控制测试集外部加载。
- `max_index_width` 控制候选索引最大列数；`budgets.validation_and_testing` 为评估阶段分配的存储预算集合（单位 MB）。
- `workload_embedder` 可选用多种计划/SQL 表示方法（如 `PlanEmbedderLSIBOW`）。

## 实验生命周期与核心模块

生命周期主线在 `swirl/__main__.py` / `main.py`：

```startLine:endLine:/home/baiyutao/SWIRL/swirl/__main__.py
12:20:.../swirl/__main__.py
```

- 解析配置、创建 `Experiment`。
- `experiment.prepare()`：解析 schema、生成/加载工作负载、构造候选索引与嵌入器。
- 创建并归一化并行环境，实例化 RL 算法模型。
- `experiment.compare()`：构建对比回调环境（测试/验证）。
- 使用回调周期性评估并保存最佳/滑动平均模型。
- `model.learn(...)` 完成训练；`experiment.finish_learning(...)` 汇总训练统计。
- `experiment.finish()` 对最终、滑动平均、最佳均值奖励模型进行全面测试/验证，导出报告与数据。

`Experiment.prepare()` 的关键步骤在 `swirl/experiment.py`：

```startLine:endLine:/home/baiyutao/SWIRL/swirl/experiment.py
71:101:/home/baiyutao/SWIRL/swirl/experiment.py
```

- 解析 schema 与列过滤；
- 基于 `WorkloadGenerator` 生成训练/验证/测试 workloads；
- 生成全局可索引列的组合与存储开销预测；
- 可选初始化 `workload_embedder`；
- 为验证/测试 workloads 赋予预算。

环境构造在 `Experiment.make_env(...)`：

```startLine:endLine:/home/baiyutao/SWIRL/swirl/experiment.py
1436:1505:/home/baiyutao/SWIRL/swirl/experiment.py
```

- 创建 `action_manager`/`observation_manager`/`reward_calculator`。
- 根据环境类型选择训练/验证/测试 workloads。
- 使用 Gym `DB-v{gym_version}` 注册环境注入组件与配置。

评估路径：

```startLine:endLine:/home/baiyutao/SWIRL/swirl/experiment.py
1107:1181:/home/baiyutao/SWIRL/swirl/experiment.py
```

- `test_model(...)`、`validate_model(...)` 对各自 workloads 构建只读环境，调用 `_evaluate_model(...)`。
- `_evaluate_model(...)` 同步归一化统计、调用 `evaluate_policy` 运行，记录性能与 SWIRL 索引选择时间，并收集推荐索引为 labels。

实验收尾与导出：

```startLine:endLine:/home/baiyutao/SWIRL/swirl/experiment.py
319:418:/home/baiyutao/SWIRL/swirl/experiment.py
```

- 对多类模型快照进行测试/验证；
- 将包含 labels 的训练数据评估（训练当作测试）也执行与收集；
- 保存 workloads（含 labels）与索引选择过程 JSON；
- 写入实验报告。

## 训练/验证/测试数据流

数据来源与分发：
- 通过 `WorkloadGenerator` 构造 `wl_training`、`wl_validation`、`wl_testing`。
- `prepare()` 中调用 `_assign_budgets_to_workloads()` 为验证/测试 workloads 随机分配预算：

```startLine:endLine:/home/baiyutao/SWIRL/swirl/experiment.py
128:136:/home/baiyutao/SWIRL/swirl/experiment.py
```

- 训练环境使用 `wl_training` 的全部样本；验证/测试环境根据需要传入单个 workload 或列表。

嵌入与观测：
- 可选 `workload_embedder`（如 `PlanEmbedderLSIBOW`）在 `prepare()` 阶段创建，供 `observation_manager` 使用形成状态特征。
- `observation_manager` 输入包含：查询类数、工作负载大小、可选嵌入器与列数量等。

动作与奖励：
- `action_manager` 暴露全局候选索引动作与存储估计；
- `reward_calculator` 使用成本改进与存储开销定义回报（如相对差值）。

评估与时间统计：
- `_evaluate_model(...)` 记录 SWIRL 决策时间并写入 `self.swirl_times`；
- 与 Extend/DB2Advis 的时间对比在 `compare()` 期间收集并在报告中聚合。

## 推荐索引标签的生成与导出

- 评估时 `_collect_index_labels(...)` 将每个 episode 的最终推荐索引集合写回 `workload.labels`，并同步到 `self.workload_labels`：

```startLine:endLine:/home/baiyutao/SWIRL/swirl/experiment.py
1325:1382:/home/baiyutao/SWIRL/swirl/experiment.py
```

- 训练数据也可被当作测试集评估，标签记录到 `test` 类型：

```startLine:endLine:/home/baiyutao/SWIRL/swirl/experiment.py
1383:1435:/home/baiyutao/SWIRL/swirl/experiment.py
```

- 收尾阶段导出两类 JSON：
  - 所有评估的 workloads（含 labels）的结构化快照：

```startLine:endLine:/home/baiyutao/SWIRL/swirl/experiment.py
137:183:/home/baiyutao/SWIRL/swirl/experiment.py
```

  - 与外部标注格式一致的 labeled workload 文件（便于下游训练/分析）：

```startLine:endLine:/home/baiyutao/SWIRL/swirl/experiment.py
1536:1653:/home/baiyutao/SWIRL/swirl/experiment.py
```

- 可选导出索引选择过程与时间统计：

```startLine:endLine:/home/baiyutao/SWIRL/swirl/experiment.py
1226:1324:/home/baiyutao/SWIRL/swirl/experiment.py
```

- 报告文件包含总体/分环境指标与时间：

```startLine:endLine:/home/baiyutao/SWIRL/swirl/experiment.py
517:997:/home/baiyutao/SWIRL/swirl/experiment.py
```

## 运行入口与示例命令

两种等价入口：项目根 `main.py` 与包入口 `swirl/__main__.py`。命令示例：

```bash
# 使用包入口
python -m swirl /home/baiyutao/SWIRL/experiments/localtest.json

# 使用根入口（如需要）
python /home/baiyutao/SWIRL/main.py /home/baiyutao/SWIRL/experiments/localtest.json
```

关键入口片段（创建实验、环境与模型）：

```startLine:endLine:/home/baiyutao/SWIRL/swirl/__main__.py
37:69:/home/baiyutao/SWIRL/swirl/__main__.py
```

评估回调与学习：

```startLine:endLine:/home/baiyutao/SWIRL/swirl/__main__.py
70:139:/home/baiyutao/SWIRL/swirl/__main__.py
```

环境工厂（供训练/验证/测试引用）：

```startLine:endLine:/home/baiyutao/SWIRL/swirl/experiment.py
1467:1504:/home/baiyutao/SWIRL/swirl/experiment.py
```

## 常见路径与产物说明

- 结果目录：`experiment_results/ID_{id}`
  - `final_model.*`、`best_mean_reward_model.zip`、`moving_average_model*.zip`
  - `vec_normalize.pkl`：归一化统计
  - `report_ID_{id}.txt`：总体训练与对比报告
  - `testing_workloads.json`、`validation_workloads.json`：评估快照（含 labels 汇总）
  - `*_workloads_with_labels_*.json`：外部对齐的 labeled workload 文件
  - `index_selection_{EnvironmentType}_{timestamp}.json`：索引选择过程与时间统计（如启用）

---

如需进一步定制：
- 更换 `workload_embedder`（`swirl/workload_embedder.py`）类型与 `representation_size`。
- 更换 `action_manager`/`observation_manager`/`reward_calculator` 名称以改变动作与状态/奖励。
- 调整 `max_index_width` 与 `budgets` 控制候选宽度与评估预算。
