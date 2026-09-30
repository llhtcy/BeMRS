# 独立 BeMRS 迁移与清理报告

## 来源与边界

来源：`root@172.31.233.171:8006:/root/lhc/HSEvo-main`，2026-09-29快照。
原项目、原实验结果、原配置未删除或覆盖。
新项目的所有依赖代码/数据均在项目内，运行时不加载旧stage包或外部工程。
这次是结构清理与独立化，不重新设计BX/BR、探针、区域更新、真实性能或预算逻辑。

最近OP基准实际通过脚本采用Mean6D、pop10、archive24、novelty开启；正式stage16 YAML中novelty仍为false。
新项目默认采用前者，与已经确认的最新baseline配置对齐。MKP/CVRP采用此前确定的完整Mean7D。

## 目录与命名

| 原位置/名称 | 新位置/名称 |
|---|---|
| behavior_guided_simplified_stage8_behavior_filter | bemrs |
| stage16包装包反向导入stage8 | 删除包装层，直接导入bemrs |
| eoh_adapter.EoH | bemrs.runner.BeMRS |
| original/eoh.py、EOH | bemrs/core/search.py、BeMRS |
| original/eoh_interface_EC.py、InterfaceEC | bemrs/core/engine.py、SearchEngine |
| original/eoh_evolution.py | bemrs/core/operators.py |
| original/interface_LLM.py | bemrs/core/llm.py |
| original/pop_greedy.py | bemrs/core/population.py |
| getParas、prob_rank注入、旧实验参数对象 | 删除，主循环直接读取cfg |
| EOH_*环境变量 | BEMRS_* |
| eoh_timing_records.csv | bemrs_timing_records.csv |
| RBF/GPR/UCB兼容方法名 | 去除未调用者，保留者按surrogate/预测排序职责命名 |

环境变量桥接作为配置传递实现保留，不再是兼容stage的方法切换器。
正式配置只提供一种方法。旧cfg中的method=stageXX、旧EOH变量不再作为新项目接口。
lineage的核心算法、父代、真实objective和generated_order字段保留；旧RBF字段改为surrogate名称。

## 保留的运行机制

1. 固定问题seed、I1初始化、不同真实分数的初始化门槛。
2. 两轮BE、多样性父代选择、冷启动max-min候选筛选。
3. BX/BR提示词、当前父代选择规则和BX-BX-BX-BR周期。
4. 跨区域BX轮次交替、其他区域top-k替换、组合耗尽回退。没有暗中改成新的混合批次策略。
5. 当前种群、完整历史评估档案、按不同objective数截断的优势档案，以及并列保留。
6. KMeans区域、区域几何、停滞计数、重建和全局最好锚点保护。
7. 原始行为过滤、批内重复、历史重复、预测rescue。
8. 全局XGBoost Direct、训练窗口、重训门槛、就绪前近中心选择。
9. Winner-Take-Most、active区域、最大余数、公平轮换、容量不足重分配。
10. 独立novelty名额、已选候选排除、基于历史档案的标准化距离。
11. 真实评估成功预算、代码去重、失败处理、生成次数上限、超时控制。
12. 计时、lineage、历史最优和行为可视化。可视化PCA不是方法搜索PCA。

## 未移入/删除的代码

- 其他baseline与其他搜索方法：HSEvo、ReEvo、PartEvo、各stage变体及消融包装包。
- 旧多算法main选择链、其他方法YAML、旧实验runner、benchmark结果、绘图结果和备份。
- UniXcoder/LocalEmbedder、对比训练、AST语义匹配、ModelScope/Transformers模型下载路径。
- RBFRidgeModel、DeterministicRBFRidge、旧GPR兼容入口、UCB/分层采样及未调用的预测接口。
- 旧Softmax-temperature+D’Hondt区域预算、旧round-robin预算分支。
- 旧score-quantile区域构建、score-separated中心选择路径、旧balanced BX调度分支。
- E1/E2/M1/M2/BE1历史算子别名。
- 未使用的GLS适配副本、离线维度扫描、临时特征测试和单独可视化脚本。
- common目录中的旧reflection/harmony/crossover/mutation提示模板与DPP任务模板。
- external_knowledge.txt（当前BeMRS提示构造没有读取）。
- 运行生成的gpt.py、copy文件、before_*备份、pycache和Apple元数据。
- 旧getParas对象、无效selection/m/n_proc/use_numba等接口参数；checkpoint载入分支未作为新接口保留。
- 个人服务器默认SOCKS代理地址，不再强制重写用户网络代理。

完整文件映射和快照哈希在 MIGRATION_MANIFEST.json。
删除符号分两轮记录于 REMOVED_SYMBOLS.json、REMOVED_SYMBOLS_ADDITIONAL.json，其他清理项见CLEANUP_EXTRAS.json。
这些是审计记录，不是运行时补丁；BeMRS运行不需要迁移脚本。

## 参数处理

| 删除/固定项 | 原因 |
|---|---|
| predictor.rbf.*、predictor.local.* | 只保留全局XGBoost，无RBF或局部模型实现 |
| predictor.model | 模型固定为XGBoost Direct |
| trust_region.elite_multiplier | 独立archive_target_distinct已取代popsize倍数 |
| trust_region.evaluation_allocation.strategy/temperature | 固定Winner-Take-Most，无temperature |
| trust_region.region_builder | 固定KMeans |
| trust_region.kmeans_require_score_gap、center_score_separation | 当前方法不启用该条件，移除选择入口 |
| trust_region.bx_parent_selection.strategy | 固定当前rank-softmax；tau_bx仍保留且有效 |
| behavior_space.full_dimensional | 固定完整选定特征空间，移除搜索PCA切换 |
| problem.behavior.pca_dim | 不参与当前搜索/预测，删除配置项 |
| operators.duplicate_recovery_parent_counts | 没有启用对应恢复算子 |
| Harmony Search的hm_size/hmcr/par/bandwidth/max_iter/mutation_rate | 属于其他方法 |
| top-level algorithm/temperature/init_pop_size | 单一方法直接入口，LLM沿用固定temperature=1，初始化使用method.initialization |
| max_centers/min_centers/ridge_alpha/分层采样参数 | 随旧预测器删除 |

2026-09-29二次清理已删除旧信赖域的length、failure tolerance、competitive gap、
shrink factor和半径收缩/扩张。停滞触发KMeans重聚类属于当前区域维护机制，继续保留，
其触发条件不依赖半径；上一轮将它与信赖域重启一并描述为删除是不准确的。
同样保留BX/BX/BX/BR固定轮换、真实目标值改进阈值和最优锚点保护。
当前接口从trust_region改为region；完整删除/保留清单及测试见REGION_CLEANUP.md。
保留完整行为描述器供feature_group=full使用，因此斜率/diagnostics计算不是被当作死代码删除。
保留stage_ratios：这里stage指探针预算阶段，不是旧stage8版本依赖。

## 独立化修复与行为边界

- 内置FeatureSubset替代实验脚本monkey patch，OP/MKP/CVRP均值输出与旧实现逐元素比对。
- 每个运行独立复制问题入口与候选模块，数据为项目内只读链接，防止多实验互相覆盖gpt.py。
- 恢复OP/MKP/CVRP/TSP构造的按指定size读取val分支。服务器原分支被注释，原main仍会记录validation finished。
  新项目不会把没有输出指标的空跑宣称为验证成功。
- 搜索结束保存best_algorithm.py，验证显式调用evaluate.py，避免自动触发大规模验证。
- TSP构造工厂原先传入构造器不接受的probe_timeout/encode_timeout，改为使用已存在的配置环境传递。
- TSP构造评估器补充select_next_node_v1识别：固定seed模板使用v1，原评估器只接受v2或无后缀名称。
- TSP构造默认encode_timeout从2秒改为5秒：固定seed在原2秒限制下超时，5秒检查通过。
  该项是明确参数变化，不能把这类新运行当作与旧2秒运行严格同配置。
- 凭据只从环境提供，不复制服务器API key。保留原LICENSE/CITATION与问题许可证，命名改变不改变来源。

## 问题完整性与已知限制

8个问题目录全部迁移：op_aco、mkp_aco、cvrp_aco、tsp_gls、tsp_constructive、tsp_aco、bpp_online、bpp_offline_aco。
保留所有已有训练/验证/测试数据、TSPLIB文件及实例生成器。
原snapshot未提供TSP-ACO和BPP-offline的BeMRS行为提取实现；BPP-online工厂引用的文件在原服务器也不存在。
新项目保留这些任务评估入口，但搜索会明确拒绝，不私自发明特征冒充已迁移。
部分任务/规模没有val数据，必须由调用者选择已有数据规模或明确生成数据。

## 验证结果

- Python源码编译检查。
- OP/MKP/CVRP固定seed特征：与服务器原描述器mean切片逐元素相同，分别6/7/7D。
- OP/MKP/CVRP/TSP-GLS/TSP构造配置与特征构造检查，后两者12D/72D。
- 三个轻量测试：可变K/晚期集中/容量重分配/跨批公平；BX远与BR近的排名抽样；完整65预算模拟搜索。
- 模拟搜索使用替代LLM/目标评估，真实XGBoost训练，覆盖BE、区域建立、novelty和预测器启动后选择。
- 三个真实固定seed评估：OP50=12.840；MKP100=22.066064914900455；CVRP50=18.815434509296。
- TSP构造200节点固定seed真实评估：16实例均值13.325519159595594。
- 未运行完整benchmark，未发起LLM生成实验。轻量真实seed评估只确认评估链路，不是性能复现实验。

## 源码与过去实验的关系

这份项目来自当前服务器源码，并以最近baseline设置为默认。旧实验可能采用更早BR策略等实现，
不能承诺新项目与所有历史实验逐步完全相同。保留快照哈希和配置是为了明确复现边界。
