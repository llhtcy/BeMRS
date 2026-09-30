# 区域模块二次清理（2026-09-29）

本次仅清理独立BeMRS项目，不修改原HSEvo-main、实验结果和问题数据。
清理前本地可恢复备份：`/Users/llhtcy/ours/BeMRS.before_trust_cleanup_20260929`。

## 删除项

| 项目 | 删除范围与原因 |
|---|---|
| 旧区域半径 | length初值/上下限、归一化半径；当前区域是最近质心划分，不以方盒限定候选 |
| 半径自适应 | failure_count/failure_tolerance、扩张/收缩次数、shrink_factor、competitive_gap以及对应配置映射 |
| 方盒成员判断 | inside/relative_distance/length_at_proposal及父代inside计数；不再以旧半径参与同目标值父代排序 |
| 目标差值约束 | 永远返回空的score-conflict函数、不会启用的KMeans score-gap判断及冲突修复路径 |
| 旧预算日志 | 永远不会执行的temperature/quality_probabilities日志分支；WTM分配公式保持不变 |
| 搜索PCA | 先读取参数再强制设为0的配置代码、不可达投影分支和多余状态；仍使用完整选定特征的StandardScaler |
| 未使用状态 | 旧_visual_trust_*、不再更新的成功计数、未读取的重建元数据、fit_all入参以及无效pass |

## 保留项及理由

- KMeans初建、BE完成后的建区、停滞后的全区域重新聚类：是多区域维护，不是半径重启。
- rebuild_tolerance=5、preserve_global_best_anchor=true：保留原有停滞计数及最优锚点保护。
- relative_improvement=0.0001、scale_window=100：真实目标改进判定仍使用原阈值，没有改成任何微小下降都更新。
- 每次显著改进更新区域真实最优及算法锚点；质心仅在重新聚类时变化。
- BX/BX/BX/BR固定3:1轮换；BX远向与BR近向rank-softmax父代选择。
- 区域候选生成预算、WTM真实评估预算、区域内部XGBoost排序、novelty独立名额。
- 优势档案、行为过滤和rescue、特征提取、真实评估与数据文件。
- 仅用于可视化的PCA：不影响搜索或特征维度。

上一轮临时编辑错误地删除了固定算子调度和KMeans维护入口，本轮已恢复。
没有把这些有效组件当作“信赖域残留”删除。

## 接口变化

`method.trust_region.*` → `method.region.*`。
`BEMRS_TRUST_REGION_*`/`BEMRS_SOFT_TRUST_REGION_EVAL_BATCH_SIZE` → `BEMRS_REGION_*`。
运行时函数、状态、lineage字段、日志、center_history文件名同步改名。
旧实验日志不修改。分析新日志的脚本应读取`region_source_id`、`region_membership_id`、
`region_allocation_id`等新字段。审计文档保留旧名字作为历史映射，不能将其当作运行代码。

## 验证

- 9个轻量单元测试：原3项加6项区域回归测试。
- 65次模拟成功评估的完整搜索，真实XGBoost、模拟LLM/行为特征/目标评估。
- 区域回归覆盖：真实评估更新及固定质心、原改进阈值、停滞重聚类与最优保护、
  3:1算子轮换、远离质心仍可归属、K=1/2/3/6聚类及真实最优锚点。
- OP/MKP/CVRP的check=true入口：只提取固定seed特征，不调用LLM或真实目标评估。
- 打包时核对问题dataset/test文件与来源快照逐字节一致。

这次移除了同目标值父代排序中旧inside优先级，因此不宣称与历史搜索轨迹逐步一致。
未运行完整benchmark或付费LLM实验。
