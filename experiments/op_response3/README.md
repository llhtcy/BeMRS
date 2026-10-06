# OP 三维响应行为实验

独立实验入口，只改变 `problem.behavior.feature_group=response3`。
其他参数继承运行时服务器的 `cfg/config.yaml`、`cfg/method/bemrs.yaml`
和 `cfg/problem/op_aco.yaml`，不修改默认 Mean6D 基线。

三个特征依次是：

1. **决策熵**：原始可行动作的启发式值按现有 ACO 规则转为概率，
   `-sum(p log p) / log(m)`，跨固定探针求平均。
2. **特征提取耗时（秒）**：单算法完整三维特征提取的 wall-clock 时间，
   包括编译、原始及扰动调用、排名与熵统计；不包括一次性探针库构建、日志写出、真实评估。
   保留秒单位；区域聚类沿用现有 StandardScaler，预测器沿用现有输入处理。
3. **输入稳定性**：所有奖励和预算乘 `1+0.05*U`，所有客户坐标加
   `0.05*median_edge_length*U`，其中各分量独立 `U ~ Uniform[-1,1]`。
   仓库固定，距离矩阵及原路径长度重新计算，当前/已访问节点固定。
   共同合法动作中采用平均并列百分位排名，稳定性为 `1-mean(abs(r-r'))`。

固定每个探针一份扰动，所有算法共享；原始/扰动函数调用使用共同随机数。
少于两个共同动作的探针不进入稳定性平均；日志保留有效对数及原候选保留率。
调用失败或超时返回三个 -1，由既有无效特征流程处理，不给失败算法伪造稳定性。
熵/稳定性可复现，时间有系统噪声，不能解释为性能或纯算法复杂度。
本实验衡量三维向量整体效果，不能单独证明每一维有效。

```bash
/root/miniconda3/envs/hsevo/bin/python experiments/op_response3/run.py --check
/root/miniconda3/envs/hsevo/bin/python experiments/op_response3/run.py
/root/miniconda3/envs/hsevo/bin/python experiments/op_response3/run.py --seeds 2222
```

默认依次运行 1111、2222、3333。结果位于此目录的 `runs/seed_*/时间戳/`。
每次运行独立目录，不覆盖先前结果；Hydra 保存实际参数快照。
参数快照可能包含运行环境中的 API 凭据，不要上传 GitHub。
`--check` 不生成算法、不调用 LLM、不执行真实评估。
