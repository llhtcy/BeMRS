# OP 熵与提取耗时：2D → 5D

默认顺序：2D/1111、2D/2222、2D/3333、5D/1111、5D/2222、5D/3333。
覆盖问题、种子、特征组、输出目录、检查模式及本地LLM连接；其他参数继承服务器当时配置。
任何任务失败后停止，不带着失败继续后续任务。每次命令创建新时间戳，不覆盖历史实验。

| 组 | 输入预测器/聚类/行为过滤的特征 |
|---|---|
| 2D | 决策熵、特征提取耗时（s） |
| 5D | 上述2维、奖励偏好均值、距离偏好均值、效率偏好均值 |

两组同一流程计算现有OP core描述器，只输出不同列，保证耗时定义一致。
时间包括编译、每实例一次候选调用及描述器统计，排除建库、首次导入及日志写入。
不计算扰动稳定性、斜率或矩阵诊断。2D仍计算偏好统计，但不把它们传给任何下游模块。
该耗时与旧response3的扰动提取耗时不在同一测量口径，不可直接视为同一个特征数值。

奖励偏好=`sum(p * percentile_rank(prize))`；距离偏好=`sum(p * percentile_rank(distance))`；
效率偏好=`sum(p * percentile_rank(prize / detour))`。
`detour=d(current,node)+d(node,depot)-d(current,depot)`，分母至少1e-12。
概率沿用现有正值处理后的启发式权重归一化，排名并列取平均。
先对各实例固定探针求均值，再对实例求均值。
奖励/效率偏好越高表示偏向高奖励/高效率；距离偏好越高表示偏向较远节点，保留旧编码方向。
测量时间存在机器负载噪声；两组都沿用已有标准化、预测和过滤，不修改其他机制。

```bash
cd /root/lhc/BeMRS-standalone
/root/miniconda3/envs/hsevo/bin/python experiments/op_entropy_cost/run.py
# 检查六个设置，不调用LLM或真实评估
/root/miniconda3/envs/hsevo/bin/python experiments/op_entropy_cost/run.py --check
# 指定组和种子
/root/miniconda3/envs/hsevo/bin/python experiments/op_entropy_cost/run.py --groups 5d --seeds 2222
```

本地模型固定为 `qwen3:8b`，通过双卡网关 `http://127.0.0.1:11434/api/chat`
调用，`think=false`，温度保留正式接口的1.0，并发继承现有配置（6）。
传输替换只发生在实验子进程，不修改正式模块；不会回退到云端模型。
返回对象保留现有 InterfaceAPI 接口，日志包含实际后端端口和模型token计数。
原LLMBatch计数仍是现有tiktoken估计口径。默认上下文继承Ollama，不新增生成策略参数。

```bash
# 用实际InterfaceAPI发6个短并发请求验证本地调用，无搜索或真实评估
/root/miniconda3/envs/hsevo/bin/python experiments/op_entropy_cost/run.py --llm-check
```

输出：`experiments/op_entropy_cost/runs_local_ollama/{2d,5d}/seed_*/时间戳/`。

## 随机状态对照

加 `--random-states`：从仓库出发，每步在全部当前合法节点中均匀随机选一个，
走到无法继续；从该路径中至少两动作合法的状态里均匀随机抽一个。
重复路径采样，去重后每实例保留12个状态（沿用原3×4的数量，不使用0.2/0.5/0.8阶段目标）。
可达路径、合法动作、已行驶距离保持自洽，不直接随机拼接访问掩码与预算。
它仍依赖问题可行性规则，但不使用奖励/距离/效率优先选择的专家策略。
固定探针种子，2D与5D同种子共用同一库；日志记录每实例bank_hash。
随机状态不保证阶段均匀覆盖。原始实例与探针数量不变。

```bash
/root/miniconda3/envs/hsevo/bin/python experiments/op_entropy_cost/run.py --random-states
/root/miniconda3/envs/hsevo/bin/python experiments/op_entropy_cost/run.py --random-states --check
```

结果独立保存到 `runs_random_states_local_ollama/{2d,5d}/seed_*/时间戳/`。
只在该脚本子进程替换探针建库函数，正式默认描述器和此前实验不变。

## 均匀分层随机状态（推荐本轮测试）

使用 `--stratified-random-states --groups 5d`。
随机可行路径中的状态按预算进度 `[0,1/3)`、`[1/3,2/3)`、`[2/3,1]`
分为三池，每池抽4个，共12个/实例、60个总探针。
按当前节点与可行动作集合跨层去重，随机抽样优先覆盖不同当前节点。
最多生成240条随机路径/实例；不足时报错，不复制探针补数。
日志记录路径数量、建库耗时秒、各池规模、去重数量和bank_hash。
不使用奖励或距离优先规则，也不做固定0.2/0.5/0.8目标状态选择。
其他参数沿用当前服务器配置；此前模式保留，结果单独写入
`runs_stratified_random_states_local_ollama/5d/seed_*/时间戳/`。
实际参数由Hydra保存在每轮快照中；其中可能包含API凭据，不要发布或提交快照。
