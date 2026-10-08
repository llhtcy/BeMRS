# CVRP 2D/5D 本地Qwen3:8B对照实验

默认先依次运行2D的1111、2222、3333，再运行5D的1111、2222、3333，共6次独立搜索。
特征与探针定义见 [CVRP_5D.md](../../CVRP_5D.md)。
默认5个实例，每实例前、中、后各4个随机可行状态，共60个探针。
脚本明确固定实例5、每层探针4、随机均匀分层、模型传输、特征组、随机种子和输出目录，
其余方法参数继承服务器正式配置。
使用与OP相同的官方BF16 vLLM适配器：`http://127.0.0.1:8000/v1/chat/completions`，关闭思考，无云端/Ollama回退。
运行命令时加载GPU 0/1/2/3上的四卡TP模型，六个任务共用一次加载，命令结束后释放，不常驻。
温度1.0、top_p=0.95、top_k=20、重复惩罚1.0；上下文40960、输出上限8192，截断时报错。
`--check`与`--dry-run`不会启动或加载模型。部署见[说明](../../deployment/vllm/README.md)。

| 组别 | 输出特征 |
|---|---|
| 2D | 决策熵、特征提取耗时（s） |
| 5D | 决策熵、特征提取耗时（s）、距离偏好、路径节约偏好、需求量偏好 |

两组使用相同状态库及核心计算/计时流程：2D只返回前两列，5D返回全部五列。
2D中三种偏好的核心统计仍计算但不作为聚类或预测器输入，以免改变时间特征的测量定义。
同一种子共享相同探针；LLM输出和墙钟计时不保证严格可复现。正式CVRP默认配置仍然是5D，脚本不改写配置文件。

```bash
cd /root/lhc/BeMRS-standalone

# 两组各三个种子的固定seed特征检查：不请求LLM，不运行真实评估器
/root/miniconda3/envs/hsevo/bin/python experiments/cvrp_entropy_cost/run.py --check

# 完整六次搜索：先2D三次，再5D三次（只有手动执行此命令才开始）
/root/miniconda3/envs/hsevo/bin/python experiments/cvrp_entropy_cost/run.py

# 只运行2D的三个种子
/root/miniconda3/envs/hsevo/bin/python experiments/cvrp_entropy_cost/run.py --groups 2d

# 指定某组某个种子
/root/miniconda3/envs/hsevo/bin/python experiments/cvrp_entropy_cost/run.py --groups 5d --seeds 2222

# 只打印任务计划，不加载模型、不执行子进程
/root/miniconda3/envs/hsevo/bin/python experiments/cvrp_entropy_cost/run.py --dry-run
```

搜索日志存放在 `experiments/cvrp_entropy_cost/runs_local_vllm/<2d或5d>/seed_<seed>/<时间>/main.log`，
检查日志在 `checks_local_vllm` 下，不覆盖历史CVRP或OP结果；旧Ollama实验保留原目录。
同一个命令中的任务顺序执行；某次失败会停止，不会跳过后伪装成全部成功。每次新调用使用新的时间戳。
