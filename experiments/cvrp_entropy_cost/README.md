# CVRP5D 本地Qwen3:8B实验

默认三个种子1111、2222、3333依次运行。特征与探针定义见 [CVRP_5D.md](../../CVRP_5D.md)。
除模型传输、随机种子和输出目录外，参数继承正式配置；不修改模型服务的GPU分配或常驻策略。
使用与OP相同的本地Ollama适配器：`http://127.0.0.1:11434/api/chat`、`think=False`、`keep_alive=5m`，没有云端回退。

```bash
cd /root/lhc/BeMRS-standalone

# 三种子固定seed特征检查：不请求LLM，不运行真实评估器
/root/miniconda3/envs/hsevo/bin/python experiments/cvrp_entropy_cost/run.py --check

# 三种子完整搜索（只有手动执行此命令才开始）
/root/miniconda3/envs/hsevo/bin/python experiments/cvrp_entropy_cost/run.py

# 指定某个种子
/root/miniconda3/envs/hsevo/bin/python experiments/cvrp_entropy_cost/run.py --seeds 2222
```

搜索日志存放在 `experiments/cvrp_entropy_cost/runs_local_ollama/5d/seed_<seed>/<时间>/main.log`，
检查日志在 `checks_local_ollama` 下，不覆盖历史CVRP或OP结果。
