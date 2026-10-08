# 官方 Qwen3-8B BF16：四卡 vLLM

只替换LLM运行环境、传输与生命周期，不改变BeMRS搜索、行为特征、父代、预测器或真实评估器。
不改写服务器含凭据的`cfg/config.yaml`。新结果与Ollama结果独立保存。

## 服务器布局

- 独立环境：`/root/lhc/bemrs-vllm-venv`，固定`vllm==0.31.0`，不更动`hsevo`环境。
- 权重：`/root/lhc/models/Qwen3-8B`，来自Qwen官方ModelScope仓库`Qwen/Qwen3-8B`。
- 五个原始Safetensors分片，BF16；没有GGUF转换、AWQ、GPTQ、FP8或4位量化。
- 下载器验证全部文件大小/SHA256及Safetensors头部dtype，保存`bemrs_checkpoint_provenance.json`。
- API：`http://127.0.0.1:8000/v1`，仅监听本机，不向公网开放。
- GPU：0、1、2、3，`tensor_parallel_size=4`：一份模型分布在四卡，并非四个独立副本。
- 每卡显存比例0.35，为服务器已有任务留余量；不足时报错，不杀其他任务。
- RTX4090无NVLink，关闭自定义all-reduce/P2P/IB，通过NCCL普通通信；四卡不保证比单卡更快。
- 上下文40960，最多6个并发序列，输出上限8192；关闭思考。
- 显式沿用旧模型调用的温度1.0、top_p=0.95、top_k=20、重复惩罚1.0，避免自动继承HF默认温度0.6。
- 使用vLLM自带的Triton/native采样后端，关闭需要额外CUDA工具链的FlashInfer采样JIT；采样参数不变，不修改全局CUDA或驱动。

## 使用

OP/CVRP已有实验脚本已经接入按需生命周期。服务在一次命令的全部种子/特征组间复用，
结束、异常或Ctrl+C后清理本命令拥有的进程组；不停止既有外部服务。
同时启动第二个受管理命令会报错，请按顺序运行。

```bash
cd /root/lhc/BeMRS-standalone

# 六个短并发请求；无搜索和真实评估
/root/miniconda3/envs/hsevo/bin/python experiments/op_entropy_cost/run.py --llm-check

# OP 5D，随机均匀分层探针，三个种子
/root/miniconda3/envs/hsevo/bin/python experiments/op_entropy_cost/run.py \
  --stratified-random-states --groups 5d --seeds 1111 2222 3333

# CVRP 2D后5D，三个种子
/root/miniconda3/envs/hsevo/bin/python experiments/cvrp_entropy_cost/run.py

# 正式main入口，其余参数由服务器配置继承；模型连接仅在运行时覆盖
/root/miniconda3/envs/hsevo/bin/python deployment/vllm/run_search.py problem=op_aco seed=1111

# 可选手动前台服务；Ctrl+C退出，后续实验可复用此服务
/root/miniconda3/envs/hsevo/bin/python -m deployment.vllm.runtime
```

服务日志：`deployment/vllm/logs/server_*.log`；实验日志使用`*_local_vllm`新目录。
未完成的历史Ollama任务可继续使用旧11434接口，不会被切换或终止。
非思考模式与权重精度是独立变量；更换推理后端本身也不保证输出数值完全一致。

## 复现部署

```bash
# 首次安装需要C/C++编译器和标准头文件，供Triton GPU内核初始化使用
apt-get install -y --no-install-recommends gcc g++ libc6-dev
/root/miniconda3/envs/hsevo/bin/python -m venv /root/lhc/bemrs-vllm-venv
/root/lhc/bemrs-vllm-venv/bin/python -m pip install 'vllm==0.31.0'
/root/miniconda3/envs/hsevo/bin/python deployment/vllm/download_model.py
```

不要向Git仓库提交权重、环境、实验快照、日志或API凭据。
