# BeMRS

独立的 Behavior-guided Multi-Region Search 项目。源码来自服务器
`/root/lhc/HSEvo-main` 的 2026-09-29 快照，整理记录见 `MIGRATION.md`。
运行时不依赖 HSEvo-main、baseline 或任何 stage 包，不需要实验脚本 monkey patch。

## 安装与运行

Python 3.10+。推荐新建环境后安装：

```bash
pip install -r requirements.txt
export OPENAI_API_KEY='你的密钥'
export OPENAI_BASE_URL='你的兼容接口地址'
python main.py problem=op_aco seed=1111
```

不含真实凭据。模型默认 `openai/qwen3-8b`，可用 `model=...` 覆盖。
LLM 请求沿用原实现 temperature=1。随机种子控制行为探针、区域抽样、预测器和初始化提示词，
不保证服务端 LLM 完全确定。

```bash
python main.py problem=mkp_aco seed=2222
python main.py problem=cvrp_aco seed=3333
python main.py problem=tsp_gls seed=1111
python main.py problem=tsp_constructive seed=1111
```

默认结果存放 `runs/<problem>/seed_<seed>/<时间>/`，包含真实生效配置、日志、lineage、计时、
历史最优代码 `best_algorithm.py` 和可视化。每个运行目录有独立的 evaluator workspace，
避免不同种子共同覆盖问题目录中的 `gpt.py`。工作目录中的数据链接均指向本项目，不指向原项目。

## 默认设置

| 参数 | 默认 |
|---|---|
| OP / MKP / CVRP 特征 | Mean 6D / 7D / 7D |
| TSP-GLS / TSP 构造特征 | compact12 / 原有72D |
| 种群 / 档案 | 10 / 24个不同目标值，保留并列 |
| 区域 / 子代生成 | 3 / 每区域父代数决定 BX、BR 数量，同轮混合筛选 |
| 总成功真实评估预算 | 210（包含固定seed评估） |
| 正式搜索真实评估 | 有效去重新候选的20%，向上取整；novelty从中预留1个 |
| 行为过滤 | atol=1e-6，rtol=1e-4，XGBoost就绪后启用，rescue开启 |
| XGBoost | 起始50样本，最近最多200样本，新增10样本重训 |
| 区域预算 | 线性 Winner-Take-Most + 最大余数整数配额 |
| BX/BR | 原概率父代选择；区域内BX、跨区域BX、BR同轮生成，BX名额均分并共用上限 |

常用覆盖：

```bash
python main.py problem=op_aco pop_size=10 method.region.archive_target_distinct=48
python main.py method.behavior_filter.rtol=0.01
python main.py method.novelty_selection.enabled=false
python main.py problem=op_aco problem.behavior.feature_group=full
```

OP/MKP/CVRP 的完整描述器保留，`feature_group=mean` 是内置投影，不是运行时修改类定义。
Probe阶段、特征计算、评估函数沿用原实现。

## 检查与评估已有算法

```bash
python main.py check=true problem=op_aco
python -m unittest discover -s tests
python evaluate.py --problem op_aco --code prompts/op_aco/seed_func.txt --size 50 --output checks/eval_seed
python evaluate.py --problem op_aco --code runs/你的运行/best_algorithm.py --size 50 --split val --output checks/eval_best_val
```

`check=true` 只构造模型/探针并提取固定seed特征，不调用LLM，不运行真实目标评估。
单元测试使用模拟LLM、模拟行为向量和模拟目标值，真实XGBoost训练仍运行。
区域参数统一为`method.region.*`，日志使用`Region`/`R1`等名称。
已移除旧半径扩缩机制；正式搜索不再使用BX/BR固定轮换，停滞后的KMeans重聚类保留。
本次生成规则、比例分母与边界条件见 `MIXED_GENERATION.md`。
`evaluate.py` 是真实评估，只运行用户提供的算法。已有非空输出目录拒绝覆盖。
验证集必须实际存在，缺失时不静默使用训练集。搜索结束不自动运行全部验证规模，避免隐藏额外成本。

## 问题支持边界

完整保留原服务器的8个问题目录、实例生成器和已有数据文件。

| 问题 | BeMRS搜索 | 独立评估入口 |
|---|---|---|
| OP-ACO、MKP-ACO、CVRP-ACO | 支持 | 支持 |
| TSP-GLS、TSP constructive | 支持 | 支持 |
| TSP-ACO、BPP-offline-ACO | 原项目未提供对应行为提取器，未新增方法 | 保留 |
| BPP-online | 原工厂引用的行为提取器文件缺失，明确报错 | 保留 |

后三类不能宣称已完成BeMRS搜索支持。评估器保留不等同于已经逐类完成运行验证。
若要支持这些问题的BeMRS搜索，需要另外定义并验证行为特征。

## 来源与许可

保留原始 `LICENSE`、`CITATION.cff` 和问题子目录中的许可证。
命名统一不改变原代码的来源与许可义务。详见 `MIGRATION.md`。
