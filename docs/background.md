Vibe Coding Coding Test｜大模型 GPU 推理部署自动优化系统


感谢你参与本次技术面试。我们邀请你完成一道 Vibe Coding Coding Test，主要考察你在 AI 辅助开发场景下，对复杂工程问题的拆解、系统设计、代码实现和技术决策能力。

请在收到题目后 48 小时内完成并提交。你可以使用任意编程语言、开源库及 AI Coding 工具。

题目：AutoPilot——大模型 GPU 推理部署自动优化系统
背景
一个模型服务需要部署到包含多种 GPU 的资源池中。不同业务场景对首 Token 延迟、Token 间延迟、吞吐、显存和成本的要求不同。

你需要实现一个自动化部署决策系统。系统根据：

模型属性；

GPU 资源和互联拓扑；

推理后端能力；

业务流量特征；

历史运行画像；

延迟、质量和成本约束；

自动生成较优的推理部署方案，并根据线上运行指标给出动态调整建议。

本题不要求进行性能压测，也不要求真实部署大型模型或 GPU 集群。

一、任务目标
请实现一个命令行工具，例如：

autopilot plan \
  --model examples/model.yaml \
  --cluster examples/cluster.yaml \
  --traffic examples/traffic.jsonl \
  --profiles examples/runtime_profiles.yaml \
  --slo examples/slo.yaml \
  --output outputs/plan/
并支持根据线上指标进行重新决策：

autopilot reconcile \
  --plan outputs/plan/deployment_plan.yaml \
  --telemetry examples/telemetry.jsonl \
  --output outputs/reconcile/
系统需要输出：

outputs/plan/
├── deployment_plan.yaml
├── alternatives.json
└── decision_report.md

outputs/reconcile/
├── actions.json
└── decision_log.md
二、输入信息
1. 模型信息
示例：

name: qwen3-32b
architecture: decoder_only

parameter_count: 32B
num_layers: 64
hidden_size: 5120
num_kv_heads: 8

supported_precisions:
  - bf16
  - fp8

max_model_len: 32768
minimum_quality_retention: 0.99
2. GPU 资源池
示例：

gpu_pools:
  - id: h800-sxm
    gpu_type: H800-80GB
    count: 8
    memory_gb: 80
    topology: nvlink
    hourly_cost_per_gpu: 3.5

  - id: l40s-pcie
    gpu_type: L40S-48GB
    count: 16
    memory_gb: 48
    topology: pcie
    hourly_cost_per_gpu: 1.5
系统需要考虑：

GPU 数量；

单卡显存；

GPU 间互联；

单位时间成本；

不同资源池是否适合较大的 TP 或 PP。

3. 推理后端能力
示例：

backends:
  vllm:
    supported_precisions:
      - bf16
      - fp8

    supported_kv_cache_dtypes:
      - auto
      - fp8

    tp_values:
      - 1
      - 2
      - 4
      - 8

    pp_values:
      - 1
      - 2

    features:
      prefix_cache: true
      chunked_prefill: true
      cuda_graph: true

    constraints:
      - "tp * pp <= gpu_count"
      - "prefix_cache requires kv_cache"
      - "fp8 requires supported_gpu_arch"

  sglang:
    # 支持的参数和约束可能不同
评测时可能增加新的后端，因此请避免将全部逻辑硬编码为：

if backend == "vllm":
    ...
elif backend == "sglang":
    ...
允许为后端实现 Adapter，但通用规划逻辑应尽量由配置驱动。

4. 业务流量
示例：

{
  "timestamp_ms": 1200,
  "input_tokens": 3150,
  "output_tokens": 210,
  "prefix_id": "customer-service-kb-v3",
  "priority": "interactive"
}
流量中可能包含：

短输入、长输出；

长输入、短输出；

突发流量；

大量共享前缀；

几乎没有前缀复用；

不同优先级；

不同时段的流量变化。

5. 历史运行画像
系统可以使用已有的历史运行数据辅助决策，例如：

profiles:
  - gpu_type: H800-80GB
    backend: vllm
    precision: bf16
    tp: 4

    maximum_prefill_tokens_per_second: 24000
    maximum_decode_tokens_per_second: 3600

    base_ttft_ms: 150
    base_itl_ms: 18

    runtime_memory_overhead_gb: 7
    communication_penalty:
      nvlink: 1.0
      pcie: 1.35
这些数据可以理解为已有系统积累的运行画像。候选人不需要生成这些数据，但需要合理使用，并处理画像缺失、数据稀疏或部分参数组合没有记录的情况。

6. 业务目标和约束
示例：

objective:
  primary: minimize_hourly_cost
  secondary: maximize_goodput

constraints:
  p95_ttft_ms: 800
  p95_itl_ms: 45
  p99_e2e_ms: 8000

  minimum_quality_retention: 0.99
  minimum_capacity_headroom: 0.20
  maximum_gpu_count: 8
系统应优先满足硬约束，再根据配置的优化目标选择方案。

三、部署方案
系统至少需要对以下配置进行决策：

gpu_pool: h800-sxm
backend: vllm

replicas: 2
tensor_parallel: 4
pipeline_parallel: 1

precision: fp8
kv_cache_dtype: fp8

max_num_seqs: 128
max_num_batched_tokens: 8192

enable_prefix_cache: true
enable_chunked_prefill: true
prefill_chunk_size: 2048

estimated:
  hourly_cost: 28.0
  peak_memory_per_gpu_gb: 72.5
  capacity_headroom: 0.24
你不需要实现以上所有参数，但至少应覆盖：

GPU Pool；

Replica 数量；

TP；

PP；

Precision；

一个 Batch 或调度参数；

一个 Cache 或显存相关参数。

四、必须完成的功能
1. Workload 分析
程序至少需要分析：

输入和输出 Token 长度分布；

P50、P90、P99；

平均请求速率和峰值请求速率；

突发流量程度；

Prefix 复用率；

潜在并发量；

Workload 更偏向 Prefill 还是 Decode；

Workload 更偏向低延迟还是高吞吐。

这些分析结果必须实际影响部署决策，而不能只出现在报告中。

2. 显存和资源可行性判断
系统需要判断一个配置是否能够运行，至少考虑：

模型权重显存；

Precision 对权重显存的影响；

KV Cache 显存；

最大上下文长度；

并发请求数；

Runtime 显存开销；

安全余量；

TP 和 PP 对模型切分的影响；

GPU 数量限制。

不要求建立完全精确的显存模型，但需要说明使用的公式、假设和安全边界。

3. 自动生成部署方案
不能只返回一套固定规则配置。

系统应：

生成多个合法候选方案；

排除明显不可行的方案；

根据资源、SLO、历史画像和 Workload 对方案评分；

输出最终推荐方案；

输出至少两个备选方案；

解释最终方案与备选方案之间的权衡。

搜索方法不限，可以使用：

启发式规则；

约束求解；

Beam Search；

Local Search；

Integer Programming；

多目标优化；

上述方法的组合。

4. 不完整数据处理
实际场景中，历史画像可能不完整。

系统需要合理处理：

某种 GPU 没有对应画像；

某个 TP 配置没有记录；

新后端只有少量数据；

Profile 数据存在异常值；

某些参数只能做定性判断。

可以使用插值、保守估算、降级规则或置信度评分，但必须明确区分：

已知事实；

模型估算；

工程假设；

尚未验证的风险。

5. 线上闭环调节
reconcile 命令需要读取一段线上 Telemetry，例如：

{
  "timestamp": "2026-06-20T10:00:00Z",
  "request_rate": 12.5,
  "queue_depth": 18,

  "p95_ttft_ms": 920,
  "p95_itl_ms": 38,

  "gpu_utilization": 0.72,
  "kv_cache_utilization": 0.94,

  "oom_count": 0,
  "error_rate": 0.001
}
系统需要根据当前部署方案和连续多个时间窗口的数据，给出可能的操作，例如：

{
  "action": "scale_replicas",
  "from": 2,
  "to": 3,
  "reason": "TTFT SLO violated for 3 consecutive windows",
  "confidence": 0.87
}
支持的操作可以包括：

增加或减少 Replica；

调整最大并发请求数；

调整 Batch Token Budget；

启用或关闭 Prefix Cache；

调整 Prefill Chunk Size；

建议切换 TP、PP 或 GPU Pool；

保持当前配置。

涉及重启或数据迁移的操作需要标注风险，不应与在线热更新操作混为一谈。

6. 防止配置震荡
闭环控制至少需要考虑：

连续多个窗口确认；

Cooldown；

Hysteresis；

最小样本量；

扩容和缩容使用不同阈值；

高风险变更需要人工确认；

指标恢复后的回退策略。

例如，系统不应因为单个时间窗口的 TTFT 波动，就不断扩容和缩容。

7. 可解释性
decision_report.md 至少需要包含：

Workload 摘要；

关键资源约束；

最终推荐配置；

方案评分；

预计显存占用；

预计容量和资源余量；

预计成本；

为什么选择该配置；

为什么没有选择更大的 TP；

为什么选择或不选择 Prefix Cache；

备选方案及适用条件；

当前结果的置信程度；

尚未验证的假设；

上线前建议补充验证的内容。

五、建议覆盖的场景
请至少准备三个示例 Scenario。

场景一：高 Prefix 复用的客服 Chat
特点：

输入较长；

输出中等；

大量请求共享知识库前缀；

TTFT 要求严格；

流量存在明显高峰。

场景二：长文本生成
特点：

输入较短；

输出较长；

Prefix 复用率低；

Decode 能力是主要瓶颈；

更关注 ITL 和输出吞吐。

场景三：成本敏感的混合流量
特点：

同时包含 Chat、RAG 和长文本生成；

流量随时间变化；

可选择高性能 GPU 或低成本 GPU；

需要在成本、延迟和容量余量之间进行权衡。

六、提交物
请提交一个可运行的代码仓库，建议结构如下：

repository/
├── README.md
├── AI_USAGE.md
├── pyproject.toml / requirements.txt
├── src/
├── tests/
├── examples/
└── outputs/
README.md
至少需要说明：

安装和运行方式；

一条完整的运行命令；

系统架构；

Workload 分析方法；

显存估算方法；

候选方案生成和评分方法；

闭环控制策略；

已知限制；

如何接入新的 GPU、模型或推理后端。

AI_USAGE.md
本次任务允许并鼓励使用 AI Coding 工具。

请简要说明：

使用了哪些 AI 工具；

AI 主要帮助完成了哪些部分；

一个被采纳的关键建议；

一个被拒绝或修正的 AI 建议；

如何验证 AI 生成的代码；

哪些关键技术判断由你独立完成。

无需提供完整私人对话记录。

测试
至少覆盖：

配置合法性；

GPU 数量约束；

显存估算；

TP、PP 和 Replica 资源计算；

SLO 判断；

方案排序；

Profile 缺失时的降级逻辑；

Reconcile 的连续窗口判断；

Cooldown 和防震荡逻辑；

高风险操作识别。

七、不要求完成
48 小时内不要求：

真实部署大型模型；

自备 GPU；

进行性能压测；

编写 CUDA Kernel；

搭建 Kubernetes 集群；

调用真实云平台 API；

实现完整的生产级调度系统；

实现 Web UI。

请优先保证以下核心链路可运行：

Model + Cluster + Traffic + Profiles
                ↓
       Feasibility Analysis
                ↓
       Deployment Planning
                ↓
       Explainable Decision
                ↓
        Telemetry Reconcile
八、评估重点
我们将重点关注：

是否正确理解模型推理的资源和性能约束；

显存、KV Cache、并发和上下文长度之间的关系；

TP、PP 和 Replica 的不同作用；

GPU 拓扑对并行方案的影响；

Prefix Cache 对不同 Workload 的适用性；

是否优先满足硬约束；

优化目标是否清晰且可配置；

数据缺失时的处理是否合理；

闭环控制是否安全，是否会产生配置震荡；

系统是否易于扩展到新的 GPU 和推理后端；

是否有效使用 AI Coding 工具并验证其输出；

是否明确说明事实、估算、假设和风险。

我们不会以代码量、界面完成度或使用 AI 工具的多少作为主要评价标准。

九、提交方式
请在 48小时 内将以下内容发送至 本邮箱：

代码仓库链接或压缩包；

一条可复现的运行命令；

示例输入和输出；

如有未完成内容，请在 README 中明确说明。

如遇到阻塞性问题，请直接联系我。

谢谢，期待看到你的方案。