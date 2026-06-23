# AutoPilot — 大模型 GPU 推理部署自动优化系统

给定模型架构、集群拓扑、流量特征、运行画像和 SLO 约束，AutoPilot 自动搜索配置空间生成最优部署方案，并基于运行时 Telemetry 持续闭环调整。

## 安装

需要 Python 3.11+。

```bash
git clone <repo-url> && cd AutoPilot
pip install -e ".[dev]"
```

依赖：pydantic, click, pyyaml, pytest (dev)。

## 运行

### Plan — 生成部署配置

```bash
autopilot plan \
  --model examples/scenario_1_customer_service/model.yaml \
  --cluster examples/scenario_1_customer_service/cluster.yaml \
  --backends examples/scenario_1_customer_service/backends.yaml \
  --traffic examples/scenario_1_customer_service/traffic.jsonl \
  --profiles examples/scenario_1_customer_service/runtime_profiles.yaml \
  --slo examples/scenario_1_customer_service/slo.yaml \
  --output outputs/plan/
```

输出：
- `deployment_plan.yaml` — 推荐配置（TP、PP、精度、Batch 参数、缓存策略）
- `alternatives.json` — 排序后的备选方案及权衡说明
- `decision_report.md` — 完整推理链

### Reconcile — 闭环调整

```bash
autopilot reconcile \
  --plan outputs/plan/deployment_plan.yaml \
  --telemetry examples/scenario_1_customer_service/telemetry.jsonl \
  --output outputs/reconcile/
```

输出：
- `actions.json` — 建议操作（含置信度和风险等级）
- `decision_log.md` — Telemetry 分析与决策日志

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                         Plan 阶段                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入 ──→ Workload 分析器 ──→ WorkloadSummary               │
│       ──→ 显存估算器     ──→ 可行性检查                      │
│       ──→ 候选方案生成器  ──→ 有效配置集合                    │
│       ──→ Profile 查询   ──→ 延迟估算                        │
│       ──→ 评分器         ──→ 排序方案 + 决策报告              │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                      Reconcile 阶段                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Telemetry ──→ 趋势分析器  ──→ 分析摘要                      │
│            ──→ 决策引擎    ──→ 候选操作                       │
│            ──→ 安全守卫    ──→ 过滤后操作                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

系统分为两个独立命令，共享同一套数据模型。`plan` 基于静态输入做离线优化；`reconcile` 基于运行中的方案和观测到的 Telemetry 做在线调整。详见：[docs/architecture.md](docs/architecture.md)

## Workload 分析

Workload 分析器从请求日志中提取流量特征：

| 特征 | 方法 |
|------|------|
| Token 长度分布 | 输入/输出 Token 的 P50/P90/P99 |
| 请求速率 | 滑动窗口 RPS + 峰值/突发比 |
| Prefix 复用 | 覆盖率（携带 prefix 的请求占比）× 复用率（1 - 唯一数/总数）|
| 并发估算 | Little 定律：L = λ × W |
| 时间模式 | 5 分钟窗口的峰谷检测 |
| 负载类型 | Prefill 密集型（输入/输出 > 3）vs Decode 密集型 |

这些特征驱动下游决策（prefix cache、chunked prefill、batch 大小）。详见：[docs/workload-analysis.md](docs/workload-analysis.md)

## 显存估算

单卡 GPU 显存按如下公式计算：

```
模型权重     = 参数量 × 精度字节数 / TP / PP
KV Cache 总量 = 2 × 层数 × KV Head 数 × Head 维度 × KV 精度字节数 / TP
               × max_num_seqs × 上下文长度
运行时开销    = ~5-7 GB（激活、CUDA Graph、框架缓冲区）
总计         = (模型权重 + KV Cache + 运行时开销) × 1.10
```

配置可行的条件：`总计 ≤ GPU 显存 × 0.95`。详见：[docs/memory-estimation.md](docs/memory-estimation.md)

## 候选方案生成与评分

**生成**：穷举 `GPU 池 × 后端 × 精度 × TP × PP × KV Cache 精度` 的笛卡尔积，并按以下规则剪枝：
- TP 必须整除 `num_kv_heads`
- PP 必须整除 `num_layers`
- `TP × PP` 不超过资源池 GPU 数
- 显存必须可容纳
- 精度质量损失满足阈值

**评分**：

```
score = w_cost × 成本分 + w_headroom × 余量分 + w_latency × 延迟余量分 + w_conf × 置信度
```

权重根据优化目标动态调整（`minimize_hourly_cost` vs `maximize_goodput`）。每个方案附带决策理由（rationale），解释各参数选择原因。详见：[docs/scoring.md](docs/scoring.md)

## 闭环控制策略

Reconciler 监控运行时 Telemetry 并决定是否调整：

| 信号 | 条件 | 操作 |
|------|------|------|
| TTFT SLO 违反 | ≥3 个连续窗口超阈值 | 降低 `max_num_seqs` 或扩容 |
| ITL SLO 违反 | ≥3 个连续窗口 | 扩容 |
| KV Cache 临界 | 平均利用率 > 95% | 降低 `max_num_seqs` |
| KV Cache 偏高 | 平均利用率 > 90% | 切换 FP8 KV Cache |
| GPU 低利用率 | ≥5 个连续窗口 < 25% | 缩容 |
| OOM 事件 | 任何 OOM | 激进降低 `max_num_seqs` |

**安全机制：**
- **连续窗口确认** — 单次波动不触发操作
- **非对称阈值** — 缩容（5 窗口）比扩容（3 窗口）更保守
- **冷却期** — 操作后 3 个窗口内禁止新操作
- **单次单操作** — 每个周期只允许一个调整

详见：[docs/closed-loop-control.md](docs/closed-loop-control.md)

## 已知限制

- 延迟估算基于简化模型（线性缩放 + 通信惩罚），生产环境需要实际 profiling 校准
- 显存估算不考虑 PagedAttention 的块级节省和 Prefix Cache 共享内存
- Profile 插值在跨 GPU 类型时置信度较低（0.5），标注为 medium confidence
- Reconcile 阈值基于经验值，需要在实际部署中持续调优
- 不支持预测性扩容，仅支持反应式调整
- 未建模 Speculative Decoding 的额外显存开销

## 扩展：接入新的 GPU、模型或推理后端

### 新 GPU

在 `cluster.yaml` 中添加条目即可，无需改代码：

```yaml
gpu_pools:
  - id: a100-nvlink
    gpu_type: A100-80GB
    count: 8
    memory_gb: 80
    topology: nvlink
    hourly_cost_per_gpu: 2.8
```

为获得准确延迟估算，建议在 `runtime_profiles.yaml` 中添加对应 profile。

### 新模型

创建 `model.yaml`，关键字段：`parameter_count`、`num_layers`（PP 整除）、`num_kv_heads`（TP 整除）、`hidden_size`（计算 head_dim）。

### 新推理后端

在 `backends.yaml` 中添加：

```yaml
backends:
  tensorrt-llm:
    supported_precisions: [bf16, fp8, int4]
    supported_kv_cache_dtypes: [auto, fp8]
    tp_values: [1, 2, 4, 8]
    pp_values: [1, 2, 4]
    features:
      prefix_cache: true
      chunked_prefill: false
      cuda_graph: true
```

然后添加对应 runtime profile。候选方案生成器会自动将新后端纳入搜索空间。

详见：[docs/extending.md](docs/extending.md)

## 设计分析

- [三场景方案分析](docs/scenario-analysis.md) — 各场景输入差异、方案选择逻辑、Reconcile 行为解读
- [关键考量点分析](docs/key-considerations.md) — 12 项核心设计决策的实现方式和验证方法

## 测试

```bash
pytest tests/ -v --cov=autopilot
# 71 个测试，93% 覆盖率
```

覆盖范围：配置合法性、GPU 约束、显存估算、TP/PP/Replica 资源计算、SLO 判断、方案排序、Profile 降级、连续窗口判断、Cooldown 防震荡、高风险操作识别。

### 端到端场景命令

**场景一：客服 Chat**（高 prefix 复用、TTFT 敏感）

```bash
# Plan
autopilot plan \
  --model examples/scenario_1_customer_service/model.yaml \
  --cluster examples/scenario_1_customer_service/cluster.yaml \
  --backends examples/scenario_1_customer_service/backends.yaml \
  --traffic examples/scenario_1_customer_service/traffic.jsonl \
  --profiles examples/scenario_1_customer_service/runtime_profiles.yaml \
  --slo examples/scenario_1_customer_service/slo.yaml \
  --output outputs/scenario_1/plan/

# Reconcile
autopilot reconcile \
  --plan outputs/scenario_1/plan/deployment_plan.yaml \
  --telemetry examples/scenario_1_customer_service/telemetry.jsonl \
  --output outputs/scenario_1/reconcile/
```

**场景二：长文本生成**（decode 密集、低 prefix 复用）

```bash
# Plan
autopilot plan \
  --model examples/scenario_2_long_generation/model.yaml \
  --cluster examples/scenario_2_long_generation/cluster.yaml \
  --backends examples/scenario_2_long_generation/backends.yaml \
  --traffic examples/scenario_2_long_generation/traffic.jsonl \
  --profiles examples/scenario_2_long_generation/runtime_profiles.yaml \
  --slo examples/scenario_2_long_generation/slo.yaml \
  --output outputs/scenario_2/plan/

# Reconcile
autopilot reconcile \
  --plan outputs/scenario_2/plan/deployment_plan.yaml \
  --telemetry examples/scenario_2_long_generation/telemetry.jsonl \
  --output outputs/scenario_2/reconcile/
```

**场景三：混合流量**（Chat + RAG + 长文本、成本敏感）

```bash
# Plan
autopilot plan \
  --model examples/scenario_3_mixed_traffic/model.yaml \
  --cluster examples/scenario_3_mixed_traffic/cluster.yaml \
  --backends examples/scenario_3_mixed_traffic/backends.yaml \
  --traffic examples/scenario_3_mixed_traffic/traffic.jsonl \
  --profiles examples/scenario_3_mixed_traffic/runtime_profiles.yaml \
  --slo examples/scenario_3_mixed_traffic/slo.yaml \
  --output outputs/scenario_3/plan/

# Reconcile
autopilot reconcile \
  --plan outputs/scenario_3/plan/deployment_plan.yaml \
  --telemetry examples/scenario_3_mixed_traffic/telemetry.jsonl \
  --output outputs/scenario_3/reconcile/
```

详见：[docs/testing.md](docs/testing.md)

## 项目结构

```
src/autopilot/
├── cli.py                        # Click CLI 入口
├── orchestrator.py               # 流程编排
├── loader.py                     # YAML/JSONL 加载
├── models/                       # Pydantic v2 数据模型
├── analyzer/workload.py          # 流量分析
├── estimator/memory.py           # 显存公式
├── estimator/profile_lookup.py   # Profile 匹配（含多级降级）
├── planner/candidate_generator.py # 搜索 + 剪枝
├── planner/scorer.py             # 多目标评分排序
├── reconciler/analyzer.py        # Telemetry 趋势分析
├── reconciler/decision_engine.py # 调整决策
├── reconciler/safeguards.py      # 防震荡机制
└── reporter/                     # 报告生成
```
