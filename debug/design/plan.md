# AutoPilot 详细设计计划

## 一、项目概述

实现一个 CLI 工具，根据模型属性、GPU 资源、推理后端能力、业务流量、历史画像和 SLO 约束，自动生成最优推理部署方案，并支持基于线上 Telemetry 的闭环动态调整。

---

## 二、技术选型

| 维度 | 选择 | 理由 |
|------|------|------|
| 语言 | Python 3.11+ | 生态丰富，适合数据处理和建模 |
| CLI 框架 | Click | 轻量、成熟、支持子命令 |
| 配置解析 | PyYAML + Pydantic v2 | YAML 解析 + 强类型校验 |
| 搜索/优化 | 自研启发式 + 约束过滤 | 不引入重依赖，可控性强 |
| 测试 | pytest | 标准选择 |
| 包管理 | pyproject.toml + pip | 简洁标准 |

---

## 三、系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                          CLI Layer (Click)                        │
│         autopilot plan / autopilot reconcile                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                     Orchestrator                                  │
│  plan_workflow() / reconcile_workflow()                           │
└───┬──────────┬──────────┬──────────┬──────────┬────────────────┘
    │          │          │          │          │
    ▼          ▼          ▼          ▼          ▼
┌───────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
│Workload│ │Memory  │ │Candidate│ │Scorer/ │ │Reconciler│
│Analyzer│ │Estimator│ │Generator│ │Ranker  │ │          │
└───────┘ └────────┘ └────────┘ └────────┘ └──────────┘
    │          │          │          │          │
    ▼          ▼          ▼          ▼          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Data Models (Pydantic)                        │
│  ModelSpec, ClusterSpec, TrafficProfile, RuntimeProfile, SLO...   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 四、模块设计

### 4.1 数据模型层 (`src/autopilot/models/`)

| 文件 | 内容 |
|------|------|
| `model_spec.py` | 模型定义：参数量、层数、hidden_size、kv_heads、精度、max_len |
| `cluster.py` | GPU Pool：类型、数量、显存、拓扑、成本 |
| `backend.py` | 推理后端能力：支持的精度、TP/PP、Features、约束（配置驱动） |
| `traffic.py` | 单条流量记录 + Workload 分析结果 |
| `profile.py` | 历史运行画像 |
| `slo.py` | 业务目标和约束 |
| `plan.py` | 部署方案数据结构 |
| `telemetry.py` | 线上指标数据 |

### 4.2 Workload 分析器 (`src/autopilot/analyzer/workload.py`)

**输入**: `traffic.jsonl`

**输出**: `WorkloadSummary`

**分析内容**:
- Token 长度分布：input/output 的 P50、P90、P99
- 请求速率：平均 RPS、峰值 RPS、突发系数（peak/avg）
- Prefix 复用率：unique prefix count / total requests with prefix
- 并发估算：基于到达率和预估处理时间（Little's Law）
- Workload 特征分类：
  - prefill_heavy vs decode_heavy（基于 input/output token 比例）
  - latency_sensitive vs throughput_oriented（基于 priority 分布）
- 时间模式：流量是否有明显的峰谷

**决策影响**:
- prefix 复用率高 → 倾向启用 prefix cache
- prefill heavy → 需要更大的 prefill chunk budget
- decode heavy → 需要关注 ITL，可能需要更多 decode 吞吐
- 突发流量高 → 需要更高的 capacity headroom 或更多 replica
- 高并发 → 影响 max_num_seqs 设置

### 4.3 显存估算器 (`src/autopilot/estimator/memory.py`)

**显存模型**:

```
total_memory_per_gpu = model_weight_memory + kv_cache_memory + runtime_overhead + safety_margin

model_weight_memory = parameter_count * bytes_per_param / tp
  - bf16: 2 bytes/param
  - fp8: 1 byte/param

kv_cache_per_token = 2 * num_layers * num_kv_heads * head_dim * kv_dtype_bytes / tp
  - head_dim = hidden_size / num_attention_heads (需要从模型spec推导或约定)
  - kv_dtype_bytes: auto(同model精度) 或 fp8(1 byte)

kv_cache_memory = kv_cache_per_token * max_num_seqs * max_context_length
  (保守取 max_model_len 或根据 P99 input+output 动态调整)

runtime_overhead = from profile (runtime_memory_overhead_gb) or default 5-8 GB
safety_margin = 10% of GPU memory
```

**可行性判断**:
- `total_memory_per_gpu <= gpu_memory_gb * 0.95`
- `tp * pp * replicas <= gpu_count`
- `tp` 必须在后端支持的值内
- `pp` 必须在后端支持的值内
- PP 场景下层需要能整除 pp

**假设和风险说明**:
- head_dim 假设为 hidden_size / (num_kv_heads * kv_head_ratio)，需要明确
- Activation memory 在推理时相对较小，暂不单独建模
- CUDA Graph 会额外占用显存，作为 runtime overhead 的一部分

### 4.4 候选方案生成器 (`src/autopilot/planner/candidate_generator.py`)

**搜索空间**:
```python
for gpu_pool in cluster.gpu_pools:
    for backend in backends:
        for precision in intersect(model.supported_precisions, backend.supported_precisions):
            for tp in backend.tp_values:
                for pp in backend.pp_values:
                    for kv_cache_dtype in backend.supported_kv_cache_dtypes:
                        # 计算可行的 replica 范围
                        # 计算 batch 参数
                        # 生成候选
```

**剪枝策略**（按顺序）:
1. **硬约束过滤**: tp*pp > gpu_count → 排除
2. **显存可行性**: 显存不够 → 排除
3. **拓扑约束**: PCIe 上大 TP（>2）性能差 → 标记惩罚（不直接排除，通过 profile 的 communication_penalty 体现）
4. **SLO 预判**: 基于 profile 估算延迟，明显超 SLO → 排除
5. **GPU 数量上限**: 超过 maximum_gpu_count → 排除
6. **质量约束**: precision 降精度超出 minimum_quality_retention → 排除

**Batch 参数决策**:
- `max_num_seqs`: 基于并发估算 × headroom，受显存约束
- `max_num_batched_tokens`: 基于 prefill chunk 需求和显存

**Cache 参数决策**:
- prefix 复用率 > 30% → enable_prefix_cache = true
- chunked_prefill: 当输入长度 P90 > prefill_chunk_size 时启用
- prefill_chunk_size: 根据 input token P50 和延迟要求调整

### 4.5 方案评分与排序 (`src/autopilot/planner/scorer.py`)

**评分维度**（加权综合）:

| 维度 | 权重来源 | 计算方式 |
|------|----------|----------|
| SLO 满足度 | 硬约束，不满足则淘汰 | 基于 profile 估算 TTFT/ITL，与 SLO 比较 |
| 成本效率 | objective.primary | hourly_cost = gpu_count * replicas * hourly_cost_per_gpu |
| 吞吐余量 | capacity_headroom 约束 | (max_throughput - estimated_demand) / max_throughput |
| 显存余量 | 安全性 | (available - used) / total |
| Profile 置信度 | 数据完整性 | 有精确 profile = 1.0, 插值 = 0.7, 估算 = 0.4 |

**排序逻辑**:
1. 过滤掉不满足硬约束的候选
2. 根据 primary objective 排序（minimize_cost → 成本升序）
3. 同等成本下按 secondary objective 排序
4. 输出 top-1 为推荐方案，top-2/3 为备选

### 4.6 Profile 查询与缺失处理 (`src/autopilot/estimator/profile_lookup.py`)

**查找策略**:
1. 精确匹配：gpu_type + backend + precision + tp
2. 近似匹配：同 gpu_type 不同 tp → 线性插值吞吐，非线性调整延迟
3. 同代 GPU 缩放：H800 无数据但 A100 有 → 按算力比缩放
4. 保守默认：完全无数据 → 使用悲观估计，标注低置信度

**置信度标注**:
- `fact`: 精确匹配 profile
- `estimated`: 插值或缩放
- `assumed`: 无数据，使用工程假设
- `unverified`: 高不确定性，建议上线前验证

### 4.7 Reconciler (`src/autopilot/reconciler/`)

#### 4.7.1 状态分析 (`analyzer.py`)

读取连续 Telemetry 窗口，分析：
- SLO 违规：哪些指标超阈值，持续多少窗口
- 资源压力：GPU 利用率、KV Cache 利用率趋势
- 错误状态：OOM 次数、错误率
- 队列深度趋势：是否持续增长

#### 4.7.2 决策引擎 (`decision_engine.py`)

**触发条件矩阵**:

| 信号 | 持续窗口 | 动作 | 风险等级 |
|------|----------|------|----------|
| TTFT > SLO, queue_depth 增长 | ≥3 | scale_replicas +1 | medium |
| ITL > SLO, GPU util > 0.9 | ≥3 | scale_replicas +1 或 reduce max_num_seqs | medium |
| KV Cache util > 0.95 | ≥2 | reduce max_num_seqs 或 reduce max_model_len | medium |
| OOM count > 0 | ≥1 | reduce max_num_seqs, 紧急 | high |
| GPU util < 0.3, 无 SLO 违规 | ≥5 | scale_replicas -1 | low |
| All metrics healthy, over-provisioned | ≥10 | scale_replicas -1 | low |
| 持续 SLO 违规, scale 已到上限 | ≥5 | 建议切换 TP/GPU Pool | high (需重启) |

#### 4.7.3 防震荡机制 (`safeguards.py`)

```python
class ReconcilerSafeguards:
    # 扩缩容使用不同阈值（Hysteresis）
    scale_up_threshold = 0.85    # 利用率超过 85% 考虑扩容
    scale_down_threshold = 0.30  # 利用率低于 30% 考虑缩容

    # Cooldown: 上次操作后最少等待时间
    cooldown_after_scale_up = 300    # 5 分钟
    cooldown_after_scale_down = 600  # 10 分钟
    cooldown_after_config_change = 900  # 15 分钟

    # 连续确认窗口
    min_windows_for_scale_up = 3
    min_windows_for_scale_down = 5   # 缩容更保守

    # 最小样本量
    min_sample_count = 50  # 每个窗口至少50个请求才有统计意义

    # 高风险变更需要人工确认
    high_risk_actions = ["change_tp", "change_pp", "change_gpu_pool", "change_precision"]

    # 回退策略：指标恢复后观察足够长时间再回退
    recovery_observation_windows = 5
```

### 4.8 报告生成 (`src/autopilot/reporter/`)

- `plan_reporter.py`: 生成 `deployment_plan.yaml`, `alternatives.json`, `decision_report.md`
- `reconcile_reporter.py`: 生成 `actions.json`, `decision_log.md`

**decision_report.md 模板**:
```markdown
# Deployment Decision Report

## Workload Summary
- ...

## Resource Constraints
- ...

## Recommended Configuration
- ...

## Scoring Breakdown
- ...

## Memory Estimation
- ...

## Capacity & Headroom
- ...

## Cost Estimation
- ...

## Decision Rationale
- Why this TP: ...
- Why this precision: ...
- Why prefix cache enabled/disabled: ...
- Why not a larger TP: ...

## Alternatives
| # | Config | Score | Trade-off |
|---|--------|-------|-----------|

## Confidence Assessment
- ...

## Unverified Assumptions
- ...

## Pre-deployment Verification Recommendations
- ...
```

---

## 五、目录结构

```
AutoPilot/
├── README.md
├── AI_USAGE.md
├── pyproject.toml
├── src/
│   └── autopilot/
│       ├── __init__.py
│       ├── cli.py                      # Click CLI 入口
│       ├── orchestrator.py             # plan/reconcile 流程编排
│       ├── models/
│       │   ├── __init__.py
│       │   ├── model_spec.py
│       │   ├── cluster.py
│       │   ├── backend.py
│       │   ├── traffic.py
│       │   ├── profile.py
│       │   ├── slo.py
│       │   ├── plan.py
│       │   └── telemetry.py
│       ├── analyzer/
│       │   ├── __init__.py
│       │   └── workload.py
│       ├── estimator/
│       │   ├── __init__.py
│       │   ├── memory.py
│       │   └── profile_lookup.py
│       ├── planner/
│       │   ├── __init__.py
│       │   ├── candidate_generator.py
│       │   └── scorer.py
│       ├── reconciler/
│       │   ├── __init__.py
│       │   ├── analyzer.py
│       │   ├── decision_engine.py
│       │   └── safeguards.py
│       └── reporter/
│           ├── __init__.py
│           ├── plan_reporter.py
│           └── reconcile_reporter.py
├── tests/
│   ├── __init__.py
│   ├── test_workload_analyzer.py
│   ├── test_memory_estimator.py
│   ├── test_candidate_generator.py
│   ├── test_scorer.py
│   ├── test_profile_lookup.py
│   ├── test_reconciler.py
│   └── test_safeguards.py
├── examples/
│   ├── scenario_1_customer_service/
│   │   ├── model.yaml
│   │   ├── cluster.yaml
│   │   ├── backends.yaml
│   │   ├── traffic.jsonl
│   │   ├── runtime_profiles.yaml
│   │   └── slo.yaml
│   ├── scenario_2_long_generation/
│   │   └── ...
│   └── scenario_3_mixed_traffic/
│       └── ...
├── outputs/
│   ├── plan/
│   └── reconcile/
├── docs/
│   └── background.md
└── debug/
    └── design/
        └── plan.md
```

---

## 六、开发阶段与优先级

### Phase 1: 基础骨架（~4h）
1. 项目初始化：pyproject.toml、目录结构、CLI 注册
2. Pydantic 数据模型定义
3. YAML/JSONL 解析与加载
4. 示例数据文件编写（3 个场景）

### Phase 2: 核心逻辑（~10h）
5. Workload 分析器
6. 显存估算器
7. 候选方案生成器（含剪枝）
8. Profile 查询与缺失处理
9. 方案评分与排序
10. Plan 流程串联

### Phase 3: 闭环调节（~6h）
11. Telemetry 分析
12. 决策引擎
13. 防震荡安全机制
14. Reconcile 流程串联

### Phase 4: 输出与测试（~6h）
15. 报告生成（decision_report.md, alternatives.json, actions.json, decision_log.md）
16. 单元测试覆盖
17. 端到端测试（3 个场景跑通）

### Phase 5: 文档与收尾（~2h）
18. README.md
19. AI_USAGE.md
20. 最终检查与清理

---

## 七、关键设计决策

### 7.1 后端能力配置驱动

不 hardcode `if backend == "vllm"`，而是通过 YAML 配置声明每个后端的能力和约束。通用规划逻辑读取配置做决策：

```python
class BackendSpec:
    name: str
    supported_precisions: list[str]
    tp_values: list[int]
    pp_values: list[int]
    features: dict[str, bool]
    constraints: list[str]  # 可解析的约束表达式
```

如需后端特有逻辑，通过 Adapter 模式扩展：
```python
class BackendAdapter(Protocol):
    def validate_config(self, config: DeploymentConfig) -> list[str]: ...
    def estimate_overhead(self, config: DeploymentConfig) -> float: ...
```

### 7.2 显存模型的透明性

所有估算公式和假设在代码中有注释，在报告中有说明。区分：
- **已知**: 从 model spec 直接计算
- **估算**: 从 profile 插值
- **假设**: 缺少数据时的工程默认值

### 7.3 搜索策略

采用 **约束过滤 + 网格搜索 + 评分排序**：
- 搜索空间本身不大（几种 GPU × 几种 TP × 几种 PP × 几种精度 ≈ 几十到几百个候选）
- 先用硬约束快速过滤到几十个合法候选
- 再评分排序
- 不需要复杂的优化算法，保持可解释性

### 7.4 Reconciler 的保守原则

- 倾向少做而不是多做
- 缩容比扩容更保守
- 涉及重启的操作标注高风险
- 单次只建议一个操作（避免同时改多个变量）
- 每个决策附带 confidence score 和 reason

---

## 八、风险与缓解

| 风险 | 缓解 |
|------|------|
| head_dim 未在模型 spec 中显式提供 | 约定 head_dim = hidden_size / num_attention_heads，或在模型 spec 中增加字段 |
| Profile 数据稀疏 | 实现多级 fallback + 置信度标注 |
| 约束表达式解析复杂 | Phase 1 先用简单字符串匹配，后续可升级为表达式引擎 |
| 显存估算不准 | 保留 10% 安全余量，报告中标注为估算 |
| 48h 时间紧张 | 严格按优先级推进，Phase 4/5 可适当精简 |

---

## 九、待确认问题

1. **num_attention_heads** 未出现在模型 spec 示例中，是否需要在模型 YAML 中新增该字段？还是假设 num_attention_heads = num_kv_heads * group_size（GQA）？
   - **暂定方案**: 在模型 spec 中增加 `num_attention_heads` 字段；对示例中的 qwen3-32b，假设 num_attention_heads = 40（5120 / 128 = 40, head_dim=128）
2. **质量保留率 (quality_retention)**: fp8 相对于 bf16 的质量损失如何量化？
   - **暂定方案**: fp8 默认 quality_retention = 0.995，可在 backend 配置中自定义
3. **流量的时间粒度**: traffic.jsonl 中的 timestamp_ms 是相对时间还是绝对时间？
   - **暂定方案**: 视为相对时间（从 0 开始的模拟流量窗口）

---

请 Review 以上计划，有任何调整或补充请告知。
