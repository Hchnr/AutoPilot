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
| `model_spec.py` | 模型定义：参数量、层数、hidden_size、num_attention_heads、kv_heads、精度、max_len |
| `cluster.py` | GPU Pool：类型、数量、显存、拓扑、成本 |
| `backend.py` | 推理后端能力：支持的精度、TP/PP、Features、约束、版本、启动参数模板（配置驱动） |
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
  - head_dim = hidden_size / num_attention_heads (从模型spec的num_attention_heads字段直接获取)
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
- head_dim = hidden_size / num_attention_heads，num_attention_heads 从模型 spec 直接读取
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

```yaml
backends:
  vllm:
    version: ">=0.6.0"
    supported_precisions:
      - bf16
      - fp8
    supported_kv_cache_dtypes:
      - auto
      - fp8
    tp_values: [1, 2, 4, 8]
    pp_values: [1, 2]
    features:
      prefix_cache: true
      chunked_prefill: true
      cuda_graph: true
      speculative_decoding: false
    constraints:
      - "tp * pp <= gpu_count"
      - "prefix_cache requires kv_cache"
      - "fp8 requires gpu_arch >= sm_89"
    default_args:
      enable-chunked-prefill: true
      disable-log-requests: true

  sglang:
    version: ">=0.4.0"
    supported_precisions:
      - bf16
      - fp8
    supported_kv_cache_dtypes:
      - auto
      - fp8
    tp_values: [1, 2, 4, 8]
    pp_values: [1]
    features:
      prefix_cache: true
      chunked_prefill: true
      cuda_graph: true
      speculative_decoding: false
    constraints:
      - "tp <= gpu_count"
      - "pp == 1"  # sglang 当前不支持 PP
      - "fp8 requires gpu_arch >= sm_89"
    default_args:
      chunked-prefill-size: 8192
      disable-radix-cache: false
```

```python
class BackendSpec:
    name: str
    version: str
    supported_precisions: list[str]
    supported_kv_cache_dtypes: list[str]
    tp_values: list[int]
    pp_values: list[int]
    features: dict[str, bool]
    constraints: list[str]  # 可解析的约束表达式
    default_args: dict[str, Any]  # 后端默认启动参数
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
| head_dim 计算依赖 num_attention_heads | 模型 spec 中已新增 num_attention_heads 字段，head_dim = hidden_size / num_attention_heads |
| Profile 数据稀疏 | 实现多级 fallback + 置信度标注 |
| 约束表达式解析复杂 | Phase 1 先用简单字符串匹配，后续可升级为表达式引擎 |
| 显存估算不准 | 保留 10% 安全余量，报告中标注为估算 |
| 48h 时间紧张 | 严格按优先级推进，Phase 4/5 可适当精简 |

---

## 九、已确认问题

1. **num_attention_heads**: ✅ 已确认在模型 spec 中新增 `num_attention_heads` 字段。
   - 对 qwen3-32b：num_attention_heads = 40（hidden_size 5120 / head_dim 128 = 40）
   - head_dim = hidden_size / num_attention_heads，直接从 spec 计算，无需额外假设

2. **质量保留率 (quality_retention)**: ✅ 已确认 fp8 默认 quality_retention = 0.995，可在 backend 配置中自定义。
   - 完善后的 backend 配置包含 vllm 和 sglang 两个后端的完整定义
   - 新增 `version`、`supported_kv_cache_dtypes`、`default_args` 等字段

3. **流量的时间粒度**: ✅ 已确认 timestamp_ms 为绝对时间（Unix 毫秒时间戳）。
   - 分析时需要根据时间窗口进行聚合统计（如按分钟/小时分桶计算 RPS）
   - 支持识别流量的时间模式（峰谷、周期性）

---

## 十、验证策略

验证目标：通过自动化测试和端到端场景运行，证明系统满足 `docs/background.md` 的全部功能要求和评估重点。

### 10.1 单元测试验证矩阵

每个单元测试对应 background.md 中的一个具体要求：

| 测试文件 | 验证的需求点 | 对应 background.md 章节 |
|----------|-------------|----------------------|
| `test_workload_analyzer.py` | Token 长度分布 P50/P90/P99、RPS 计算、突发系数、Prefix 复用率、并发估算、Workload 分类 | 四.1 Workload 分析 |
| `test_memory_estimator.py` | 模型权重显存、Precision 影响、KV Cache 显存、TP/PP 切分、安全余量、GPU 数量限制 | 四.2 显存和资源可行性判断 |
| `test_candidate_generator.py` | 多候选生成、不可行方案排除、搜索空间覆盖、GPU 数量约束 | 四.3 自动生成部署方案 |
| `test_scorer.py` | SLO 满足度、成本排序、吞吐余量、方案排序逻辑 | 四.3 方案评分排序 |
| `test_profile_lookup.py` | 精确匹配、近似插值、跨 GPU 缩放、保守默认、置信度标注 | 四.4 不完整数据处理 |
| `test_reconciler.py` | Telemetry 分析、操作生成（scale/adjust/switch）、reason 和 confidence | 四.5 线上闭环调节 |
| `test_safeguards.py` | 连续窗口确认、Cooldown、Hysteresis、最小样本量、扩缩容不同阈值、高风险识别 | 四.6 防止配置震荡 |

### 10.2 关键断言设计

#### A. Workload 分析结果实际影响决策（非仅展示）

```python
def test_workload_drives_decision():
    """验证: 分析结果必须实际影响部署决策，而不能只出现在报告中"""
    # 高 prefix 复用率 → enable_prefix_cache = true
    high_prefix_traffic = generate_traffic(prefix_reuse_rate=0.8)
    plan_high = run_plan(traffic=high_prefix_traffic)
    assert plan_high.enable_prefix_cache is True

    # 低 prefix 复用率 → enable_prefix_cache = false
    low_prefix_traffic = generate_traffic(prefix_reuse_rate=0.05)
    plan_low = run_plan(traffic=low_prefix_traffic)
    assert plan_low.enable_prefix_cache is False

def test_decode_heavy_affects_config():
    """验证: decode-heavy workload 影响 batch 参数"""
    decode_heavy_traffic = generate_traffic(avg_input=100, avg_output=2000)
    plan = run_plan(traffic=decode_heavy_traffic)
    # decode-heavy 应关注 ITL，max_num_seqs 不应过大
    assert plan.max_num_seqs <= expected_conservative_limit

def test_burst_traffic_affects_replicas():
    """验证: 突发流量影响 replica 数量或 headroom"""
    bursty_traffic = generate_traffic(burst_ratio=5.0)
    plan_bursty = run_plan(traffic=bursty_traffic)
    steady_traffic = generate_traffic(burst_ratio=1.2)
    plan_steady = run_plan(traffic=steady_traffic)
    assert plan_bursty.replicas >= plan_steady.replicas
```

#### B. 显存估算准确性和约束执行

```python
def test_memory_model_correctness():
    """验证: 显存公式各项计算正确"""
    # 已知: qwen3-32b, bf16, tp=4
    # model_weight = 32e9 * 2 / 4 = 16 GB per GPU
    result = estimate_memory(model=qwen3_32b, precision="bf16", tp=4)
    assert abs(result.model_weight_per_gpu_gb - 16.0) < 0.1

def test_infeasible_rejected():
    """验证: 显存不够的配置一定被排除"""
    # L40S-48GB 跑 bf16 tp=1, 模型权重就要 64GB, 一定不可行
    candidates = generate_candidates(model=qwen3_32b, gpu=l40s_48gb)
    for c in candidates:
        if c.precision == "bf16" and c.tp == 1:
            assert False, "不可行方案未被过滤"

def test_tp_divides_heads():
    """验证: TP 必须能整除 num_kv_heads"""
    candidates = generate_candidates(model=qwen3_32b)
    for c in candidates:
        assert qwen3_32b.num_kv_heads % c.tp == 0
```

#### C. 方案排序和评分

```python
def test_minimize_cost_ranking():
    """验证: primary=minimize_cost 时，推荐方案是成本最低的可行方案"""
    plans = run_plan(slo=cost_first_slo)
    assert plans.recommended.hourly_cost <= plans.alternatives[0].hourly_cost

def test_slo_hard_constraint():
    """验证: 所有推荐方案都满足 SLO 硬约束"""
    plans = run_plan(slo=strict_slo)
    assert plans.recommended.estimated_p95_ttft_ms <= strict_slo.p95_ttft_ms
    assert plans.recommended.estimated_p95_itl_ms <= strict_slo.p95_itl_ms

def test_at_least_two_alternatives():
    """验证: 输出至少两个备选方案"""
    plans = run_plan()
    assert len(plans.alternatives) >= 2
```

#### D. 不完整数据降级

```python
def test_missing_profile_fallback():
    """验证: 缺失 profile 时系统不崩溃，使用保守估计并标注低置信度"""
    result = run_plan(profiles=empty_profiles)
    assert result.recommended is not None
    assert result.confidence < 0.7  # 置信度应明确低于有完整 profile 时

def test_interpolation_between_tp():
    """验证: 同 GPU 不同 TP 可以插值"""
    # 有 tp=2 和 tp=8 的 profile，查询 tp=4
    lookup = profile_lookup(gpu="H800", tp=4, profiles=partial_profiles)
    assert lookup.source == "interpolated"
    assert lookup.confidence < 1.0
```

#### E. Reconcile 闭环验证

```python
def test_slo_violation_triggers_action():
    """验证: 连续多窗口 SLO 违反触发操作"""
    telemetry = generate_telemetry(p95_ttft_ms=[920, 950, 930], windows=3)
    actions = reconcile(plan=current_plan, telemetry=telemetry, slo_ttft=800)
    assert len(actions) > 0
    assert actions[0].action in ["scale_replicas", "adjust_max_num_seqs"]

def test_single_spike_no_action():
    """验证: 单窗口波动不触发操作（防震荡）"""
    telemetry = generate_telemetry(p95_ttft_ms=[920, 500, 500], windows=3)
    actions = reconcile(plan=current_plan, telemetry=telemetry, slo_ttft=800)
    assert len(actions) == 0

def test_cooldown_respected():
    """验证: cooldown 期间不产生新操作"""
    actions = reconcile(
        plan=current_plan,
        telemetry=violation_telemetry,
        last_action_time=recent_time,  # cooldown 内
    )
    assert len(actions) == 0

def test_scale_down_more_conservative():
    """验证: 缩容比扩容更保守"""
    # 轻微低负载不缩容
    low_load = generate_telemetry(gpu_util=[0.3, 0.35, 0.32], windows=3)
    actions_low = reconcile(plan=current_plan, telemetry=low_load)
    # 需要更长时间/更低负载才缩容
    assert len(actions_low) == 0  # 3 个窗口不够缩容

def test_high_risk_action_flagged():
    """验证: 涉及重启的变更标记为高风险"""
    telemetry = generate_telemetry(sustained_overload=True, windows=10)
    actions = reconcile(plan=current_plan, telemetry=telemetry)
    tp_changes = [a for a in actions if a.action == "change_tp"]
    for a in tp_changes:
        assert a.risk_level == "high"
        assert a.requires_restart is True
```

### 10.3 端到端场景验证

三个场景直接对应 background.md 第五节要求：

#### 场景一：高 Prefix 复用的客服 Chat

| 预期行为 | 验证方式 |
|----------|----------|
| prefix_cache 启用 | 断言 `plan.enable_prefix_cache == True` |
| 选择 NVLink GPU（大 TP 需要高带宽） | 断言 `plan.gpu_pool == "h800-sxm"` |
| TTFT 满足严格 SLO | 断言 `plan.estimated_p95_ttft_ms <= slo.p95_ttft_ms` |
| 流量高峰时有足够 headroom | 断言 `plan.capacity_headroom >= slo.minimum_capacity_headroom` |
| decision_report 解释了为什么开启 prefix cache | 检查报告包含 prefix cache 相关 rationale |

#### 场景二：长文本生成

| 预期行为 | 验证方式 |
|----------|----------|
| 关注 ITL 而非 TTFT | 方案选择偏向 decode 吞吐而非 prefill 速度 |
| KV Cache 占用大 → 可能用 fp8 kv_cache_dtype | 断言当显存紧张时选择 fp8 kv cache |
| max_num_seqs 较低（长序列占用大） | 断言 `plan.max_num_seqs < 场景一的 max_num_seqs` |
| prefix_cache 关闭（复用率低） | 断言 `plan.enable_prefix_cache == False` |

#### 场景三：成本敏感的混合流量

| 预期行为 | 验证方式 |
|----------|----------|
| 可能选择 L40S（更低成本） | 在成本约束下验证 GPU pool 选择逻辑 |
| 在成本和延迟间有明确权衡 | alternatives 中展示不同成本/延迟取舍 |
| 流量变化 → reconcile 能给出调整建议 | 用变化流量的 telemetry 测试 reconcile |

#### 端到端运行验证脚本

```bash
#!/bin/bash
# test_e2e.sh - 端到端验证
set -e

echo "=== Scenario 1: 高 Prefix 复用客服 Chat ==="
autopilot plan \
  --model examples/scenario_1_customer_service/model.yaml \
  --cluster examples/scenario_1_customer_service/cluster.yaml \
  --backends examples/scenario_1_customer_service/backends.yaml \
  --traffic examples/scenario_1_customer_service/traffic.jsonl \
  --profiles examples/scenario_1_customer_service/runtime_profiles.yaml \
  --slo examples/scenario_1_customer_service/slo.yaml \
  --output outputs/scenario_1/plan/

# 验证输出文件存在且非空
[ -s outputs/scenario_1/plan/deployment_plan.yaml ]
[ -s outputs/scenario_1/plan/alternatives.json ]
[ -s outputs/scenario_1/plan/decision_report.md ]

echo "=== Scenario 1: Reconcile ==="
autopilot reconcile \
  --plan outputs/scenario_1/plan/deployment_plan.yaml \
  --telemetry examples/scenario_1_customer_service/telemetry.jsonl \
  --output outputs/scenario_1/reconcile/

[ -s outputs/scenario_1/reconcile/actions.json ]
[ -s outputs/scenario_1/reconcile/decision_log.md ]

echo "=== Scenario 2: 长文本生成 ==="
autopilot plan \
  --model examples/scenario_2_long_generation/model.yaml \
  --cluster examples/scenario_2_long_generation/cluster.yaml \
  --backends examples/scenario_2_long_generation/backends.yaml \
  --traffic examples/scenario_2_long_generation/traffic.jsonl \
  --profiles examples/scenario_2_long_generation/runtime_profiles.yaml \
  --slo examples/scenario_2_long_generation/slo.yaml \
  --output outputs/scenario_2/plan/

echo "=== Scenario 3: 成本敏感混合流量 ==="
autopilot plan \
  --model examples/scenario_3_mixed_traffic/model.yaml \
  --cluster examples/scenario_3_mixed_traffic/cluster.yaml \
  --backends examples/scenario_3_mixed_traffic/backends.yaml \
  --traffic examples/scenario_3_mixed_traffic/traffic.jsonl \
  --profiles examples/scenario_3_mixed_traffic/runtime_profiles.yaml \
  --slo examples/scenario_3_mixed_traffic/slo.yaml \
  --output outputs/scenario_3/plan/

echo "✅ All e2e scenarios passed"
```

### 10.4 评估重点覆盖度验证

对照 background.md 第八节"评估重点"，逐条设计验证：

| 评估重点 | 验证方法 | 测试位置 |
|----------|----------|----------|
| 正确理解模型推理的资源和性能约束 | 显存公式单元测试 + 不可行方案过滤 | `test_memory_estimator.py` |
| 显存、KV Cache、并发和上下文长度之间的关系 | 参数化测试：改变 max_num_seqs 观察显存变化 | `test_memory_estimator.py` |
| TP、PP 和 Replica 的不同作用 | TP 减小单卡显存、PP 支持超大模型、Replica 增加吞吐 | `test_candidate_generator.py` |
| GPU 拓扑对并行方案的影响 | PCIe 上大 TP 有 communication_penalty | `test_scorer.py` |
| Prefix Cache 对不同 Workload 的适用性 | 高/低复用率场景的差异化决策 | `test_workload_analyzer.py` + e2e |
| 是否优先满足硬约束 | SLO 不满足 → 方案被淘汰 | `test_scorer.py` |
| 优化目标是否清晰且可配置 | 切换 primary objective 观察排序变化 | `test_scorer.py` |
| 数据缺失时的处理是否合理 | 空 profile / 部分 profile 场景 | `test_profile_lookup.py` |
| 闭环控制是否安全 | 防震荡全套测试 | `test_safeguards.py` |
| 系统是否易于扩展到新的 GPU 和推理后端 | 添加新后端 YAML 即可使用，无需改代码 | `test_backend_extensibility.py` |
| 是否明确说明事实、估算、假设和风险 | decision_report 内容检查 | e2e 输出校验 |

### 10.5 输出产物格式验证

```python
def test_deployment_plan_schema():
    """验证 deployment_plan.yaml 至少包含 background.md 要求的字段"""
    plan = load_yaml("outputs/plan/deployment_plan.yaml")
    required_fields = [
        "gpu_pool", "replicas", "tensor_parallel", "pipeline_parallel",
        "precision", "kv_cache_dtype", "max_num_seqs",  # batch 参数
        "enable_prefix_cache",  # cache 参数
    ]
    for field in required_fields:
        assert field in plan, f"缺少必需字段: {field}"

def test_alternatives_has_tradeoff():
    """验证备选方案包含权衡说明"""
    alternatives = load_json("outputs/plan/alternatives.json")
    assert len(alternatives) >= 2
    for alt in alternatives:
        assert "tradeoff" in alt or "score" in alt

def test_decision_report_completeness():
    """验证 decision_report.md 包含所有要求的章节"""
    report = read_file("outputs/plan/decision_report.md")
    required_sections = [
        "Workload",           # Workload 摘要
        "Resource",           # 关键资源约束
        "Recommended",        # 最终推荐配置
        "Score" or "Scoring", # 方案评分
        "Memory",             # 预计显存占用
        "Capacity",           # 预计容量和资源余量
        "Cost",               # 预计成本
        "Rationale",          # 决策原因
        "Alternative",        # 备选方案
        "Confidence",         # 置信程度
        "Assumption",         # 尚未验证的假设
        "Verification",       # 上线前建议
    ]
    for section in required_sections:
        assert section.lower() in report.lower(), f"报告缺少章节: {section}"

def test_reconcile_action_schema():
    """验证 actions.json 包含必要字段"""
    actions = load_json("outputs/reconcile/actions.json")
    for action in actions:
        assert "action" in action
        assert "reason" in action
        assert "confidence" in action
        if action.get("requires_restart"):
            assert "risk_level" in action
```

### 10.6 测试运行命令

```bash
# 全量测试
pytest tests/ -v --tb=short

# 单模块验证
pytest tests/test_workload_analyzer.py -v
pytest tests/test_memory_estimator.py -v
pytest tests/test_candidate_generator.py -v
pytest tests/test_scorer.py -v
pytest tests/test_profile_lookup.py -v
pytest tests/test_reconciler.py -v
pytest tests/test_safeguards.py -v

# 端到端
bash tests/test_e2e.sh

# 覆盖率
pytest tests/ --cov=src/autopilot --cov-report=term-missing
```

---

计划已就绪，所有待确认问题已明确，验证策略已完善，可以开始实施。
