# 关键考量点分析

本文档逐项说明系统在设计和实现中如何处理每个关键考量点。

---

## 1. 模型推理的资源和性能约束

**实现方式**：显存估算器（`estimator/memory.py`）建模推理过程的三大显存消费：

```
总显存 = (模型权重 + KV Cache + 运行时开销) × 1.10
```

- **模型权重**：`param_count × bytes_per_param / TP / PP`
- **KV Cache**：`2 × layers × kv_heads × head_dim × kv_bytes / TP × max_num_seqs × context_length`
- **运行时开销**：固定 7 GB（覆盖激活、CUDA Graph、框架缓冲区）
- **安全余量**：小计 × 10%

可行性判断：`total ≤ gpu_memory × 0.95`（预留 5% 给 OS/驱动）。

**验证**：11 个单元测试覆盖各因子的独立变化和组合效果。

---

## 2. 显存、KV Cache、并发和上下文长度之间的关系

**核心公式**：

```
kv_cache_memory = kv_per_token × max_num_seqs × context_length
```

其中：
- `kv_per_token = 2 × layers × kv_heads × head_dim × dtype_bytes / TP`
- `context_length = min(input_p99 + output_p99, max_model_len)`

**关键洞察**：
- 并发（max_num_seqs）和上下文长度对显存的影响是**乘法关系**
- TP 同时降低权重和 KV Cache 的单卡占用
- FP8 KV Cache 将这部分显存减半，是高并发场景的重要优化手段

**测试验证**：
- `test_concurrency_affects_memory`：16 并发 → 64 并发，显存显著增加
- `test_context_length_affects_memory`：4000 上下文 → 8000 上下文，显存增加
- `test_kv_cache_fp8_halves`：FP8 KV 精确为 BF16 的一半

---

## 3. TP、PP 和 Replica 的不同作用

| 维度 | 切分对象 | 影响范围 | 约束 |
|------|----------|----------|------|
| TP | 权重 + KV Cache（每层内切分） | 单卡显存 ↓，通信开销 ↑ | 必须整除 num_kv_heads |
| PP | 权重（层间切分） | 单卡权重 ↓，pipeline bubble ↑ | 必须整除 num_layers |
| Replica | 无切分，独立副本 | 总吞吐 ↑，成本 ↑ | TP × PP × Replicas ≤ pool.count |

**代码实现**：
- 候选生成器对 TP、PP、Replica 三维度穷举
- 硬约束在评分前剪枝：`num_kv_heads % TP == 0`、`num_layers % PP == 0`
- 资源约束：`TP × PP × Replicas ≤ max_gpu_count` 且 `≤ pool.count`

**设计决策**：PP 引入 pipeline bubble（效率损失约 `(PP-1)/PP`），在线服务通常 PP=1 最优，除非单卡装不下模型。三个场景均选 PP=1 是合理的。

---

## 4. GPU 拓扑对并行方案的影响

**实现方式**：Profile Lookup 中根据拓扑应用通信惩罚因子：

| 拓扑 | 通信惩罚 | 典型场景 |
|------|----------|----------|
| NVLink | 1.0 | H800 SXM |
| PCIe | 1.3~1.4 | L40S |

延迟估算公式：
```
TTFT = base_ttft × (input_p90 / 1000) × communication_penalty × load_factor
ITL = base_itl × communication_penalty × load_factor
```

**实际效果**：
- 场景一 TTFT SLO=800ms，H800 NVLink 估算 605ms（满足）
- 同配置 L40S PCIe 估算 ~787ms（逼近 SLO 边界，失去余量）

系统自动选择拓扑更优的方案，因为评分器中 latency_score 考虑了 SLO 余量。

---

## 5. Prefix Cache 对不同 Workload 的适用性

**启用条件**（必须同时满足）：
1. `prefix_coverage = requests_with_prefix / total_requests > 0.30`
2. `prefix_reuse_rate = 1 - unique_prefixes / requests_with_prefix > 0.30`

**设计理由**：
- 单看复用率不够：场景二复用率 50% 但覆盖率仅 9%，91% 请求无法受益
- 单看覆盖率不够：如果每个请求用不同 prefix，缓存命中率为 0

**场景验证**：

| 场景 | 覆盖率 | 复用率 | 决策 | 理由 |
|------|--------|--------|------|------|
| 一 | 100% | 99% | 启用 | 全部请求受益，极高命中率 |
| 二 | 9% | 50% | 关闭 | 大部分请求无 prefix |
| 三 | 51% | 99% | 启用 | 超半数请求受益，命中率极高 |

---

## 6. 是否优先满足硬约束

**两阶段过滤确保硬约束优先**：

**阶段一 — 候选生成（generate_candidates）**：
- `num_kv_heads % TP == 0` → TP 配置合法性
- `num_layers % PP == 0` → PP 配置合法性
- `TP × PP ≤ pool.count` → 物理资源限制
- `total_memory ≤ gpu_memory × 0.95` → 显存可行性
- `TP × PP × Replicas ≤ max_gpu_count` → SLO 资源预算

不满足任一条件的方案**直接丢弃**，不进入评分。

**阶段二 — 评分（score_and_rank）**：
- 估算 TTFT > SLO → 方案被过滤
- 估算 ITL > SLO → 方案被过滤
- 容量余量 < minimum_capacity_headroom → 方案被过滤

**只有通过所有硬约束的方案才参与软指标排序**（成本、余量、延迟）。

---

## 7. 优化目标是否清晰且可配置

**配置方式**：SLO YAML 文件中定义：

```yaml
objective:
  primary: minimize_hourly_cost  # 或 maximize_goodput
  secondary: maximize_goodput
```

**权重映射**：

| 目标 | cost | headroom | latency | confidence |
|------|------|----------|---------|------------|
| minimize_hourly_cost | 0.5 | 0.2 | 0.2 | 0.1 |
| maximize_goodput | 0.2 | 0.3 | 0.4 | 0.1 |

**扩展性**：新增优化目标只需在 scorer 中添加一组权重。评分函数结构不变：
```
final_score = Σ(weight_i × score_i)
```

---

## 8. 数据缺失时的处理是否合理

**Profile Lookup 四级降级**：

| 级别 | 条件 | 置信度 | 处理方式 |
|------|------|--------|----------|
| 精确匹配 | GPU + 后端 + 精度 + TP 完全匹配 | 1.0 | 直接使用 profile 数据 |
| TP 插值 | 同 GPU/后端/精度，不同 TP | 0.7 | 线性插值 |
| 跨 GPU 缩放 | 不同 GPU 类型 | 0.5 | 保守 0.8× 缩放 |
| 默认值 | 无相关 profile | 0.3 | 保守假设（TTFT=500ms, ITL=30ms）|

**保守性原则**：
- 跨 GPU 缩放使用 0.8× 而非 1.0×，假设未知 GPU 性能较差
- 低置信度方案在评分中被 confidence 权重惩罚
- 决策报告对低置信度方案标注更强的验证建议

---

## 9. 闭环控制是否安全，是否会产生配置震荡

**四重防震荡机制**：

### 连续窗口确认
- 扩容：需要连续 **3** 个窗口（15 分钟）违反
- 缩容：需要连续 **5** 个窗口（25 分钟）低利用率
- 单次波动不触发任何操作

### 非对称阈值
- 扩容门槛低（3 窗口）：SLO 违反的用户影响大
- 缩容门槛高（5 窗口）：误缩容的恢复成本大于轻微过配

### Cooldown 冷却期
- 任何操作后强制 3 窗口（15 分钟）冷却
- 冷却期内无论信号多强均不允许新操作
- 让上一次操作有时间生效

### 单次单操作
- 即使检测到多个信号，每个 reconcile 周期只输出一个操作
- 更容易归因每次变更的效果
- 避免冲突调整叠加

**中断重置**：如果连续违反序列中间出现一个正常窗口，计数器归零重新开始。

---

## 10. 系统是否易于扩展到新的 GPU 和推理后端

**纯配置驱动，无需改代码**：

| 扩展项 | 操作 | 改动范围 |
|--------|------|----------|
| 新 GPU 类型 | 在 `cluster.yaml` 加条目 | 1 个配置文件 |
| 新推理后端 | 在 `backends.yaml` 加条目 | 1 个配置文件 |
| 新模型 | 创建 `model.yaml` | 1 个配置文件 |
| 新 runtime profile | 在 `runtime_profiles.yaml` 加条目 | 1 个配置文件 |

候选生成器自动对 `gpu_pools × backends × precisions × TP × PP × KV_dtype` 做笛卡尔积。新加任何维度的选项，系统会自动探索包含它的组合。

无 profile 时系统仍能工作（降级到默认值 + 低置信度标注），不会阻塞决策。

---

## 11. 是否有效使用 AI Coding 工具并验证其输出

**使用方式**：Claude Code (Opus 4)，CLI 交互式开发。

**被采纳的关键建议**：Profile Lookup 的多级降级策略 — AI 建议四级 fallback + 递减置信度，形成完整信息链。

**被拒绝的建议**：Prefix Cache 单阈值判断 — AI 最初只检查 `prefix_reuse_rate > 0.3`，在场景二中 9% 覆盖率 + 50% 复用率会错误启用。修正为覆盖率 × 复用率双重检查。

**验证策略**：
- 71 个单元测试 + 端到端测试，93% 覆盖率
- 对抗性测试（空数据、极端输入）
- 人工审查决策报告的推理自洽性

---

## 12. 是否明确说明事实、估算、假设和风险

决策报告（`decision_report.md`）结构化分离四类信息：

| 章节 | 性质 | 内容 |
|------|------|------|
| Resource Constraints | 事实 | 直接来自输入文件的模型/集群/SLO 参数 |
| Workload Summary | 事实 | 从 traffic.jsonl 计算的统计量 |
| Memory Estimation | 估算 | 基于公式计算，标注计算方法 |
| Estimated Latency | 估算 | 基于 profile 预测，标注 SLO 对比 |
| Confidence Assessment | 置信度 | 明确标注数据来源质量 |
| Unverified Assumptions | 假设 | 显式列出未验证的前提 |
| Pre-deployment Verification | 风险缓解 | 针对每个假设给出验证方法 |

**具体假设清单**（在报告中明确声明）：
- 激活内存相对权重和 KV Cache 可忽略
- CUDA Graph 内存包含在运行时开销估算中
- 通信惩罚因子基于历史 profile，未针对当前部署实测

**风险缓解建议**（在报告中明确给出）：
1. 用代表性流量跑压测验证 TTFT/ITL
2. 峰值负载时监控 GPU 显存确认余量
3. 验证 KV Cache 利用率在峰值并发下 <90%
4. 跑质量评估确认 FP8 满足 retention 阈值
5. 用生产流量验证 prefix cache 命中率
