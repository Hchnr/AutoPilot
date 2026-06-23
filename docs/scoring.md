# 候选方案生成与评分

## 候选方案生成

### 搜索空间

生成器穷举以下笛卡尔积：

```
GPU 池 × 后端 × 精度 × TP × PP × KV Cache 精度
```

以示例输入为例（2 池 × 2 后端 × 2 精度 × 4 TP 值 × 2 PP 值 × 2 KV 精度），剪枝前最多产生 128 个原始组合。

### 剪枝规则（硬约束）

每个组合逐一检查，任一不满足即淘汰：

| 规则 | 原因 |
|------|------|
| `num_kv_heads % TP == 0` | GQA 要求 KV Head 均匀分布到各 TP rank |
| `num_layers % PP == 0` | Pipeline 各阶段必须等层数 |
| `TP × PP ≤ pool.count` | 不能超过可用 GPU 数 |
| `总显存 ≤ GPU 显存 × 0.95` | 必须装得下 |
| `质量保持率 ≥ 阈值` | FP8 约 ~0.5% 质量损失；FP8 KV 再加 ~0.2% |
| `总 GPU 数 ≤ max_gpu_count` | SLO 定义的资源预算 |

### Batch 参数选择

通过剪枝的候选方案，根据 workload 计算 batch 参数：

- **max_num_seqs**：`min(estimated_concurrency × 1.5, 显存限制上限)`
  - 上限取决于 GPU 显存：256（≥80GB）、128（≥48GB）、64（其他）
- **max_num_batched_tokens**：`min(input_p90 × max_num_seqs × 0.5, 16384)`
  - 限制在 2048 到 16384 之间
- **enable_prefix_cache**：仅当 prefix 覆盖率 > 30% 且复用率 > 30%
- **enable_chunked_prefill**：当输入 P90 > 2048 tokens
- **prefill_chunk_size**：根据输入 P50 选择 4096 / 2048 / 1024

### 副本数估算

```
needed_replicas = ceil(peak_rps × output_p50 / 单副本最大 decode 吞吐 × (1 + headroom))
```

限制在 `[1, pool.count / (TP × PP)]` 范围内，并受 `max_gpu_count` 进一步约束。

## 评分

### 基于 Profile 的延迟估算

通过 Profile Lookup 引擎估算每个候选方案的预期延迟：

1. **精确匹配**（置信度 1.0）— 有对应 GPU/后端/精度/TP 的 profile
2. **TP 插值**（置信度 0.7）— 同 GPU/后端/精度，不同 TP → 线性插值
3. **跨 GPU 缩放**（置信度 0.5）— 不同 GPU 类型 → 保守 0.8× 缩放
4. **默认值**（置信度 0.3）— 无相关 profile → 保守假设

延迟估算包含：
```
TTFT = base_ttft × (input_p90 / 1000) × 通信惩罚 × 负载因子
ITL  = base_itl × 通信惩罚 × 负载因子
```

其中：
- `通信惩罚` = 1.0（NVLink）或 1.3-1.4（PCIe）
- `负载因子` = 1 + 0.1 × max(0, 每副本并发 - 10)

### 评分函数

通过延迟和余量硬约束的候选方案进入评分：

```python
cost_score     = 1 / (1 + hourly_cost / 100)      # 成本越低 → 分越高
headroom_score = min(capacity_headroom, 1.0)       # 余量越大 → 分越高
latency_score  = (ttft_margin + itl_margin) / 2    # SLO 余量越大 → 分越高
```

最终加权分数取决于优化目标：

| 目标 | cost | headroom | latency | confidence |
|------|------|----------|---------|------------|
| minimize_hourly_cost | 0.5 | 0.2 | 0.2 | 0.1 |
| maximize_goodput | 0.2 | 0.3 | 0.4 | 0.1 |

### 决策理由生成

每个方案附带可读的决策解释：
- 为什么选这个 TP 值（显存压力、通信拓扑）
- 为什么选这个 PP 值（层数可容纳 vs pipeline 开销）
- 为什么选这个精度（成本 vs 质量权衡）
- 为什么启用/关闭 Prefix Cache（workload 模式）

## 输出

方案按分数降序排列。最高分方案为 `recommended`，后续 1-3 个为 `alternatives`。每个方案包含：
- 完整配置参数
- 预估延迟（TTFT、ITL）
- 预估成本（每小时、每月）
- 容量余量百分比
- 置信度分数
- 决策理由（rationale）
