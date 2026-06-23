# Workload 分析

## 目的

Workload 分析器将原始请求日志（`traffic.jsonl`）转化为结构化的 `WorkloadSummary`，驱动所有下游决策。目标是刻画系统需要处理的*工作类型*，从而做出合适的配置选择。

## 输入格式

`traffic.jsonl` 每行一个 JSON 对象：

```json
{
  "timestamp_ms": 1719100800000,
  "input_tokens": 3500,
  "output_tokens": 250,
  "prefix_id": "customer-service-kb-v3",
  "priority": "interactive"
}
```

## 提取特征

### Token 长度分布

计算输入和输出 Token 的 P50、P90、P99。直接影响：
- 显存规划（KV Cache 分配使用 P99）
- Chunked Prefill 决策（输入 P90 > 2048 时启用）
- Prefill 分块大小选择

### 请求速率（RPS）

使用 1 分钟滑动窗口计算：
- **avg_rps** = 总请求数 / 总时长
- **peak_rps** = max(窗口内请求数 / 窗口时长)
- **burst_ratio** = peak_rps / avg_rps

突发比 > 2.0 表示有显著流量尖峰，容量规划需要预留余量。

### Prefix 复用分析

两个指标共同决定 Prefix Cache 是否有益：

1. **覆盖率** = 携带 prefix 的请求数 / 总请求数
   - 有多少比例的流量能从缓存中受益？
2. **复用率** = 1 - (唯一 prefix 数 / 携带 prefix 的请求总数)
   - 在有 prefix 的请求中，同一 prefix 被复用的频率如何？

只有当**覆盖率 > 30% 且复用率 > 30%** 时才启用 Prefix Cache。这避免了两种无效场景：
- 低覆盖率：大部分请求没有 prefix，缓存收益微小
- 低复用率：每个请求用不同 prefix，缓存命中率极低

### 并发估算

使用 Little 定律：`L = λ × W`

其中：
- λ = avg_rps
- W = 估算处理时间 = output_tokens × 20ms（decode）+ input_tokens × 0.5ms（prefill）

估算结果用于设定 `max_num_seqs` 和计算容量需求。

### 时间模式检测

将观测窗口划分为 5 分钟桶，检查峰谷 RPS 差异是否超过均值的 50%。存在时间模式意味着系统可能受益于基于时间的扩缩策略（当前版本未实现，但会标记给运维人员）。

### 负载分类

- **Prefill 密集型**：输入/输出 Token 比 > 3.0 → 优先保证 prefill 吞吐
- **延迟敏感型**：> 50% 请求为 "interactive" 优先级 → 更严格的 TTFT/ITL 目标
- **Decode 密集型**：输出相对输入较长 → ITL 优化更重要

## 特征如何驱动决策

| 特征 | 影响 |
|------|------|
| input_tokens_p90 > 2048 | 启用 Chunked Prefill |
| prefix 覆盖率 > 30% 且复用率 > 30% | 启用 Prefix Cache |
| burst_ratio | 容量余量大小 |
| estimated_concurrency | max_num_seqs 初始值 |
| is_prefill_heavy | Prefill 分块大小调优 |
| is_latency_sensitive | 评分器权重调整 |
