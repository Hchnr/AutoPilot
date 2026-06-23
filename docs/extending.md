# 扩展 AutoPilot

本文档说明如何接入新的 GPU、模型或推理后端。

## 接入新 GPU 类型

1. **在 `cluster.yaml` 中添加条目**：

```yaml
gpu_pools:
  - id: a100-nvlink
    gpu_type: A100-80GB
    count: 8
    memory_gb: 80
    topology: nvlink
    hourly_cost_per_gpu: 2.8
```

无需代码改动 — 系统会自动评估此资源池与所有后端和精度的组合。

2. **（可选）添加 runtime profile** 以获得更准确的延迟估算：

```yaml
profiles:
  - gpu_type: A100-80GB
    backend: vllm
    precision: bf16
    tp: 4
    maximum_prefill_tokens_per_second: 20000
    maximum_decode_tokens_per_second: 3000
    base_ttft_ms: 180
    base_itl_ms: 20
    runtime_memory_overhead_gb: 7
    communication_penalty:
      nvlink: 1.0
      pcie: 1.35
```

无 profile 时，系统会从已有 profile 进行跨 GPU 缩放（置信度 0.5）。有 profile 时，估算为精确匹配（置信度 1.0）。

## 接入新模型

1. **创建 `model.yaml`**：

```yaml
name: llama-3-70b
architecture: decoder_only
parameter_count: "70B"
num_layers: 80
hidden_size: 8192
num_attention_heads: 64
num_kv_heads: 8
supported_precisions:
  - bf16
  - fp8
max_model_len: 131072
minimum_quality_retention: 0.99
```

系统使用的关键字段：
- `parameter_count` — 解析为数值（支持 "7B"、"70B"、"405B" 写法）
- `num_kv_heads` — 必须能被你想支持的 TP 值整除
- `num_layers` — 必须能被 PP 值整除
- `hidden_size` — 用于计算 `head_dim = hidden_size / num_attention_heads`

2. **（可选）添加模型专属 profile** 到 `runtime_profiles.yaml`。

## 接入新推理后端

1. **在 `backends.yaml` 中添加**：

```yaml
backends:
  tensorrt-llm:
    supported_precisions:
      - bf16
      - fp8
      - int4
    supported_kv_cache_dtypes:
      - auto
      - fp8
    tp_values: [1, 2, 4, 8]
    pp_values: [1]
    features:
      prefix_cache: true
      chunked_prefill: true
      cuda_graph: true
      in_flight_batching: true
    constraints:
      - "tp <= gpu_count"
      - "fp8 requires gpu_arch >= sm_89"
    default_args:
      max-tokens-in-paged-kv-cache: null
```

系统读取的字段：
- `supported_precisions` — 与模型支持的精度取交集
- `supported_kv_cache_dtypes` — 在候选方案生成时穷举
- `tp_values` / `pp_values` — 定义并行度搜索空间
- `features` — 用于判断 prefix_cache 和 chunked_prefill 的可用性

2. **添加对应 runtime profile**：

```yaml
profiles:
  - gpu_type: H800-80GB
    backend: tensorrt-llm
    precision: bf16
    tp: 4
    maximum_prefill_tokens_per_second: 28000
    maximum_decode_tokens_per_second: 4000
    base_ttft_ms: 120
    base_itl_ms: 15
    runtime_memory_overhead_gb: 8
    communication_penalty:
      nvlink: 1.0
      pcie: 1.30
```

## Profile 对方案质量的影响

| Profile 可用性 | 置信度 | 影响 |
|---------------|--------|------|
| 精确匹配 | 1.0 | 延迟估算可靠，评分可信 |
| 同 GPU/后端，不同 TP | 0.7 | 插值，精度合理 |
| 不同 GPU 类型 | 0.5 | 保守缩放，方案标记为中等置信度 |
| 无相关 profile | 0.3 | 默认假设，方案标记为低置信度并附验证建议 |

系统始终会产出方案，但会清楚地传达估算不确定时的情况。低置信度方案在决策报告中包含更强的验证建议。

## 数据格式参考

完整文件格式参见 `examples/` 下的示例。Pydantic 模型定义在 `src/autopilot/models/` 中，包含精确的 schema 和验证规则。
