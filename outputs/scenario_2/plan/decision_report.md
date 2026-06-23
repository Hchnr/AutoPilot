# Deployment Decision Report

## Workload Summary

- Total requests: 300
- Input tokens: P50=487, P90=763, P99=792
- Output tokens: P50=2760, P90=3738, P99=3943
- Average RPS: 0.08, Peak RPS: 0.18
- Burst ratio: 2.18
- Prefix reuse rate: 50.00%
- Estimated concurrency: 4.7
- Workload type: decode-heavy
- Latency sensitivity: high
- Time pattern: no significant pattern

## Resource Constraints

- Model: qwen3-32b (32B params, 64 layers)
- Max model length: 32768
- SLO: P95 TTFT ≤ 2000.0ms, P95 ITL ≤ 35.0ms
- Max GPU count: 8
- Min quality retention: 0.99
- Min capacity headroom: 15%

## Recommended Configuration

- GPU Pool: h800-sxm (H800-80GB)
- Backend: vllm
- Replicas: 1
- Tensor Parallel: 4
- Pipeline Parallel: 1
- Precision: fp8
- KV Cache Dtype: auto
- Max Num Seqs: 16
- Max Batched Tokens: 6104
- Prefix Cache: disabled
- Chunked Prefill: disabled

## Scoring Breakdown

- Overall score: 0.8638
- Optimization objective: minimize_hourly_cost
- Confidence: 1.00

## Memory Estimation

- Estimated peak memory per GPU: 21.0 GB
- Model weight per GPU: 7.5 GB
- KV cache per token: 65536 bytes

## Capacity & Headroom

- Estimated capacity headroom: 87.95%
- Required minimum: 15%

## Cost Estimation

- Total GPUs: 4
- Estimated hourly cost: $14.00
- Estimated monthly cost: $10080

## Decision Rationale

- Why this tp: TP=4，将模型切分到 4 张 GPU，减少单卡显存压力；NVLink 互联支持大 TP 的高效通信
- Why this pp: PP=1，模型规模在 TP 切分后单卡可容纳，无需流水线
- Why this precision: 使用 FP8 量化，显存减半且质量损失极小(~0.5%)
- Why this prefix_cache: 关闭 Prefix Cache，流量 prefix 复用率仅 50%，收益有限且增加显存开销

## Estimated Latency

- P95 TTFT: 99ms (SLO: 2000.0ms)
- P95 ITL: 16ms (SLO: 35.0ms)

## Alternatives

| # | GPU Pool | Backend | TP | PP | Precision | Replicas | Cost/hr | Score | Trade-off |
|---|----------|---------|----|----|-----------|----------|---------|-------|-----------|
| 1 | h800-sxm | vllm | 4 | 1 | fp8 | 1 | $14.0 | 0.864 | similar profile |
| 2 | h800-sxm | sglang | 4 | 1 | bf16 | 1 | $14.0 | 0.858 | higher latency |
| 3 | h800-sxm | sglang | 4 | 1 | bf16 | 1 | $14.0 | 0.858 | higher latency |

## Confidence Assessment

- **High confidence** (1.00): Profile data available for this configuration

## Unverified Assumptions

- Activation memory is assumed negligible relative to model weights and KV cache
- CUDA graph memory is included in runtime overhead estimate
- Communication penalty factors are based on historical profiles, not measured for this specific deployment

## Pre-deployment Verification Recommendations

1. Run a short stress test with representative traffic to verify TTFT and ITL
2. Monitor GPU memory usage during peak load to confirm headroom
3. Validate KV cache utilization stays below 90% under peak concurrency
4. Run quality evaluation to confirm fp8 quality meets retention threshold
