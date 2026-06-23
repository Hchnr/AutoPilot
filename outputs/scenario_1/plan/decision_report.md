# Deployment Decision Report

## Workload Summary

- Total requests: 500
- Input tokens: P50=3562, P90=4654, P99=4967
- Output tokens: P50=303, P90=468, P99=498
- Average RPS: 0.14, Peak RPS: 0.48
- Burst ratio: 3.47
- Prefix reuse rate: 99.40%
- Estimated concurrency: 1.1
- Workload type: prefill-heavy
- Latency sensitivity: high
- Time pattern: yes (peak/valley detected)

## Resource Constraints

- Model: qwen3-32b (32B params, 64 layers)
- Max model length: 32768
- SLO: P95 TTFT ≤ 800.0ms, P95 ITL ≤ 45.0ms
- Max GPU count: 8
- Min quality retention: 0.99
- Min capacity headroom: 20%

## Recommended Configuration

- GPU Pool: h800-sxm (H800-80GB)
- Backend: vllm
- Replicas: 1
- Tensor Parallel: 4
- Pipeline Parallel: 1
- Precision: fp8
- KV Cache Dtype: auto
- Max Num Seqs: 16
- Max Batched Tokens: 16384
- Prefix Cache: enabled
- Chunked Prefill: enabled

## Scoring Breakdown

- Overall score: 0.8204
- Optimization objective: minimize_hourly_cost
- Confidence: 1.00

## Memory Estimation

- Estimated peak memory per GPU: 21.8 GB
- Model weight per GPU: 7.5 GB
- KV cache per token: 65536 bytes

## Capacity & Headroom

- Estimated capacity headroom: 96.51%
- Required minimum: 20%

## Cost Estimation

- Total GPUs: 4
- Estimated hourly cost: $14.00
- Estimated monthly cost: $10080

## Decision Rationale

- Why this tp: TP=4，将模型切分到 4 张 GPU，减少单卡显存压力；NVLink 互联支持大 TP 的高效通信
- Why this pp: PP=1，模型规模在 TP 切分后单卡可容纳，无需流水线
- Why this precision: 使用 FP8 量化，显存减半且质量损失极小(~0.5%)
- Why this prefix_cache: 启用 Prefix Cache，流量 prefix 复用率 99%，可显著减少重复 prefill 计算

## Estimated Latency

- P95 TTFT: 605ms (SLO: 800.0ms)
- P95 ITL: 16ms (SLO: 45.0ms)

## Alternatives

| # | GPU Pool | Backend | TP | PP | Precision | Replicas | Cost/hr | Score | Trade-off |
|---|----------|---------|----|----|-----------|----------|---------|-------|-----------|
| 1 | h800-sxm | vllm | 4 | 1 | fp8 | 1 | $14.0 | 0.820 | similar profile |
| 2 | h800-sxm | sglang | 4 | 1 | bf16 | 1 | $14.0 | 0.812 | higher latency |
| 3 | h800-sxm | sglang | 4 | 1 | bf16 | 1 | $14.0 | 0.812 | higher latency |

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
5. Verify prefix cache hit rate with production traffic
