# Reconcile Decision Log

## Telemetry Analysis Summary

- Windows analyzed: 6
- TTFT: avg=908ms, max=1100ms, trend=increasing
- ITL: avg=40ms, max=43ms, trend=stable
- GPU utilization: avg=83%, max=90%
- KV cache utilization: avg=86%, max=92%
- OOM events: 0, error rate: 0.0020

## Current Deployment

- GPU Pool: h800-sxm
- Backend: vllm
- Replicas: 1
- TP: 4
- Precision: fp8
- Max Num Seqs: 16

## Recommended Actions

### Action 1: scale_replicas

- **Change**: `replicas` from `1` to `2`
- **Reason**: TTFT SLO violated for 4 consecutive windows, scaling up to handle load
- **Confidence**: 0.87
- **Risk level**: low

## Safety Notes

- Actions are limited to one per reconcile cycle to avoid compounding changes
- Scale-down operations require more consecutive windows of low utilization
- High-risk changes (TP/PP/precision) require manual confirmation
- A cooldown period is enforced between consecutive actions
