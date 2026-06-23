"""Plan 报告生成器."""

from autopilot.models import (
    DeploymentPlan,
    ModelSpec,
    PlanResult,
    SloConfig,
    WorkloadSummary,
)


def generate_plan_report(
    result: PlanResult,
    workload: WorkloadSummary,
    slo: SloConfig,
    model: ModelSpec,
) -> str:
    """生成 decision_report.md."""
    plan = result.recommended
    lines = []

    lines.append("# Deployment Decision Report\n")

    # Workload Summary
    lines.append("## Workload Summary\n")
    lines.append(f"- Total requests: {workload.total_requests}")
    lines.append(
        f"- Input tokens: P50={workload.input_tokens_p50:.0f}, P90={workload.input_tokens_p90:.0f}, P99={workload.input_tokens_p99:.0f}"
    )
    lines.append(
        f"- Output tokens: P50={workload.output_tokens_p50:.0f}, P90={workload.output_tokens_p90:.0f}, P99={workload.output_tokens_p99:.0f}"
    )
    lines.append(
        f"- Average RPS: {workload.avg_rps:.2f}, Peak RPS: {workload.peak_rps:.2f}"
    )
    lines.append(f"- Burst ratio: {workload.burst_ratio:.2f}")
    lines.append(f"- Prefix reuse rate: {workload.prefix_reuse_rate:.2%}")
    lines.append(f"- Estimated concurrency: {workload.estimated_concurrency:.1f}")
    lines.append(
        f"- Workload type: {'prefill-heavy' if workload.is_prefill_heavy else 'decode-heavy'}"
    )
    lines.append(
        f"- Latency sensitivity: {'high' if workload.is_latency_sensitive else 'throughput-oriented'}"
    )
    lines.append(
        f"- Time pattern: {'yes (peak/valley detected)' if workload.has_time_pattern else 'no significant pattern'}"
    )
    lines.append("")

    # Resource Constraints
    lines.append("## Resource Constraints\n")
    lines.append(
        f"- Model: {model.name} ({model.parameter_count} params, {model.num_layers} layers)"
    )
    lines.append(f"- Max model length: {model.max_model_len}")
    lines.append(
        f"- SLO: P95 TTFT ≤ {slo.constraints.p95_ttft_ms}ms, P95 ITL ≤ {slo.constraints.p95_itl_ms}ms"
    )
    lines.append(f"- Max GPU count: {slo.constraints.maximum_gpu_count}")
    lines.append(
        f"- Min quality retention: {slo.constraints.minimum_quality_retention}"
    )
    lines.append(
        f"- Min capacity headroom: {slo.constraints.minimum_capacity_headroom:.0%}"
    )
    lines.append("")

    # Recommended Configuration
    lines.append("## Recommended Configuration\n")
    lines.append(f"- GPU Pool: {plan.gpu_pool} ({plan.gpu_type})")
    lines.append(f"- Backend: {plan.backend}")
    lines.append(f"- Replicas: {plan.replicas}")
    lines.append(f"- Tensor Parallel: {plan.tensor_parallel}")
    lines.append(f"- Pipeline Parallel: {plan.pipeline_parallel}")
    lines.append(f"- Precision: {plan.precision}")
    lines.append(f"- KV Cache Dtype: {plan.kv_cache_dtype}")
    lines.append(f"- Max Num Seqs: {plan.max_num_seqs}")
    lines.append(f"- Max Batched Tokens: {plan.max_num_batched_tokens}")
    lines.append(
        f"- Prefix Cache: {'enabled' if plan.enable_prefix_cache else 'disabled'}"
    )
    lines.append(
        f"- Chunked Prefill: {'enabled' if plan.enable_chunked_prefill else 'disabled'}"
    )
    lines.append("")

    # Scoring Breakdown
    lines.append("## Scoring Breakdown\n")
    lines.append(f"- Overall score: {plan.score:.4f}")
    lines.append(f"- Optimization objective: {slo.objective.primary}")
    lines.append(f"- Confidence: {plan.confidence:.2f}")
    lines.append("")

    # Memory Estimation
    lines.append("## Memory Estimation\n")
    lines.append(
        f"- Estimated peak memory per GPU: {plan.estimated_peak_memory_per_gpu_gb:.1f} GB"
    )
    if result.memory_details:
        lines.append(
            f"- Model weight per GPU: {result.memory_details.get('model_weight_per_gpu_gb', 0):.1f} GB"
        )
        lines.append(
            f"- KV cache per token: {result.memory_details.get('kv_cache_per_token_bytes', 0):.0f} bytes"
        )
    lines.append("")

    # Capacity & Headroom
    lines.append("## Capacity & Headroom\n")
    lines.append(
        f"- Estimated capacity headroom: {plan.estimated_capacity_headroom:.2%}"
    )
    lines.append(f"- Required minimum: {slo.constraints.minimum_capacity_headroom:.0%}")
    lines.append("")

    # Cost Estimation
    lines.append("## Cost Estimation\n")
    total_gpus = plan.replicas * plan.tensor_parallel * plan.pipeline_parallel
    lines.append(f"- Total GPUs: {total_gpus}")
    lines.append(f"- Estimated hourly cost: ${plan.estimated_hourly_cost:.2f}")
    lines.append(f"- Estimated monthly cost: ${plan.estimated_hourly_cost * 720:.0f}")
    lines.append("")

    # Decision Rationale
    lines.append("## Decision Rationale\n")
    for key, reason in plan.rationale.items():
        lines.append(f"- Why this {key}: {reason}")
    lines.append("")

    # Estimated Latency
    lines.append("## Estimated Latency\n")
    lines.append(
        f"- P95 TTFT: {plan.estimated_p95_ttft_ms:.0f}ms (SLO: {slo.constraints.p95_ttft_ms}ms)"
    )
    lines.append(
        f"- P95 ITL: {plan.estimated_p95_itl_ms:.0f}ms (SLO: {slo.constraints.p95_itl_ms}ms)"
    )
    lines.append("")

    # Alternatives
    lines.append("## Alternatives\n")
    lines.append(
        "| # | GPU Pool | Backend | TP | PP | Precision | Replicas | Cost/hr | Score | Trade-off |"
    )
    lines.append(
        "|---|----------|---------|----|----|-----------|----------|---------|-------|-----------|"
    )
    for i, alt in enumerate(result.alternatives, 1):
        tradeoff = _describe_tradeoff(plan, alt)
        lines.append(
            f"| {i} | {alt.gpu_pool} | {alt.backend} | {alt.tensor_parallel} | "
            f"{alt.pipeline_parallel} | {alt.precision} | {alt.replicas} | "
            f"${alt.estimated_hourly_cost:.1f} | {alt.score:.3f} | {tradeoff} |"
        )
    lines.append("")

    # Confidence Assessment
    lines.append("## Confidence Assessment\n")
    if plan.confidence >= 0.8:
        lines.append(
            f"- **High confidence** ({plan.confidence:.2f}): Profile data available for this configuration"
        )
    elif plan.confidence >= 0.5:
        lines.append(
            f"- **Medium confidence** ({plan.confidence:.2f}): Profile interpolated from similar configurations"
        )
    else:
        lines.append(
            f"- **Low confidence** ({plan.confidence:.2f}): Limited profile data, using conservative estimates"
        )
    lines.append("")

    # Unverified Assumptions
    lines.append("## Unverified Assumptions\n")
    lines.append(
        "- Activation memory is assumed negligible relative to model weights and KV cache"
    )
    lines.append("- CUDA graph memory is included in runtime overhead estimate")
    lines.append(
        "- Communication penalty factors are based on historical profiles, not measured for this specific deployment"
    )
    if plan.confidence < 0.7:
        lines.append(
            "- Latency estimates are based on interpolated/default profiles and may differ from actual performance"
        )
    lines.append("")

    # Pre-deployment Verification
    lines.append("## Pre-deployment Verification Recommendations\n")
    lines.append(
        "1. Run a short stress test with representative traffic to verify TTFT and ITL"
    )
    lines.append("2. Monitor GPU memory usage during peak load to confirm headroom")
    lines.append(
        "3. Validate KV cache utilization stays below 90% under peak concurrency"
    )
    if plan.precision == "fp8":
        lines.append(
            "4. Run quality evaluation to confirm fp8 quality meets retention threshold"
        )
    if plan.enable_prefix_cache:
        lines.append(
            f"{'5' if plan.precision == 'fp8' else '4'}. Verify prefix cache hit rate with production traffic"
        )
    lines.append("")

    return "\n".join(lines)


def _describe_tradeoff(recommended: DeploymentPlan, alternative: DeploymentPlan) -> str:
    """描述备选方案相对于推荐方案的权衡."""
    parts = []
    if alternative.estimated_hourly_cost > recommended.estimated_hourly_cost:
        parts.append("higher cost")
    elif alternative.estimated_hourly_cost < recommended.estimated_hourly_cost:
        parts.append("lower cost")

    if alternative.estimated_p95_ttft_ms < recommended.estimated_p95_ttft_ms:
        parts.append("lower latency")
    elif alternative.estimated_p95_ttft_ms > recommended.estimated_p95_ttft_ms:
        parts.append("higher latency")

    if (
        alternative.estimated_capacity_headroom
        > recommended.estimated_capacity_headroom
    ):
        parts.append("more headroom")

    return ", ".join(parts) if parts else "similar profile"
