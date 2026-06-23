"""Reconcile 报告生成器."""

from autopilot.models.telemetry import ReconcileResult


def generate_reconcile_report(result: ReconcileResult) -> str:
    """生成 decision_log.md."""
    lines = []
    lines.append("# Reconcile Decision Log\n")

    # Analysis Summary
    lines.append("## Telemetry Analysis Summary\n")
    analysis = result.analysis_summary
    if analysis.get("status") == "no_data":
        lines.append("- No telemetry data available\n")
    else:
        lines.append(f"- Windows analyzed: {analysis.get('windows', 0)}")
        if "ttft" in analysis:
            ttft = analysis["ttft"]
            lines.append(
                f"- TTFT: avg={ttft['avg']:.0f}ms, max={ttft['max']:.0f}ms, trend={ttft['trend']}"
            )
        if "itl" in analysis:
            itl = analysis["itl"]
            lines.append(
                f"- ITL: avg={itl['avg']:.0f}ms, max={itl['max']:.0f}ms, trend={itl['trend']}"
            )
        if "gpu_utilization" in analysis:
            gpu = analysis["gpu_utilization"]
            lines.append(
                f"- GPU utilization: avg={gpu['avg']:.0%}, max={gpu['max']:.0%}"
            )
        if "kv_cache_utilization" in analysis:
            kv = analysis["kv_cache_utilization"]
            lines.append(
                f"- KV cache utilization: avg={kv['avg']:.0%}, max={kv['max']:.0%}"
            )
        if "errors" in analysis:
            err = analysis["errors"]
            lines.append(
                f"- OOM events: {err['total_oom']}, error rate: {err['avg_error_rate']:.4f}"
            )
    lines.append("")

    # Current Plan
    lines.append("## Current Deployment\n")
    plan = result.current_plan
    if plan:
        lines.append(f"- GPU Pool: {plan.get('gpu_pool', 'N/A')}")
        lines.append(f"- Backend: {plan.get('backend', 'N/A')}")
        lines.append(f"- Replicas: {plan.get('replicas', 'N/A')}")
        lines.append(f"- TP: {plan.get('tensor_parallel', 'N/A')}")
        lines.append(f"- Precision: {plan.get('precision', 'N/A')}")
        lines.append(f"- Max Num Seqs: {plan.get('max_num_seqs', 'N/A')}")
    lines.append("")

    # Actions
    lines.append("## Recommended Actions\n")
    if not result.actions:
        lines.append(
            "**No action needed.** Current deployment is within acceptable parameters.\n"
        )
    else:
        for i, action in enumerate(result.actions, 1):
            lines.append(f"### Action {i}: {action.action}\n")
            lines.append(
                f"- **Change**: `{action.field}` from `{action.from_value}` to `{action.to_value}`"
            )
            lines.append(f"- **Reason**: {action.reason}")
            lines.append(f"- **Confidence**: {action.confidence:.2f}")
            lines.append(f"- **Risk level**: {action.risk_level}")
            if action.requires_restart:
                lines.append(
                    "- ⚠️ **Requires restart**: This change requires service restart and may cause brief downtime"
                )
            lines.append("")

    # Safety Notes
    lines.append("## Safety Notes\n")
    lines.append(
        "- Actions are limited to one per reconcile cycle to avoid compounding changes"
    )
    lines.append(
        "- Scale-down operations require more consecutive windows of low utilization"
    )
    lines.append("- High-risk changes (TP/PP/precision) require manual confirmation")
    lines.append("- A cooldown period is enforced between consecutive actions")
    lines.append("")

    return "\n".join(lines)
