"""决策引擎 - 基于 Telemetry 分析生成调整操作."""

from autopilot.models import DeploymentPlan, ReconcileAction
from autopilot.models.telemetry import TelemetryRecord
from autopilot.reconciler.safeguards import SafeguardEngine


# 默认 SLO (从 plan 的 estimated 值推断，或使用保守默认)
DEFAULT_TTFT_SLO = 800.0
DEFAULT_ITL_SLO = 45.0

# 连续窗口确认数
CONSECUTIVE_WINDOWS_SCALE_UP = 3
CONSECUTIVE_WINDOWS_SCALE_DOWN = 5

# 阈值
KV_CACHE_HIGH_THRESHOLD = 0.90
KV_CACHE_CRITICAL_THRESHOLD = 0.95
GPU_UTIL_LOW_THRESHOLD = 0.25
GPU_UTIL_HIGH_THRESHOLD = 0.85


def decide_actions(
    plan: DeploymentPlan,
    telemetry: list[TelemetryRecord],
    analysis: dict,
) -> list[ReconcileAction]:
    """根据 Telemetry 分析结果决策操作.

    决策逻辑:
    1. SLO 违反 → 考虑扩容或调参
    2. 资源过载 → 扩容
    3. 资源闲置 → 缩容（更保守）
    4. KV Cache 压力 → 调整 max_num_seqs 或建议扩容
    """
    if not telemetry or analysis.get("status") == "no_data":
        return []

    safeguard = SafeguardEngine()
    actions: list[ReconcileAction] = []

    # --- SLO 违反检测 ---
    ttft_violations = _count_consecutive_violations(
        [r.p95_ttft_ms for r in telemetry], DEFAULT_TTFT_SLO
    )
    itl_violations = _count_consecutive_violations(
        [r.p95_itl_ms for r in telemetry], DEFAULT_ITL_SLO
    )

    # TTFT 违反
    if ttft_violations >= CONSECUTIVE_WINDOWS_SCALE_UP:
        # 优先调参（低风险），其次扩容
        if _can_adjust_batch(plan):
            action = ReconcileAction(
                action="adjust_max_num_seqs",
                field="max_num_seqs",
                from_value=plan.max_num_seqs,
                to_value=max(plan.max_num_seqs - 32, 16),
                reason=f"TTFT SLO violated for {ttft_violations} consecutive windows, reducing concurrency to lower queuing delay",
                confidence=0.85,
                risk_level="low",
                requires_restart=False,
            )
            if safeguard.is_allowed(action):
                actions.append(action)
        else:
            action = ReconcileAction(
                action="scale_replicas",
                field="replicas",
                from_value=plan.replicas,
                to_value=plan.replicas + 1,
                reason=f"TTFT SLO violated for {ttft_violations} consecutive windows, scaling up to handle load",
                confidence=0.87,
                risk_level="low",
                requires_restart=False,
            )
            if safeguard.is_allowed(action):
                actions.append(action)

    # ITL 违反
    if itl_violations >= CONSECUTIVE_WINDOWS_SCALE_UP and not actions:
        action = ReconcileAction(
            action="scale_replicas",
            field="replicas",
            from_value=plan.replicas,
            to_value=plan.replicas + 1,
            reason=f"ITL SLO violated for {itl_violations} consecutive windows",
            confidence=0.82,
            risk_level="low",
            requires_restart=False,
        )
        if safeguard.is_allowed(action):
            actions.append(action)

    # --- KV Cache 压力 ---
    kv_utils = [r.kv_cache_utilization for r in telemetry]
    avg_kv = sum(kv_utils) / len(kv_utils)

    if avg_kv > KV_CACHE_CRITICAL_THRESHOLD and not actions:
        action = ReconcileAction(
            action="adjust_max_num_seqs",
            field="max_num_seqs",
            from_value=plan.max_num_seqs,
            to_value=max(int(plan.max_num_seqs * 0.75), 16),
            reason=f"KV cache utilization critical ({avg_kv:.0%}), reducing max_num_seqs to prevent OOM",
            confidence=0.90,
            risk_level="medium",
            requires_restart=False,
        )
        if safeguard.is_allowed(action):
            actions.append(action)
    elif avg_kv > KV_CACHE_HIGH_THRESHOLD and not actions:
        # 建议启用 fp8 kv cache（如果当前未启用）
        if plan.kv_cache_dtype != "fp8":
            action = ReconcileAction(
                action="change_kv_cache_dtype",
                field="kv_cache_dtype",
                from_value=plan.kv_cache_dtype,
                to_value="fp8",
                reason=f"KV cache utilization high ({avg_kv:.0%}), switching to fp8 KV cache to save memory",
                confidence=0.75,
                risk_level="medium",
                requires_restart=True,
            )
            if safeguard.is_allowed(action):
                actions.append(action)

    # --- 资源闲置 → 缩容（更保守） ---
    gpu_utils = [r.gpu_utilization for r in telemetry]
    consecutive_low = _count_consecutive_below(gpu_utils, GPU_UTIL_LOW_THRESHOLD)

    if (
        consecutive_low >= CONSECUTIVE_WINDOWS_SCALE_DOWN
        and plan.replicas > 1
        and not actions
    ):
        action = ReconcileAction(
            action="scale_replicas",
            field="replicas",
            from_value=plan.replicas,
            to_value=plan.replicas - 1,
            reason=f"GPU utilization below {GPU_UTIL_LOW_THRESHOLD:.0%} for {consecutive_low} consecutive windows, scaling down",
            confidence=0.70,
            risk_level="low",
            requires_restart=False,
        )
        if safeguard.is_allowed(action):
            actions.append(action)

    # --- OOM 检测 ---
    total_oom = sum(r.oom_count for r in telemetry)
    if total_oom > 0 and not actions:
        action = ReconcileAction(
            action="adjust_max_num_seqs",
            field="max_num_seqs",
            from_value=plan.max_num_seqs,
            to_value=max(int(plan.max_num_seqs * 0.6), 8),
            reason=f"OOM detected ({total_oom} events), aggressively reducing max_num_seqs",
            confidence=0.95,
            risk_level="medium",
            requires_restart=False,
        )
        if safeguard.is_allowed(action):
            actions.append(action)

    # 如果所有检测都正常，返回空列表（保持当前配置）
    return actions


def _count_consecutive_violations(values: list[float], threshold: float) -> int:
    """从末尾计算连续超过阈值的窗口数."""
    count = 0
    for v in reversed(values):
        if v > threshold:
            count += 1
        else:
            break
    return count


def _count_consecutive_below(values: list[float], threshold: float) -> int:
    """从末尾计算连续低于阈值的窗口数."""
    count = 0
    for v in reversed(values):
        if v < threshold:
            count += 1
        else:
            break
    return count


def _can_adjust_batch(plan: DeploymentPlan) -> bool:
    """是否还有调整 batch 参数的空间."""
    return plan.max_num_seqs > 32
