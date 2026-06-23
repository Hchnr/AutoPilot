"""方案评分与排序."""

from autopilot.estimator.profile_lookup import ProfileLookup
from autopilot.models import (
    ClusterSpec,
    DeploymentPlan,
    SloConfig,
    WorkloadSummary,
)


def _estimate_latency(
    plan: DeploymentPlan,
    workload: WorkloadSummary,
    profile_lookup: ProfileLookup,
    cluster: ClusterSpec,
) -> tuple[float, float, float]:
    """估算延迟 (p95_ttft_ms, p95_itl_ms, confidence).

    基于 profile 查询结果估算延迟。
    """
    # 查找 GPU type
    gpu_pool = next((p for p in cluster.gpu_pools if p.id == plan.gpu_pool), None)
    if gpu_pool is None:
        return 999999.0, 999999.0, 0.0

    result = profile_lookup.lookup(
        gpu_type=gpu_pool.gpu_type,
        backend=plan.backend,
        precision=plan.precision,
        tp=plan.tensor_parallel,
    )
    profile = result.profile
    confidence = result.confidence

    # 通信惩罚
    topology = gpu_pool.topology
    comm_penalty = profile.communication_penalty.get(topology, 1.2)

    # TTFT 估算: base_ttft * (input_tokens_p95 / 1000) * comm_penalty * load_factor
    # load_factor: 基于并发/replica
    concurrency_per_replica = workload.estimated_concurrency / max(plan.replicas, 1)
    load_factor = 1.0 + 0.1 * max(0, concurrency_per_replica - 10)

    estimated_ttft = (
        profile.base_ttft_ms
        * (workload.input_tokens_p90 / 1000.0)
        * comm_penalty
        * load_factor
    )

    # ITL 估算: base_itl * comm_penalty * load_factor
    estimated_itl = profile.base_itl_ms * comm_penalty * load_factor

    return estimated_ttft, estimated_itl, confidence


def _compute_cost(plan: DeploymentPlan, cluster: ClusterSpec) -> float:
    """计算每小时成本."""
    gpu_pool = next((p for p in cluster.gpu_pools if p.id == plan.gpu_pool), None)
    if gpu_pool is None:
        return float("inf")

    total_gpus = plan.replicas * plan.tensor_parallel * plan.pipeline_parallel
    return total_gpus * gpu_pool.hourly_cost_per_gpu


def _compute_capacity_headroom(
    plan: DeploymentPlan,
    workload: WorkloadSummary,
    profile_lookup: ProfileLookup,
    cluster: ClusterSpec,
) -> float:
    """计算容量余量."""
    gpu_pool = next((p for p in cluster.gpu_pools if p.id == plan.gpu_pool), None)
    if gpu_pool is None:
        return 0.0

    result = profile_lookup.lookup(
        gpu_type=gpu_pool.gpu_type,
        backend=plan.backend,
        precision=plan.precision,
        tp=plan.tensor_parallel,
    )
    profile = result.profile

    # 最大吞吐 = decode_tokens/s * replicas
    max_throughput = profile.maximum_decode_tokens_per_second * plan.replicas
    # 需求 = peak_rps * avg_output_tokens
    demand = workload.peak_rps * workload.output_tokens_p50

    if max_throughput <= 0:
        return 0.0
    return (max_throughput - demand) / max_throughput


def score_and_rank(
    candidates: list[DeploymentPlan],
    workload: WorkloadSummary,
    slo: SloConfig,
    profile_lookup: ProfileLookup,
    cluster: ClusterSpec,
) -> list[DeploymentPlan]:
    """对候选方案评分并排序.

    评分维度:
    1. SLO 满足度（硬约束，不满足则淘汰）
    2. 成本效率
    3. 吞吐余量
    4. 置信度
    """
    scored: list[DeploymentPlan] = []
    constraints = slo.constraints

    for plan in candidates:
        # 估算延迟
        est_ttft, est_itl, confidence = _estimate_latency(
            plan, workload, profile_lookup, cluster
        )
        plan.estimated_p95_ttft_ms = est_ttft
        plan.estimated_p95_itl_ms = est_itl
        plan.confidence = confidence

        # 硬约束: SLO
        if est_ttft > constraints.p95_ttft_ms and confidence > 0.5:
            continue
        if est_itl > constraints.p95_itl_ms and confidence > 0.5:
            continue

        # 成本
        hourly_cost = _compute_cost(plan, cluster)
        plan.estimated_hourly_cost = hourly_cost

        # 容量余量
        headroom = _compute_capacity_headroom(plan, workload, profile_lookup, cluster)
        plan.estimated_capacity_headroom = headroom

        # 容量余量硬约束
        if headroom < constraints.minimum_capacity_headroom and confidence > 0.5:
            continue

        # 综合评分（越高越好）
        # 归一化成本: 1 / (1 + cost)
        cost_score = 1.0 / (1.0 + hourly_cost / 100.0)
        # 余量分
        headroom_score = min(headroom, 1.0)
        # 延迟裕度
        ttft_margin = max(0, 1.0 - est_ttft / constraints.p95_ttft_ms)
        itl_margin = max(0, 1.0 - est_itl / constraints.p95_itl_ms)
        latency_score = (ttft_margin + itl_margin) / 2.0

        # 权重取决于优化目标
        if slo.objective.primary == "minimize_hourly_cost":
            score = cost_score * 0.5 + headroom_score * 0.2 + latency_score * 0.2 + confidence * 0.1
        else:  # maximize_goodput
            score = latency_score * 0.4 + headroom_score * 0.3 + cost_score * 0.2 + confidence * 0.1

        plan.score = score

        # 生成 rationale
        plan.rationale = _generate_rationale(plan, workload, cluster)

        scored.append(plan)

    # 排序
    scored.sort(key=lambda p: p.score, reverse=True)
    return scored


def _generate_rationale(
    plan: DeploymentPlan, workload: WorkloadSummary, cluster: ClusterSpec
) -> dict[str, str]:
    """为方案生成决策理由."""
    rationale = {}

    # TP 选择
    rationale["tp"] = f"TP={plan.tensor_parallel}，将模型切分到 {plan.tensor_parallel} 张 GPU，减少单卡显存压力"
    if plan.tensor_parallel > 2:
        gpu_pool = next((p for p in cluster.gpu_pools if p.id == plan.gpu_pool), None)
        if gpu_pool and gpu_pool.topology == "nvlink":
            rationale["tp"] += "；NVLink 互联支持大 TP 的高效通信"

    # PP 选择
    if plan.pipeline_parallel > 1:
        rationale["pp"] = f"PP={plan.pipeline_parallel}，模型层数多({plan.pipeline_parallel}路流水)，进一步降低单卡显存"
    else:
        rationale["pp"] = "PP=1，模型规模在 TP 切分后单卡可容纳，无需流水线"

    # Precision
    if plan.precision == "fp8":
        rationale["precision"] = "使用 FP8 量化，显存减半且质量损失极小(~0.5%)"
    else:
        rationale["precision"] = "使用 BF16 全精度，保证最高推理质量"

    # Prefix cache
    if plan.enable_prefix_cache:
        rationale["prefix_cache"] = (
            f"启用 Prefix Cache，流量 prefix 复用率 {workload.prefix_reuse_rate:.0%}，"
            "可显著减少重复 prefill 计算"
        )
    else:
        rationale["prefix_cache"] = (
            f"关闭 Prefix Cache，流量 prefix 复用率仅 {workload.prefix_reuse_rate:.0%}，"
            "收益有限且增加显存开销"
        )

    return rationale
