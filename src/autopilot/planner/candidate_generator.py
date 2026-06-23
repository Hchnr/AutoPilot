"""候选方案生成器 - 穷举搜索空间并剪枝."""

from autopilot.estimator.memory import MemoryEstimator
from autopilot.models import (
    BackendsConfig,
    ClusterSpec,
    DeploymentPlan,
    ModelSpec,
    SloConfig,
    WorkloadSummary,
)


def _decide_max_num_seqs(workload: WorkloadSummary, gpu_memory_gb: float) -> int:
    """根据 workload 并发估算和 GPU 显存决定 max_num_seqs."""
    # 基础值：并发估算 * 1.5 headroom
    base = max(int(workload.estimated_concurrency * 1.5), 16)
    # 显存越大允许越多并发
    if gpu_memory_gb >= 80:
        return min(base, 256)
    elif gpu_memory_gb >= 48:
        return min(base, 128)
    else:
        return min(base, 64)


def _decide_batch_tokens(workload: WorkloadSummary, max_num_seqs: int) -> int:
    """决定 max_num_batched_tokens."""
    # 取 input P90 * max_num_seqs 的合理上界
    prefill_budget = int(workload.input_tokens_p90 * max_num_seqs * 0.5)
    return max(min(prefill_budget, 16384), 2048)


def _decide_prefix_cache(workload: WorkloadSummary) -> bool:
    """根据 prefix 复用率决定是否开启 prefix cache.

    需要同时满足:
    - 有足够多的请求携带 prefix (>30% 的请求)
    - prefix 本身有较好的复用率 (>0.3)
    """
    prefix_coverage = workload.requests_with_prefix / max(workload.total_requests, 1)
    return prefix_coverage > 0.3 and workload.prefix_reuse_rate > 0.3


def _decide_chunked_prefill(workload: WorkloadSummary) -> bool:
    """根据输入长度决定是否开启 chunked prefill."""
    return workload.input_tokens_p90 > 2048


def _decide_prefill_chunk_size(workload: WorkloadSummary) -> int:
    """决定 prefill chunk size."""
    p50 = workload.input_tokens_p50
    if p50 > 4096:
        return 4096
    elif p50 > 2048:
        return 2048
    else:
        return 1024


def generate_candidates(
    model: ModelSpec,
    cluster: ClusterSpec,
    backends: BackendsConfig,
    workload: WorkloadSummary,
    memory_estimator: MemoryEstimator,
    slo: SloConfig,
) -> list[DeploymentPlan]:
    """生成所有可行候选方案.

    搜索空间: gpu_pool × backend × precision × tp × pp × kv_cache_dtype
    剪枝: 显存不可行、GPU 数量不够、TP 不能整除 kv_heads、SLO 硬约束
    """
    candidates = []
    constraints = slo.constraints

    for pool in cluster.gpu_pools:
        for backend_name, backend in backends.backends.items():
            # 交集精度
            common_precisions = [
                p
                for p in model.supported_precisions
                if p in backend.supported_precisions
            ]

            for precision in common_precisions:
                for tp in backend.tp_values:
                    # TP 必须能整除 num_kv_heads
                    if model.num_kv_heads % tp != 0:
                        continue

                    for pp in backend.pp_values:
                        # 层数必须能整除 PP
                        if model.num_layers % pp != 0:
                            continue

                        # GPU 数量约束: tp * pp 不能超过 pool count
                        gpus_per_replica = tp * pp
                        if gpus_per_replica > pool.count:
                            continue

                        for kv_dtype in backend.supported_kv_cache_dtypes:
                            # 质量约束: fp8 precision
                            quality_retention = 1.0
                            if precision == "fp8":
                                quality_retention = 0.995
                            if kv_dtype == "fp8" and precision != "fp8":
                                quality_retention *= 0.998
                            if (
                                quality_retention
                                < constraints.minimum_quality_retention
                            ):
                                continue

                            # 决定 batch 参数
                            max_num_seqs = _decide_max_num_seqs(
                                workload, pool.memory_gb
                            )
                            max_batched_tokens = _decide_batch_tokens(
                                workload, max_num_seqs
                            )

                            # 决定上下文长度（用 P99 input + output）
                            context_len = int(
                                min(
                                    workload.input_tokens_p99
                                    + workload.output_tokens_p99,
                                    model.max_model_len,
                                )
                            )

                            # 显存可行性
                            feasible, estimated_mem = memory_estimator.is_feasible(
                                precision=precision,
                                kv_dtype=kv_dtype,
                                tp=tp,
                                pp=pp,
                                max_num_seqs=max_num_seqs,
                                context_length=context_len,
                                gpu_memory_gb=pool.memory_gb,
                            )
                            if not feasible:
                                continue

                            # 计算可用 replica 数
                            max_replicas = pool.count // gpus_per_replica
                            if max_replicas < 1:
                                continue

                            # 根据吞吐需求估算需要的 replica
                            # 简化：确保有足够 headroom
                            needed_replicas = _estimate_replicas(
                                workload,
                                max_num_seqs,
                                constraints.minimum_capacity_headroom,
                            )
                            replicas = min(max(needed_replicas, 1), max_replicas)

                            # GPU 总数约束
                            total_gpus = replicas * gpus_per_replica
                            if total_gpus > constraints.maximum_gpu_count:
                                # 缩减 replica
                                replicas = (
                                    constraints.maximum_gpu_count // gpus_per_replica
                                )
                                if replicas < 1:
                                    continue
                                total_gpus = replicas * gpus_per_replica

                            # Cache 决策
                            enable_prefix_cache = _decide_prefix_cache(
                                workload
                            ) and backend.features.get("prefix_cache", False)
                            enable_chunked = _decide_chunked_prefill(
                                workload
                            ) and backend.features.get("chunked_prefill", False)
                            chunk_size = _decide_prefill_chunk_size(workload)

                            # 成本
                            hourly_cost = total_gpus * pool.hourly_cost_per_gpu

                            candidate = DeploymentPlan(
                                gpu_pool=pool.id,
                                gpu_type=pool.gpu_type,
                                backend=backend_name,
                                replicas=replicas,
                                tensor_parallel=tp,
                                pipeline_parallel=pp,
                                precision=precision,
                                kv_cache_dtype=kv_dtype,
                                max_num_seqs=max_num_seqs,
                                max_num_batched_tokens=max_batched_tokens,
                                enable_prefix_cache=enable_prefix_cache,
                                enable_chunked_prefill=enable_chunked,
                                prefill_chunk_size=chunk_size,
                                estimated_hourly_cost=hourly_cost,
                                estimated_peak_memory_per_gpu_gb=estimated_mem,
                            )
                            candidates.append(candidate)

    return candidates


def _estimate_replicas(
    workload: WorkloadSummary, max_num_seqs: int, headroom: float
) -> int:
    """估算需要的 replica 数（基于并发）."""
    # 每个 replica 能处理 max_num_seqs 个并发
    needed_capacity = workload.estimated_concurrency * (1 + headroom)
    # 考虑突发
    needed_capacity *= workload.burst_ratio
    replicas = int(needed_capacity / max_num_seqs) + 1
    return max(replicas, 1)
