"""候选方案生成器测试."""

import pytest

from autopilot.estimator.memory import MemoryEstimator
from autopilot.models import (
    BackendsConfig,
    ClusterSpec,
    GpuPool,
    ModelSpec,
    SloConfig,
    SloConstraints,
    SloObjective,
    WorkloadSummary,
)
from autopilot.planner.candidate_generator import generate_candidates


@pytest.fixture
def model():
    return ModelSpec(
        name="qwen3-32b",
        architecture="decoder_only",
        parameter_count="32B",
        num_layers=64,
        hidden_size=5120,
        num_attention_heads=40,
        num_kv_heads=8,
        supported_precisions=["bf16", "fp8"],
        max_model_len=32768,
    )


@pytest.fixture
def cluster():
    return ClusterSpec(
        gpu_pools=[
            GpuPool(
                id="h800",
                gpu_type="H800-80GB",
                count=8,
                memory_gb=80,
                topology="nvlink",
                hourly_cost_per_gpu=3.5,
            ),
            GpuPool(
                id="l40s",
                gpu_type="L40S-48GB",
                count=16,
                memory_gb=48,
                topology="pcie",
                hourly_cost_per_gpu=1.5,
            ),
        ]
    )


@pytest.fixture
def backends():
    return BackendsConfig.from_dict(
        {
            "backends": {
                "vllm": {
                    "supported_precisions": ["bf16", "fp8"],
                    "supported_kv_cache_dtypes": ["auto", "fp8"],
                    "tp_values": [1, 2, 4, 8],
                    "pp_values": [1, 2],
                    "features": {"prefix_cache": True, "chunked_prefill": True},
                }
            }
        }
    )


@pytest.fixture
def workload():
    return WorkloadSummary(
        input_tokens_p50=2000,
        input_tokens_p90=4000,
        input_tokens_p99=6000,
        output_tokens_p50=300,
        output_tokens_p90=600,
        output_tokens_p99=1000,
        avg_rps=10.0,
        peak_rps=20.0,
        burst_ratio=2.0,
        prefix_reuse_rate=0.8,
        total_requests=500,
        requests_with_prefix=400,
        estimated_concurrency=8.0,
        is_prefill_heavy=True,
        is_latency_sensitive=True,
        has_time_pattern=True,
        peak_window_rps=20.0,
        valley_window_rps=5.0,
    )


@pytest.fixture
def slo():
    return SloConfig(
        objective=SloObjective(primary="minimize_hourly_cost"),
        constraints=SloConstraints(p95_ttft_ms=800, p95_itl_ms=45, maximum_gpu_count=8),
    )


class TestCandidateGenerator:
    """候选方案生成器测试集."""

    def test_generates_candidates(self, model, cluster, backends, workload, slo):
        """验证: 能生成候选方案."""
        estimator = MemoryEstimator(model=model)
        candidates = generate_candidates(
            model, cluster, backends, workload, estimator, slo
        )
        assert len(candidates) > 0

    def test_tp_divides_kv_heads(self, model, cluster, backends, workload, slo):
        """验证: TP 必须整除 num_kv_heads."""
        estimator = MemoryEstimator(model=model)
        candidates = generate_candidates(
            model, cluster, backends, workload, estimator, slo
        )
        for c in candidates:
            assert model.num_kv_heads % c.tensor_parallel == 0

    def test_gpu_count_constraint(self, model, cluster, backends, workload, slo):
        """验证: 总 GPU 数不超过约束."""
        estimator = MemoryEstimator(model=model)
        candidates = generate_candidates(
            model, cluster, backends, workload, estimator, slo
        )
        for c in candidates:
            total_gpus = c.replicas * c.tensor_parallel * c.pipeline_parallel
            assert total_gpus <= slo.constraints.maximum_gpu_count

    def test_infeasible_memory_excluded(self, model, cluster, backends, workload, slo):
        """验证: 显存不足的配置被排除."""
        estimator = MemoryEstimator(model=model)
        candidates = generate_candidates(
            model, cluster, backends, workload, estimator, slo
        )
        # bf16 tp=1 在 L40S-48GB 上不可行
        for c in candidates:
            if (
                c.gpu_pool == "l40s"
                and c.precision == "bf16"
                and c.tensor_parallel == 1
            ):
                pytest.fail("不可行方案未被过滤: L40S bf16 tp=1")

    def test_pp_divides_layers(self, model, cluster, backends, workload, slo):
        """验证: PP 必须整除层数."""
        estimator = MemoryEstimator(model=model)
        candidates = generate_candidates(
            model, cluster, backends, workload, estimator, slo
        )
        for c in candidates:
            assert model.num_layers % c.pipeline_parallel == 0

    def test_prefix_cache_influenced_by_workload(self, model, cluster, backends, slo):
        """验证: prefix_cache 决策受 workload 影响."""
        estimator = MemoryEstimator(model=model)

        # 高 prefix 复用
        high_prefix = WorkloadSummary(
            input_tokens_p50=2000,
            input_tokens_p90=4000,
            input_tokens_p99=6000,
            output_tokens_p50=300,
            output_tokens_p90=600,
            output_tokens_p99=1000,
            avg_rps=10.0,
            peak_rps=20.0,
            burst_ratio=2.0,
            prefix_reuse_rate=0.8,
            total_requests=500,
            requests_with_prefix=400,
            estimated_concurrency=8.0,
            is_prefill_heavy=True,
            is_latency_sensitive=True,
            has_time_pattern=False,
        )
        candidates_high = generate_candidates(
            model, cluster, backends, high_prefix, estimator, slo
        )
        has_prefix_cache = any(c.enable_prefix_cache for c in candidates_high)
        assert has_prefix_cache, "高复用场景应有启用 prefix cache 的方案"

        # 低 prefix 复用
        low_prefix = WorkloadSummary(
            input_tokens_p50=200,
            input_tokens_p90=500,
            input_tokens_p99=800,
            output_tokens_p50=1500,
            output_tokens_p90=2500,
            output_tokens_p99=3500,
            avg_rps=5.0,
            peak_rps=8.0,
            burst_ratio=1.6,
            prefix_reuse_rate=0.05,
            total_requests=300,
            requests_with_prefix=10,
            estimated_concurrency=5.0,
            is_prefill_heavy=False,
            is_latency_sensitive=False,
            has_time_pattern=False,
        )
        candidates_low = generate_candidates(
            model, cluster, backends, low_prefix, estimator, slo
        )
        all_no_cache = all(not c.enable_prefix_cache for c in candidates_low)
        assert all_no_cache, "低复用场景不应启用 prefix cache"

    def test_total_gpu_count(self, model, cluster, backends, workload, slo):
        """验证: TP × PP × Replicas = 方案使用的总 GPU 数."""
        estimator = MemoryEstimator(model=model)
        candidates = generate_candidates(
            model, cluster, backends, workload, estimator, slo
        )
        for c in candidates:
            total_gpus = c.tensor_parallel * c.pipeline_parallel * c.replicas
            assert total_gpus >= 1
            # 总 GPU 数不能超过对应资源池可用数
            pool = next(p for p in cluster.gpu_pools if p.id == c.gpu_pool)
            assert total_gpus <= pool.count

    def test_replicas_scale_with_load(self, model, cluster, backends, slo):
        """验证: 更高流量需要更多副本."""
        estimator = MemoryEstimator(model=model)

        low_load = WorkloadSummary(
            input_tokens_p50=1000,
            input_tokens_p90=2000,
            input_tokens_p99=3000,
            output_tokens_p50=200,
            output_tokens_p90=400,
            output_tokens_p99=600,
            avg_rps=2.0,
            peak_rps=3.0,
            burst_ratio=1.5,
            prefix_reuse_rate=0.0,
            total_requests=100,
            requests_with_prefix=0,
            estimated_concurrency=2.0,
            is_prefill_heavy=True,
            is_latency_sensitive=True,
            has_time_pattern=False,
        )
        high_load = WorkloadSummary(
            input_tokens_p50=1000,
            input_tokens_p90=2000,
            input_tokens_p99=3000,
            output_tokens_p50=200,
            output_tokens_p90=400,
            output_tokens_p99=600,
            avg_rps=50.0,
            peak_rps=80.0,
            burst_ratio=1.6,
            prefix_reuse_rate=0.0,
            total_requests=3000,
            requests_with_prefix=0,
            estimated_concurrency=40.0,
            is_prefill_heavy=True,
            is_latency_sensitive=True,
            has_time_pattern=False,
        )
        candidates_low = generate_candidates(
            model, cluster, backends, low_load, estimator, slo
        )
        candidates_high = generate_candidates(
            model, cluster, backends, high_load, estimator, slo
        )

        max_replicas_low = (
            max(c.replicas for c in candidates_low) if candidates_low else 1
        )
        max_replicas_high = (
            max(c.replicas for c in candidates_high) if candidates_high else 1
        )
        assert max_replicas_high >= max_replicas_low
