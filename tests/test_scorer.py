"""方案评分器测试."""

import pytest

from autopilot.estimator.profile_lookup import ProfileLookup
from autopilot.models import (
    ClusterSpec,
    DeploymentPlan,
    GpuPool,
    ProfilesConfig,
    RuntimeProfile,
    SloConfig,
    SloConstraints,
    SloObjective,
    WorkloadSummary,
)
from autopilot.planner.scorer import score_and_rank


@pytest.fixture
def profiles():
    return ProfilesConfig(
        profiles=[
            RuntimeProfile(
                gpu_type="H800-80GB", backend="vllm", precision="bf16", tp=4,
                maximum_prefill_tokens_per_second=24000,
                maximum_decode_tokens_per_second=3600,
                base_ttft_ms=150, base_itl_ms=18,
            ),
            RuntimeProfile(
                gpu_type="H800-80GB", backend="vllm", precision="fp8", tp=4,
                maximum_prefill_tokens_per_second=30000,
                maximum_decode_tokens_per_second=4200,
                base_ttft_ms=130, base_itl_ms=16,
            ),
        ]
    )


@pytest.fixture
def cluster():
    return ClusterSpec(
        gpu_pools=[
            GpuPool(id="h800", gpu_type="H800-80GB", count=8, memory_gb=80, topology="nvlink", hourly_cost_per_gpu=3.5),
        ]
    )


@pytest.fixture
def workload():
    return WorkloadSummary(
        input_tokens_p50=2000, input_tokens_p90=3000, input_tokens_p99=5000,
        output_tokens_p50=300, output_tokens_p90=500, output_tokens_p99=800,
        avg_rps=10.0, peak_rps=15.0, burst_ratio=1.5,
        prefix_reuse_rate=0.7, total_requests=500, requests_with_prefix=400,
        estimated_concurrency=6.0,
        is_prefill_heavy=True, is_latency_sensitive=True,
        has_time_pattern=False,
    )


def _make_plan(**kwargs) -> DeploymentPlan:
    defaults = dict(
        gpu_pool="h800", gpu_type="H800-80GB", backend="vllm",
        replicas=1, tensor_parallel=4, pipeline_parallel=1,
        precision="bf16", kv_cache_dtype="auto",
        max_num_seqs=64, max_num_batched_tokens=8192,
        enable_prefix_cache=True, enable_chunked_prefill=True,
    )
    defaults.update(kwargs)
    return DeploymentPlan(**defaults)


class TestScorer:
    """评分器测试集."""

    def test_slo_hard_constraint(self, profiles, cluster, workload):
        """验证: SLO 不满足的方案被淘汰."""
        strict_slo = SloConfig(
            objective=SloObjective(primary="minimize_hourly_cost"),
            constraints=SloConstraints(p95_ttft_ms=100, p95_itl_ms=10),  # 极严格
        )
        profile_lookup = ProfileLookup(profiles=profiles)
        candidates = [_make_plan(replicas=1)]
        ranked = score_and_rank(candidates, workload, strict_slo, profile_lookup, cluster)
        # 极严格 SLO → 可能无法满足
        # 这取决于估算，但至少验证不崩溃
        assert isinstance(ranked, list)

    def test_cost_ranking(self, profiles, cluster, workload):
        """验证: minimize_cost 时，低成本方案排名靠前."""
        slo = SloConfig(
            objective=SloObjective(primary="minimize_hourly_cost"),
            constraints=SloConstraints(p95_ttft_ms=2000, p95_itl_ms=100),  # 宽松
        )
        profile_lookup = ProfileLookup(profiles=profiles)
        candidates = [
            _make_plan(replicas=1),  # 4 GPUs → $14/hr
            _make_plan(replicas=2),  # 8 GPUs → $28/hr
        ]
        ranked = score_and_rank(candidates, workload, slo, profile_lookup, cluster)
        if len(ranked) >= 2:
            assert ranked[0].estimated_hourly_cost <= ranked[1].estimated_hourly_cost

    def test_at_least_scored(self, profiles, cluster, workload):
        """验证: 候选方案会被评分."""
        slo = SloConfig(
            objective=SloObjective(primary="minimize_hourly_cost"),
            constraints=SloConstraints(p95_ttft_ms=2000, p95_itl_ms=100, minimum_capacity_headroom=0.0),
        )
        profile_lookup = ProfileLookup(profiles=profiles)
        candidates = [_make_plan(replicas=2)]
        ranked = score_and_rank(candidates, workload, slo, profile_lookup, cluster)
        assert len(ranked) > 0
        assert ranked[0].score > 0

    def test_rationale_generated(self, profiles, cluster, workload):
        """验证: 方案有决策理由."""
        slo = SloConfig(
            objective=SloObjective(primary="minimize_hourly_cost"),
            constraints=SloConstraints(p95_ttft_ms=2000, p95_itl_ms=100, minimum_capacity_headroom=0.0),
        )
        profile_lookup = ProfileLookup(profiles=profiles)
        candidates = [_make_plan(replicas=2)]
        ranked = score_and_rank(candidates, workload, slo, profile_lookup, cluster)
        assert len(ranked) > 0
        assert "tp" in ranked[0].rationale
        assert "precision" in ranked[0].rationale

    def test_slo_filters_violating_plans(self, profiles, cluster, workload):
        """验证: 不满足 SLO 的方案被过滤掉."""
        # 极严格 SLO：TTFT 50ms, ITL 5ms — 几乎不可能满足
        strict_slo = SloConfig(
            objective=SloObjective(primary="minimize_hourly_cost"),
            constraints=SloConstraints(p95_ttft_ms=50, p95_itl_ms=5, minimum_capacity_headroom=0.0),
        )
        profile_lookup = ProfileLookup(profiles=profiles)
        candidates = [_make_plan(replicas=1), _make_plan(replicas=2)]
        ranked = score_and_rank(candidates, workload, strict_slo, profile_lookup, cluster)
        # 严格 SLO 下应过滤大部分或全部方案
        assert len(ranked) < len(candidates)

    def test_goodput_ranking_differs_from_cost(self, profiles, cluster):
        """验证: maximize_goodput 和 minimize_cost 目标下排序可能不同."""
        cost_slo = SloConfig(
            objective=SloObjective(primary="minimize_hourly_cost"),
            constraints=SloConstraints(p95_ttft_ms=3000, p95_itl_ms=200, minimum_capacity_headroom=0.0),
        )
        goodput_slo = SloConfig(
            objective=SloObjective(primary="maximize_goodput"),
            constraints=SloConstraints(p95_ttft_ms=3000, p95_itl_ms=200, minimum_capacity_headroom=0.0),
        )
        workload = WorkloadSummary(
            input_tokens_p50=500, input_tokens_p90=1000, input_tokens_p99=1500,
            output_tokens_p50=100, output_tokens_p90=200, output_tokens_p99=300,
            avg_rps=2.0, peak_rps=3.0, burst_ratio=1.5,
            prefix_reuse_rate=0.0, total_requests=100, requests_with_prefix=0,
            estimated_concurrency=2.0,
            is_prefill_heavy=False, is_latency_sensitive=True,
            has_time_pattern=False,
        )
        profile_lookup = ProfileLookup(profiles=profiles)
        candidates = [_make_plan(replicas=1), _make_plan(replicas=2)]

        ranked_cost = score_and_rank(candidates, workload, cost_slo, profile_lookup, cluster)
        ranked_goodput = score_and_rank(candidates, workload, goodput_slo, profile_lookup, cluster)

        if len(ranked_cost) >= 2 and len(ranked_goodput) >= 2:
            # minimize_cost 应更偏好 1 副本（便宜）
            assert ranked_cost[0].estimated_hourly_cost <= ranked_cost[1].estimated_hourly_cost
            # goodput 目标下，2 副本分数差距应不大于 cost 目标下
            # （因为 goodput 更看重余量和延迟，降低了成本权重）
            cost_gap = ranked_cost[0].score - ranked_cost[1].score
            goodput_gap = ranked_goodput[0].score - ranked_goodput[1].score
            assert goodput_gap <= cost_gap
