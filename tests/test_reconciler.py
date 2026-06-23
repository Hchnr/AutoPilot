"""Reconciler 决策引擎测试."""

import pytest

from autopilot.models import DeploymentPlan
from autopilot.models.telemetry import TelemetryRecord
from autopilot.reconciler.analyzer import analyze_telemetry
from autopilot.reconciler.decision_engine import decide_actions


@pytest.fixture
def current_plan():
    return DeploymentPlan(
        gpu_pool="h800", gpu_type="H800-80GB", backend="vllm",
        replicas=2, tensor_parallel=4, pipeline_parallel=1,
        precision="bf16", kv_cache_dtype="auto",
        max_num_seqs=64, max_num_batched_tokens=8192,
        enable_prefix_cache=True, enable_chunked_prefill=True,
    )


def _make_telemetry(
    count: int = 5,
    ttft: float = 500,
    itl: float = 30,
    gpu_util: float = 0.6,
    kv_util: float = 0.7,
    oom: int = 0,
) -> list[TelemetryRecord]:
    """生成 telemetry 数据."""
    return [
        TelemetryRecord(
            timestamp=f"2026-06-20T10:{i*5:02d}:00Z",
            request_rate=10.0,
            queue_depth=5,
            p95_ttft_ms=ttft,
            p95_itl_ms=itl,
            gpu_utilization=gpu_util,
            kv_cache_utilization=kv_util,
            oom_count=oom,
            error_rate=0.001,
        )
        for i in range(count)
    ]


class TestReconciler:
    """Reconcile 决策引擎测试集."""

    def test_slo_violation_triggers_action(self, current_plan):
        """验证: 连续多窗口 SLO 违反触发操作."""
        telemetry = _make_telemetry(count=5, ttft=920)  # 全部超过 800ms SLO
        analysis = analyze_telemetry(telemetry)
        actions = decide_actions(current_plan, telemetry, analysis)
        assert len(actions) > 0
        assert actions[0].action in ["scale_replicas", "adjust_max_num_seqs"]

    def test_single_spike_no_action(self, current_plan):
        """验证: 单窗口波动不触发操作."""
        # 只有最后一个窗口违反
        telemetry = _make_telemetry(count=4, ttft=500)
        telemetry.append(
            TelemetryRecord(
                timestamp="2026-06-20T10:20:00Z",
                request_rate=10.0, queue_depth=5,
                p95_ttft_ms=920, p95_itl_ms=30,
                gpu_utilization=0.6, kv_cache_utilization=0.7,
            )
        )
        analysis = analyze_telemetry(telemetry)
        actions = decide_actions(current_plan, telemetry, analysis)
        assert len(actions) == 0

    def test_low_gpu_util_not_immediate_scale_down(self, current_plan):
        """验证: 缩容比扩容更保守（需要更多窗口）."""
        # 只有3个窗口低利用率 → 不应缩容（需要5个）
        telemetry = _make_telemetry(count=3, gpu_util=0.15)
        analysis = analyze_telemetry(telemetry)
        actions = decide_actions(current_plan, telemetry, analysis)
        # 应该没有缩容操作
        scale_downs = [a for a in actions if a.action == "scale_replicas" and a.to_value < current_plan.replicas]
        assert len(scale_downs) == 0

    def test_sustained_low_util_scales_down(self, current_plan):
        """验证: 持续低利用率触发缩容."""
        telemetry = _make_telemetry(count=6, gpu_util=0.15)
        analysis = analyze_telemetry(telemetry)
        actions = decide_actions(current_plan, telemetry, analysis)
        scale_downs = [a for a in actions if a.action == "scale_replicas" and a.to_value < current_plan.replicas]
        assert len(scale_downs) > 0

    def test_oom_triggers_reduction(self, current_plan):
        """验证: OOM 触发 max_num_seqs 降低."""
        telemetry = _make_telemetry(count=3, oom=2)
        analysis = analyze_telemetry(telemetry)
        actions = decide_actions(current_plan, telemetry, analysis)
        assert len(actions) > 0
        assert actions[0].action == "adjust_max_num_seqs"
        assert actions[0].to_value < current_plan.max_num_seqs

    def test_kv_cache_critical_triggers_action(self, current_plan):
        """验证: KV cache 过高触发调整."""
        telemetry = _make_telemetry(count=5, kv_util=0.96)
        analysis = analyze_telemetry(telemetry)
        actions = decide_actions(current_plan, telemetry, analysis)
        assert len(actions) > 0

    def test_healthy_system_no_action(self, current_plan):
        """验证: 正常运行不产生操作."""
        telemetry = _make_telemetry(count=5, ttft=500, itl=30, gpu_util=0.6, kv_util=0.7)
        analysis = analyze_telemetry(telemetry)
        actions = decide_actions(current_plan, telemetry, analysis)
        assert len(actions) == 0

    def test_action_has_required_fields(self, current_plan):
        """验证: 操作包含必需字段."""
        telemetry = _make_telemetry(count=5, ttft=920)
        analysis = analyze_telemetry(telemetry)
        actions = decide_actions(current_plan, telemetry, analysis)
        for action in actions:
            assert action.action != ""
            assert action.reason != ""
            assert 0 < action.confidence <= 1.0
            assert action.risk_level in ["low", "medium", "high"]

    def test_empty_telemetry(self, current_plan):
        """验证: 空 telemetry 不崩溃."""
        analysis = analyze_telemetry([])
        actions = decide_actions(current_plan, [], analysis)
        assert actions == []
