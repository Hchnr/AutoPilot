"""Workload 分析器测试."""

import pytest

from autopilot.analyzer.workload import analyze_workload
from autopilot.models.traffic import TrafficRecord


def _make_records(
    count: int = 100,
    input_range=(500, 2000),
    output_range=(100, 500),
    prefix_ids=None,
    priority="interactive",
    burst=False,
) -> list[TrafficRecord]:
    """生成测试流量数据."""
    import random

    random.seed(42)
    base_ts = 1719100800000
    records = []
    for i in range(count):
        if burst and i < count // 5:
            ts = base_ts + random.randint(0, 60_000)  # 20% 集中在1分钟
        else:
            ts = base_ts + random.randint(0, 3_600_000)  # 分散在1小时
        records.append(
            TrafficRecord(
                timestamp_ms=ts,
                input_tokens=random.randint(*input_range),
                output_tokens=random.randint(*output_range),
                prefix_id=random.choice(prefix_ids) if prefix_ids else None,
                priority=priority,
            )
        )
    return records


class TestWorkloadAnalyzer:
    """Workload 分析器测试集."""

    def test_token_distribution(self):
        """验证 Token 长度分布计算."""
        records = _make_records(100, input_range=(1000, 3000), output_range=(200, 600))
        result = analyze_workload(records)

        assert result.input_tokens_p50 > 0
        assert result.input_tokens_p90 >= result.input_tokens_p50
        assert result.input_tokens_p99 >= result.input_tokens_p90
        assert result.output_tokens_p50 > 0
        assert result.output_tokens_p90 >= result.output_tokens_p50

    def test_rps_calculation(self):
        """验证 RPS 计算."""
        records = _make_records(200)
        result = analyze_workload(records)

        assert result.avg_rps > 0
        assert result.peak_rps >= result.avg_rps
        assert result.burst_ratio >= 1.0

    def test_burst_detection(self):
        """验证突发流量检测."""
        steady_records = _make_records(200, burst=False)
        bursty_records = _make_records(200, burst=True)

        steady_result = analyze_workload(steady_records)
        bursty_result = analyze_workload(bursty_records)

        assert bursty_result.burst_ratio > steady_result.burst_ratio

    def test_prefix_reuse_high(self):
        """验证高 prefix 复用率."""
        # 所有请求都用同一个 prefix
        records = _make_records(100, prefix_ids=["shared-prefix"])
        result = analyze_workload(records)

        # 1 unique / 100 total_with_prefix = 0.99 reuse
        assert result.prefix_reuse_rate > 0.9
        assert result.requests_with_prefix == 100

    def test_prefix_reuse_low(self):
        """验证低 prefix 复用率."""
        # 每个请求用唯一 prefix
        records = []
        for i in range(100):
            records.append(
                TrafficRecord(
                    timestamp_ms=1719100800000 + i * 1000,
                    input_tokens=1000,
                    output_tokens=200,
                    prefix_id=f"unique-{i}",
                )
            )
        result = analyze_workload(records)
        # 100 unique / 100 total = 0.0 reuse rate
        assert result.prefix_reuse_rate == 0.0

    def test_prefix_reuse_none(self):
        """验证无 prefix 时."""
        records = _make_records(100, prefix_ids=None)
        result = analyze_workload(records)
        assert result.prefix_reuse_rate == 0.0
        assert result.requests_with_prefix == 0

    def test_workload_classification_prefill_heavy(self):
        """验证 prefill-heavy 分类."""
        records = _make_records(100, input_range=(5000, 10000), output_range=(100, 200))
        result = analyze_workload(records)
        assert result.is_prefill_heavy is True

    def test_workload_classification_decode_heavy(self):
        """验证 decode-heavy 分类."""
        records = _make_records(100, input_range=(100, 300), output_range=(1000, 3000))
        result = analyze_workload(records)
        assert result.is_prefill_heavy is False

    def test_latency_sensitive(self):
        """验证延迟敏感分类."""
        records = _make_records(100, priority="interactive")
        result = analyze_workload(records)
        assert result.is_latency_sensitive is True

        records = _make_records(100, priority="batch")
        result = analyze_workload(records)
        assert result.is_latency_sensitive is False

    def test_concurrency_estimation(self):
        """验证并发估算."""
        records = _make_records(200)
        result = analyze_workload(records)
        assert result.estimated_concurrency > 0

    def test_empty_records_raises(self):
        """验证空记录报错."""
        with pytest.raises(ValueError):
            analyze_workload([])
