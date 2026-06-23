"""Workload 分析器 - 分析流量特征."""

import statistics
from collections import defaultdict

from autopilot.models.traffic import TrafficRecord, WorkloadSummary


def _percentile(data: list[float], p: float) -> float:
    """计算百分位数."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (p / 100.0) * (len(sorted_data) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(sorted_data) - 1)
    frac = idx - lower
    return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac


def _compute_rps(records: list[TrafficRecord], window_ms: int = 60_000) -> dict:
    """按时间窗口计算 RPS."""
    if not records:
        return {"avg_rps": 0.0, "peak_rps": 0.0, "burst_ratio": 1.0}

    timestamps = [r.timestamp_ms for r in records]
    min_ts = min(timestamps)
    max_ts = max(timestamps)
    duration_s = (max_ts - min_ts) / 1000.0

    if duration_s <= 0:
        return {"avg_rps": float(len(records)), "peak_rps": float(len(records)), "burst_ratio": 1.0}

    avg_rps = len(records) / duration_s

    # 按窗口分桶计算峰值
    buckets: defaultdict[int, int] = defaultdict(int)
    for ts in timestamps:
        bucket_id = (ts - min_ts) // window_ms
        buckets[bucket_id] += 1

    window_s = window_ms / 1000.0
    window_rps_values = [count / window_s for count in buckets.values()]
    peak_rps = max(window_rps_values) if window_rps_values else avg_rps
    burst_ratio = peak_rps / avg_rps if avg_rps > 0 else 1.0

    return {"avg_rps": avg_rps, "peak_rps": peak_rps, "burst_ratio": burst_ratio}


def _compute_time_pattern(records: list[TrafficRecord], window_ms: int = 300_000) -> dict:
    """检测流量时间模式（峰谷）."""
    if len(records) < 10:
        return {"has_time_pattern": False, "peak_window_rps": 0.0, "valley_window_rps": 0.0}

    timestamps = [r.timestamp_ms for r in records]
    min_ts = min(timestamps)

    buckets: defaultdict[int, int] = defaultdict(int)
    for ts in timestamps:
        bucket_id = (ts - min_ts) // window_ms
        buckets[bucket_id] += 1

    if len(buckets) < 3:
        return {"has_time_pattern": False, "peak_window_rps": 0.0, "valley_window_rps": 0.0}

    window_s = window_ms / 1000.0
    rps_values = [count / window_s for count in buckets.values()]
    peak_rps = max(rps_values)
    valley_rps = min(rps_values)
    mean_rps = statistics.mean(rps_values)

    # 如果峰谷差异超过均值的 50%，认为有时间模式
    has_pattern = (peak_rps - valley_rps) > mean_rps * 0.5

    return {
        "has_time_pattern": has_pattern,
        "peak_window_rps": peak_rps,
        "valley_window_rps": valley_rps,
    }


def analyze_workload(records: list[TrafficRecord]) -> WorkloadSummary:
    """分析流量特征，输出 WorkloadSummary."""
    if not records:
        raise ValueError("流量记录为空，无法分析")

    # Token 长度分布
    input_tokens = [r.input_tokens for r in records]
    output_tokens = [r.output_tokens for r in records]

    # RPS
    rps_info = _compute_rps(records)

    # Prefix 复用
    total_requests = len(records)
    requests_with_prefix = sum(1 for r in records if r.prefix_id)
    unique_prefixes = len(set(r.prefix_id for r in records if r.prefix_id))
    prefix_reuse_rate = 0.0
    if requests_with_prefix > 0:
        # 复用率 = 1 - (unique / total_with_prefix)，越高表示复用越多
        prefix_reuse_rate = 1.0 - (unique_prefixes / requests_with_prefix)

    # 并发估算 (Little's Law: L = λ * W)
    avg_input = statistics.mean(input_tokens)
    avg_output = statistics.mean(output_tokens)
    # 假设处理时间约为 output_tokens * 20ms (decode) + input_tokens * 0.5ms (prefill)
    estimated_processing_time_s = (avg_output * 0.02 + avg_input * 0.0005)
    estimated_concurrency = rps_info["avg_rps"] * estimated_processing_time_s

    # Workload 分类
    avg_io_ratio = avg_input / max(avg_output, 1)
    is_prefill_heavy = avg_io_ratio > 3.0  # 输入是输出的 3 倍以上

    interactive_count = sum(1 for r in records if r.priority == "interactive")
    is_latency_sensitive = interactive_count / total_requests > 0.5

    # 时间模式
    time_pattern = _compute_time_pattern(records)

    return WorkloadSummary(
        input_tokens_p50=_percentile(input_tokens, 50),
        input_tokens_p90=_percentile(input_tokens, 90),
        input_tokens_p99=_percentile(input_tokens, 99),
        output_tokens_p50=_percentile(output_tokens, 50),
        output_tokens_p90=_percentile(output_tokens, 90),
        output_tokens_p99=_percentile(output_tokens, 99),
        avg_rps=rps_info["avg_rps"],
        peak_rps=rps_info["peak_rps"],
        burst_ratio=rps_info["burst_ratio"],
        prefix_reuse_rate=prefix_reuse_rate,
        total_requests=total_requests,
        requests_with_prefix=requests_with_prefix,
        estimated_concurrency=estimated_concurrency,
        is_prefill_heavy=is_prefill_heavy,
        is_latency_sensitive=is_latency_sensitive,
        has_time_pattern=time_pattern["has_time_pattern"],
        peak_window_rps=time_pattern["peak_window_rps"],
        valley_window_rps=time_pattern["valley_window_rps"],
    )
