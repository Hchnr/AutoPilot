"""Telemetry 分析器 - 分析线上指标趋势."""

from autopilot.models.telemetry import TelemetryRecord


def analyze_telemetry(records: list[TelemetryRecord]) -> dict:
    """分析 Telemetry 数据，产出趋势摘要.

    Returns:
        分析摘要 dict，包含各指标的趋势和违反情况。
    """
    if not records:
        return {"status": "no_data", "windows": 0}

    n = len(records)

    # 各指标序列
    ttft_values = [r.p95_ttft_ms for r in records]
    itl_values = [r.p95_itl_ms for r in records]
    gpu_util_values = [r.gpu_utilization for r in records]
    kv_util_values = [r.kv_cache_utilization for r in records]
    request_rates = [r.request_rate for r in records]
    queue_depths = [r.queue_depth for r in records]
    error_rates = [r.error_rate for r in records]
    oom_counts = [r.oom_count for r in records]

    return {
        "status": "analyzed",
        "windows": n,
        "ttft": {
            "values": ttft_values,
            "avg": sum(ttft_values) / n,
            "max": max(ttft_values),
            "min": min(ttft_values),
            "trend": _trend(ttft_values),
        },
        "itl": {
            "values": itl_values,
            "avg": sum(itl_values) / n,
            "max": max(itl_values),
            "min": min(itl_values),
            "trend": _trend(itl_values),
        },
        "gpu_utilization": {
            "values": gpu_util_values,
            "avg": sum(gpu_util_values) / n,
            "max": max(gpu_util_values),
            "min": min(gpu_util_values),
        },
        "kv_cache_utilization": {
            "values": kv_util_values,
            "avg": sum(kv_util_values) / n,
            "max": max(kv_util_values),
            "min": min(kv_util_values),
        },
        "request_rate": {
            "values": request_rates,
            "avg": sum(request_rates) / n,
            "max": max(request_rates),
        },
        "queue_depth": {
            "avg": sum(queue_depths) / n,
            "max": max(queue_depths),
        },
        "errors": {
            "total_oom": sum(oom_counts),
            "avg_error_rate": sum(error_rates) / n,
            "max_error_rate": max(error_rates),
        },
    }


def _trend(values: list[float]) -> str:
    """简单趋势判断."""
    if len(values) < 3:
        return "stable"
    first_half = sum(values[: len(values) // 2]) / (len(values) // 2)
    second_half = sum(values[len(values) // 2 :]) / (len(values) - len(values) // 2)
    ratio = second_half / first_half if first_half > 0 else 1.0
    if ratio > 1.15:
        return "increasing"
    elif ratio < 0.85:
        return "decreasing"
    return "stable"
