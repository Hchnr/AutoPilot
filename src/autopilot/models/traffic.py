"""流量数据结构."""

from pydantic import BaseModel


class TrafficRecord(BaseModel):
    """单条流量记录."""

    timestamp_ms: int  # 绝对时间 Unix 毫秒
    input_tokens: int
    output_tokens: int
    prefix_id: str | None = None
    priority: str = "interactive"  # "interactive" | "batch" | "background"


class WorkloadSummary(BaseModel):
    """Workload 分析结果."""

    # Token 长度分布
    input_tokens_p50: float
    input_tokens_p90: float
    input_tokens_p99: float
    output_tokens_p50: float
    output_tokens_p90: float
    output_tokens_p99: float

    # 请求速率
    avg_rps: float
    peak_rps: float
    burst_ratio: float  # peak / avg

    # Prefix 复用
    prefix_reuse_rate: float  # unique prefix / total with prefix
    total_requests: int
    requests_with_prefix: int

    # 并发估算
    estimated_concurrency: float

    # 分类
    is_prefill_heavy: bool  # input_tokens 占比高
    is_latency_sensitive: bool  # interactive 请求占比高

    # 时间模式
    has_time_pattern: bool  # 是否有明显峰谷
    peak_window_rps: float = 0.0
    valley_window_rps: float = 0.0
