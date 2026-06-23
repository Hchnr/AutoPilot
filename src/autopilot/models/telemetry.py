"""线上 Telemetry 数据."""

from pydantic import BaseModel


class TelemetryRecord(BaseModel):
    """单个时间窗口的 Telemetry."""

    timestamp: str  # ISO format
    request_rate: float
    queue_depth: int = 0

    p95_ttft_ms: float
    p95_itl_ms: float

    gpu_utilization: float
    kv_cache_utilization: float

    oom_count: int = 0
    error_rate: float = 0.0


class ReconcileAction(BaseModel):
    """Reconcile 建议的操作."""

    action: str  # scale_replicas, adjust_max_num_seqs, etc.
    field: str = ""  # 修改的字段名
    from_value: int | float | bool | str = ""
    to_value: int | float | bool | str = ""
    reason: str
    confidence: float
    risk_level: str = "low"  # low | medium | high
    requires_restart: bool = False


class ReconcileResult(BaseModel):
    """Reconcile 命令的完整输出."""

    actions: list[ReconcileAction]
    analysis_summary: dict = {}
    current_plan: dict = {}
