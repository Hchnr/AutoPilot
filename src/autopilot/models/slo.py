"""SLO 约束定义."""

from pydantic import BaseModel


class SloConstraints(BaseModel):
    """延迟和资源约束."""

    p95_ttft_ms: float = 1000.0
    p95_itl_ms: float = 50.0
    p99_e2e_ms: float = 10000.0

    minimum_quality_retention: float = 0.99
    minimum_capacity_headroom: float = 0.20
    maximum_gpu_count: int = 16


class SloObjective(BaseModel):
    """优化目标."""

    primary: str = "minimize_hourly_cost"  # minimize_hourly_cost | maximize_goodput
    secondary: str = "maximize_goodput"


class SloConfig(BaseModel):
    """完整 SLO 配置."""

    objective: SloObjective
    constraints: SloConstraints
