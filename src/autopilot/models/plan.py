"""部署方案数据结构."""

from pydantic import BaseModel


class DeploymentPlan(BaseModel):
    """部署方案."""

    gpu_pool: str
    gpu_type: str
    backend: str

    replicas: int
    tensor_parallel: int
    pipeline_parallel: int

    precision: str
    kv_cache_dtype: str

    max_num_seqs: int
    max_num_batched_tokens: int

    enable_prefix_cache: bool
    enable_chunked_prefill: bool
    prefill_chunk_size: int = 2048

    # 估算指标
    estimated_hourly_cost: float = 0.0
    estimated_peak_memory_per_gpu_gb: float = 0.0
    estimated_capacity_headroom: float = 0.0
    estimated_p95_ttft_ms: float = 0.0
    estimated_p95_itl_ms: float = 0.0

    # 评分
    score: float = 0.0
    confidence: float = 1.0

    # 决策理由
    rationale: dict[str, str] = {}


class PlanResult(BaseModel):
    """Plan 命令的完整输出."""

    recommended: DeploymentPlan
    alternatives: list[DeploymentPlan]
    workload_summary: dict = {}
    memory_details: dict = {}
