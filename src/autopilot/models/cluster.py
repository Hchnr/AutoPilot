"""GPU 集群资源定义."""

from pydantic import BaseModel


class GpuPool(BaseModel):
    """单个 GPU 资源池."""

    id: str
    gpu_type: str  # e.g. "H800-80GB"
    count: int
    memory_gb: float
    topology: str  # "nvlink" | "pcie"
    hourly_cost_per_gpu: float


class ClusterSpec(BaseModel):
    """集群配置，包含多个 GPU Pool."""

    gpu_pools: list[GpuPool]
