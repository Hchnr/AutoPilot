"""历史运行画像."""

from pydantic import BaseModel


class RuntimeProfile(BaseModel):
    """单个配置组合的运行画像."""

    gpu_type: str
    backend: str
    precision: str
    tp: int

    # 吞吐
    maximum_prefill_tokens_per_second: float
    maximum_decode_tokens_per_second: float

    # 延迟基线
    base_ttft_ms: float
    base_itl_ms: float

    # 资源
    runtime_memory_overhead_gb: float = 7.0

    # 通信惩罚
    communication_penalty: dict[str, float] = {"nvlink": 1.0, "pcie": 1.35}


class ProfilesConfig(BaseModel):
    """所有运行画像."""

    profiles: list[RuntimeProfile]
