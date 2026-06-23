"""推理后端能力定义."""

from typing import Any

from pydantic import BaseModel


class BackendSpec(BaseModel):
    """单个推理后端的能力声明."""

    name: str
    version: str = ">=0.1.0"
    supported_precisions: list[str]
    supported_kv_cache_dtypes: list[str] = ["auto"]
    tp_values: list[int]
    pp_values: list[int]
    features: dict[str, bool] = {}
    constraints: list[str] = []
    default_args: dict[str, Any] = {}


class BackendsConfig(BaseModel):
    """所有后端配置."""

    backends: dict[str, BackendSpec]

    @classmethod
    def from_dict(cls, data: dict) -> "BackendsConfig":
        """从 YAML dict 构建，自动为每个 backend 注入 name 字段."""
        backends = {}
        for name, spec in data.get("backends", {}).items():
            spec["name"] = name
            backends[name] = BackendSpec(**spec)
        return cls(backends=backends)
