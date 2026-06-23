"""模型 Spec 数据结构."""

from pydantic import BaseModel, field_validator


class ModelSpec(BaseModel):
    """模型架构和属性定义."""

    name: str
    architecture: str = "decoder_only"

    parameter_count: str  # e.g. "32B"
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int

    supported_precisions: list[str]  # e.g. ["bf16", "fp8"]
    max_model_len: int
    minimum_quality_retention: float = 0.99

    @property
    def head_dim(self) -> int:
        """head_dim = hidden_size / num_attention_heads."""
        return self.hidden_size // self.num_attention_heads

    @property
    def param_count_billion(self) -> float:
        """解析参数量为浮点数（单位: 十亿）."""
        s = self.parameter_count.upper().replace("B", "")
        return float(s)

    @property
    def param_count(self) -> int:
        """总参数量."""
        return int(self.param_count_billion * 1e9)

    @field_validator("supported_precisions")
    @classmethod
    def validate_precisions(cls, v: list[str]) -> list[str]:
        valid = {"bf16", "fp16", "fp8", "int8", "int4"}
        for p in v:
            if p not in valid:
                raise ValueError(f"不支持的精度: {p}, 可选: {valid}")
        return v
