"""显存估算器."""

from autopilot.models.model_spec import ModelSpec


# 精度对应的 bytes per parameter
PRECISION_BYTES = {
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "int8": 1,
    "int4": 0.5,
}

# KV cache dtype 对应的 bytes
KV_DTYPE_BYTES = {
    "auto": None,  # 取决于 model precision
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
}


class MemoryEstimator:
    """显存估算器，基于模型 spec 计算各项显存占用."""

    def __init__(self, model: ModelSpec):
        self.model = model

    def model_weight_memory(self, precision: str, tp: int, pp: int = 1) -> float:
        """模型权重显存 (GB per GPU).

        公式: param_count * bytes_per_param / tp / pp / 1e9
        """
        bytes_per_param = PRECISION_BYTES.get(precision, 2)
        total_bytes = self.model.param_count * bytes_per_param
        per_gpu_bytes = total_bytes / tp / pp
        return per_gpu_bytes / (1024**3)

    def kv_cache_per_token(self, kv_dtype: str, tp: int) -> float:
        """单个 token 的 KV cache 显存 (bytes).

        公式: 2 * num_layers * num_kv_heads * head_dim * kv_dtype_bytes / tp
        """
        if kv_dtype == "auto":
            # auto 模式下跟随最小支持精度（通常 bf16 = 2 bytes）
            kv_bytes = 2
        else:
            kv_bytes = KV_DTYPE_BYTES.get(kv_dtype, 2)

        return (
            2  # K + V
            * self.model.num_layers
            * self.model.num_kv_heads
            * self.model.head_dim
            * kv_bytes
            / tp
        )

    def kv_cache_memory(
        self, kv_dtype: str, tp: int, max_num_seqs: int, context_length: int
    ) -> float:
        """KV cache 总显存 (GB per GPU).

        公式: kv_per_token * max_num_seqs * context_length / 1e9
        """
        per_token = self.kv_cache_per_token(kv_dtype, tp)
        total_bytes = per_token * max_num_seqs * context_length
        return total_bytes / (1024**3)

    def total_memory_per_gpu(
        self,
        precision: str,
        kv_dtype: str,
        tp: int,
        pp: int,
        max_num_seqs: int,
        context_length: int,
        runtime_overhead_gb: float = 7.0,
    ) -> float:
        """总显存估算 (GB per GPU).

        公式: model_weight + kv_cache + runtime_overhead + safety_margin
        """
        weight_mem = self.model_weight_memory(precision, tp, pp)
        kv_mem = self.kv_cache_memory(kv_dtype, tp, max_num_seqs, context_length)
        subtotal = weight_mem + kv_mem + runtime_overhead_gb
        safety_margin = subtotal * 0.10  # 10% 安全余量
        return subtotal + safety_margin

    def is_feasible(
        self,
        precision: str,
        kv_dtype: str,
        tp: int,
        pp: int,
        max_num_seqs: int,
        context_length: int,
        gpu_memory_gb: float,
        runtime_overhead_gb: float = 7.0,
    ) -> tuple[bool, float]:
        """判断配置是否在显存内可行.

        Returns:
            (is_feasible, estimated_memory_gb)
        """
        estimated = self.total_memory_per_gpu(
            precision=precision,
            kv_dtype=kv_dtype,
            tp=tp,
            pp=pp,
            max_num_seqs=max_num_seqs,
            context_length=context_length,
            runtime_overhead_gb=runtime_overhead_gb,
        )
        # 使用 95% 的 GPU 显存作为上限
        return estimated <= gpu_memory_gb * 0.95, estimated
