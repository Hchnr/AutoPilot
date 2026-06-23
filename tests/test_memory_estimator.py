"""显存估算器测试."""

import pytest

from autopilot.estimator.memory import MemoryEstimator
from autopilot.models.model_spec import ModelSpec


@pytest.fixture
def qwen3_32b():
    return ModelSpec(
        name="qwen3-32b",
        architecture="decoder_only",
        parameter_count="32B",
        num_layers=64,
        hidden_size=5120,
        num_attention_heads=40,
        num_kv_heads=8,
        supported_precisions=["bf16", "fp8"],
        max_model_len=32768,
    )


@pytest.fixture
def estimator(qwen3_32b):
    return MemoryEstimator(model=qwen3_32b)


class TestMemoryEstimator:
    """显存估算器测试集."""

    def test_model_weight_bf16_tp4(self, estimator):
        """验证: qwen3-32b, bf16, tp=4 → ~16 GB per GPU."""
        result = estimator.model_weight_memory(precision="bf16", tp=4)
        # 32e9 * 2 bytes / 4 GPUs / 1024^3 = ~14.9 GB
        assert abs(result - 14.9) < 1.0

    def test_model_weight_fp8_tp4(self, estimator):
        """验证: fp8 权重减半."""
        bf16_mem = estimator.model_weight_memory(precision="bf16", tp=4)
        fp8_mem = estimator.model_weight_memory(precision="fp8", tp=4)
        assert abs(fp8_mem - bf16_mem / 2) < 0.5

    def test_model_weight_tp1_vs_tp4(self, estimator):
        """验证: TP=1 是 TP=4 的 4 倍."""
        tp1 = estimator.model_weight_memory(precision="bf16", tp=1)
        tp4 = estimator.model_weight_memory(precision="bf16", tp=4)
        assert abs(tp1 / tp4 - 4.0) < 0.01

    def test_model_weight_pp2(self, estimator):
        """验证: PP=2 减半单卡显存."""
        pp1 = estimator.model_weight_memory(precision="bf16", tp=4, pp=1)
        pp2 = estimator.model_weight_memory(precision="bf16", tp=4, pp=2)
        assert abs(pp1 / pp2 - 2.0) < 0.01

    def test_kv_cache_per_token(self, estimator):
        """验证 KV cache per token 计算."""
        # 2 * 64 layers * 8 kv_heads * 128 head_dim * 2 bytes / tp=4
        # = 2 * 64 * 8 * 128 * 2 / 4 = 65536 bytes
        result = estimator.kv_cache_per_token(kv_dtype="auto", tp=4)
        assert result == 65536.0

    def test_kv_cache_fp8_halves(self, estimator):
        """验证 fp8 KV cache 比 auto(bf16) 减半."""
        auto_kv = estimator.kv_cache_per_token(kv_dtype="auto", tp=4)
        fp8_kv = estimator.kv_cache_per_token(kv_dtype="fp8", tp=4)
        assert abs(fp8_kv - auto_kv / 2) < 1.0

    def test_total_memory(self, estimator):
        """验证总显存估算."""
        total = estimator.total_memory_per_gpu(
            precision="bf16",
            kv_dtype="auto",
            tp=4,
            pp=1,
            max_num_seqs=64,
            context_length=4096,
        )
        # model_weight(~14.9) + kv(64*4096*65536/1e9 ~16.4) + runtime(7) + 10% safety
        assert total > 0
        assert total < 80  # 应该小于单张 H800

    def test_feasible_tp4_h800(self, estimator):
        """验证: tp=4, H800 80GB, bf16 应可行."""
        feasible, mem = estimator.is_feasible(
            precision="bf16",
            kv_dtype="auto",
            tp=4,
            pp=1,
            max_num_seqs=32,
            context_length=4096,
            gpu_memory_gb=80,
        )
        assert feasible is True
        assert mem < 80

    def test_infeasible_tp1_bf16(self, estimator):
        """验证: tp=1, bf16 在 48GB GPU 上不可行."""
        feasible, mem = estimator.is_feasible(
            precision="bf16",
            kv_dtype="auto",
            tp=1,
            pp=1,
            max_num_seqs=64,
            context_length=4096,
            gpu_memory_gb=48,
        )
        assert feasible is False
        assert mem > 48

    def test_concurrency_affects_memory(self, estimator):
        """验证: 增加并发 → 增加显存."""
        mem_low = estimator.total_memory_per_gpu(
            precision="bf16",
            kv_dtype="auto",
            tp=4,
            pp=1,
            max_num_seqs=16,
            context_length=4096,
        )
        mem_high = estimator.total_memory_per_gpu(
            precision="bf16",
            kv_dtype="auto",
            tp=4,
            pp=1,
            max_num_seqs=128,
            context_length=4096,
        )
        assert mem_high > mem_low

    def test_context_length_affects_memory(self, estimator):
        """验证: 增加上下文长度 → 增加显存."""
        mem_short = estimator.total_memory_per_gpu(
            precision="bf16",
            kv_dtype="auto",
            tp=4,
            pp=1,
            max_num_seqs=32,
            context_length=2048,
        )
        mem_long = estimator.total_memory_per_gpu(
            precision="bf16",
            kv_dtype="auto",
            tp=4,
            pp=1,
            max_num_seqs=32,
            context_length=16384,
        )
        assert mem_long > mem_short
