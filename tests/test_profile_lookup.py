"""Profile 查询测试."""

import pytest

from autopilot.estimator.profile_lookup import ProfileLookup
from autopilot.models.profile import ProfilesConfig, RuntimeProfile


@pytest.fixture
def profiles():
    return ProfilesConfig(
        profiles=[
            RuntimeProfile(
                gpu_type="H800-80GB",
                backend="vllm",
                precision="bf16",
                tp=2,
                maximum_prefill_tokens_per_second=13000,
                maximum_decode_tokens_per_second=2000,
                base_ttft_ms=200,
                base_itl_ms=22,
            ),
            RuntimeProfile(
                gpu_type="H800-80GB",
                backend="vllm",
                precision="bf16",
                tp=8,
                maximum_prefill_tokens_per_second=40000,
                maximum_decode_tokens_per_second=5000,
                base_ttft_ms=180,
                base_itl_ms=20,
            ),
            RuntimeProfile(
                gpu_type="H800-80GB",
                backend="vllm",
                precision="fp8",
                tp=4,
                maximum_prefill_tokens_per_second=30000,
                maximum_decode_tokens_per_second=4200,
                base_ttft_ms=130,
                base_itl_ms=16,
            ),
        ]
    )


class TestProfileLookup:
    """Profile 查询测试集."""

    def test_exact_match(self, profiles):
        """验证: 精确匹配."""
        lookup = ProfileLookup(profiles=profiles)
        result = lookup.lookup("H800-80GB", "vllm", "fp8", 4)
        assert result.source == "exact"
        assert result.confidence == 1.0
        assert result.profile.tp == 4

    def test_interpolation(self, profiles):
        """验证: 同 GPU 不同 TP 插值."""
        lookup = ProfileLookup(profiles=profiles)
        result = lookup.lookup("H800-80GB", "vllm", "bf16", 4)
        assert result.source == "interpolated"
        assert result.confidence < 1.0
        # 插值结果应在 tp=2 和 tp=8 之间
        assert result.profile.maximum_prefill_tokens_per_second > 13000
        assert result.profile.maximum_prefill_tokens_per_second < 40000

    def test_missing_falls_to_default(self, profiles):
        """验证: 完全缺失时返回保守默认."""
        lookup = ProfileLookup(profiles=profiles)
        result = lookup.lookup("UNKNOWN-GPU", "unknown_backend", "int4", 16)
        assert result.source == "default"
        assert result.confidence <= 0.3

    def test_low_confidence_on_scaled(self, profiles):
        """验证: 跨 GPU 缩放时置信度低."""
        lookup = ProfileLookup(profiles=profiles)
        result = lookup.lookup("L40S-48GB", "vllm", "bf16", 2)
        assert result.confidence < 0.8

    def test_empty_profiles(self):
        """验证: 空 profile 列表不崩溃."""
        lookup = ProfileLookup(profiles=ProfilesConfig(profiles=[]))
        result = lookup.lookup("H800-80GB", "vllm", "bf16", 4)
        assert result.source == "default"
        assert result.confidence <= 0.3
        assert result.profile is not None
