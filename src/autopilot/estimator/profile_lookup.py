"""Profile 查询与缺失处理."""

from autopilot.models.profile import ProfilesConfig, RuntimeProfile


class ProfileLookupResult:
    """Profile 查询结果，包含置信度."""

    def __init__(
        self,
        profile: RuntimeProfile,
        source: str,  # "exact" | "interpolated" | "scaled" | "default"
        confidence: float,
    ):
        self.profile = profile
        self.source = source
        self.confidence = confidence


class ProfileLookup:
    """Profile 查询引擎，支持多级 fallback."""

    def __init__(self, profiles: ProfilesConfig):
        self.profiles = profiles.profiles

    def lookup(
        self,
        gpu_type: str,
        backend: str,
        precision: str,
        tp: int,
    ) -> ProfileLookupResult:
        """查找最匹配的 profile.

        查找策略:
        1. 精确匹配
        2. 同 GPU + 后端，不同 TP → 线性插值
        3. 同代 GPU 缩放
        4. 保守默认
        """
        # 1. 精确匹配
        exact = self._exact_match(gpu_type, backend, precision, tp)
        if exact:
            return ProfileLookupResult(exact, source="exact", confidence=1.0)

        # 2. 同 GPU + 后端 + 精度，不同 TP → 插值
        interpolated = self._interpolate_tp(gpu_type, backend, precision, tp)
        if interpolated:
            return ProfileLookupResult(interpolated, source="interpolated", confidence=0.7)

        # 3. 同代 GPU 缩放（简化：取同后端+精度+tp 的任意 GPU profile，按算力比缩放）
        scaled = self._scale_gpu(gpu_type, backend, precision, tp)
        if scaled:
            return ProfileLookupResult(scaled, source="scaled", confidence=0.5)

        # 4. 保守默认
        default = self._default_profile(gpu_type, backend, precision, tp)
        return ProfileLookupResult(default, source="default", confidence=0.3)

    def _exact_match(
        self, gpu_type: str, backend: str, precision: str, tp: int
    ) -> RuntimeProfile | None:
        for p in self.profiles:
            if (
                p.gpu_type == gpu_type
                and p.backend == backend
                and p.precision == precision
                and p.tp == tp
            ):
                return p
        return None

    def _interpolate_tp(
        self, gpu_type: str, backend: str, precision: str, tp: int
    ) -> RuntimeProfile | None:
        """同 GPU/backend/precision 下，不同 TP 的 profile 进行线性插值."""
        same_config = [
            p
            for p in self.profiles
            if p.gpu_type == gpu_type and p.backend == backend and p.precision == precision
        ]
        if len(same_config) < 2:
            return None

        # 找到最近的上下界
        sorted_profiles = sorted(same_config, key=lambda p: p.tp)
        lower = None
        upper = None
        for p in sorted_profiles:
            if p.tp <= tp:
                lower = p
            if p.tp >= tp and upper is None:
                upper = p

        if lower is None or upper is None or lower.tp == upper.tp:
            # 只有一侧，用最近的
            nearest = lower or upper
            if nearest is None:
                return None
            return self._scale_tp(nearest, tp)

        # 线性插值
        ratio = (tp - lower.tp) / (upper.tp - lower.tp)
        return RuntimeProfile(
            gpu_type=gpu_type,
            backend=backend,
            precision=precision,
            tp=tp,
            maximum_prefill_tokens_per_second=(
                lower.maximum_prefill_tokens_per_second
                + ratio * (upper.maximum_prefill_tokens_per_second - lower.maximum_prefill_tokens_per_second)
            ),
            maximum_decode_tokens_per_second=(
                lower.maximum_decode_tokens_per_second
                + ratio * (upper.maximum_decode_tokens_per_second - lower.maximum_decode_tokens_per_second)
            ),
            base_ttft_ms=(
                lower.base_ttft_ms + ratio * (upper.base_ttft_ms - lower.base_ttft_ms)
            ),
            base_itl_ms=(
                lower.base_itl_ms + ratio * (upper.base_itl_ms - lower.base_itl_ms)
            ),
            runtime_memory_overhead_gb=(
                lower.runtime_memory_overhead_gb
                + ratio * (upper.runtime_memory_overhead_gb - lower.runtime_memory_overhead_gb)
            ),
            communication_penalty=lower.communication_penalty,
        )

    def _scale_tp(self, base: RuntimeProfile, target_tp: int) -> RuntimeProfile:
        """根据 TP 变化线性缩放."""
        tp_ratio = target_tp / base.tp
        return RuntimeProfile(
            gpu_type=base.gpu_type,
            backend=base.backend,
            precision=base.precision,
            tp=target_tp,
            # TP 增大 → prefill 吞吐近似线性增长
            maximum_prefill_tokens_per_second=base.maximum_prefill_tokens_per_second * tp_ratio,
            # TP 增大 → decode 吞吐增长但受通信开销限制，用 0.8 系数
            maximum_decode_tokens_per_second=base.maximum_decode_tokens_per_second * (tp_ratio ** 0.8),
            # TP 增大 → 延迟略有增加（通信开销）
            base_ttft_ms=base.base_ttft_ms * (1 + 0.05 * (target_tp - base.tp)),
            base_itl_ms=base.base_itl_ms * (1 + 0.03 * (target_tp - base.tp)),
            runtime_memory_overhead_gb=base.runtime_memory_overhead_gb,
            communication_penalty=base.communication_penalty,
        )

    def _scale_gpu(
        self, gpu_type: str, backend: str, precision: str, tp: int
    ) -> RuntimeProfile | None:
        """跨 GPU 类型缩放（简化版：找到同 backend+precision+tp 的 profile）."""
        candidates = [
            p
            for p in self.profiles
            if p.backend == backend and p.precision == precision and p.tp == tp
        ]
        if not candidates:
            # 放宽 tp 约束
            candidates = [
                p
                for p in self.profiles
                if p.backend == backend and p.precision == precision
            ]
        if not candidates:
            return None

        # 取第一个，用保守系数 0.8 缩放
        base = candidates[0]
        return RuntimeProfile(
            gpu_type=gpu_type,
            backend=backend,
            precision=precision,
            tp=tp,
            maximum_prefill_tokens_per_second=base.maximum_prefill_tokens_per_second * 0.8,
            maximum_decode_tokens_per_second=base.maximum_decode_tokens_per_second * 0.8,
            base_ttft_ms=base.base_ttft_ms * 1.2,
            base_itl_ms=base.base_itl_ms * 1.2,
            runtime_memory_overhead_gb=base.runtime_memory_overhead_gb,
            communication_penalty=base.communication_penalty,
        )

    def _default_profile(
        self, gpu_type: str, backend: str, precision: str, tp: int
    ) -> RuntimeProfile:
        """完全无数据时的保守默认值."""
        return RuntimeProfile(
            gpu_type=gpu_type,
            backend=backend,
            precision=precision,
            tp=tp,
            maximum_prefill_tokens_per_second=5000.0,  # 保守估计
            maximum_decode_tokens_per_second=1000.0,
            base_ttft_ms=300.0,
            base_itl_ms=30.0,
            runtime_memory_overhead_gb=8.0,
            communication_penalty={"nvlink": 1.0, "pcie": 1.5},
        )
