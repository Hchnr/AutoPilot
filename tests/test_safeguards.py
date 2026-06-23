"""防震荡安全机制测试."""

import pytest

from autopilot.models.telemetry import ReconcileAction
from autopilot.reconciler.safeguards import SafeguardEngine


def _make_action(action: str = "scale_replicas", **kwargs) -> ReconcileAction:
    defaults = dict(
        action=action,
        field="replicas",
        from_value=2,
        to_value=3,
        reason="test",
        confidence=0.85,
        risk_level="low",
        requires_restart=False,
    )
    defaults.update(kwargs)
    return ReconcileAction(**defaults)


class TestSafeguards:
    """防震荡安全机制测试集."""

    def test_cooldown_blocks_action(self):
        """验证: cooldown 期间阻止新操作."""
        engine = SafeguardEngine(
            cooldown_windows=3,
            last_action_window=8,
            current_window=10,  # 距上次操作只有 2 个窗口
        )
        action = _make_action()
        assert engine.is_allowed(action) is False

    def test_cooldown_allows_after_period(self):
        """验证: cooldown 过后允许操作."""
        engine = SafeguardEngine(
            cooldown_windows=3,
            last_action_window=5,
            current_window=10,  # 距上次操作 5 个窗口
        )
        action = _make_action()
        assert engine.is_allowed(action) is True

    def test_no_previous_action_allowed(self):
        """验证: 无之前操作时允许."""
        engine = SafeguardEngine(
            cooldown_windows=3,
            last_action_window=None,
            current_window=0,
        )
        action = _make_action()
        assert engine.is_allowed(action) is True

    def test_single_action_per_cycle(self):
        """验证: 每个周期只允许一个操作."""
        engine = SafeguardEngine(cooldown_windows=3, last_action_window=None, current_window=10)
        action1 = _make_action(action="scale_replicas")
        action2 = _make_action(action="adjust_max_num_seqs")
        assert engine.is_allowed(action1) is True
        assert engine.is_allowed(action2) is False

    def test_high_risk_flagged(self):
        """验证: 高风险操作被标记."""
        engine = SafeguardEngine(cooldown_windows=3, last_action_window=None, current_window=10)
        action = _make_action(action="change_tp")
        engine.is_allowed(action)
        assert action.risk_level == "high"
        assert action.requires_restart is True

    def test_scale_replicas_low_risk(self):
        """验证: scale_replicas 是低风险."""
        engine = SafeguardEngine(cooldown_windows=3, last_action_window=None, current_window=10)
        action = _make_action(action="scale_replicas")
        engine.is_allowed(action)
        assert action.risk_level == "low"

    def test_classify_risk_change_precision(self):
        """验证: 修改 precision 是高风险."""
        action = _make_action(action="change_precision")
        SafeguardEngine.classify_risk(action)
        assert action.risk_level == "high"
        assert action.requires_restart is True

    def test_classify_risk_adjust_batch(self):
        """验证: 调整 batch 参数是中等风险."""
        action = _make_action(action="adjust_max_num_seqs")
        SafeguardEngine.classify_risk(action)
        assert action.risk_level == "medium"
        assert action.requires_restart is False

    def test_classify_risk_change_gpu_pool(self):
        """验证: 切换 GPU 池是高风险."""
        action = _make_action(action="change_gpu_pool")
        SafeguardEngine.classify_risk(action)
        assert action.risk_level == "high"
        assert action.requires_restart is True

    def test_classify_risk_change_tp(self):
        """验证: 修改 TP 是高风险."""
        action = _make_action(action="change_tp")
        SafeguardEngine.classify_risk(action)
        assert action.risk_level == "high"
        assert action.requires_restart is True

    def test_classify_risk_change_kv_cache(self):
        """验证: 修改 KV cache 精度是高风险（需要重启）."""
        action = _make_action(action="change_kv_cache_dtype")
        SafeguardEngine.classify_risk(action)
        assert action.risk_level == "high"
        assert action.requires_restart is True
