"""防震荡安全机制."""

from autopilot.models.telemetry import ReconcileAction


# 高风险操作：需要重启的操作
HIGH_RISK_ACTIONS = {
    "change_tp",
    "change_pp",
    "change_gpu_pool",
    "change_precision",
    "change_kv_cache_dtype",
}


class SafeguardEngine:
    """防震荡安全守卫.

    机制:
    - Cooldown: 上次操作后的冷却期
    - Hysteresis: 扩容/缩容使用不同阈值
    - 高风险标记: 涉及重启的操作需要人工确认
    - 单次只允许一个操作
    """

    def __init__(
        self,
        cooldown_windows: int = 3,
        last_action_window: int | None = None,
        current_window: int = 0,
    ):
        self.cooldown_windows = cooldown_windows
        self.last_action_window = last_action_window
        self.current_window = current_window
        self._action_count = 0

    def is_allowed(self, action: ReconcileAction) -> bool:
        """检查操作是否被允许.

        Rules:
        1. Cooldown 期间不允许新操作
        2. 单次只允许一个操作
        3. 标记高风险操作
        """
        # Cooldown 检查
        if self.last_action_window is not None:
            if (self.current_window - self.last_action_window) < self.cooldown_windows:
                return False

        # 单次只允许一个操作
        if self._action_count >= 1:
            return False

        # 标记高风险
        if action.action in HIGH_RISK_ACTIONS:
            action.risk_level = "high"
            action.requires_restart = True

        self._action_count += 1
        return True

    @staticmethod
    def classify_risk(action: ReconcileAction) -> ReconcileAction:
        """对操作进行风险分类."""
        if action.action in HIGH_RISK_ACTIONS:
            action.risk_level = "high"
            action.requires_restart = True
        elif action.action in {"scale_replicas"}:
            action.risk_level = "low"
            action.requires_restart = False
        else:
            action.risk_level = "medium"
            action.requires_restart = False
        return action
