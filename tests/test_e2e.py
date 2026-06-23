"""端到端集成测试."""

import json
from pathlib import Path

import pytest
import yaml

from autopilot.orchestrator import plan_workflow, reconcile_workflow


EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs" / "test_e2e"


@pytest.fixture(autouse=True)
def clean_outputs():
    """测试前清理输出目录."""
    import shutil

    if OUTPUTS_DIR.exists():
        shutil.rmtree(OUTPUTS_DIR)
    yield
    # 不清理方便查看


class TestEndToEndScenario1:
    """场景一: 高 Prefix 复用客服 Chat."""

    SCENARIO = EXAMPLES_DIR / "scenario_1_customer_service"

    def test_plan_generates_output(self):
        """验证: plan 生成完整输出."""
        output = OUTPUTS_DIR / "scenario_1" / "plan"
        plan_workflow(
            model_path=str(self.SCENARIO / "model.yaml"),
            cluster_path=str(self.SCENARIO / "cluster.yaml"),
            backends_path=str(self.SCENARIO / "backends.yaml"),
            traffic_path=str(self.SCENARIO / "traffic.jsonl"),
            profiles_path=str(self.SCENARIO / "runtime_profiles.yaml"),
            slo_path=str(self.SCENARIO / "slo.yaml"),
            output_dir=str(output),
        )
        # 验证文件存在
        assert (output / "deployment_plan.yaml").exists()
        assert (output / "alternatives.json").exists()
        assert (output / "decision_report.md").exists()

        # 验证 plan 内容
        with open(output / "deployment_plan.yaml") as f:
            plan = yaml.safe_load(f)
        assert plan["enable_prefix_cache"] is True  # 高 prefix 复用
        assert plan["tensor_parallel"] >= 2  # 需要多卡

        # 验证 alternatives
        with open(output / "alternatives.json") as f:
            alts = json.load(f)
        assert len(alts) >= 1

        # 验证 report 包含关键章节
        report = (output / "decision_report.md").read_text()
        for section in [
            "Workload",
            "Resource",
            "Recommended",
            "Score",
            "Memory",
            "Capacity",
            "Cost",
            "Rationale",
            "Alternative",
            "Confidence",
            "Assumption",
            "Verification",
        ]:
            assert section.lower() in report.lower(), f"报告缺少章节: {section}"

    def test_reconcile_detects_slo_violation(self):
        """验证: reconcile 检测到 SLO 违反并建议扩容."""
        # 先生成 plan
        plan_output = OUTPUTS_DIR / "scenario_1" / "plan"
        plan_workflow(
            model_path=str(self.SCENARIO / "model.yaml"),
            cluster_path=str(self.SCENARIO / "cluster.yaml"),
            backends_path=str(self.SCENARIO / "backends.yaml"),
            traffic_path=str(self.SCENARIO / "traffic.jsonl"),
            profiles_path=str(self.SCENARIO / "runtime_profiles.yaml"),
            slo_path=str(self.SCENARIO / "slo.yaml"),
            output_dir=str(plan_output),
        )

        # 运行 reconcile
        recon_output = OUTPUTS_DIR / "scenario_1" / "reconcile"
        result = reconcile_workflow(
            plan_path=str(plan_output / "deployment_plan.yaml"),
            telemetry_path=str(self.SCENARIO / "telemetry.jsonl"),
            output_dir=str(recon_output),
        )

        # Telemetry 有 SLO 违反，应产生操作
        assert len(result.actions) > 0
        assert (recon_output / "actions.json").exists()
        assert (recon_output / "decision_log.md").exists()


class TestEndToEndScenario2:
    """场景二: 长文本生成."""

    SCENARIO = EXAMPLES_DIR / "scenario_2_long_generation"

    def test_plan_disables_prefix_cache(self):
        """验证: 低 prefix 复用时关闭 prefix cache."""
        output = OUTPUTS_DIR / "scenario_2" / "plan"
        result = plan_workflow(
            model_path=str(self.SCENARIO / "model.yaml"),
            cluster_path=str(self.SCENARIO / "cluster.yaml"),
            backends_path=str(self.SCENARIO / "backends.yaml"),
            traffic_path=str(self.SCENARIO / "traffic.jsonl"),
            profiles_path=str(self.SCENARIO / "runtime_profiles.yaml"),
            slo_path=str(self.SCENARIO / "slo.yaml"),
            output_dir=str(output),
        )
        assert result.recommended.enable_prefix_cache is False

    def test_reconcile_stable(self):
        """验证: 稳定运行时不产生操作."""
        plan_output = OUTPUTS_DIR / "scenario_2" / "plan"
        plan_workflow(
            model_path=str(self.SCENARIO / "model.yaml"),
            cluster_path=str(self.SCENARIO / "cluster.yaml"),
            backends_path=str(self.SCENARIO / "backends.yaml"),
            traffic_path=str(self.SCENARIO / "traffic.jsonl"),
            profiles_path=str(self.SCENARIO / "runtime_profiles.yaml"),
            slo_path=str(self.SCENARIO / "slo.yaml"),
            output_dir=str(plan_output),
        )
        recon_output = OUTPUTS_DIR / "scenario_2" / "reconcile"
        result = reconcile_workflow(
            plan_path=str(plan_output / "deployment_plan.yaml"),
            telemetry_path=str(self.SCENARIO / "telemetry.jsonl"),
            output_dir=str(recon_output),
        )
        assert len(result.actions) == 0


class TestEndToEndScenario3:
    """场景三: 成本敏感混合流量."""

    SCENARIO = EXAMPLES_DIR / "scenario_3_mixed_traffic"

    def test_plan_cost_sensitive(self):
        """验证: 成本敏感场景产出方案."""
        output = OUTPUTS_DIR / "scenario_3" / "plan"
        result = plan_workflow(
            model_path=str(self.SCENARIO / "model.yaml"),
            cluster_path=str(self.SCENARIO / "cluster.yaml"),
            backends_path=str(self.SCENARIO / "backends.yaml"),
            traffic_path=str(self.SCENARIO / "traffic.jsonl"),
            profiles_path=str(self.SCENARIO / "runtime_profiles.yaml"),
            slo_path=str(self.SCENARIO / "slo.yaml"),
            output_dir=str(output),
        )
        # 应有备选方案可供比较
        assert len(result.alternatives) >= 1
        # 成本估算应有值
        assert result.recommended.estimated_hourly_cost > 0
