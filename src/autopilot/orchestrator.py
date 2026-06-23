"""流程编排器 - 串联 plan 和 reconcile 的完整流程."""

import json
from pathlib import Path

import yaml

from autopilot.analyzer.workload import analyze_workload
from autopilot.estimator.memory import MemoryEstimator
from autopilot.estimator.profile_lookup import ProfileLookup
from autopilot.loader import (
    load_backends,
    load_cluster,
    load_deployment_plan,
    load_model_spec,
    load_profiles,
    load_slo,
    load_telemetry,
    load_traffic,
)
from autopilot.models import PlanResult, ReconcileResult
from autopilot.planner.candidate_generator import generate_candidates
from autopilot.planner.scorer import score_and_rank
from autopilot.reconciler.analyzer import analyze_telemetry
from autopilot.reconciler.decision_engine import decide_actions
from autopilot.reporter.plan_reporter import generate_plan_report
from autopilot.reporter.reconcile_reporter import generate_reconcile_report


def plan_workflow(
    model_path: str,
    cluster_path: str,
    backends_path: str,
    traffic_path: str,
    profiles_path: str,
    slo_path: str,
    output_dir: str,
) -> PlanResult:
    """执行完整的 Plan 流程."""
    # 1. 加载输入
    model = load_model_spec(model_path)
    cluster = load_cluster(cluster_path)
    backends = load_backends(backends_path)
    traffic = load_traffic(traffic_path)
    profiles = load_profiles(profiles_path)
    slo = load_slo(slo_path)

    # 2. Workload 分析
    workload = analyze_workload(traffic)

    # 3. 初始化估算器
    memory_estimator = MemoryEstimator(model=model)
    profile_lookup = ProfileLookup(profiles=profiles)

    # 4. 生成候选方案
    candidates = generate_candidates(
        model=model,
        cluster=cluster,
        backends=backends,
        workload=workload,
        memory_estimator=memory_estimator,
        slo=slo,
    )

    # 5. 评分排序
    ranked = score_and_rank(
        candidates=candidates,
        workload=workload,
        slo=slo,
        profile_lookup=profile_lookup,
        cluster=cluster,
    )

    # 6. 构造结果
    if not ranked:
        raise RuntimeError("无法生成任何可行部署方案，请检查约束条件")

    recommended = ranked[0]
    alternatives = ranked[1:4]  # top 2-3 备选

    result = PlanResult(
        recommended=recommended,
        alternatives=alternatives,
        workload_summary=workload.model_dump(),
        memory_details={
            "model_weight_per_gpu_gb": memory_estimator.model_weight_memory(
                precision=recommended.precision, tp=recommended.tensor_parallel
            ),
            "kv_cache_per_token_bytes": memory_estimator.kv_cache_per_token(
                kv_dtype=recommended.kv_cache_dtype, tp=recommended.tensor_parallel
            ),
        },
    )

    # 7. 输出
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # deployment_plan.yaml
    plan_data = recommended.model_dump()
    with open(output / "deployment_plan.yaml", "w") as f:
        yaml.dump(plan_data, f, default_flow_style=False, allow_unicode=True)

    # alternatives.json
    alt_data = [a.model_dump() for a in alternatives]
    with open(output / "alternatives.json", "w") as f:
        json.dump(alt_data, f, indent=2, ensure_ascii=False)

    # decision_report.md
    report = generate_plan_report(result, workload, slo, model)
    with open(output / "decision_report.md", "w") as f:
        f.write(report)

    return result


def reconcile_workflow(
    plan_path: str,
    telemetry_path: str,
    output_dir: str,
) -> ReconcileResult:
    """执行完整的 Reconcile 流程."""
    # 1. 加载输入
    current_plan = load_deployment_plan(plan_path)
    telemetry = load_telemetry(telemetry_path)

    # 2. 分析 Telemetry
    analysis = analyze_telemetry(telemetry)

    # 3. 决策
    actions = decide_actions(
        plan=current_plan,
        telemetry=telemetry,
        analysis=analysis,
    )

    result = ReconcileResult(
        actions=actions,
        analysis_summary=analysis,
        current_plan=current_plan.model_dump(),
    )

    # 4. 输出
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # actions.json
    actions_data = [a.model_dump() for a in actions]
    with open(output / "actions.json", "w") as f:
        json.dump(actions_data, f, indent=2, ensure_ascii=False)

    # decision_log.md
    report = generate_reconcile_report(result)
    with open(output / "decision_log.md", "w") as f:
        f.write(report)

    return result
