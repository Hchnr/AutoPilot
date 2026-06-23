"""AutoPilot CLI - 大模型 GPU 推理部署自动优化系统."""

import click

from autopilot.orchestrator import plan_workflow, reconcile_workflow


@click.group()
def main():
    """AutoPilot: 大模型 GPU 推理部署自动优化系统."""
    pass


@main.command()
@click.option("--model", required=True, type=click.Path(exists=True), help="模型 spec YAML")
@click.option("--cluster", required=True, type=click.Path(exists=True), help="集群配置 YAML")
@click.option("--backends", required=True, type=click.Path(exists=True), help="推理后端配置 YAML")
@click.option("--traffic", required=True, type=click.Path(exists=True), help="流量数据 JSONL")
@click.option("--profiles", required=True, type=click.Path(exists=True), help="运行画像 YAML")
@click.option("--slo", required=True, type=click.Path(exists=True), help="SLO 约束 YAML")
@click.option("--output", required=True, type=click.Path(), help="输出目录")
def plan(model, cluster, backends, traffic, profiles, slo, output):
    """生成最优推理部署方案."""
    plan_workflow(
        model_path=model,
        cluster_path=cluster,
        backends_path=backends,
        traffic_path=traffic,
        profiles_path=profiles,
        slo_path=slo,
        output_dir=output,
    )


@main.command()
@click.option("--plan", "plan_path", required=True, type=click.Path(exists=True), help="当前部署方案")
@click.option("--telemetry", required=True, type=click.Path(exists=True), help="线上指标 JSONL")
@click.option("--output", required=True, type=click.Path(), help="输出目录")
def reconcile(plan_path, telemetry, output):
    """基于线上指标进行闭环调整."""
    reconcile_workflow(
        plan_path=plan_path,
        telemetry_path=telemetry,
        output_dir=output,
    )


if __name__ == "__main__":
    main()
