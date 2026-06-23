"""数据加载工具."""

import json
from pathlib import Path

import yaml

from autopilot.models import (
    BackendsConfig,
    ClusterSpec,
    DeploymentPlan,
    ModelSpec,
    ProfilesConfig,
    SloConfig,
    TelemetryRecord,
    TrafficRecord,
)


def load_yaml(path: str | Path) -> dict:
    """加载 YAML 文件."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str | Path) -> list[dict]:
    """加载 JSONL 文件."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_model_spec(path: str | Path) -> ModelSpec:
    """加载模型 spec."""
    data = load_yaml(path)
    return ModelSpec(**data)


def load_cluster(path: str | Path) -> ClusterSpec:
    """加载集群配置."""
    data = load_yaml(path)
    return ClusterSpec(**data)


def load_backends(path: str | Path) -> BackendsConfig:
    """加载后端配置."""
    data = load_yaml(path)
    return BackendsConfig.from_dict(data)


def load_traffic(path: str | Path) -> list[TrafficRecord]:
    """加载流量数据."""
    records = load_jsonl(path)
    return [TrafficRecord(**r) for r in records]


def load_profiles(path: str | Path) -> ProfilesConfig:
    """加载运行画像."""
    data = load_yaml(path)
    return ProfilesConfig(**data)


def load_slo(path: str | Path) -> SloConfig:
    """加载 SLO 配置."""
    data = load_yaml(path)
    return SloConfig(**data)


def load_telemetry(path: str | Path) -> list[TelemetryRecord]:
    """加载 Telemetry 数据."""
    records = load_jsonl(path)
    return [TelemetryRecord(**r) for r in records]


def load_deployment_plan(path: str | Path) -> DeploymentPlan:
    """加载已有部署方案."""
    data = load_yaml(path)
    return DeploymentPlan(**data)
