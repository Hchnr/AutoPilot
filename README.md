# AutoPilot

大模型 GPU 推理部署自动优化系统 — 基于工程公式和约束求解，自动生成最优推理部署配置方案，并在运行时闭环调整。

## 设计思路

传统的大模型推理部署面临大量手动调参工作：TP/PP 切分策略、精度选择、Batch 大小、KV Cache 管理等需要考虑模型架构、硬件拓扑、流量特征的交叉影响。AutoPilot 将这些领域知识编码为公式和规则，自动完成搜索空间探索和方案评估。

核心架构：
```
输入 (模型/集群/流量/画像/SLO)
  └→ Workload 分析器 → 流量特征摘要
  └→ 候选方案生成器 → 搜索空间穷举 + 显存剪枝
  └→ Profile 查询 + 延迟估算
  └→ 多目标评分 → 排序输出推荐方案

线上闭环:
  Telemetry → 趋势分析 → 决策引擎 → 防震荡守卫 → 操作建议
```

## 快速开始

```bash
pip install -e ".[dev]"
```

### Plan — 生成部署方案

```bash
autopilot plan \
  --model examples/scenario_1_customer_service/model.yaml \
  --cluster examples/scenario_1_customer_service/cluster.yaml \
  --backends examples/scenario_1_customer_service/backends.yaml \
  --traffic examples/scenario_1_customer_service/traffic.jsonl \
  --profiles examples/scenario_1_customer_service/runtime_profiles.yaml \
  --slo examples/scenario_1_customer_service/slo.yaml \
  --output outputs/plan/
```

输出:
- `deployment_plan.yaml` — 推荐部署配置
- `alternatives.json` — 备选方案
- `decision_report.md` — 决策报告（含完整推理过程）

### Reconcile — 闭环调整

```bash
autopilot reconcile \
  --plan outputs/plan/deployment_plan.yaml \
  --telemetry examples/scenario_1_customer_service/telemetry.jsonl \
  --output outputs/reconcile/
```

输出:
- `actions.json` — 建议操作
- `decision_log.md` — 决策日志

## 项目结构

```
src/autopilot/
├── cli.py                  # CLI 入口
├── orchestrator.py         # 流程编排
├── loader.py               # 数据加载
├── models/                 # Pydantic 数据模型
│   ├── model_spec.py       # 模型架构定义
│   ├── cluster.py          # 集群资源定义
│   ├── backend.py          # 推理后端能力定义
│   ├── traffic.py          # 流量数据结构
│   ├── profile.py          # 运行画像
│   ├── slo.py              # SLO 约束
│   ├── plan.py             # 部署方案
│   └── telemetry.py        # Telemetry 数据
├── analyzer/
│   └── workload.py         # Workload 分析器
├── estimator/
│   ├── memory.py           # 显存估算器
│   └── profile_lookup.py   # Profile 查询引擎
├── planner/
│   ├── candidate_generator.py  # 候选方案生成 + 剪枝
│   └── scorer.py           # 多目标评分排序
├── reconciler/
│   ├── analyzer.py         # Telemetry 趋势分析
│   ├── decision_engine.py  # 调整决策引擎
│   └── safeguards.py       # 防震荡安全机制
└── reporter/
    ├── plan_reporter.py    # Plan 报告生成
    └── reconcile_reporter.py  # Reconcile 报告生成
```

## 核心公式

### 显存估算

```
模型权重 = param_count × bytes_per_param / TP / PP
KV Cache = 2 × layers × kv_heads × head_dim × kv_dtype_bytes / TP × max_seqs × context_len
总显存 = (权重 + KV Cache + 运行时开销) × 1.10
```

### 评分函数

```
score = w_cost × cost_score + w_headroom × headroom_score + w_latency × latency_score + w_conf × confidence
```

权重根据优化目标（minimize_cost vs maximize_goodput）动态调整。

## 示例场景

| 场景 | 特征 | 关键决策 |
|------|------|----------|
| 客服 Chat | 高 prefix 复用、TTFT 敏感 | 启用 Prefix Cache + Chunked Prefill |
| 长文本生成 | 输出长、ITL 敏感 | 关闭 Prefix Cache、优化 decode 吞吐 |
| 混合流量 | Chat + RAG + 长文本 | FP8 降低成本、平衡 prefill/decode |

## 测试

```bash
pytest tests/ -v --cov=autopilot
```

## 技术栈

- Python 3.11+
- Pydantic v2 — 数据模型与验证
- Click — CLI 框架
- PyYAML — 配置解析
- pytest — 测试框架
