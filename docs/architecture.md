# 系统架构

## 概览

AutoPilot 由一系列可组合的模块组成，每个模块职责单一：

```
src/autopilot/
├── cli.py                     # Click CLI — 解析参数，调用 orchestrator
├── orchestrator.py            # 串联各模块完成 plan/reconcile 流程
├── loader.py                  # 将 YAML/JSONL 反序列化为 Pydantic 模型
├── models/                    # Pydantic v2 数据契约
│   ├── model_spec.py          # 大模型架构定义
│   ├── cluster.py             # GPU 资源池规格
│   ├── backend.py             # 推理后端能力定义
│   ├── traffic.py             # 请求级流量记录 + WorkloadSummary
│   ├── profile.py             # 运行画像
│   ├── slo.py                 # SLO 目标与约束
│   ├── plan.py                # 部署方案 + PlanResult
│   └── telemetry.py           # 运行时指标 + ReconcileAction/Result
├── analyzer/
│   └── workload.py            # 流量 → WorkloadSummary
├── estimator/
│   ├── memory.py              # 架构 → GPU 显存需求
│   └── profile_lookup.py      # 配置 → 延迟估算（含多级降级）
├── planner/
│   ├── candidate_generator.py # 搜索空间穷举 + 剪枝
│   └── scorer.py              # 多目标评分排序
├── reconciler/
│   ├── analyzer.py            # Telemetry → 趋势摘要
│   ├── decision_engine.py     # 趋势 → 调整操作
│   └── safeguards.py          # 防震荡控制
└── reporter/
    ├── plan_reporter.py       # PlanResult → decision_report.md
    └── reconcile_reporter.py  # ReconcileResult → decision_log.md
```

## 数据流 — Plan

```
model.yaml ─────────┐
cluster.yaml ───────┤
backends.yaml ──────┤──→ loader.py ──→ 类型化模型
traffic.jsonl ──────┤                      │
profiles.yaml ──────┤                      ▼
slo.yaml ───────────┘              orchestrator.plan_workflow()
                                           │
                    ┌──────────────────────┼───────────────────────┐
                    ▼                      ▼                       ▼
           workload.analyze()    memory.MemoryEstimator    profile_lookup.ProfileLookup
                    │                      │                       │
                    └──────────────────────┼───────────────────────┘
                                           ▼
                              candidate_generator.generate_candidates()
                                           │
                                           ▼
                              scorer.score_and_rank()
                                           │
                                           ▼
                              PlanResult（推荐方案 + 备选方案）
                                           │
                                           ▼
                              reporter → deployment_plan.yaml
                                        → alternatives.json
                                        → decision_report.md
```

## 数据流 — Reconcile

```
deployment_plan.yaml ──┐
telemetry.jsonl ───────┘──→ loader.py ──→ 类型化模型
                                              │
                                              ▼
                                   orchestrator.reconcile_workflow()
                                              │
                              ┌────────────────┼────────────────┐
                              ▼                                 ▼
                   analyzer.analyze_telemetry()      decision_engine.decide_actions()
                              │                                 │
                              └────────────────┬────────────────┘
                                               ▼
                                    safeguards.is_allowed()
                                               │
                                               ▼
                                    ReconcileResult（操作列表）
                                               │
                                               ▼
                                    reporter → actions.json
                                             → decision_log.md
```

## 设计原则

1. **类型化契约** — 模块间所有数据通过 Pydantic 模型传递，带验证规则
2. **确定性** — 相同输入始终产生相同输出（评分无随机性）
3. **显式置信度** — 每个估算值携带置信度分数（0-1），标明数据质量
4. **安全降级** — Profile 数据缺失时使用保守估算并标注低置信度
5. **关注点分离** — Plan 阶段纯离线；Reconcile 阶段纯反应式；共享模型但不共享状态
