# 测试策略与覆盖说明

本文档说明测试如何覆盖系统的各个关键判断逻辑，以及每个测试验证的具体行为。

## 运行测试

```bash
pytest tests/ -v --cov=autopilot
# 71 个测试，93% 覆盖率
```

## 覆盖矩阵

| 测试需求 | 测试文件 | 关键用例 |
|----------|----------|----------|
| 配置合法性 | test_candidate_generator | TP 整除 KV Head、PP 整除层数 |
| GPU 数量约束 | test_candidate_generator | 总 GPU 不超限、TP×PP×Replica 计算 |
| 显存估算 | test_memory_estimator | 权重/KV/总量计算、可行性判断 |
| TP、PP 和 Replica 资源计算 | test_candidate_generator + test_memory_estimator | TP 切分权重、PP 切分层数、Replica 随流量扩展 |
| SLO 判断 | test_scorer | 硬约束过滤、延迟 margin 计算 |
| 方案排序 | test_scorer | 成本排序、不同目标权重差异 |
| Profile 缺失降级 | test_profile_lookup | 精确/插值/缩放/默认四级 |
| 连续窗口判断 | test_reconciler | 精确边界（2 窗口不触发、3 窗口触发） |
| Cooldown 和防震荡 | test_safeguards | 冷却期阻断、单次单操作 |
| 高风险操作识别 | test_safeguards | 各类操作风险分级 |

---

## 1. 配置合法性

**文件**: `tests/test_candidate_generator.py`

| 用例 | 验证内容 |
|------|----------|
| `test_tp_divides_kv_heads` | 所有候选方案的 TP 值必须整除模型的 num_kv_heads |
| `test_pp_divides_layers` | 所有候选方案的 PP 值必须整除模型的 num_layers |
| `test_infeasible_memory_excluded` | bf16 TP=1 在 48GB GPU 上不可行的配置被排除 |
| `test_prefix_cache_influenced_by_workload` | Prefix Cache 启用条件依赖覆盖率和复用率双重检查 |

**验证思路**: 对生成器输出的所有候选方案逐一断言硬约束，确保非法配置不会泄漏到评分阶段。

---

## 2. GPU 数量约束

**文件**: `tests/test_candidate_generator.py`

| 用例 | 验证内容 |
|------|----------|
| `test_gpu_count_constraint` | TP × PP × Replicas ≤ SLO 定义的 maximum_gpu_count |
| `test_total_gpu_count` | TP × PP × Replicas ≤ 资源池可用 GPU 数 |
| `test_replicas_scale_with_load` | 高流量场景生成更多副本 |

**验证思路**: 同时验证 SLO 层面和物理资源层面的 GPU 约束。

---

## 3. 显存估算

**文件**: `tests/test_memory_estimator.py`

| 用例 | 验证内容 |
|------|----------|
| `test_model_weight_bf16_tp4` | Qwen3-32B bf16 TP=4 → ~14.9 GB（数值验证）|
| `test_model_weight_fp8_tp4` | FP8 权重为 BF16 的一半 |
| `test_model_weight_tp1_vs_tp4` | TP=1 是 TP=4 的 4 倍 |
| `test_model_weight_pp2` | PP=2 减半单卡显存 |
| `test_kv_cache_per_token` | 按公式验证单 token KV 字节数（65536 bytes）|
| `test_kv_cache_fp8_halves` | FP8 KV 为 auto(bf16) 的一半 |
| `test_total_memory` | 总显存在合理范围内 |
| `test_feasible_tp4_h800` | TP=4 H800 80GB 可行 |
| `test_infeasible_tp1_bf16` | TP=1 bf16 在 48GB 上不可行 |
| `test_concurrency_affects_memory` | 增加并发 → 增加显存 |
| `test_context_length_affects_memory` | 增加上下文 → 增加显存 |

**验证思路**: 对公式的每个因子独立测试（精度、TP、PP、并发、上下文），确保乘除关系正确。

---

## 4. TP、PP 和 Replica 资源计算

**文件**: `tests/test_memory_estimator.py` + `tests/test_candidate_generator.py`

| 用例 | 验证内容 |
|------|----------|
| `test_model_weight_tp1_vs_tp4` | TP 切分线性降低单卡权重 |
| `test_model_weight_pp2` | PP 切分线性降低单卡权重 |
| `test_total_gpu_count` | 总 GPU = TP × PP × Replicas |
| `test_replicas_scale_with_load` | 副本数随流量需求增加 |

**验证思路**: TP 和 PP 影响单卡显存，Replica 影响总资源和吞吐。三者的乘积关系通过组合验证。

---

## 5. SLO 判断

**文件**: `tests/test_scorer.py`

| 用例 | 验证内容 |
|------|----------|
| `test_slo_hard_constraint` | 极严格 SLO 下评分器不崩溃 |
| `test_slo_filters_violating_plans` | 不满足 TTFT/ITL 约束的方案被过滤 |
| `test_cost_ranking` | 满足 SLO 的前提下按成本排序 |

**验证思路**: 使用极严格 SLO（50ms TTFT, 5ms ITL）验证方案确实被过滤；使用宽松 SLO 验证排序逻辑。

---

## 6. 方案排序

**文件**: `tests/test_scorer.py`

| 用例 | 验证内容 |
|------|----------|
| `test_cost_ranking` | minimize_cost 目标下低成本方案排前 |
| `test_goodput_ranking_differs_from_cost` | 不同优化目标产生不同的分数差距 |
| `test_at_least_scored` | 方案有分数且大于 0 |
| `test_rationale_generated` | 方案包含 tp、precision 等决策理由 |

**验证思路**: 通过构造同配置不同副本数的方案对，验证排序受目标函数权重影响。

---

## 7. Profile 缺失时的降级逻辑

**文件**: `tests/test_profile_lookup.py`

| 用例 | 验证内容 |
|------|----------|
| `test_exact_match` | 精确匹配 → confidence=1.0, source="exact" |
| `test_interpolation` | 同 GPU 不同 TP → 线性插值，值在两端之间 |
| `test_missing_falls_to_default` | 完全未知配置 → 保守默认值, confidence≤0.3 |
| `test_low_confidence_on_scaled` | 跨 GPU 缩放 → confidence<0.8 |
| `test_empty_profiles` | 空 profile 列表 → 不崩溃，返回默认 |

**验证思路**: 逐级验证降级路径，确保置信度随数据可用性递减且估算值保守。

---

## 8. Reconcile 的连续窗口判断

**文件**: `tests/test_reconciler.py`

| 用例 | 验证内容 |
|------|----------|
| `test_single_spike_no_action` | 1 个窗口违反 → 不触发 |
| `test_exactly_two_windows_no_action` | 恰好 2 个窗口违反 → 不触发（阈值为 3）|
| `test_exactly_three_windows_triggers` | 恰好 3 个窗口违反 → 触发操作 |
| `test_slo_violation_triggers_action` | 5 个连续窗口违反 → 触发 |
| `test_four_windows_low_util_no_scale_down` | 4 个窗口低利用率 → 不触发缩容（阈值为 5）|
| `test_five_windows_low_util_triggers_scale_down` | 5 个窗口低利用率 → 触发缩容 |
| `test_interrupted_violation_resets_counter` | 中间正常窗口重置连续计数 |

**验证思路**: 在精确边界上测试（N-1 不触发、N 触发），确认连续性检查正确。

---

## 9. Cooldown 和防震荡逻辑

**文件**: `tests/test_safeguards.py`

| 用例 | 验证内容 |
|------|----------|
| `test_cooldown_blocks_action` | 冷却期内（距上次操作 <3 窗口）阻止新操作 |
| `test_cooldown_allows_after_period` | 冷却期后（距上次操作 ≥3 窗口）允许操作 |
| `test_no_previous_action_allowed` | 无历史操作时允许操作 |
| `test_single_action_per_cycle` | 同一周期第二个操作被拒绝 |

**验证思路**: 模拟不同时间点的操作请求，验证 cooldown 和单次单操作约束。

---

## 10. 高风险操作识别

**文件**: `tests/test_safeguards.py`

| 用例 | 验证内容 |
|------|----------|
| `test_high_risk_flagged` | change_tp 被标记为高风险 + requires_restart |
| `test_scale_replicas_low_risk` | scale_replicas 是低风险 |
| `test_classify_risk_change_precision` | change_precision → high, restart=True |
| `test_classify_risk_change_gpu_pool` | change_gpu_pool → high, restart=True |
| `test_classify_risk_change_tp` | change_tp → high, restart=True |
| `test_classify_risk_change_kv_cache` | change_kv_cache_dtype → high, restart=True |
| `test_classify_risk_adjust_batch` | adjust_max_num_seqs → medium, restart=False |

**验证思路**: 逐类型验证风险分级和重启标记，确保高风险操作不会被静默执行。

---

## 端到端场景测试

**文件**: `tests/test_e2e.py`

| 场景 | 验证内容 |
|------|----------|
| 场景一（客服 Chat）| Plan 启用 prefix cache；Reconcile 检测 SLO 违反 |
| 场景二（长文本）| Plan 关闭 prefix cache；Reconcile 无操作（稳定）|
| 场景三（混合流量）| Plan 输出成本估算和备选方案 |

端到端测试确保各模块串联后整体行为符合预期，不会因模块接口不匹配导致错误。
