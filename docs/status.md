# 当前状态与已知问题

此页区分代码能力、验证范围和待解决问题。不是生产承诺。默认值、版本与接口以[代码事实参考](reference.md)为准；测试结果以本次实际运行输出为准，不在各文档重复维护“最新通过数”。

## 已实现

- 默认研究模式与可关闭的基础模式；多源结构化规划、独立来源有限并发、局部成功保留。
- 四个真实 `@tool` 接入初搜、补搜和专题回读；输入约束、ToolMessage / artifact、调用编号和结果事件。
- 知识文件增量同步、可选轮询；有限新闻候选的向量缓存与正文获取。
- 版本化新闻原文、分段阅读成果、精确引用位置；严格任务键复用、FTS5 + 向量召回、可修复投影。
- 有限阅读/专题任务池，进程内模型与网络限流、租约、取消和过期结果提交防护。
- CLI 持久化检查点与研究运行管理；补搜、答案重写、双维评审、通过答案缓存。

## 已确认且尚未修复

| 问题 | 代码证据 | 影响/建议 |
|---|---|---|
| 阅读数字检查使用子串，未完整校验单位/口径 | [research/chains.py](../src/agentic_rag/research/chains.py) 的 `validate_reading`、`resolve_selection` | 原文有 12 不证明概括中的 2 正确；应核对完整数值、单位、对象及范围 |
| 重试未完整分类永久/暂时错误 | [graph/chains.py](../src/agentic_rag/graph/chains.py)、[scheduler.py](../src/agentic_rag/research/scheduler.py) | 401/404 等可能无效重试；需分层错误策略与总预算 |
| `RESEARCH_TIMEOUT` 每批重置 | [scheduler.py](../src/agentic_rag/research/scheduler.py) 的 `run` | 不是端到端截止时间，仍缺总时间/总调用预算 |
| 专题 summary 的 500 字仅是字段描述 | [research/chains.py](../src/agentic_rag/research/chains.py) 的 `SpecialistResult` | 不是 Pydantic 硬限制；专题建议与提示开销可能超出证据字符预算 |
| 阅读/专题记忆缺完整可信审核与撤回机制 | [service.py](../src/agentic_rag/research/service.py)、[store.py](../src/agentic_rag/research/store.py) | 已保存不等于事实已经独立核实；专题投影不等于最终评审通过 |
| 有限召回候选后才校验日期和当前版本 | [memory.py](../src/agentic_rag/research/memory.py) | 有效日期内文章可能没进入候选，不能保证高召回 |
| RAGAS 收集 `documents` 而非实际 `evidence_context`；题库域不匹配 | [评估脚本](../evaluation/run_evaluation.py)、[数据集](../evaluation/dataset.json) | 先修取样，再建立财经质量基准；旧分数不代表当前业务质量 |

以上是审阅发现，不是本轮文档任务顺便修复的功能。具体题目和原始反例见[面试题对照](Agent求职面试题与参考答案.html)。

## 适用边界

- 本地单用户展示项目，无网页前端、跨问题聊天记忆、多租户权限、完整安全沙箱或分布式队列。
- 工具为程序受控执行，不是自由 ReAct；公开网络只用搜索摘要，不能声称已读完整研报。
- 研究记忆只覆盖实际抓取/读过的新闻；API 条件召回与摘要向量排序可能漏掉关键文章。
- 限流是进程内的，GPU 是否真正并行取决于服务；外部模型计算不保证严格一次。
- 字符预算不是精确 token 预算；所有提示与专题建议的总长度还需治理。
- 知识索引重建不是原子切换；监听跨进程无写锁；嵌入距离等有库默认依赖。
- 数据库、原文版本和事件没有自动归档清理；模型同名换权重可通过摘要区分，但嵌入同路径换权重需版本化配置。
- 现有评审会误判。通过核验、流程完整阅读、测试全绿，都不能证明预测准确或报道真实。

## 验证证据怎样看

运行 `python -m pytest -q` 看本次回归；`python scripts/docs_sync.py --check` 看代码文档是否同步。离线测试使用真实本地数据结构/线程/数据库及外部服务替身，证明范围是协议和控制行为。

历史记录：预测流程早期回归为 75 项与 3 个子测试，持久研究升级为 106 项与 3 个子测试，工具升级为 133 项与 3 个子测试。这些是阶段性记录，不是当前测试数量。此前本机新闻/模型实测详见[预测记录](forecast-flow-update.md)和[研究记录](research-memory-guide.md)；本次文档维护没有重跑真实新闻质量评测。
