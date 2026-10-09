# 评估说明：测试通过与回答质量是两件事

入口是 [run_evaluation.py](../evaluation/run_evaluation.py)，数据集是 [dataset.json](../evaluation/dataset.json)。本文以仓库代码为准，不把历史分数当当前效果证明。

## 验证与评测入口

- `python -m pytest -q`：离线回归，检查工具、预算、并发、索引、引用、缓存与恢复。模型和外部服务大多用替身，不证明预测准确率。
- `python scripts/docs_sync.py --check`：公开文档的一致性门禁，检查源码事实、链接、SVG 与审阅指纹；不依赖本地面试资料，也不证明自然语言说明或回答本身正确。
- `python -m agentic_rag.evaluation`：冻结财经材料上的多种检索对照、ranx 标准指标/配对检验，以及已有研究的只读 Trace 导出；不调用生成模型或模型裁判，Dense/重排按需调用本地检索模型。当前 10 题合成回归样本只测排序相关性，未自动计算答案质量；见[实验基线与 Trace](实验基线与Trace.md)和[检索竞技场](检索竞技场.md)。
- `python evaluation/run_evaluation.py`：实际执行问题并调用 RAGAS 裁判，会使用配置模型、嵌入和相关数据源，也可能写正常研究数据。不是无副作用模拟。

## 当前脚本怎样取样

先 `ensure_index()`，再 `build_graph()`，逐题 `graph.invoke`，记录答案和实际 `evidence_context`，评分后写 `evaluation/results.csv`。没有像 CLI 一样注入 SqliteSaver，也没有 CLI 预热、运行锁和完整事件展示；不是可恢复 CLI 的等价评测。

2026-10-01 已修复取样口径：由 [actual_contexts](../src/agentic_rag/evaluation/answer_samples.py) 只读取生成器实际看到的 `evidence_context`。缺少字段的旧记录拒绝评分，不回退到未被模型看见的全文；空上下文保持空。目前把整个装箱上下文作为一个评分单元，因此ContextPrecision不能解释为逐篇新闻的排名精度；排名评测使用另一个带qrels的检索入口。这修正了评分输入，不代表RAGAS裁判本身不会误判，本轮也没有把未执行的RAGAS评分报成实测成绩。

## 脚本实际启用的指标

| 指标类 | 主要观察什么 | 限制 |
|---|---|---|
| `Faithfulness` | 答案声明能否被裁判收到的上下文支持 | 受取样和裁判误判影响 |
| `ResponseRelevancy` | 回答是否与问题相关 | 不证明事实正确 |
| `LLMContextPrecisionWithReference` | 参考答案条件下的相关上下文排序 | 不是简单相关文档占比 |
| `LLMContextRecall` | 参考答案的信息是否被上下文覆盖 | 依赖参考答案质量 |

裁判用 `EVAL_MODEL`，未设置时跟随 `LLM_MODEL`；模板显式填写时以模板为准。脚本设 `RunConfig(timeout=600, max_workers=1)`，是保守配置，不是所有硬件必须串行。

## 运行

```bat
conda activate agent
python -m pip install -e ".[eval,dev]"
python evaluation\run_evaluation.py
```

先确认 Ollama 与模型可用，按需配置新闻 API Key。路由可能选择外部来源，不保证完全不联网。CSV 会覆盖同名输出，需历史对比时先另存旧结果。

## 数据集的现实边界

当前 8 道英文题讲 RAG 概念，但 `knowledge/` 已偏财经分析方法，`docs/` 不参与检索。数据域不匹配，旧结果不代表当前新闻预测质量，也没有“十几道题就能发现大多数回退”的保证。

业务验收现在另用下面的真实新闻开发集，旧英文数据集保留为概念演示，不混入财经得分。分开测召回、覆盖、任务完成、成本与延迟，关键事实仍需复核。

新增概率规则和候选加权排序也需要业务级验证：规则测试只能证明指定反例能被拦截，不能证明概率校准准确；固定权重通过单元测试也不等于最优召回/排序。应分别建立目标事件与估值时点明确的概率样例，以及包含相关性标注的新闻候选集，才有依据比较改动前后效果。

已知问题统一见[当前状态](status.md)。

## 真实新闻的配对验收

入口 [acceptance.py](../src/agentic_rag/evaluation/acceptance.py)，题目与必要条件在 [cases.json](../evaluation/suites/news_acceptance_v1/cases.json)。20题来自本机13篇已有新闻，包含黄金、企业财务、短剧产业、制造业项目和会谈报道。来源真实性未获独立证明；这是检验“忠于这批原文”的开发验收集，不是独立财经金标准。

```bat
python -m agentic_rag.evaluation.acceptance freeze --output evaluation/runs/news-frozen
python -m agentic_rag.evaluation.acceptance run --snapshot evaluation/runs/news-frozen/snapshot.json --output evaluation/runs/news-acceptance --seconds 360 --calls 30
```

冻结动作只读本机研究库，按版本定位全文，核对每道题的标注引文确实存在，并删除旧研究的相关性/影响评分，避免历史模型意见污染输入。运行使用真正的本地模型和生产图，从 `grade_documents` 检查点开始；不调用路由、联网搜索或跨问题记忆召回。两组接收完全相同的文章全文：basic 跳过持久阅读和专题，research 执行它们；因此比较的是“额外研究步骤是否值得”，不能解释成端到端联网召回对照。

两组顺序逐题交替，保留配置中的思考开关，禁用最终答案缓存，独立保存研究库/检查点/向量库。研究组内部相同文章的阅读可按配方复用，复用事件会留档；不是每题冷启动性能。每题独立进程，模型调用/执行预算之外另有进程墙钟看门狗。失败、超时不会从分母消失；缺失用量标未知。源码中途变化立即停止，不把不同实现混在一份成绩中。

产物包含 `manifest.json`（源码、输入指纹与配置）、逐题 JSON/日志（答案、实际上下文、模型评审、用量、阶段事件）、`summary.json`、SQLite 研究/检查点。原文和完整记录只保存在被忽略的 `evaluation/runs/`，不自动公开。

`workflow_passed` 是模型和程序流程门禁；`checks_passed` 只是数字/关键词/合法引用等必要条件；两者共同通过也不自动等于语义正确。人工审阅状态单列。20道题共享13篇材料，不能当20个独立事件做显著性结论；每题一次的观测P95也不是线上SLA。最终结果见[当前状态](status.md)。

修复前后对比用 [acceptance_compare.py](../scripts/acceptance_compare.py)。它先核对正文、元数据、问题、来源和引文完全相同，再对照结果；缺一条记录就失败，不缩小分母。修正过严的必要条件时，可用 `--labels evaluation/suites/news_acceptance_v1/cases.json` 另算一列当前规则成绩，旧JSON与原成绩保持不变，报告列出变更题号。2026-10-01至02实跑中的N16（“而非”误报）和N08（正确未来上线时间无需额外复述“试营运”）属于此类，不能算模型能力提升。

后续复核又发现N18照抄原文的明显倍数矛盾、N17历史预测的日期归属不足，以及N20基础模式正确算术说明被预计值护栏误拦。修复与额外复跑必须使用新目录/源码指纹，不能覆盖前一轮并声称从未失败。必要条件通过、流程通过、逐条原文语义复核应分开记录；助手复核不冒称独立人工双审。

第三轮20题全部通过必要条件后，逐条复核又发现N19将季度GDP笼统归入“8月数据”；已追加统计期规则、提示和独立基础/研究模式复跑。它说明关键词检查不能替代语义核对，不能只挑通过数字发布质量结论。

[live_acceptance.py](../scripts/live_acceptance.py)另执行新问题的完整CLI：预热、路由、实际新闻查询、全文处理、专题、生成与核验，保留完整日志和隔离研究库。它不是上述冻结重放，传输路径也单独记录；通过SSH隧道不能证明公网TLS已经恢复。原文与完整运行记录默认留本机，公开仓库仅保存实现、题目定义与概括性验证结果。
