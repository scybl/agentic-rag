# 评估说明：测试通过与回答质量是两件事

入口是 [run_evaluation.py](../evaluation/run_evaluation.py)，数据集是 [dataset.json](../evaluation/dataset.json)。本文以仓库代码为准，不把历史分数当当前效果证明。

## 两条路径

- `python -m pytest -q`：离线回归，检查工具、预算、并发、索引、引用、缓存与恢复。模型和外部服务大多用替身，不证明预测准确率。
- `python scripts/docs_sync.py --check`：公开文档的一致性门禁，检查源码事实、链接、SVG 与审阅指纹；不依赖本地面试资料，也不证明自然语言说明或回答本身正确。
- `python evaluation/run_evaluation.py`：实际执行问题并调用 RAGAS 裁判，会使用配置模型、嵌入和相关数据源，也可能写正常研究数据。不是无副作用模拟。

## 当前脚本怎样取样

先 `ensure_index()`，再 `build_graph()`，逐题 `graph.invoke`，记录答案和 `documents` 正文，评分后写 `evaluation/results.csv`。没有像 CLI 一样注入 SqliteSaver，也没有 CLI 预热、运行锁和完整事件展示；不是可恢复 CLI 的等价评测。

**关键缺口：取样用 `documents`，不是生成器实际看到的 `evidence_context`。** 筛选、预算截取和专题回读后，两者可能不同。不能声称已测精确的最终证据忠实度。

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

下一轮应先修正实际上下文采样，再建立带来源/时点的业务集：日期约束、多源联合、局部失败、资料不足、情景预测、冲突新闻、引用、数字单位、复用。分开测召回、覆盖、任务完成、成本与延迟，关键事实人工复核。

新增概率规则和候选加权排序也需要业务级验证：规则测试只能证明指定反例能被拦截，不能证明概率校准准确；固定权重通过单元测试也不等于最优召回/排序。应分别建立目标事件与估值时点明确的概率样例，以及包含相关性标注的新闻候选集，才有依据比较改动前后效果。

这些是待改进方案，不表示已经实现。已知问题统一见[当前状态](status.md)。
