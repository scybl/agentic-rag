# 多源智能体 RAG

一个以本地 Ollama 为模型入口的命令行研究项目：联合稳定知识、私有新闻和公开搜索，保存可追溯的阅读成果，在有限预算内补证据、做专题分析并核验回答。

它适合展示受控工具封装、有界并发、版本化复用、检查点恢复和执行追踪；当前是本地单用户原型，不是生产级预测服务，也没有网页前端。

## 从这里开始

- [文档首页](docs/index.md)：按学习目标选择阅读顺序。
- [运行与维护](docs/operations.md)：Conda、提问、增量索引、监听、恢复与诊断。
- [技术详解](docs/技术细节详解.md)：逐层对照实际代码。
- [代码事实参考](docs/reference.md)：自动生成的配置、命令、工具输入与精确流程边表。
- [当前状态与已知问题](docs/status.md)：已实现能力、验证范围和未修缺口。

## 当前主流程

![默认研究模式：规划、多源工具取证、阅读复用、专题回读、生成与有界修正](docs/agentic-rag-research-flow.svg)

默认开启研究模式。图是 8 个学习阶段概览，不是函数一一映射；相关性、充分性和专题任务合并展示。精确连接由源码生成在代码参考中。关闭研究模式后的流程见[基础模式图](docs/agentic-rag-core-architecture.svg)。

已接入的四项工具是 `search_knowledge`、`search_news`、`search_web`、`read_news`。它们真正通过 `@tool`、输入 Schema、ToolMessage 和证据 artifact 执行，不是只挂装饰器。模型生成结构化计划，程序受控调用；不是无限自由 ReAct。

## 快速开始（Windows CMD / Anaconda Prompt）

需要 Python 3.10+、可访问的 Ollama 和已安装模型。在项目根目录，用已有 Conda 环境：

```bat
conda activate agent
python -m pip install -e ".[dev]"
if not exist .env copy .env.example .env
ollama pull qwen2.5:3b
python -m agentic_rag.ingestion
agentic-rag
```

`qwen2.5:3b` 是仓库示例，不要求替换已有模型；已有 `.env` 中的 `LLM_MODEL` 要与实际模型匹配。首次嵌入可能下载权重，离线使用需先准备本地目录。不要覆盖已有配置，也不要把 PowerShell 的 `&` 前缀复制到 CMD。

不用 Conda 时，可用 `python -m venv .venv` 并执行 `.venv\Scripts\activate`，其余安装步骤相同。更多细节见[运行指南](docs/operations.md)。

单次提问、查看工具：

```bat
agentic-rag "什么是事件研究法？"
agentic-rag --list-tools
```

私有新闻源需要本机 `NEWS_API_KEY`；不要提交真实密钥。未配置时仍可运行其他来源，但路由选中新闻会记录失败，不保证自动避开。

## 使用自己的资料

给模型看的 Markdown、TXT、PDF 放入 `knowledge/`。给人看的解释放在 `docs/`，默认不会进入知识检索。

文件新增、修改或删除后运行：

```bat
agentic-rag-ingest
```

这是增量同步，未变文件跳过。监听默认关闭；仅重启问答不会自动刷新已有非空索引。需要监听时显式使用 `agentic-rag-ingest --watch`，或设置后台监听，详见运行指南。

新闻不用先全量建库：API 召回有限候选，标题/摘要缓存辅助排序，只抓取入选正文。研究模式保存版本化原文与阅读成果，下一次兼容任务可复用。

## 研究成果与恢复

```bat
agentic-rag --runs
agentic-rag --status RUN_ID
agentic-rag --inspect-reading READING_ID
agentic-rag --resume RUN_ID
agentic-rag --resume RUN_ID --retry-failed
agentic-rag --repair-memory
```

编号要换成真实值。恢复完成研究展示历史结果，不刷新新闻；要新资料请重新提问。检查点不是跨问题聊天记忆。模型、阅读配方或工作流版本变化时，未完成旧研究需重新发起，兼容阅读成果仍能复用。

模型并发、线程数与网络并发分别限制，但都是进程内约束；`RESEARCH_TIMEOUT` 是一批调度的超时，不是整个研究总耗时。详细边界见[研究指南](docs/research-memory-guide.md)。

## 代码结构

| 位置 | 职责 |
|---|---|
| [cli.py](src/agentic_rag/cli.py) | 提问、预热、恢复、执行事件展示 |
| [graph/](src/agentic_rag/graph/build.py) | 状态图、节点、规划/生成/评审链 |
| [tools/](src/agentic_rag/tools/__init__.py) | 四项工具、契约、调用编号与完整证据返回 |
| [research/](src/agentic_rag/research/service.py) | 持久任务、阅读、专题、混合记忆、资源上限 |
| [ingestion.py](src/agentic_rag/ingestion.py) | 知识增量索引与可选监听 |
| [news_retrieval.py](src/agentic_rag/news_retrieval.py) | 新闻候选、缓存排序、入选正文与降级 |
| [evidence.py](src/agentic_rag/evidence.py) | 去重、片段选择、编号与字符预算 |
| [analysis_cache.py](src/agentic_rag/analysis_cache.py) | 核验通过的答案缓存 |
| [evaluation/](evaluation/run_evaluation.py) | 现有 RAGAS 脚本；评估域与取样仍有缺口 |
| [scripts/docs_sync.py](scripts/docs_sync.py) | 代码事实生成与文档漂移门禁 |

默认运行数据在 `.chroma/`，可能包含私有原文、问题和结果；可选本地模型在 `.models/`。它们不提交到 Git，实际路径可由配置覆盖。

## 学习与面试资料

[工具封装](docs/tools-guide.md) · [预测流程](docs/forecast-flow-update.md) · [研究记忆与恢复](docs/research-memory-guide.md) · [RAG 基础](docs/rag_fundamentals.md) · [模式与取舍](docs/agentic_rag_patterns.md) · [RAGAS 评估](docs/evaluation_ragas.md) · [45 道面试题与代码对照](docs/Agent求职面试题与参考答案.html)

终端显示来源、实际查询/日期、工具编号、证据数量、任务状态、复用和判断摘要；不输出隐藏逐字思维链。网页结果仍是搜索摘要，“核验通过”不等于预测正确。

## 开发与文档同步

```bat
python -m pytest -q
python scripts\docs_sync.py --check
```

测试主要验证协议和流程，外部服务大多使用替身，不代表真实模型质量。最新通过数以实际输出为准，历史记录见状态页。

修改代码后先 `--refresh` 自动更新事实参考与 HTML 行号，再人工核对解释和两图；完成审阅才执行 `--acknowledge-review`。检查已接入 pytest 和独立 CI。流程与边界见[文档维护约定](docs/documentation-maintenance.md)：自动门禁能发现未审阅变更，不能自动证明中文解释永远正确。

## 许可证

项目元数据声明 MIT。
