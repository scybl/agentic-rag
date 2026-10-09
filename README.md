# 多源智能体 RAG

一个以本地 Ollama 为模型入口的命令行研究项目：联合稳定知识、私有新闻和公开搜索，保存可追溯的阅读成果，在有限预算内补证据、做专题分析并核验回答。

它适合展示受控工具封装、有界并发、版本化复用、检查点恢复和执行追踪；当前是本地单用户原型，不是生产级预测服务，也没有网页前端。

交互模式现在支持连续追问：可以先问某家公司，再追问“它的风险”或明确编号的“第二条”。系统先还原问题，不确定时请求澄清；长会话按需压缩，原记录可以回查。每一题仍独立取证，不把旧回答当作新闻事实。新闻侧的展示目标是持续获得去重后的增量资讯，不承诺全量覆盖。

## 从这里开始

当前发布候选为 **1.0.0rc1**。已实修公网TLS并保留真实验收记录；适用范围和待解决问题仍以下方状态页为准，不承诺任意问题都成功。需要尽量接近本次Python 3.12环境时，可使用 `python -m pip install -e ".[dev]" -c constraints-tested.txt`；该文件只约束主要依赖，不是跨平台完整锁文件。

- [文档首页](docs/index.md)：按学习目标选择阅读顺序。
- [运行与维护](docs/operations.md)：Conda、提问、增量索引、监听、恢复与诊断。
- [技术详解](docs/技术细节详解.md)：逐层对照实际代码。
- [代码事实参考](docs/reference.md)：自动生成的配置、命令、工具输入与精确流程边表。
- [当前状态与已知问题](docs/status.md)：已实现能力、验证范围和未修缺口。

## 当前主流程

![默认研究模式：规划、多源工具取证、阅读复用、专题回读、生成与有界修正](docs/agentic-rag-research-flow.svg)

默认开启研究模式。图是 8 个学习阶段概览，不是函数一一映射；先筛选再精读，充分性检查与专题任务合并展示。精确连接由源码生成在代码参考中。关闭研究模式后的流程见[基础模式图](docs/agentic-rag-core-architecture.svg)。

开放新闻检索现采用「全量匹配摘要 → 每批12篇轻量筛选 → 有限核心正文 → 全部分段阅读」。缺失、重复或跑题的正文从候选池补位；摘要筛选和阅读成果分别缓存。阅读队列不再受旧的20／200个任务上限截断，但线程数、累计时间、模型调用和重试仍有预算。未读完会停在可恢复节点，不再拿失败提示当作已完成研究。

已接入四项取证工具 `search_knowledge`、`search_news`、`search_web`、`read_news`，以及四项规则工具：模型输出校验、自适应重试决策和两项概率契约。它们真正通过 `@tool`、输入 Schema、ToolMessage 和 artifact 执行，不是只挂装饰器。模型生成结构化计划，程序受控调用；不是无限自由 ReAct。

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

连续问答默认在交互模式开启；输入 `/new` 开始新话题，`/history` 查看记录，`/turn 2` 回看第 2 轮。退出后用 `agentic-rag --conversation 会话编号` 接续；`--no-conversation` 保留每题独立模式。详见[会话记忆与压缩](docs/conversation-guide.md)。

交互模式等待输入超过 5 分钟会自动退出，并尝试通知 Ollama 卸载当前模型；这不是研究执行超时。`agentic-rag --idle-timeout 60` 可调整等待时间，设为 `0` 可关闭。单次提问和正常退出也会请求卸载；同一 Ollama 被其他程序使用时需注意影响，释放是否成功以服务响应为准。

私有新闻源需要本机 `NEWS_API_KEY`；不要提交真实密钥。未配置时仍可运行其他来源，但路由选中新闻会记录失败，不保证自动避开。

## 使用自己的资料

给模型看的 Markdown、TXT、PDF 放入 `knowledge/`。给人看的解释放在 `docs/`，默认不会进入知识检索。

文件新增、修改或删除后运行：

```bat
agentic-rag-ingest
```

这是增量同步，未变文件跳过。监听默认关闭；仅重启问答不会自动刷新已有非空索引。需要监听时显式使用 `agentic-rag-ingest --watch`，或设置后台监听，详见运行指南。

新闻不用先全量建库：模型生成查询，程序执行并留档。未指定候选数量时取完匹配摘要页；明确“最近50篇、影响最大的20篇”时，先按主主题建立最新50篇集合，再以幅度、范围、持续性、信息增量、证据强度评估潜在影响，读取前20篇原文。影响评分可复用，但不是实证价格贡献，也不等于语义相似度。正文分析、专题任务和最终逐篇清单均有数量检查，不足时标记未完成。阅读成果保存原文位置、事件背景及事实/预测类型；发布时间不能替代事件时间。详见[研究指南](docs/research-memory-guide.md)。

人物与机构逐项检查是否进入实际查询；主题是检索线索，程序在最多四组查询内补充，并展示未覆盖主题，不因主题标签与查询措辞不同而拒绝整个计划。新闻临时 HTTP 故障最多尝试三次；仍失败时保留已取回部分，明确标记分页未取完，不将服务故障解释成“没有新闻”。

## 研究成果与恢复

```bat
agentic-rag --runs
agentic-rag --status RUN_ID
agentic-rag --inspect-reading READING_ID
agentic-rag --resume RUN_ID
agentic-rag --resume RUN_ID --retry-failed
agentic-rag --repair-memory
```

编号要换成真实值。恢复已结束且有答案的研究可离线展示历史结果，不预热模型、不启动监听、不写入新事件、不刷新新闻；要新资料请重新提问。检查点不是跨问题聊天记忆。模型、阅读配方或工作流版本变化时，未完成旧研究需重新发起，兼容阅读成果仍能复用。

模型并发、线程数与网络并发分别限制，但都是进程内约束；`RESEARCH_TIMEOUT` 是一批调度的超时。整场另有 `RESEARCH_TOTAL_TIMEOUT`（默认1800秒）与 `RESEARCH_MAX_MODEL_CALLS`（默认160次，含重试），恢复继承已消耗额度；这是协作式限制，不是远端推理强杀或精确Token额度。详细边界见[研究指南](docs/research-memory-guide.md)。

## 代码结构

| 位置 | 职责 |
|---|---|
| [cli.py](src/agentic_rag/cli.py) | 提问、预热、恢复、执行事件展示 |
| [conversation.py](src/agentic_rag/conversation.py) | 会话持久化、追问还原、抽取式压缩与原问答回读 |
| [token_usage.py](src/agentic_rag/token_usage.py) | 真实 token 计量、逐调用/逐步骤计时、研究累计统计 |
| [console.py](src/agentic_rag/console.py) | 默认精简数字视图，实际查询参数和异常不隐藏 |
| [research/inspection.py](src/agentic_rag/research/inspection.py) | 只读状态诊断，识别已保存节点结果但尚待推进的检查点 |
| [graph/](src/agentic_rag/graph/build.py) | 状态图、节点、规划/生成/评审链 |
| [tools/](src/agentic_rag/tools/__init__.py) | 四项取证工具、输出校验、自适应重试决策与两项概率规则、调用审计 |
| [research/](src/agentic_rag/research/service.py) | 持久任务、阅读、专题、混合记忆、资源上限 |
| [ingestion.py](src/agentic_rag/ingestion.py) | 知识增量索引与可选监听 |
| [news_retrieval.py](src/agentic_rag/news_retrieval.py) | 新闻候选、缓存排序、入选正文与降级 |
| [news_plan.py](src/agentic_rag/news_plan.py) / [news_ranking.py](src/agentic_rag/news_ranking.py) | 新闻查询编译、统一过滤、全部候选的向量与 TF-IDF 加权排序 |
| [evidence.py](src/agentic_rag/evidence.py) | 去重、片段选择、编号与字符预算 |
| [evidence_audit.py](src/agentic_rag/evidence_audit.py) | 时间、口径、反向证据、价格基准及因果传导的逐项核验约束 |
| [analysis_cache.py](src/agentic_rag/analysis_cache.py) | 核验通过的答案缓存 |
| [evaluation/](src/agentic_rag/evaluation/__main__.py) | 冻结财经样本上的重复实验与结果比较 |
| [retrieval/](src/agentic_rag/retrieval/arena.py) | 不同检索方案的离线对照实验 |
| [telemetry/](src/agentic_rag/telemetry/schema.py) | 研究运行记录的统一格式与导出 |
| [评估材料与脚本](docs/evaluation_ragas.md) | 合成检索回归、真实新闻配对模型验收和RAGAS实际上下文取样；独立质量评估仍有缺口 |
| [scripts/docs_sync.py](scripts/docs_sync.py) | 代码事实生成与文档漂移门禁 |

默认运行数据在 `.chroma/`，可能包含私有原文、问题和结果；可选本地模型在 `.models/`。它们不提交到 Git，实际路径可由配置覆盖。

## 学习资料与执行观察

[工具封装](docs/tools-guide.md) · [预测流程](docs/forecast-flow-update.md) · [研究记忆与恢复](docs/research-memory-guide.md) · [RAG 基础](docs/rag_fundamentals.md) · [模式与取舍](docs/agentic_rag_patterns.md) · [RAGAS 评估](docs/evaluation_ragas.md)

新增能力的运行示例和验证范围见[实验基线与 Trace](docs/实验基线与Trace.md)和[检索竞技场](docs/检索竞技场.md)。

工具调用显示名称、调用者、理由和结果。完整模型返回会执行输出类型校验；截断则直接进入重试决策，不接纳半截内容。格式、额度或服务失败时，`plan_model_retry` 在资源上限内决定反馈修正、扩容、延时、关闭思考、切分或停止。概率类问题另执行生成前证据检查与交付前答案检查；这些规则不是概率计算模型。

规划阶段还会逐项列出全部八个工具本轮是“必须调用、满足条件时调用、还是跳过”，并给出调用者和理由。`validate_model_output` 每轮必须执行，`plan_model_retry` 仅在失败后调用；政策利率概率题还会强制使用本地方法、新闻和公开搜索，证据不足时先用 `read_news` 回读已有原文。

终端显示实际查询条件、候选过滤数量、入选权重、证据数、任务状态和复用。详细模式补充调用编号、完整参数和判断说明；事件随研究编号保存，不输出隐藏逐字思维链。网页结果仍是搜索摘要，“核验通过”不等于预测正确。

默认按步骤显示一行核心数字：耗时、调用数、输入/生成 token、状态；实际查询词、日期、取证条数和异常仍显示。最后只保留一块「Token 最终合计 / 耗时统计」。`agentic-rag -v` 展开完整规划理由、逐调用用量和核验说明；计量仍保存所有调用，重试/回读/重写不漏算，未知不冒充零，思考是生成的子项。口径见[Token 统计说明](docs/operations.md#token-统计怎么看)。无需重建索引。

末尾区分本次执行/研究历史/程序累计时间。并发模型请求耗时之和在 `-v` 下单独列出，不能当作实际等待时长；旧记录缺失计时显示未知。`--status RUN_ID` 显示停止原因、进度与可恢复状态，`--status RUN_ID -v` 展开完整记录；查询不会重跑模型。口径见[计时说明](docs/operations.md#计时怎么看)。

支持思考的模型可设置 `LLM_REASONING=true`；证据综合、专题与回答遵循该开关，规划、相关性筛选、摘录和答案评审首轮关闭深度思考。自适应重试也可能临时关闭思考。仓库小模型示例默认关闭，具体资源策略见[技术详解](docs/技术细节详解.md)。

## 升级现有工作区

拉取后重新执行 `python -m pip install -e ".[dev]"`，安装新闻词法排序新增的 `scikit-learn` 依赖。无需因文档、日志或候选排序更新而重建知识索引；不要清空研究数据库。工作流和阅读配方已升级，不兼容的未完成研究需重新提问，完成记录仍可查看，兼容成果仍可复用。

## 开发与文档同步

```bat
python -m pytest -q
python scripts\docs_sync.py --check
```

测试主要验证协议和流程，外部服务大多使用替身，不代表真实模型质量。最新通过数以实际输出为准，历史记录见状态页。

修改代码后先 `--refresh` 更新事实参考与公开 HTML 行号，再人工核对解释和两图；完成审阅才执行 `--acknowledge-review`。检查已接入 pytest 和独立 CI。公开仓库只包含项目代码、测试、配置示例、必要知识样例与使用说明；面试、简历、后续规划、临时导入和私有运行数据只留本地，不发布，也不进入文档检查或默认知识库。干净克隆无需这些文件。维护流程与边界见[文档维护约定](docs/documentation-maintenance.md)。

## 许可证

项目元数据声明 MIT。
