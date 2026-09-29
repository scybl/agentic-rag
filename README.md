# 🧭 多源智能体 RAG

一个面向真实资料分析的本地智能问答项目。

它不仅会从资料中寻找内容，还会根据问题选择合适的信息来源、核对证据是否足够，并在回答存在缺口时继续查找或修正。最终答案会列出实际使用的证据，方便复核。

项目当前专注于核心问答链路，采用命令行交互，不包含网页前端和 Docker 部署文件。

## 项目流程一览

![持久化多 Agent 研究：规划、并行检索、阅读复用、专题分析、补搜和核验](docs/agentic-rag-research-flow.svg)

默认启用持久化研究模式。先读[研究记忆与恢复指南](docs/research-memory-guide.md)，再结合[技术细节详解](docs/技术细节详解.md)理解基础 RAG；旧核心图保留作此前版本参考。

## 项目亮点

- **多来源联合分析**：可以结合本地知识、私有新闻和公开网络回答问题。
- **证据驱动**：重要事实附带证据编号和来源，不把模型记忆当成可靠资料。
- **主动发现缺口**：资料不足时会说明缺什么，并在有限次数内补充搜索。
- **回答自我检查**：不仅检查有没有依据，还检查是否真正完成了用户的问题。
- **本地优先**：问答模型、嵌入模型和知识索引均可在本机运行。
- **过程透明**：终端展示来源选择、实际查询、证据数量和最终检查结果。
- **真正有界并发**：工作线程、模型请求和网络请求分别限流，任务有独立目标和状态。
- **阅读成果长期复用**：原文分段、精确引用、内容版本和阅读覆盖率落盘，不因换问题而重复阅读同一版本。
- **增强检索记忆**：Chroma 多层向量投影 + SQLite FTS5 关键词召回；向量故障可修复，不丢阅读结果。
- **中断后继续**：LangGraph 检查点与子任务日志协作，完成的任务直接复用，失败任务按预算重试。
- **可替换领域资料**：将自己的 Markdown、TXT 或 PDF 放入知识目录即可建立专属知识库。

## 适合展示的场景

- 对近期财经新闻进行来源核验和影响分析；
- 将实时新闻与稳定的分析方法结合起来；
- 根据已有事实做有条件的趋势判断，并明确假设与不确定性；
- 面向企业资料、业务规则或研究笔记构建本地知识助手；
- 演示一个能够规划、检索、反思和自我修正的智能体工作流。

## 快速开始

环境要求：Python **3.10+** 和 [Ollama](https://ollama.com/download)。

```bat
:: 1. 准备本地对话模型
ollama pull qwen2.5:3b

:: 2. 获取并安装项目
git clone https://github.com/Slahnia/agentic-rag.git
cd agentic-rag
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[eval,dev]"

:: 3. 创建本地配置
copy .env.example .env

:: 4. 为 knowledge 目录建立索引
agentic-rag-ingest

:: 5. 启动交互式问答
agentic-rag
```

也可以直接提出一个问题：

```bat
agentic-rag "什么是事件研究法？"
```

私有新闻功能需要在本机 `.env` 中配置自己的 `NEWS_API_KEY`；不使用该功能时，仍可体验本地知识问答和公开网络搜索。

已有 Conda `agent` 环境可以直接 `conda activate agent`，然后 `python -m pip install -e ".[dev]"`。不要覆盖已有 `.env`；新配置缺省生效，无需为此次升级重建知识库。

每次研究会显示编号。管理命令中的编号需要替换为真实值：

```bat
agentic-rag --runs
agentic-rag --status RUN_ID
agentic-rag --resume RUN_ID
agentic-rag --resume RUN_ID --retry-failed
agentic-rag --inspect-reading READING_ID
agentic-rag --repair-memory
```

默认 `AGENT_WORKERS=4`、`LLM_CONCURRENCY=2`、`RESEARCH_MAX_TASKS=20`、`RESEARCH_MAX_ROUNDS=3`。显存紧张时把模型请求并发改为1；这不影响其他任务并行。知识目录监听保持默认关闭。

## 使用自己的资料

将需要检索的 `.md`、`.txt` 或 `.pdf` 文件放入 `knowledge/`，然后运行：

```bat
agentic-rag-ingest
```

再次启动 `agentic-rag` 后，新资料即可参与回答。

`docs/` 用来存放供人阅读的项目文档，不会作为问答证据；`knowledge/` 才是提供给模型检索的知识材料。

## 示例问题

| 问题 | 预期用途 |
|---|---|
| “如何判断一条财经新闻是否可信？” | 使用本地分析方法 |
| “今天有哪些生猪行业新闻？” | 查询近期新闻 |
| “这些新闻可能怎样影响相关企业的收入和成本？” | 联合新闻与本地知识 |
| “基于现有证据，未来价格可能出现哪些情景？” | 条件分析与不确定性说明 |
| “查询私有新闻库未覆盖的公开信息” | 补充公开网络资料 |

## 你会在终端看到什么

程序会展示一份可以核对的执行摘要，例如：

```text
[流程 1 · 规划数据源]
  选择：本地知识库 + 新闻 API

[流程 2–5 · 收集并合并证据]
  本地知识库：4 条
  新闻 API：5 条

[流程 6b · 检查证据是否足以分析]
  可进行条件分析：是
  缺口：消费需求数据不足，需要在回答中说明

[流程 9 · 核验答案依据]
  事实与推断依据：通过
  是否完成用户任务：通过
```

这些是来源、查询和判断结果的摘要，不会输出模型隐藏的逐字思维链。

## 项目结构

```text
agentic-rag/
├── src/agentic_rag/                 # 项目核心代码
│   ├── __init__.py                  # Python 包入口
│   ├── config.py                    # 读取环境变量和项目配置
│   ├── cli.py                       # 命令行问答入口与执行过程展示
│   ├── ingestion.py                 # 将本地资料同步到知识索引
│   ├── evidence.py                  # 证据去重、截取、编号和上下文整理
│   ├── analysis_cache.py            # 保存已经核验通过的分析结果
│   ├── ollama_connection.py         # 管理本机 Ollama 的连接方式
│   ├── news_api.py                  # 访问私有新闻服务
│   ├── news_plan.py                 # 整理新闻关键词和查询时间
│   ├── news_retrieval.py            # 召回、筛选和排序新闻候选
│   ├── news_index.py                # 管理本地新闻向量缓存
│   ├── research/                    # 持久任务、分段阅读、混合记忆、专题 Agent 与资源上限
│   ├── graph/                       # 智能体工作流
│   │   ├── __init__.py
│   │   ├── state.py                 # 定义各步骤共享的数据状态
│   │   ├── chains.py                # 规划、评估、生成和反思所用的模型链
│   │   ├── nodes.py                 # 每个工作流节点的具体行为
│   │   └── build.py                 # 连接节点并构建完整流程图
│   └── tools/
│       ├── __init__.py
│       └── web_search.py            # DuckDuckGo 公开网络搜索
│
├── knowledge/                       # 提供给模型检索的业务知识材料
│
├── docs/                            # 供人阅读的文档，不参与知识检索
│   ├── 技术细节详解.md              # 完整技术实现、设计权衡和面试问答
│   ├── agentic-rag-core-architecture.svg  # 核心流程图
│   ├── forecast-flow-update.md      # 预测流程专项说明
│   ├── research-memory-guide.md     # 新研究架构、数据存储、恢复命令与已知边界
│   ├── agentic-rag-research-flow.svg # 当前持久化研究流程图
│   ├── rag_fundamentals.md          # RAG 基础知识
│   ├── agentic_rag_patterns.md      # 智能体 RAG 常见模式
│   └── evaluation_ragas.md          # RAGAS 评估说明
│
├── evaluation/
│   ├── dataset.json                 # 评估问题与参考答案
│   └── run_evaluation.py            # 运行 RAGAS 离线评估
│
├── scripts/
│   └── diagnose_ollama.py           # 排查 Ollama 与系统代理连接问题
│
├── tests/                           # 自动化测试
│   ├── test_ingestion_incremental.py # 知识索引增量同步测试
│   ├── test_news_api.py             # 新闻接口参数与安全测试
│   ├── test_news_plan.py            # 新闻查询计划测试
│   ├── test_news_retrieval.py       # 新闻召回与降级测试
│   ├── test_news_index.py           # 新闻向量缓存测试
│   ├── test_news_graph.py           # 多来源工作流测试
│   ├── test_evidence.py             # 证据处理测试
│   ├── test_generation_check.py     # 回答核验与引用测试
│   ├── test_forecast_flow.py        # 预测、补搜与重写流程测试
│   ├── test_analysis_cache.py       # 分析缓存测试
│   ├── test_cli_trace.py            # 命令行执行摘要测试
│   └── test_ollama_connection.py    # Ollama 连接与代理隔离测试
│
├── .env.example                     # 可复制的配置模板
├── pyproject.toml                   # 依赖、包信息和命令行入口
└── README.md                        # 项目介绍和快速开始
```

运行后还会生成 `.chroma/`（知识索引与缓存）和可选的 `.models/`（本地嵌入模型）；它们属于本机运行数据，不提交到仓库。

## 项目文档

- [技术细节详解（推荐先读）](docs/技术细节详解.md)：完整实现机制、设计权衡、可靠性边界和面试问答。
- [研究记忆与恢复指南（本次升级）](docs/research-memory-guide.md)：有界并发、版本化阅读、向量投影与恢复。
- [当前研究流程图](docs/agentic-rag-research-flow.svg)：对应默认开启的多 Agent 研究模式。
- [核心流程图](docs/agentic-rag-core-architecture.svg)：项目主要数据流的可视化说明。
- [预测流程优化说明](docs/forecast-flow-update.md)：预测类问题的专项改进记录。
- [RAG 基础](docs/rag_fundamentals.md)：RAG 的基础概念。
- [智能体 RAG 模式](docs/agentic_rag_patterns.md)：常见工作流模式。
- [RAGAS 评估](docs/evaluation_ragas.md)：评估指标入门。

## 当前状态

- 核心问答链路可用；
- 本地知识索引与新闻检索已接入；
- 本次自动化验证：`106 passed, 3 subtests passed`，包含并发、增量阅读、混合检索与持久化恢复；可用 `python -m pytest -q` 复验；
- 项目以本地单用户、命令行使用为主。

## 后续方向

- 增加多轮对话记忆；
- 完善财经领域评估集；
- 增加面向大规模数据的归档、权限和分布式运行治理；
- 探索 GraphRAG 与长期事件记忆。

## 许可证

MIT
