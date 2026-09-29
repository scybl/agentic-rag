# 代码事实参考（自动生成）

> 由 `python scripts/docs_sync.py --refresh` 从当前源码 AST 与 `.env.example` 生成；不要手改。
> 不导入业务模块，不读取个人 `.env`。表达式是代码默认规则，不是本机有效配置。

解释与操作见[文档首页](index.md)、[运行指南](operations.md)及[维护约定](documentation-maintenance.md)。

## 配置声明

`EVAL_MODEL` 未设置时跟随 `LLM_MODEL`；`DOCUMENTS_DIR`、`KB_DESCRIPTION` 是旧名回退。
路径配置相对项目根目录解析；直接新闻配置另列。

| 环境变量（含回退） | Settings 字段 | 源码默认表达式 |
| --- | --- | --- |
| LLM_MODEL | llm_model | `os.getenv('LLM_MODEL', 'qwen2.5:3b')` |
| EVAL_MODEL / LLM_MODEL | eval_model | `os.getenv('EVAL_MODEL', os.getenv('LLM_MODEL', 'qwen2.5:3b'))` |
| OLLAMA_BASE_URL | ollama_base_url | `os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')` |
| OLLAMA_WARMUP_ENABLED | ollama_warmup_enabled | `_boolean('OLLAMA_WARMUP_ENABLED', True)` |
| OLLAMA_WARMUP_ATTEMPTS | ollama_warmup_attempts | `int(os.getenv('OLLAMA_WARMUP_ATTEMPTS', '10'))` |
| OLLAMA_WARMUP_RETRY_SECONDS | ollama_warmup_retry_seconds | `float(os.getenv('OLLAMA_WARMUP_RETRY_SECONDS', '3'))` |
| OLLAMA_KEEP_ALIVE | ollama_keep_alive | `os.getenv('OLLAMA_KEEP_ALIVE', '30m')` |
| LLM_TEMPERATURE | temperature | `float(os.getenv('LLM_TEMPERATURE', '0'))` |
| LLM_MAX_ATTEMPTS | llm_max_attempts | `int(os.getenv('LLM_MAX_ATTEMPTS', '6'))` |
| LLM_REASONING | llm_reasoning | `_boolean('LLM_REASONING', False)` |
| LLM_CONTEXT_WINDOW | llm_context_window | `int(os.getenv('LLM_CONTEXT_WINDOW', '8192'))` |
| LLM_MAX_OUTPUT_TOKENS | llm_max_output_tokens | `int(os.getenv('LLM_MAX_OUTPUT_TOKENS', '2048'))` |
| EMBEDDING_MODEL | embedding_model | `os.getenv('EMBEDDING_MODEL', 'BAAI/bge-small-zh-v1.5')` |
| CHROMA_DIR | chroma_dir | `field(default_factory=lambda: _path('CHROMA_DIR', '.chroma'))` |
| CHROMA_COLLECTION | collection_name | `os.getenv('CHROMA_COLLECTION', 'knowledge_base')` |
| KNOWLEDGE_DIR / DOCUMENTS_DIR | knowledge_dir | `field(default_factory=lambda: _path('KNOWLEDGE_DIR', os.getenv('DOCUMENTS_DIR', 'knowledge')))` |
| RETRIEVAL_K | retrieval_k | `int(os.getenv('RETRIEVAL_K', '4'))` |
| GENERATION_MAX_DOCUMENTS | generation_max_documents | `int(os.getenv('GENERATION_MAX_DOCUMENTS', '9'))` |
| GENERATION_CONTEXT_CHARS | generation_context_chars | `int(os.getenv('GENERATION_CONTEXT_CHARS', '6000'))` |
| NEWS_RETRIEVAL_K | news_retrieval_k | `int(os.getenv('NEWS_RETRIEVAL_K', '5'))` |
| NEWS_CANDIDATE_K | news_candidate_k | `int(os.getenv('NEWS_CANDIDATE_K', '30'))` |
| NEWS_VECTOR_CACHE_ENABLED | news_vector_cache_enabled | `_boolean('NEWS_VECTOR_CACHE_ENABLED', True)` |
| MAX_RETRIES | max_retries | `int(os.getenv('MAX_RETRIES', '2'))` |
| MAX_ANSWER_REVISIONS | max_answer_revisions | `int(os.getenv('MAX_ANSWER_REVISIONS', '2'))` |
| RESEARCH_ENABLED | research_enabled | `_boolean('RESEARCH_ENABLED', True)` |
| RESEARCH_DB | research_db | `field(default_factory=lambda: _path('RESEARCH_DB', '.chroma/research.sqlite3'))` |
| CHECKPOINT_DB | checkpoint_db | `field(default_factory=lambda: _path('CHECKPOINT_DB', '.chroma/checkpoints.sqlite3'))` |
| MEMORY_VECTOR_DIR | memory_vector_dir | `field(default_factory=lambda: _path('MEMORY_VECTOR_DIR', '.chroma/memory'))` |
| AGENT_WORKERS | agent_workers | `int(os.getenv('AGENT_WORKERS', '4'))` |
| LLM_CONCURRENCY | llm_concurrency | `int(os.getenv('LLM_CONCURRENCY', '2'))` |
| IO_CONCURRENCY | io_concurrency | `int(os.getenv('IO_CONCURRENCY', '4'))` |
| RESEARCH_MAX_TASKS | research_max_tasks | `int(os.getenv('RESEARCH_MAX_TASKS', '20'))` |
| RESEARCH_MAX_ROUNDS | research_max_rounds | `int(os.getenv('RESEARCH_MAX_ROUNDS', '3'))` |
| TASK_MAX_ATTEMPTS | task_max_attempts | `int(os.getenv('TASK_MAX_ATTEMPTS', '2'))` |
| TASK_LEASE_SECONDS | task_lease_seconds | `int(os.getenv('TASK_LEASE_SECONDS', '180'))` |
| TASK_WAIT_SECONDS | task_wait_seconds | `int(os.getenv('TASK_WAIT_SECONDS', '240'))` |
| LLM_REQUEST_TIMEOUT | llm_request_timeout | `int(os.getenv('LLM_REQUEST_TIMEOUT', '120'))` |
| RESEARCH_TIMEOUT | research_timeout | `int(os.getenv('RESEARCH_TIMEOUT', '900'))` |
| READING_CHUNK_CHARS | reading_chunk_chars | `int(os.getenv('READING_CHUNK_CHARS', '2200'))` |
| MEMORY_RECALL_K | memory_recall_k | `int(os.getenv('MEMORY_RECALL_K', '3'))` |
| SPECIALIST_COUNT | specialist_count | `int(os.getenv('SPECIALIST_COUNT', '3'))` |
| KNOWLEDGE_WATCH_ENABLED | knowledge_watch_enabled | `_boolean('KNOWLEDGE_WATCH_ENABLED', False)` |
| KNOWLEDGE_WATCH_INTERVAL | knowledge_watch_interval | `float(os.getenv('KNOWLEDGE_WATCH_INTERVAL', '2'))` |
| ANALYSIS_CACHE_ENABLED | analysis_cache_enabled | `_boolean('ANALYSIS_CACHE_ENABLED', True)` |
| ANALYSIS_CACHE_PATH | analysis_cache_path | `field(default_factory=lambda: _path('ANALYSIS_CACHE_PATH', '.chroma/analysis_cache.sqlite3'))` |
| ANALYSIS_CACHE_TTL_SECONDS | analysis_cache_ttl_seconds | `int(os.getenv('ANALYSIS_CACHE_TTL_SECONDS', '3600'))` |
| KNOWLEDGE_DESCRIPTION / KB_DESCRIPTION | knowledge_description | `os.getenv('KNOWLEDGE_DESCRIPTION', os.getenv('KB_DESCRIPTION', '本地财经新闻分析方法知识库，包含新闻来源核验与可信度分级、企业财务影响与重要性分析、事件研究法与市场反应评估；用于解释分析框架和方法，不包含实时新闻事实或个股投资建议。'))` |
| NEWS_DESCRIPTION | news_description | `os.getenv('NEWS_DESCRIPTION', '只读同花顺新闻库，持续更新，覆盖产经新闻、区域经济、公司新闻、国际财经、财经评论、财经要闻、宏观经济、金融市场和财经人物；支持按关键词、日期和栏目查询，适合回答近期或历史财经新闻问题，不包含通用知识教程和非新闻类公司内部资料。')` |
| WEB_DESCRIPTION | web_description | `os.getenv('WEB_DESCRIPTION', 'DuckDuckGo 公开网络搜索，返回相关网页的标题、链接和摘要；适合查询本地文档与私有新闻库未覆盖的公开信息和非财经时效信息；结果来自公开互联网，不代表用户的私有资料，也不保证来源权威性。')` |

### src/agentic_rag/news_api.py 的新闻配置

| 常量 | 值 |
| --- | --- |
| DEFAULT_BASE_URL | `'https://106.54.27.114:3001'` |

读取的环境变量：`NEWS_API_BASE_URL`、`NEWS_API_KEY`

### src/agentic_rag/news_index.py 的新闻配置

| 常量 | 值 |
| --- | --- |
| DEFAULT_INDEX_DIR | `PROJECT_ROOT / '.chroma' / 'news'` |
| DEFAULT_COLLECTION | `'tonghuashun_news'` |

读取的环境变量：`NEWS_CHROMA_COLLECTION`、`NEWS_CHROMA_DIR`

### 配置模板示例（不含个人配置）

模板显式填值会覆盖动态回退。`NEWS_API_KEY` 仅为注释占位，需本机提供。

| 变量 | 示例值 |
| --- | --- |
| LLM_MODEL | `qwen2.5:3b` |
| OLLAMA_BASE_URL | `http://localhost:11434` |
| OLLAMA_WARMUP_ENABLED | `true` |
| OLLAMA_WARMUP_ATTEMPTS | `10` |
| OLLAMA_WARMUP_RETRY_SECONDS | `3` |
| OLLAMA_KEEP_ALIVE | `30m` |
| LLM_TEMPERATURE | `0` |
| LLM_MAX_ATTEMPTS | `6` |
| LLM_REASONING | `false` |
| LLM_CONTEXT_WINDOW | `8192` |
| LLM_MAX_OUTPUT_TOKENS | `2048` |
| EVAL_MODEL | `qwen2.5:3b` |
| EMBEDDING_MODEL | `BAAI/bge-small-zh-v1.5` |
| CHROMA_DIR | `.chroma` |
| CHROMA_COLLECTION | `knowledge_base` |
| KNOWLEDGE_DIR | `knowledge` |
| RETRIEVAL_K | `4` |
| GENERATION_MAX_DOCUMENTS | `9` |
| GENERATION_CONTEXT_CHARS | `6000` |
| NEWS_RETRIEVAL_K | `5` |
| NEWS_CANDIDATE_K | `30` |
| NEWS_VECTOR_CACHE_ENABLED | `true` |
| MAX_RETRIES | `2` |
| MAX_ANSWER_REVISIONS | `2` |
| KNOWLEDGE_WATCH_ENABLED | `false` |
| KNOWLEDGE_WATCH_INTERVAL | `2` |
| ANALYSIS_CACHE_ENABLED | `true` |
| ANALYSIS_CACHE_PATH | `.chroma/analysis_cache.sqlite3` |
| ANALYSIS_CACHE_TTL_SECONDS | `3600` |
| RESEARCH_ENABLED | `true` |
| RESEARCH_DB | `.chroma/research.sqlite3` |
| CHECKPOINT_DB | `.chroma/checkpoints.sqlite3` |
| MEMORY_VECTOR_DIR | `.chroma/memory` |
| AGENT_WORKERS | `4` |
| LLM_CONCURRENCY | `2` |
| IO_CONCURRENCY | `4` |
| RESEARCH_MAX_TASKS | `20` |
| RESEARCH_MAX_ROUNDS | `3` |
| SPECIALIST_COUNT | `3` |
| TASK_MAX_ATTEMPTS | `2` |
| TASK_LEASE_SECONDS | `180` |
| TASK_WAIT_SECONDS | `240` |
| LLM_REQUEST_TIMEOUT | `120` |
| RESEARCH_TIMEOUT | `900` |
| READING_CHUNK_CHARS | `2200` |
| MEMORY_RECALL_K | `3` |
| KNOWLEDGE_DESCRIPTION | `本地财经新闻分析方法知识库，包含新闻来源核验与可信度分级、企业财务影响与重要性分析、事件研究法与市场反应评估；用于解释分析框架和方法，不包含实时新闻事实或个股投资建议。` |
| NEWS_DESCRIPTION | `只读同花顺新闻库，持续更新，覆盖产经新闻、区域经济、公司新闻、国际财经、财经评论、财经要闻、宏观经济、金融市场和财经人物；支持按关键词、日期和栏目查询，适合回答近期或历史财经新闻问题，不包含通用知识教程和非新闻类公司内部资料。` |
| WEB_DESCRIPTION | `DuckDuckGo 公开网络搜索，返回相关网页的标题、链接和摘要；适合查询本地文档与私有新闻库未覆盖的公开信息和非财经时效信息；结果来自公开互联网，不代表用户的私有资料，也不保证来源权威性。` |
| NEWS_API_BASE_URL | `https://106.54.27.114:3001` |
| NEWS_CHROMA_DIR | `.chroma/news` |
| NEWS_CHROMA_COLLECTION | `tonghuashun_news` |

## 命令入口与参数

入口由 pyproject.toml 声明：

```toml
agentic-rag = "agentic_rag.cli:main"
agentic-rag-ingest = "agentic_rag.ingestion:main"
```

### `src/agentic_rag/cli.py`

| 解析器/子命令 | 参数 | 声明 |
| --- | --- | --- |
| parser | 'question' | nargs='*'; help='Question to ask (omit for interactive mode)' |
| parser | '--resume' | metavar='RUN_ID'; help='恢复已保存研究；不与新问题同时使用' |
| parser | '--runs' | action='store_true'; help='列出最近研究，不调用模型' |
| parser | '--status' | metavar='RUN_ID'; help='查看任务与最近执行事件' |
| parser | '--retry-failed' | action='store_true'; help='与 --resume 配合，重新尝试失败子任务' |
| parser | '--inspect-reading' | metavar='READING_ID'; help='查看阅读成果、原文位置与版本' |
| parser | '--repair-memory' | action='store_true'; help='仅重试向量投影，不重新阅读' |
| parser | '--list-tools' | action='store_true'; help='查看实际注册的工具及参数，不调用模型或初始化数据库' |
| parser | '-v' / '--verbose' | action='store_true'; help='Show additional exception details; queries and time choices are always shown' |

### `src/agentic_rag/ingestion.py`

| 解析器/子命令 | 参数 | 声明 |
| --- | --- | --- |
| parser | '--rebuild' | action='store_true'; help='忽略清单并全量重建索引' |
| parser | '--watch' | action='store_true'; help='初始同步后持续监听知识目录' |
| parser | '--interval' | type=float; default=settings.knowledge_watch_interval; help='监听模式的检查间隔秒数' |

### `src/agentic_rag/news_index.py`

| 解析器/子命令 | 参数 | 声明 |
| --- | --- | --- |
| commands | 'preview' | help='Inspect API data without creating an index' |
| preview | '--max-items' | type=int; default=20 |
| commands | 'sync' | help='Index the newest articles incrementally' |
| sync | '--max-items' | type=int; default=1000 |
| sync | '--batch-size' | type=int; default=32 |
| commands | 'search' | help='Search the local news index' |
| search | 'query' |  |
| search | '--k' | type=int; default=5 |

## 工具公开输入（源码声明，不是运行时 JSON Schema）

完整的运行时 Schema 请运行 `agentic-rag --list-tools`；程序注入的 RunnableConfig 不在公开参数中。
声明中的校验器与共享类型同样属于契约，不能仅靠字段表判断所有规则。

### `search_knowledge`

检索本地 knowledge 中的稳定知识和分析方法，返回原始文档片段及来源。

不负责提供最新新闻；只在实际调用时初始化索引和嵌入，条数使用 RETRIEVAL_K。

实现：[tools/knowledge.py](../src/agentic_rag/tools/knowledge.py)；装饰器：`tool('search_knowledge', args_schema=KnowledgeSearchInput, response_format='content_and_artifact')`

| 字段 | 类型 | 默认/约束声明 |
| --- | --- | --- |
| query | `Query` | `Field(description='要寻找的分析方法、领域概念或稳定知识，不用于查询实时新闻')` |

### `search_news`

按短关键词和指定日期检索私有新闻，候选向量排序后读取入选正文。

每页最多20条，候选和入选总量受配置约束；不会自动添加时间限制。
保留部分失败和摘要降级信息，不把 API 错误等同于没有相关新闻。

实现：[tools/news.py](../src/agentic_rag/tools/news.py)；装饰器：`tool('search_news', args_schema=NewsSearchInput, response_format='content_and_artifact')`

| 字段 | 类型 | 默认/约束声明 |
| --- | --- | --- |
| semantic_query | `Query` | `Field(description='原始研究问题，用于候选新闻的语义排序')` |
| queries | `list[Keyword]` | `Field(min_length=1, max_length=4, description='1至4组分别查询的短主题词；只传一个空字符串表示不限关键词')` |
| start | `str` | `Field(default='', description='已确认的新闻起始日 YYYY-MM-DD；空字符串表示不限制')` |
| end | `str` | `Field(default='', description='已确认的新闻结束日 YYYY-MM-DD；空字符串表示不限制')` |
| section | `str` | `Field(default='', max_length=100, description='用户指定栏目；空字符串表示不限')` |

### `read_news`

回读当前证据中已保存新闻的相关原文，返回版本、绝对位置和逐字引用。

最多两篇、总计最多1600字符；不是再次联网抓取。需程序注入证据白名单。
摘要来源仍标为摘要；资料未保存时明确说明，不伪装成全文。

实现：[tools/news_reading.py](../src/agentic_rag/tools/news_reading.py)；装饰器：`tool('read_news', args_schema=NewsReadInput, response_format='content_and_artifact')`

| 字段 | 类型 | 默认/约束声明 |
| --- | --- | --- |
| evidence_ids | `list[EvidenceId]` | `Field(min_length=1, max_length=2, description='当前证据中要回读的E编号，最多两篇；不可传URL、路径或任意版本号')` |
| goal | `Query` | `Field(description='这次需要核对的具体事实或口径，用于挑选相关原文片段')` |

### `search_web`

搜索公开网页，返回带网址的搜索摘要，不保证获取网页全文。

用于私有新闻之外的公开核验与补充，不替代本地知识库或私有新闻接口。
网络故障明确报错，由调用方保留其他来源的成功结果。

实现：[tools/web_search.py](../src/agentic_rag/tools/web_search.py)；装饰器：`tool('search_web', args_schema=WebSearchInput, response_format='content_and_artifact')`

| 字段 | 类型 | 默认/约束声明 |
| --- | --- | --- |
| query | `Query` | `Field(description='公开网络搜索词，用于补充或核对公开资料')` |
| max_results | `int` | `Field(default=4, ge=1, le=10, strict=True, description='最多返回几条摘要，范围1至10')` |

## 版本与切块常量

| 来源 | 常量 | 值 |
| --- | --- | --- |
| src/agentic_rag/research/service.py | READING_VERSION | `'reader-v2-sentence-selection-schema2'` |
| src/agentic_rag/research/service.py | SPECIALIST_VERSION | `'specialist-v4-tool-reread'` |
| src/agentic_rag/research/service.py | WORKFLOW_VERSION | `'research-workflow-v2-tools'` |
| src/agentic_rag/graph/nodes.py | ANALYSIS_PROMPT_VERSION | `'research-v2-reading-specialists-reread'` |
| src/agentic_rag/ingestion.py | CHUNK_SIZE | `800` |
| src/agentic_rag/ingestion.py | CHUNK_OVERLAP | `120` |
| src/agentic_rag/ingestion.py | MANIFEST_VERSION | `1` |

## 研究模式精确边表

条件是路由函数返回值；具体预算和终止规则仍需读 nodes/service。

| 从 | 条件 | 到 |
| --- | --- | --- |
| START | 直接 | initialize_research |
| initialize_research | 直接 | route |
| route | 直接 | recall_memory |
| recall_memory | 直接 | collect_sources |
| collect_sources | 直接 | read_documents |
| read_documents | 直接 | grade_documents |
| dispatch_specialists | generate | generate |
| dispatch_specialists | supplement | supplement_sources |
| grade_documents | 直接 | assess_evidence |
| assess_evidence | generate | dispatch_specialists |
| assess_evidence | supplement | supplement_sources |
| supplement_sources | 直接 | read_documents |
| generate | 直接 | evaluate_generation |
| revise_answer | 直接 | evaluate_generation |
| evaluate_generation | finish | END |
| evaluate_generation | supplement | supplement_sources |
| evaluate_generation | revise | revise_answer |

## 基础模式精确边表

条件是路由函数返回值；具体预算和终止规则仍需读 nodes/service。

| 从 | 条件 | 到 |
| --- | --- | --- |
| START | 直接 | route |
| route | 直接 | collect_sources |
| collect_sources | 直接 | grade_documents |
| grade_documents | 直接 | assess_evidence |
| assess_evidence | generate | generate |
| assess_evidence | supplement | supplement_sources |
| supplement_sources | 直接 | grade_documents |
| generate | 直接 | evaluate_generation |
| revise_answer | 直接 | evaluate_generation |
| evaluate_generation | finish | END |
| evaluate_generation | supplement | supplement_sources |
| evaluate_generation | revise | revise_answer |
