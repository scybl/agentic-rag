# 运行与维护

以下 Windows 命令示例用 **CMD / Anaconda Prompt**。PowerShell 执行带引号的程序路径才需要前缀 `&`；不要把它复制到 CMD。参数清单见[代码参考](reference.md)。

## 1. 用现有 Conda 环境

在项目根目录执行：

```bat
conda activate agent
python -c "import sys; print(sys.executable)"
python -m pip install -e ".[dev]"
```

确认解释器是你要用的 `agent` 环境。`-e` 是可编辑安装，修改项目源码后重启 CLI 即可加载；新增依赖或更改命令入口后需重新安装。

不用 Conda 时可创建虚拟环境：

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
```

仓库要求 Python 3.10+；依赖下限是声明，不表示每种旧版依赖组合都已验证。没有锁定全部依赖版本。

## 2. 准备模型和配置

仅首次且没有 `.env` 时复制模板：

```bat
if not exist .env copy .env.example .env
ollama pull qwen2.5:3b
ollama list
```

这是仓库示例模型，不强制替换已有模型。`LLM_MODEL` 必须匹配本机模型名称；Ollama 服务须可连接。嵌入模型首次使用可能需要下载，离线时应配置已有本地模型目录。

不要覆盖现有 `.env`。新增设置未填写时使用代码默认值；修改设置后重启进程。私有新闻需要自己的 `NEWS_API_KEY`，不配置时可使用知识和公开搜索，新闻源失败会被记录，并不保证路由自动避开它。

## 3. 放资料并初建

把供模型查阅的 `.md`、`.txt`、`.pdf` 放进 `knowledge/`：

```bat
python -m agentic_rag.ingestion
```

等价命令 `agentic-rag-ingest`。完成后退出，不会持续监听。`docs/` 里的项目说明不是默认检索材料。

## 4. 提问与查看工具

```bat
agentic-rag --list-tools
agentic-rag "什么是事件研究法？"
agentic-rag
```

交互模式在 `>` 后输入问题，`exit` / `quit` 或 Ctrl+C 退出；空输入也退出。入口不在 PATH 时用 `python -m agentic_rag.cli`。`pyproject.toml` 把命令映射到 `agentic_rag.cli:main`，不是运行一个配置文件。

每次问题是独立研究，不自动接上上一问的聊天上下文。终端展示计划、查询、日期、工具调用、任务进度、复用和评审摘要，不输出隐藏逐字思维链。`-v` 补充异常类型，查询信息平时已显示。

## 5. 文件变化后的增量同步

```bat
agentic-rag-ingest
```

处理新增/修改/删除，跳过未变文件，不是每次全库重建。非空索引不会仅因重启问答自动同步。

按需选择一种监听方式：

- 独立进程：`agentic-rag-ingest --watch --interval 2`，先同步，再轮询，Ctrl+C 停止。
- 问答后台：在 `.env` 设 `KNOWLEDGE_WATCH_ENABLED=true`，重启 CLI；默认是关闭。

二者共用增量逻辑，但跨进程没有知识索引写锁，不建议同时启动多个写入者。`--rebuild` 会重置当前配置集合再重建，仅确实需要时使用。普通文档解释更新不需执行它。

## 6. 研究恢复和成果检查

将占位编号替换成终端显示的真实值：

```bat
agentic-rag --runs
agentic-rag --status RUN_ID
agentic-rag --inspect-reading READING_ID
agentic-rag --resume RUN_ID
agentic-rag --resume RUN_ID --retry-failed
agentic-rag --repair-memory
```

`--runs`、`--status`、`--inspect-reading` 不调用对话模型；可能初始化/打开研究数据库。`--list-tools` 连研究数据库也不初始化。

`--resume` 完成任务展示保存结果，不更新新闻；CLI 仍可能先按配置预热模型。未完成任务若模型/阅读/工作流版本不兼容，应重新提问；兼容阅读成果继续复用。`--retry-failed` 重置失败子任务预算，从保存材料重新进入阅读，完成任务仍复用。

`--repair-memory` 补写向量，不重新阅读，可能加载嵌入模型；失败时停止，不能因命令返回就假定全部修好，应看 `failed` 和 `remaining`。不要删数据库来解除运行锁。

## 7. 新闻缓存的独立操作

```bat
python -m agentic_rag.news_index preview --max-items 20
python -m agentic_rag.news_index sync --max-items 1000 --batch-size 32
python -m agentic_rag.news_index search "生猪" --k 5
```

`preview` 只看接口数据；`sync` 把有限数量的标题/摘要写入候选向量缓存，不代表全部历史归档，不会执行阅读 Agent，也不创建定时任务。正常问答会缓存遇到的候选，不要求先批量同步。

## 8. 数据、备份与故障定位

默认 `.chroma/` 保存索引、原文、问题、任务、检查点和分析；不要提交 Git。路径可由配置改写，不能假定都在默认目录。备份时停止相关进程，成套保存研究库与检查点及需要的索引；不要只复制活动 SQLite 主文件而忽略 WAL。

故障按层定位：解释器/安装 → Ollama 服务与模型 → 本机连接/代理 → 嵌入模型与知识索引 → 新闻认证/接口 → 工具事件 → 证据与反思。

```bat
python scripts\diagnose_ollama.py
python -m pytest -q
python scripts\docs_sync.py --check
```

诊断脚本会连接配置的 Ollama；测试和文档检查无需真实新闻或模型服务。503 有多种原因，不能仅增加重试就断定解决。已知限制见[状态页](status.md)。
