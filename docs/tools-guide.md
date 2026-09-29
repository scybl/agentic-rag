# 工具层的边界与实际调用

当前工具层暴露四种取证能力。它们使用真正的 LangChain `@tool`、参数 Schema 和 `ToolMessage`，已经接入首次检索、后续补搜与专题回读，不是装饰性的演示代码。

本文解释边界；输入声明和注册信息由[代码事实参考](reference.md)随源码生成，运行操作见[运行指南](operations.md)，待修问题集中在[当前状态](status.md)。

主流程仍是受控的 LangGraph 工作流：模型输出结构化计划，程序根据计划执行已注册工具。现在没有改成模型原生 `tool_calls` 驱动的自由 ReAct 循环，也没有因此增加一次工具选择模型调用。

## 哪些能力值得抽出来

| 工具 | 负责什么 | 谁实际调用 | 明确不负责什么 |
|---|---|---|---|
| `search_knowledge` | 检索稳定知识与方法，返回原始文档片段 | `retrieve` 节点 | 最新新闻、答案生成、索引增量策略 |
| `search_news` | 校验查询条件，调用既有混合新闻检索服务，转换成统一证据 | `news_api` 节点，首次收集和补搜共用 | 自行决定日期、再次设计分页和向量缓存 |
| `search_web` | 获取公开网页摘要并保留网址，明确报告网络失败 | `web_search` 节点，首次收集和补搜共用 | 声称摘要是网页全文 |
| `read_news` | 回读当前证据中的已保存新闻版本，给出精确位置 | 专题 Agent 请求回读时 | 任意路径访问、重新抓取、无限回读 |

这四项都有明确输入、可观察的取证结果，也可能由不同流程复用。调度器、阅读成果入库、向量补写、相关性评估与答案反思没有为了凑数量注册成工具：前几项是运行基础设施，后几项是模型判断和流程控制。

## 一次问题实际怎样经过工具

1. `route` 用结构化输出选择数据源、查询词、日期和核对因素。规划提示中的工具目录由真实工具定义生成，避免描述与代码各写一套。
2. `collect_sources` 在有限线程池中调用原有三个来源节点。节点只翻译输入与写回状态，不再处理新闻记录转换或直接操作外部客户端。
3. `execute_tool` 校验参数，创建调用编号，发送真正的 `ToolCall` 给对应 `BaseTool.invoke`。
4. 工具调用原有领域服务。分页、候选排序、缓存复用与网络并发仍由相应服务控制，不在新工具层重复实现。
5. 返回的 `ToolMessage.content` 是有界预览；`artifact` 保留完整 `Document`、原文片段、警告与新闻事件。图节点消费完整证据，不拿预览代替正文。
6. 需要补搜时仍经过同一工具入口。专题需要回读时，程序注入本次 E 编号到新闻版本的映射，执行 `read_news` 后把原文交给同一个专题继续分析。

`tool_call_id` 是本项目执行器生成的真实调用关联号，不是模型输出的思维过程。它把开始、API 中间事件、完成或失败串起来；不能单凭这个字段宣称实现了通用 ReAct。

## 目录职责

- [tools/__init__.py](../src/agentic_rag/tools/__init__.py)：四个工具的显式目录，以及数据源到工具的映射。没有插件扫描或动态注册框架。
- [tools/contracts.py](../src/agentic_rag/tools/contracts.py)：公共输入规则、程序注入的 `ToolContext`、完整证据 `EvidenceBundle`。
- [tools/execution.py](../src/agentic_rag/tools/execution.py)：一次调用、真实工具消息与事件。不负责重试、调度或改写用户目标。
- [tools/knowledge.py](../src/agentic_rag/tools/knowledge.py)、[news.py](../src/agentic_rag/tools/news.py)、[web_search.py](../src/agentic_rag/tools/web_search.py)、[news_reading.py](../src/agentic_rag/tools/news_reading.py)：各能力的输入约束和实现适配。
- [graph/nodes.py](../src/agentic_rag/graph/nodes.py)：编排与状态更新。`_search_source` 是三个检索节点共用的薄适配。
- [research/service.py](../src/agentic_rag/research/service.py)：派发任务、阅读复用与专题结果汇总；回读已经交给工具。

句子编号是阅读与回读都需要的纯文本处理，所以移到 [evidence.py](../src/agentic_rag/evidence.py)。仅共享这项真实重复需求，没有再建一个宽泛的 `utils` 层。

## 参数约束与资源边界

所有公开输入禁止额外字段。知识与网络查询不能为空，新闻支持 1–4 组短关键词；单独的空关键词表示查询最新新闻。日期允许留空，提供日期时校验真实日期与起止顺序，工具不会自己补默认回看期。

网页条数限制为 1–10；新闻每页最多 20 条，候选总量仍受配置限制。原文回读最多两个 E 编号、最多 1600 个原文字符；一次专题最多回读一轮仍由研究服务控制。工具输入不暴露 API Key、数据库路径、任意 URL 或版本号。

`ToolContext` 由程序通过 `RunnableConfig` 注入，不进入模型看到的 Schema。它只提供事件出口，以及回读所需的证据映射和版本读取函数，不提供整个 GraphState。这是当前单用户流程中的能力边界，不等于多租户鉴权或操作系统沙箱。

输入校验在执行前完成；原文工具先检查所有 E 编号，再开始读取。不同任务各自传入上下文，未使用全局可变的“当前研究”变量。

## 结果与错误不能混淆

| 观察状态 | 含义 | 后续处理 |
|---|---|---|
| `ok` | 正常取得证据 | 节点把完整证据写回状态 |
| `empty` | 请求正常完成但没有证据 | 可以继续核对查询、范围或补搜 |
| `degraded` | 有证据，但某查询失败或正文降级为摘要 | 保留证据并展示警告 |
| `error` | 领域服务返回失败且没有可用证据 | 不伪装成正常零结果 |
| `failed` 事件 | 参数、上下文或执行过程抛出异常 | 记录错误类型，来源收集层保留其他成功来源 |

本次修正了公开搜索吞掉异常并返回空列表的问题。现在网络故障会抛出明确的 `ToolException`，在终端和来源错误中可见。

工具执行器不额外重试，避免与模型和任务层叠加。既有模型重试对 HTTP 状态分类不足的问题仍待单独修复，不能把本次封装说成已经解决所有重试问题。事件只记录公开参数，不记录认证请求头或运行上下文；它仍不是完整的隐私脱敏与审计系统。

## 如何观察和单独验证

在项目 Conda 环境中查看注册工具及参数，不调用模型、不打开研究数据库：

```bat
conda activate agent
agentic-rag --list-tools
```

正常提问方式不变：

```bat
agentic-rag "分析近期生猪新闻对价格的影响"
```

终端会出现如下格式的真实运行事件，下面只演示字段，并非一次真实新闻执行记录：

```text
[工具 search_news · 调用编号]
  调用参数：{"semantic_query": "...", "queries": ["生猪"], "start": "", "end": "", "section": ""}
[工具 search_news · 相同调用编号]
  结果：成功/正常零结果/降级返回/失败；证据 N 条；耗时 T 秒
```

研究模式下可用原来的 `agentic-rag --status RUN_ID` 查询落盘的工具事件。搜索事件在图线程输出；专题回读的事件还带所属任务标识，避免工作线程直接抢写终端。

在 Python 中单独调用新闻工具的最小例子如下。它会访问你配置的新闻服务，不会调用问答模型：

```python
from agentic_rag.tools import search_news
from agentic_rag.tools.execution import execute_tool

message = execute_tool(search_news, {
    "semantic_query": "猪肉价格的供需因素",
    "queries": ["生猪", "饲料"],
    "start": "",
    "end": "",
    "section": "",
})
print(message.tool_call_id)
print(message.content)  # 有界预览
for document in message.artifact.documents:
    print(document.metadata, document.page_content)
```

直接调用 `tool.invoke(普通参数字典)` 通常只返回 content；需要 `ToolMessage` 和 artifact 时，应传完整 ToolCall，或使用上述统一入口。`read_news` 还需要程序注入上下文，不能只凭一个 E 编号在所有研究中随意读取。

## 验证范围与兼容性

[test_tools.py](../tests/test_tools.py) 覆盖 Schema、参数拒绝、真实 ToolMessage、完整 artifact、错误与零结果区分、原文白名单和偏移、上下文并发隔离、三来源节点接入、补搜、日志落盘和 CLI 工具目录；专题集成测试验证回读确实调用工具且成果仍可复用。工具升级阶段曾完成 `133 passed, 3 subtests passed`，其中新增 27 项工具测试；这是历史验证，不是随后的最新测试数。当前回归请运行 `python -m pytest -q`。外部 API、模型和搜索服务使用替身，不以离线测试证明真实新闻覆盖率或模型质量。

没有引入新依赖，不需修改 `.env`、重建知识索引或清空已有阅读成果。因专题回读契约变化，专题任务版本与工作流版本已升级：旧版本尚未完成的研究不继续混用新流程，请重新提交问题；原文与兼容的阅读成果仍可复用。完成的旧研究记录不会删除。

参考：[LangChain 官方工具文档](https://docs.langchain.com/oss/python/langchain/tools)。本项目采用已安装的 `langchain_core.tools.tool` 和 `RunnableConfig`，未为了使用较新的运行时包装而另加一套 Agent 框架。
