# 工具层的边界与实际调用

当前工具层包含四种取证能力和四种确定性规则工具。它们使用真正的 LangChain `@tool`、参数 Schema 和 `ToolMessage`，已经接入首次检索、后续补搜、专题回读、模型输出校验、自适应重试、概率证据检查和答案交付检查，不是装饰性的演示代码。

本文解释边界；输入声明和注册信息由[代码事实参考](reference.md)随源码生成，运行操作见[运行指南](operations.md)，待修问题集中在[当前状态](status.md)。

主流程仍是受控的 LangGraph 工作流：模型输出结构化计划，程序根据计划执行已注册工具。现在没有改成模型原生 `tool_calls` 驱动的自由 ReAct 循环，也没有因此增加一次工具选择模型调用。

## 哪些能力值得抽出来

| 工具 | 负责什么 | 谁实际调用 | 明确不负责什么 |
|---|---|---|---|
| `search_knowledge` | 检索稳定知识与方法，返回原始文档片段 | `retrieve` 节点 | 最新新闻、答案生成、索引增量策略 |
| `search_news` | 校验新闻参数类型/日期，执行实际查询组，按时间、栏目与来源过滤，返回候选评分及预算内正文 | `news_api` 节点，首次收集和补搜共用 | 自行放宽条件、从人物/主题标签自动另造查询 |
| `search_web` | 获取公开网页摘要并保留网址，明确报告网络失败 | `web_search` 节点，首次收集和补搜共用 | 声称摘要是网页全文 |
| `read_news` | 回读当前证据中的已保存新闻版本，给出精确位置 | 专题请求，或主流程概率证据不足时 | 任意路径访问、重新抓取、无限回读 |
| `validate_model_output` | 按目标 Pydantic Schema 或非空文本契约检查每次模型返回，列出字段路径和原因 | 所有模型链由程序强制调用 | 生成或猜测缺失字段、替代语义质量评审 |
| `plan_model_retry` | 根据失败类型、已用 token、上下文余量和硬上限选择扩容、延时、关闭思考、切分或停止 | 模型失败后由统一反馈环调用 | 无限提高资源、掩盖客户端/程序错误 |
| `validate_probability_evidence` | 检查概率问题是否具有数值、方法和非搜索摘要的直接证据 | `assess_evidence` 节点强制调用 | 用宏观观点代替概率数据、生成概率 |
| `validate_probability_answer` | 检查事件、时点、0–100%数值、方法、引用和证据契约 | `evaluate_generation` 节点强制调用 | 替代事实核验或决定宏观逻辑是否合理 |

八项工具都有明确输入和可观察结果。输出校验及概率契约由程序强制执行，自适应工具只在模型失败后执行；模型不能选择跳过应执行的规则。它们只处理适合确定性判断的契约，复杂的证据解释仍由模型评审。调度器、阅读成果入库、向量补写和一般相关性评估没有为了凑数量注册成工具。

工具目录同时维护每项工具的默认调用者和适用条件。规划完成后，程序会为全部八项工具生成 `required / conditional / skipped` 决策；终端逐项解释。模型输出校验始终是 `required`，自适应重试是 `conditional`，其余不适用工具不会为了提高调用数量而运行，但不能静默消失。`--list-tools` 也会显示这些调用策略。

## 一次问题实际怎样经过工具

1. 每个模型链声明目标输出类型；结构化链另声明允许/必填字段和“禁止额外字段”，通过 Ollama 原生 `json_schema` 约束生成。非截断响应由 `validate_model_output` 独立复核，文本链检查非空文本，不要求答案本身是 JSON。
2. 结构错误、空输出、截断、请求超时及可恢复服务故障进入 `plan_model_retry`。工具根据实际输入/输出 token 与硬上限选择反馈修正、临时扩容、延长超时、关闭思考、切分输入或停止；半截长输出不会回塞扩大下一次上下文。
3. `route` 用通过校验的结构化输出选择数据源、查询词、日期和核对因素。规划提示中的工具目录由真实工具定义生成，避免描述与代码各写一套。
4. `collect_sources` 在有限线程池中调用原有三个来源节点。节点只翻译输入与写回状态，不再处理新闻记录转换或直接操作外部客户端。
5. `execute_tool` 先强制要求调用者和调用理由，再校验参数、创建调用编号，发送真正的 `ToolCall` 给对应 `BaseTool.invoke`。
6. 工具调用原有领域服务。分页、候选排序、缓存复用与网络并发仍由相应服务控制，不在新工具层重复实现。
7. 返回的 `ToolMessage.content` 是有界预览；`artifact` 保留完整 `Document`、原文片段、警告与新闻事件。图节点消费完整证据，不拿预览代替正文。
8. 需要补搜时仍经过同一工具入口。专题需要回读时，程序注入本次 E 编号到新闻版本的映射，执行 `read_news` 后把原文交给同一个专题继续分析。
9. 概率任务在生成前执行证据契约，在交付前执行答案契约。搜索摘要、宏观新闻或机构方向观点不能单独让概率问题通过。
10. 政策利率概率题会强制规划三种取证工具。初次概率证据契约未通过且已有持久化新闻版本时，主流程先执行 `read_news` 定向回读概率数值、方法和数据时点，再重新执行证据契约；仍不足才进入联网补搜。

`tool_call_id` 是本项目执行器生成的真实调用关联号，不是模型输出的思维过程。它把开始、API 中间事件、完成或失败串起来；不能单凭这个字段宣称实现了通用 ReAct。

## 目录职责

- [tools/__init__.py](../src/agentic_rag/tools/__init__.py)：取证工具与规则工具的显式目录，以及数据源到工具的映射。没有插件扫描或动态注册框架。
- [tools/contracts.py](../src/agentic_rag/tools/contracts.py)：公共输入规则、程序注入的 `ToolContext`、完整证据、规则结果与模型重试计划。
- [tools/execution.py](../src/agentic_rag/tools/execution.py)：一次调用、真实工具消息与事件。不负责重试、调度或改写用户目标。
- [tools/knowledge.py](../src/agentic_rag/tools/knowledge.py)、[news.py](../src/agentic_rag/tools/news.py)、[web_search.py](../src/agentic_rag/tools/web_search.py)、[news_reading.py](../src/agentic_rag/tools/news_reading.py)：各取证能力的输入约束和实现适配。
- [tools/guardrails.py](../src/agentic_rag/tools/guardrails.py)：模型输出检查、自适应重试决策、概率任务识别及两项概率契约；只做确定性控制，不生成研究结论。
- [graph/nodes.py](../src/agentic_rag/graph/nodes.py)：编排与状态更新。`_search_source` 是三个检索节点共用的薄适配。
- [research/service.py](../src/agentic_rag/research/service.py)：派发任务、阅读复用与专题结果汇总；回读已经交给工具。

句子编号是阅读与回读都需要的纯文本处理，所以移到 [evidence.py](../src/agentic_rag/evidence.py)。仅共享这项真实重复需求，没有再建一个宽泛的 `utils` 层。

## 参数约束与资源边界

所有公开输入禁止额外字段。新闻支持1–4组短关键词，每组最多8个AND词、200字符，栏目最多80字符；规划、工具与HTTP客户端共用校验。正文预算 `result_limit` 为1–100，显式候选 `candidate_limit` 为1–200（0表示未指定），`ranking_mode` 区分相关性与潜在影响。日期、来源等条件不能暗中放宽；只指定“最新N篇”时不自动添加日期窗口。

开放相关性检索在 `search_news` 内先取完摘要，再通过 `news_screening.py` 每批12篇进行轻量筛选；初筛与正文主题核对结果按模型、规则、问题、材料指纹缓存。正文缺失、重复、主题不符时在同一候选池补位，终端记录补位原因和核心文章数。限定母集/影响排序保持原合同，不能悄悄用其他文章补足排名。正文数量控制核心规模；入选后的分段队列不再按总任务数截断。

人物/机构必须进入实际查询，主题是检索线索而非额外API过滤字段。普通查询在四组预算内补主题；明确最新数量时只用主主题建立候选集合，扩展查询公开标为未执行，不能混入母集。`coverage` 表示正文覆盖目标，不代表已经全文阅读。

网页条数限制为1–10，新闻每页最多20。未指定候选数时分页至结束；指定最新N篇时先校验元数据、去重并判断实质主题关系，达到N篇相关候选即停止（最后一页可能多返回少量，只有选定N篇进入最终比较与候选向量缓存）。N小于20时按N条取页，筛选批次最多取尚缺数量，不再为了凑20条或5条评分批次多扫历史；不足则继续翻页。同名词误命中的排除理由与评分仍留档。扫描2000条仍不足时显式报错；不会拿无关项凑数。`retrieval_complete` 表示请求范围完成，`all_matches_exhausted` 单独标明是否遍历全库匹配；不混淆这两个概念。下一页失败仍处理前页已取回但待评分的候选，保持未完成标记。候选、评分、入选状态通过artifact和事件保存。原文回读仍限两个E编号、1600字符、一轮。工具输入不暴露密钥、数据库路径或任意URL。

阅读优先级采用 70% 向量相关性、20% 标题/摘要 TF-IDF 相似度、10% 查询组覆盖率；没有可用向量时改为 90% TF-IDF、10% 覆盖率，不退回接口顺序。语义距离转换为 `1/(1+max(0,distance))`，这些权重是可审计启发式，不是经过标定的热度、可信度或概率。相关性排序会兼顾各查询组；明确要求最新/最早时尊重时间排序。正文阅读总量仍受后续研究预算约束，检索覆盖说明进入评估与生成上下文，不能声称已阅读全部候选。

影响排序不使用上述相似度作为影响分：模型给每条候选的幅度、范围、持续性、信息增量、证据强度各0–4分，按30%/20%/20%/15%/15%转换成100分，保存方向、传导机制和支持字段编号，程序回填原文。每批最多5篇，受模型槽位限制；模型/规则/问题/材料指纹相同才复用评分。评分失败不能回退成“相似度影响排名”。这是基于摘要的潜在影响评估，不是事件研究法算出的市场贡献。嵌套校验与重试事件保留独立工具编号和父调用编号。

`ToolContext` 由程序通过 `RunnableConfig` 注入，不进入模型看到的 Schema。它强制携带 `caller` 和 `reason`，并提供事件出口、模型输出目标 Schema，以及回读所需的证据映射和版本读取函数，不提供整个 GraphState。缺少调用者或理由的调用会在工具执行前被拒绝。这是当前单用户流程中的能力边界，不等于多租户鉴权或操作系统沙箱。

执行器先继承当前 `RunnableConfig`，再合并工具上下文，保留流写入器、检查点命名空间与回调。不能用只有 `tool_context` 的新配置覆盖，否则图内补搜的事件可能抛出 `checkpoint_ns` 错误；测试覆盖 SQLite 恢复后的真实自定义流路径。

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

工具执行器本身不重试；模型反馈环集中处理输出与可恢复传输故障，客户端错误和未知错误不会靠扩大资源掩盖。任务调度器仍负责进程级失败与持久恢复。事件只记录有界参数，不记录认证请求头；它仍不是完整的隐私脱敏与审计系统。

自适应尝试次数由 `LLM_ADAPTIVE_MAX_ATTEMPTS` 控制；动态输出、上下文和单次请求时间分别受 `LLM_ADAPTIVE_MAX_OUTPUT_TOKENS`、`LLM_ADAPTIVE_MAX_CONTEXT_WINDOW`、`LLM_ADAPTIVE_MAX_REQUEST_TIMEOUT` 限制。它们是硬上限，不是每次请求的固定配置。批量研究的 `RESEARCH_TIMEOUT` 仍是可恢复调度边界，不会被单个模型自行延长。

模型链实际尝试上限取 `LLM_MAX_ATTEMPTS` 与自适应次数的较小值。低于首轮参数的 adaptive 值不会把已配置首轮请求压小；参数组合需一起调整。规则的概率检测主要依赖关键词、材料类型和格式，不校准概率或证明统计口径一致；详见状态页。

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

加 `-v` 时可见如下详细事件格式；默认视图保留调用者/理由、结果、异常、查询条件和入选评分，但不逐行展开全部参数。下面只演示字段，并非一次真实新闻执行记录：

```text
[工具 search_news · 调用编号]
  调用者：主流程/首次证据收集
  使用原因：按研究计划取得近期新闻事实
  调用参数：{"semantic_query":"...","queries":["马斯克 AI"],"people":["马斯克"],"organizations":[],"topics":["AI"],"source_names":["路透"],"published_after":"2026-09-29T12:00:00+08:00","published_before":"2026-09-30T12:00:00+08:00","section":"科技","sort_by":"newest","coverage":"broad","result_limit":10}
[工具 search_news · 相同调用编号]
  结果：成功/正常零结果/降级返回/失败；证据 N 条；耗时 T 秒
```

`agentic-rag --status RUN_ID` 查看摘要，`--status RUN_ID -v` 查看落盘完整事件。搜索事件在图线程输出；专题回读的事件还带所属任务标识，避免工作线程直接抢写终端。工具计数包含规则工具，不等于模型请求次数，也不等于外部搜索次数。

在 Python 中单独调用新闻工具的最小例子如下。它会访问你配置的新闻服务，不会调用问答模型：

```python
from agentic_rag.tools import search_news
from agentic_rag.tools.execution import execute_tool

from agentic_rag.tools.contracts import ToolContext

message = execute_tool(search_news, {
    "semantic_query": "猪肉价格的供需因素",
    "queries": ["生猪", "饲料"],
    "people": [],
    "organizations": [],
    "topics": ["生猪", "饲料"],
    "source_names": [],
    "start": "",
    "end": "",
    "published_after": "",
    "published_before": "",
    "section": "",
    "sort_by": "relevance",
    "coverage": "focused",
    "result_limit": 5,
}, context=ToolContext(caller="手工诊断", reason="核对新闻工具返回结构"))
print(message.tool_call_id)
print(message.content)  # 有界预览
for document in message.artifact.documents:
    print(document.metadata, document.page_content)
```

直接调用 `tool.invoke(普通参数字典)` 通常只返回 content；需要 `ToolMessage` 和 artifact 时，应传完整 ToolCall，或使用上述统一入口。`read_news` 还需要程序注入上下文，不能只凭一个 E 编号在所有研究中随意读取。

## 验证范围与兼容性

[test_tools.py](../tests/test_tools.py) 覆盖 Schema、参数拒绝、真实 ToolMessage、完整 artifact、错误与零结果区分、原文白名单和偏移、上下文并发隔离、三来源节点接入、补搜、日志落盘和 CLI 工具目录；专题集成测试验证回读确实调用工具且成果仍可复用。工具升级阶段曾完成 `133 passed, 3 subtests passed`，其中新增 27 项工具测试；这是历史验证，不是随后的最新测试数。当前回归请运行 `python -m pytest -q`。外部 API、模型和搜索服务使用替身，不以离线测试证明真实新闻覆盖率或模型质量。

最新候选词法评分新增 `scikit-learn` 依赖，更新代码后执行 `python -m pip install -e ".[dev]"`。不要覆盖 `.env`，无需仅因工具/日志/排序更新重建知识索引或清空阅读成果。模型反馈、阅读/专题规则与全候选检索已升级版本：不兼容的未完成研究需重新提问；原文与兼容阅读成果仍可复用，完成记录不会删除。版本常量以代码参考为准。

参考：[LangChain 官方工具文档](https://docs.langchain.com/oss/python/langchain/tools)。本项目采用已安装的 `langchain_core.tools.tool` 和 `RunnableConfig`，未为了使用较新的运行时包装而另加一套 Agent 框架。
