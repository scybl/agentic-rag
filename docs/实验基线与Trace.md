# 实验基线与 Trace：第一阶段实现

本文说明已实现的冻结财经样本协议、当前排序基线的可重复实验、运行产物校验、配对比较，以及研究日志到统一 Trace 的转换。检索适配器与 ranx 评分使用同一协议，详见[检索竞技场](检索竞技场.md)。尚未实现的上下文编译器和 HTML 报告不作为当前能力；现有累计时间与调用预算见[研究指南](research-memory-guide.md)。

## 1. 本阶段能展示什么

可以实际演示两个过程：对同一份材料与问题多次运行现有排序，证明输出、代码和输入可追溯；将一场持久研究导出为父子 Span 和事件，定位模型消耗、任务重试、复用以及未返回操作。

当前实验适配器 `current_tfidf` 直接调用生产代码 `news_ranking.rank_candidates` 的无向量降级分支：标题双权重，字符 1–3 gram TF-IDF，词法权重 0.9、查询覆盖权重 0.1。冻结语料中的全部文章都视为一次查询召回的候选，因此覆盖分为常量。实验保存完整排名，Top-k 只标记入选项，不删除剩余候选。

这是一条**候选排序基线**。它不调用 Ollama，不访问新闻 API，不建立 Chroma，不执行完整研究图，不裁判最终答案，也不测 API 的候选召回率。零对话模型 Token 来自实际没有调用模型，不能解释成完整研究零成本。向量混合分支和生产链路的查询多样性补位不在这一基线里。

## 2. 冻结样本协议

样本文件：[finance_smoke_v1/suite.json](../evaluation/suites/finance_smoke_v1/suite.json)。第一版是 11 篇合成材料、10 个工程回归问题，覆盖事实、时间敏感、同名实体、多跳、预测、概率、无答案、冲突、单位和未知日期。机构、产区和数值均为虚构，不是金融事实数据集。

每题记录问题类型、dev/test 标记、是否可回答、qrels、参考答案、关键证据、禁止结论、风险义务和人工评分规则。qrels 的 relevance 取 0–3；大于零表示材料相关，**相关不等于足以回答**。例如一篇利率评论可以相关，但不足以回答精确概率。无答案样本允许没有正相关材料，不能给它编一个正确答案。

协议拒绝重复材料/问题 ID、重复 qrels、悬空引用、非法相关性等级、无正例的可回答问题以及无时区日期。未知日期用 null 表示。快照整体哈希覆盖语料、问题和标注；改一个标签也会改变实验身份。

默认只跑 dev。当前 dev/test 共用合成语料，只演示隔离流程，不能据此宣称独立测试集的泛化性能。正式质量评测仍需真实冻结材料、人工复核和独立测试集。参考答案和风险标签现在用于保存评测契约，尚未自动评分。

## 3. 运行和比较

在项目根目录、已安装项目的 Python 环境执行，CMD 和 PowerShell 均可使用：

```text
python -m agentic_rag.evaluation validate evaluation/suites/finance_smoke_v1/suite.json
python -m agentic_rag.evaluation run evaluation/suites/finance_smoke_v1/suite.json --split all --repeat 3 --output evaluation/runs/demo-a
python -m agentic_rag.evaluation run evaluation/suites/finance_smoke_v1/suite.json --split all --repeat 3 --output evaluation/runs/demo-b
python -m agentic_rag.evaluation compare evaluation/runs/demo-a evaluation/runs/demo-b --output evaluation/reports/demo-comparison.json
```

输出目录必须不存在；再次实验请换新目录。显式导出/比较也拒绝覆盖已有输出文件。程序不会自动删旧结果。正常退出为 0，样本执行有失败为 1，配置/产物/导出错误为 2，中断为 130。

```text
evaluation/runs/demo-a/
├── suite.json          输入材料、问题和 qrels 的完整冻结副本
├── manifest.json       实验编号、配置、快照哈希、源码指纹与依赖版本
├── observations.json   每题每次运行的完整排序、耗时和错误类型
├── summary.json        样本数、失败数及观测耗时；质量指标暂为 null
└── traces/             每题每次运行的一份内部 Trace JSON
```

清单保存实际排序相关源码的内容指纹，即使有未提交代码也能区分实现；Git commit 只作辅助定位。记录 Python、平台、scikit-learn、NumPy、SciPy、Pydantic 和检索模型依赖版本，不读取 `.env` 或把密钥写入清单。本文的 `current_tfidf` 未使用模型和提示，`model_revision`、`prompt_version` 均为 null；第二阶段模型适配器则记录本地模型内容哈希。

每完成一题便以同目录临时文件和原子替换保存结果，避免半份 JSON。中断会保留已完成样本并将清单标为 interrupted，比较器拒绝将它作为完整实验。普通单题异常记录为失败并继续，失败样本保留在比较分母中；错误正文不保存，仅记录异常类型。

比较前逐项核对 suite、corpus、观察结果及 Trace 指纹，验证样本是否缺失/重复、排名是否连续、候选是否完整、入选标志是否符合 Top-k。失败样本的排名一致性为 null，不把“两次都失败”当成一致成功。

改变 Top-k 时需要明确声明实验变量：

```text
python -m agentic_rag.evaluation run evaluation/suites/finance_smoke_v1/suite.json --top-k 2 --output evaluation/runs/k2
python -m agentic_rag.evaluation run evaluation/suites/finance_smoke_v1/suite.json --top-k 4 --output evaluation/runs/k4
python -m agentic_rag.evaluation compare evaluation/runs/k2 evaluation/runs/k4 --allow-change top_k --output evaluation/reports/k-comparison.json
```

`sort_by` 和 `implementation` 也可显式声明；数据集、样本划分和重复次数必须一致。实现/依赖/平台变化默认阻止比较。允许的变化会完整写入报告，不能隐藏多变量变化。哈希用于发现意外漂移，不是数字签名，不能防止有人同时篡改数据和清单。

`compare` 报告比较完整排名、Top-k 入选结果与配对耗时，不输出质量或显著性结论；第二阶段新增的 `score` 使用 ranx 计算标准指标，`score --against` 执行配对检验。耗时含冷启动、操作系统调度和运行噪声，当前中位数不构成严谨性能优势证据。

## 4. 导出真实研究 Trace

先用现有命令 `agentic-rag --runs` 找到研究编号，再替换下面的 `YOUR_RUN_ID`。数据库参数指向实际研究数据库；仓库默认位置为 `.chroma/research.sqlite3`。

```text
python -m agentic_rag.evaluation export-trace --database .chroma/research.sqlite3 --run-id YOUR_RUN_ID --output evaluation/reports/research-trace.json
```

导出通过 SQLite `mode=ro` 打开，在同一只读事务中读取运行及全部事件；不会调用 ResearchStore 的建库/迁移逻辑，也不受终端“最近 100 条”的展示限制。不存在的库或研究会失败，不会悄悄创建空库。正在执行中的研究可导出一个时间点的快照，稍后新事件不会自动进入旧文件。

新事件可以形成以下关系：

```text
研究 run
└── 本次执行 session（恢复后新建另一个）
    └── 图节点 node
        ├── 工具调用 tool
        └── 任务尝试 task（完整任务键 + 尝试次数）
            ├── 模型调用 llm
            └── 工具调用 tool
```

终端短任务编号仍用于显示，Trace 关联使用完整键。不同节点/执行中的尝试不会混为一次，重复导出稳定生成相同 ID，同一记录重放不会重复计量。任务复用作为事件保存，不虚构一个模型调用。候选事件保留每个候选的排名、分数分解、权重和入选状态；为了避免 URL 查询参数泄露，候选 ID 在研究导出中转为稳定哈希。

`Span.elapsed_seconds` 使用原始记录的单调时钟测量；时间戳表示事件发生或旧库接收时间，两者有来源标记。没有耗时的旧记录保持 null，不用接收时间差补造。研究根节点不填整个“创建至结束”的时间为执行耗时，因为恢复间隔可能包含用户等待。结束前中断的调用保持 pending，不能算成功或零消耗。

遥测默认采用字段白名单，不导出用户问题正文、Prompt、模型回答/思考、工具参数、goal、新闻标题/正文或错误消息。保留问题哈希、模型/工作流版本（原记录存在时）、调用 ID、Token 数值、重试、状态和排名。旧运行缺少节点关联/模型版本/结束事件时只保留能证明的层级，不根据附近事件猜测。未知 Token 仍为 null，推理 Token 属于生成 Token 的分项，不能再次相加。

内部 `TraceExporter` 提供稳定接口，`JsonExporter` 已实现；`BestEffortExporter` 可作为运行时旁路出口，失败计数后不抛给业务层。当前真实研究采用事后只读导出，因此写报告失败不会触发研究重试。OpenTelemetry/OpenInference、Phoenix、检查点专属 Span 和单文件 HTML 报告尚未接入。

## 5. 代码位置和验证范围

| 位置 | 责任 |
|---|---|
| [evaluation/contracts.py](../src/agentic_rag/evaluation/contracts.py) | 冻结语料、题目、qrels、实验配置和哈希 |
| [evaluation/runner.py](../src/agentic_rag/evaluation/runner.py) | 复用既有排序，保留每题结果和版本 |
| [evaluation/compare.py](../src/agentic_rag/evaluation/compare.py) | 产物完整性和配对比较 |
| [telemetry/schema.py](../src/agentic_rag/telemetry/schema.py) | Span/Event v1 和父子关系校验 |
| [telemetry/adapters.py](../src/agentic_rag/telemetry/adapters.py) | 旧事件兼容、父子映射和字段白名单 |
| [telemetry/exporters.py](../src/agentic_rag/telemetry/exporters.py) | 只读事务、原子 JSON 导出和故障隔离出口 |
| [test_telemetry.py](../tests/test_telemetry.py) | 真实 LangGraph、SQLite、线程池、LangChain 回调的接线验证 |
| [test_experiment.py](../tests/test_experiment.py) | 输入拒绝、重复实验、中断、失败保留和产物漂移验证 |

测试涵盖并发任务归属、长任务键、失败后第二次尝试、恢复复用、独立运行隔离、结束事件先到、重复事件重放、旧记录缺失、字段脱敏、非法父子树以及原子写入失败。数据集 JSON 纳入文档审阅指纹；生成的 runs/reports 默认忽略，不随代码发布。

本阶段实际验收：同一批 10 题、每题 3 次，两次独立运行产生 30 对结果，排序变化为 0；另对本地已有研究完成只读导出。这个结果证明固定输入下的排序重复性和导出可用性，不证明新算法提升，因为本阶段尚未引入新排序算法。

后续检索对照沿用这些冻结输入和输出协议；真实13篇新闻、16题来源定位结果见[检索竞技场](检索竞技场.md)，20题真实模型配对验收见[评估说明](evaluation_ragas.md)。这些是开发集，并非独立双审金标准；代表性线上端到端质量仍需继续建设。
