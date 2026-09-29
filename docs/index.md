# 项目文档首页

这里是给人阅读的说明，不作为默认 RAG 知识库。所有“当前实现”均以仓库代码为依据；本地 `.env` 可以覆盖默认值，文档不展示个人密钥或假设个人配置。历史验证单独标注，不冒充最新版实测。

## 从哪里开始

| 你想做什么 | 先读什么 |
|---|---|
| 安装、提问、换环境、更新知识 | [运行与维护](operations.md) |
| 理解嵌入、Chroma 和 RAG | [基础概念](rag_fundamentals.md) |
| 按代码学习完整执行链 | [技术详解](技术细节详解.md) |
| 理解为什么这样拆分 Agent | [模式与取舍](agentic_rag_patterns.md) |
| 理解四个工具的实际用途 | [工具层](tools-guide.md) |
| 理解存储、并发、阅读复用和恢复 | [研究记忆指南](research-memory-guide.md) |
| 理解预测为何不能只复述资料 | [预测流程](forecast-flow-update.md) |
| 看命令、配置、工具签名、图的准确边表 | [代码事实参考（自动生成）](reference.md) |
| 看已完成什么、还缺什么 | [当前状态与已知问题](status.md) |
| 理解自动测试和效果评估区别 | [RAGAS 评估](evaluation_ragas.md) |
| 面试逐题对照 | [45 道面试题 HTML](Agent求职面试题与参考答案.html) |
| 修改代码后更新说明 | [文档维护约定](documentation-maintenance.md) |

## 两张图分别讲什么

- [研究模式 SVG](agentic-rag-research-flow.svg)：默认模式，8 个学习阶段，合并展示相关性、充分性和专题分析。
- [基础模式 SVG](agentic-rag-core-architecture.svg)：`RESEARCH_ENABLED=false` 的当前流程，不再是旧版本架构存档。

它们是学习总览，不包含全部函数。精确分支由 `graph/build.py` 生成在代码参考中；修改图拓扑时必须人工复核两图。

## 文档和代码如何保持同步

`python scripts/docs_sync.py --check` 校验生成参考、相对链接、HTML 代码定位、SVG 结构，以及代码/文档是否有未确认变动。检查已纳入 pytest 和独立 CI 工作流，无需启动模型。

参数与拓扑自动抽取，设计理由人工维护。自动检查不能证明所有中文解释永远正确，因此源码变动会要求重新审阅，而不是自动盖上“已最新”的标签。
