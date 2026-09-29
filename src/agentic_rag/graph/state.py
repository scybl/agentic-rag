"""在 LangGraph 节点之间流转的共享状态。"""

from typing import Any, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    run_id: str
    model_revision: str
    reading_recipe: str
    workflow_revision: str
    research_documents: list[Document]
    research_started: float
    memory_documents: list[Document]
    reading_reports: list[dict[str, Any]]
    specialist_findings: list[dict[str, Any]]
    evidence_signature: str
    no_progress_rounds: int
    question: str  # 当前问题；主流程保留原意，具体检索词单独记录
    original_question: str  # 用户实际提出的原始问题
    datasource: str  # 兼容展示，例如 "news_api+vectorstore"
    selected_sources: list[str]  # 本轮需要联合查询的一个或多个来源
    source_queries: dict[str, str]  # 针对每个来源生成的专用查询
    task_type: str  # forecast / analysis / factual
    evidence_needs: list[str]  # 本题所需的具体证据因素
    query_history: list[str]  # 已实际执行的查询，避免无效重复
    document_grades: dict[str, bool]  # 同一会话内已筛选的文本指纹
    evidence_assessment: dict[str, Any]  # 整批证据的充分性、已知与缺口
    pending_news_queries: list[str]  # 针对缺口的待执行新闻查询
    pending_web_queries: list[str]  # 针对缺口的待执行网络查询
    next_action: str  # 补搜、生成、重写或结束
    revision_feedback: str  # 反思产生的可执行修正要求
    answer_revisions: int  # 答案重写次数，独立于补搜次数
    answer_review: dict[str, Any]  # 事实与任务完成情况的独立判定
    generation_complete: bool  # 是否真正回答了问题
    evidence_context: str  # 实际发送给生成器与核验器的证据片段
    plan_summary: str  # 可展示给用户的数据源选择说明（不是隐藏思维链）
    news_search_plan: dict[str, Any]  # 模型的时间建议，以及校验后实际采用的新闻查询条件
    news_trace: list[dict[str, Any]]  # 新闻请求、分页、缓存与失败的可核对记录
    documents_by_source: dict[str, list[Document]]  # 各来源分别返回的证据
    source_errors: dict[str, str]  # 单个来源失败时保留错误，不中断其他来源
    documents: list[Document]  # 为回答收集的证据
    answer_documents: list[Document]  # 受上下文预算限制、真正参与生成的证据
    generation: str  # 最终答案
    retries: int  # 已执行补搜轮次：研究模式 research_max_rounds，基础模式 max_retries
    analysis_cache_key: str  # 问题、证据和模型共同决定的缓存键
    analysis_cache_hit: bool  # 是否直接复用了已经核验过的分析
    generation_grounded: bool  # 最终回答是否通过证据依据检查
    generation_check: str  # 可展示给用户的核验结果摘要
