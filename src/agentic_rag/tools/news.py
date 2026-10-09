"""新闻工具适配层：模型参数、既有混合检索服务和统一证据之间的边界。"""

from datetime import date
from typing import Annotated, Literal

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import Field, StringConstraints, model_validator

from ..news_plan import parse_news_timestamp, validate_api_query
from ..news_retrieval import retrieve_news
from .contracts import EvidenceBundle, Query, ToolInput, context_from


Keyword = Annotated[str, StringConstraints(strip_whitespace=True, max_length=200)]


class NewsSearchInput(ToolInput):
    semantic_query: Query = Field(description="原始研究问题，用于候选新闻的语义排序")
    queries: list[Keyword] = Field(min_length=1, max_length=4, description="1至4组分别查询的短主题词；只传一个空字符串表示不限关键词")
    people: list[Keyword] = Field(default_factory=list, max_length=4, description="查询计划识别的人物约束")
    organizations: list[Keyword] = Field(default_factory=list, max_length=4, description="查询计划识别的机构或公司约束")
    topics: list[Keyword] = Field(default_factory=list, max_length=6, description="查询计划识别的主题线索；并非逐篇硬过滤条件")
    source_names: list[Keyword] = Field(default_factory=list, max_length=4, description="用户明确指定的信息源；为空则不限制")
    start: str = Field(default="", description="已确认的新闻起始日 YYYY-MM-DD；空字符串表示不限制")
    end: str = Field(default="", description="已确认的新闻结束日 YYYY-MM-DD；空字符串表示不限制")
    published_after: str = Field(default="", description="精确发布时间下界 ISO 8601，必须带时区")
    published_before: str = Field(default="", description="精确发布时间上界 ISO 8601，必须带时区")
    section: str = Field(default="", max_length=80, description="用户指定栏目；空字符串表示不限")
    sort_by: Literal["relevance", "newest", "oldest"] = Field(default="relevance", description="结果排序")
    coverage: Literal["focused", "broad", "exhaustive"] = Field(default="focused", description="正文覆盖目标；未指定candidate_limit时取完匹配候选")
    result_limit: int = Field(default=5, ge=1, le=100, description="用户要求的正文分析数量")
    candidate_limit: int = Field(default=0, ge=0, le=200, description="用户明确要求的候选数量；0表示未指定")
    ranking_mode: Literal["relevance", "impact"] = "relevance"

    @model_validator(mode="after")
    def valid_scope(self):
        for query in self.queries:
            validate_api_query(query, self.section)
        for value in (self.start, self.end):
            if value and date.fromisoformat(value).isoformat() != value:
                raise ValueError("新闻日期必须使用 YYYY-MM-DD")
        if self.start and self.end and self.start > self.end:
            raise ValueError("新闻起始日不能晚于结束日")
        after = parse_news_timestamp(self.published_after)
        before = parse_news_timestamp(self.published_before)
        if after and before and after > before:
            raise ValueError("精确新闻起始时间不能晚于结束时间")
        if len(self.queries) > 1 and "" in self.queries:
            raise ValueError("不限关键词不能与其他查询混用")
        return self


def news_document(item: dict) -> Document | None:
    """集中保留新闻身份、时间与全文/摘要标识，不把搜索摘要伪装成全文。"""
    title = str(item.get("title") or "").strip()
    content = str(item.get("content") or item.get("summary") or "").strip()
    if not title and not content:
        return None
    published_at = str(item.get("published_at") or "").strip()
    section = str(item.get("section") or "").strip()
    parts = [f"{label}：{value}" for label, value in (("标题", title), ("发布时间", published_at), ("栏目", section)) if value]
    if content:
        parts.append(f"正文：\n{content}")
    url = str(item.get("canonical_url") or item.get("url") or "").strip()
    source_name = str(item.get("source_name") or "news_api").strip()
    scope = item.get("_content_scope", "full" if item.get("content") else "summary")
    return Document(page_content="\n".join(parts), metadata={
        "source": url or source_name, "source_name": source_name, "title": title,
        "published_at": published_at, "section": section,
        "article_id": str(item.get("article_id") or ""), "url": url,
        "content_hash": str(item.get("content_hash") or ""), "source_type": "news_api",
        "content_kind": "news_article" if scope == "full" else "news_summary",
        "impact": item.get("impact", {}),
        "candidate_id": item.get("candidate_id", ""),
    })


@tool("search_news", args_schema=NewsSearchInput, response_format="content_and_artifact")
def search_news(semantic_query: str, queries: list[str], config: RunnableConfig,
                people: list[str] | None = None, organizations: list[str] | None = None,
                topics: list[str] | None = None, source_names: list[str] | None = None,
                start: str = "", end: str = "", published_after: str = "",
                published_before: str = "", section: str = "", sort_by: str = "relevance",
                coverage: str = "focused", result_limit: int = 5, candidate_limit: int = 0,
                ranking_mode: str = "relevance") -> tuple[str, EvidenceBundle]:
    """执行模型生成的完整新闻查询：主题/人物/机构、时间、栏目、来源、排序和数量。

    未指定候选数时取完摘要页；明确最新N篇时按日期顺序核对主题、时间与来源，收齐N篇后停止。
    不自动添加日期；受控扫描不足N篇时显式报告未完成。
    全部匹配候选及权重保留在 artifact 和研究事件，正文预算外的候选不是无关新闻。
    保留部分失败和摘要降级信息，不把 API 错误等同于没有相关新闻。
    """
    from ..news_screening import screen_news
    emit = context_from(config).emit
    result = retrieve_news(semantic_query=semantic_query, api_query=queries[0], api_queries=queries,
                           start=start, end=end, published_after=published_after,
                           published_before=published_before, section=section,
                           source_names=source_names or [], sort_by=sort_by,
                           result_k=result_limit,
                           candidate_limit=candidate_limit, ranking_mode=ranking_mode,
                           summary_screener=lambda items, stage: screen_news(items, semantic_query, stage=stage, emit=emit),
                           on_event=emit)
    documents = [doc for item in result.items if (doc := news_document(item)) is not None]
    warnings = [result.api_error or "新闻 API 查询失败"] if result.api_failed else []
    if any(event.get("kind") == "news_article_error" for event in result.events):
        warnings.append("部分文章正文读取失败；开放检索已从摘要候选补位，限定母集保留摘要标记，不能宣称全文已读")
    for event in result.events:
        if event.get("kind") == "news_core_ready" and event["selected"] < event["limit"]:
            warnings.append(f"核心正文仅获得 {event['selected']}/{event['limit']} 篇；候选已耗尽，不用无关或缺失正文凑数")
    return EvidenceBundle(documents=documents, warnings=warnings, events=result.events,
                          candidates=result.candidates, retrieval_complete=result.retrieval_complete).as_response()
