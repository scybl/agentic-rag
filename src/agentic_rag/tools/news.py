"""新闻工具适配层：模型参数、既有混合检索服务和统一证据之间的边界。"""

from datetime import date
from typing import Annotated

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import Field, StringConstraints, model_validator

from ..news_retrieval import retrieve_news
from .contracts import EvidenceBundle, Query, ToolInput, context_from


Keyword = Annotated[str, StringConstraints(strip_whitespace=True, max_length=200)]


class NewsSearchInput(ToolInput):
    semantic_query: Query = Field(description="原始研究问题，用于候选新闻的语义排序")
    queries: list[Keyword] = Field(min_length=1, max_length=4, description="1至4组分别查询的短主题词；只传一个空字符串表示不限关键词")
    start: str = Field(default="", description="已确认的新闻起始日 YYYY-MM-DD；空字符串表示不限制")
    end: str = Field(default="", description="已确认的新闻结束日 YYYY-MM-DD；空字符串表示不限制")
    section: str = Field(default="", max_length=100, description="用户指定栏目；空字符串表示不限")

    @model_validator(mode="after")
    def valid_scope(self):
        for value in (self.start, self.end):
            if value and date.fromisoformat(value).isoformat() != value:
                raise ValueError("新闻日期必须使用 YYYY-MM-DD")
        if self.start and self.end and self.start > self.end:
            raise ValueError("新闻起始日不能晚于结束日")
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
    })


@tool("search_news", args_schema=NewsSearchInput, response_format="content_and_artifact")
def search_news(semantic_query: str, queries: list[str], config: RunnableConfig,
                start: str = "", end: str = "", section: str = "") -> tuple[str, EvidenceBundle]:
    """按短关键词和指定日期检索私有新闻，候选向量排序后读取入选正文。

    每页最多20条，候选和入选总量受配置约束；不会自动添加时间限制。
    保留部分失败和摘要降级信息，不把 API 错误等同于没有相关新闻。
    """
    result = retrieve_news(semantic_query=semantic_query, api_query=queries[0], api_queries=queries,
                           start=start, end=end, section=section, on_event=context_from(config).emit)
    documents = [doc for item in result.items if (doc := news_document(item)) is not None]
    warnings = [result.api_error or "新闻 API 查询失败"] if result.api_failed else []
    if any(event.get("kind") == "news_article_error" for event in result.events):
        warnings.append("部分文章正文读取失败，已明确使用摘要证据")
    return EvidenceBundle(documents=documents, warnings=warnings, events=result.events).as_response()
