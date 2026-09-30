"""由 DuckDuckGo 提供支持的网络搜索数据源（无需 API 密钥）。"""

import logging
import re

from ddgs import DDGS
from langchain_core.documents import Document
from langchain_core.tools import ToolException, tool
from pydantic import Field

from .contracts import EvidenceBundle, Query, ToolInput

logger = logging.getLogger(__name__)


def date_hint(item):
    """搜索服务的日期和摘要日期仅作线索，不冒充核实过的网页发布时间。"""
    value = item.get("date") or item.get("published_at")
    if value:
        return str(value)[:80]
    match = re.match(r"\s*((?:[A-Z][a-z]{2,8}\s+\d{1,2},\s+20\d{2})|(?:20\d{2}[-年/]\d{1,2}[-月/]\d{1,2}日?))", item.get("body", ""))
    return match.group(1) if match else ""


class WebSearchInput(ToolInput):
    query: Query = Field(description="公开网络搜索词，用于补充或核对公开资料")
    max_results: int = Field(default=4, ge=1, le=10, strict=True, description="最多返回几条摘要，范围1至10")


@tool("search_web", args_schema=WebSearchInput, response_format="content_and_artifact")
def search_web(query: str, max_results: int = 4) -> tuple[str, EvidenceBundle]:
    """搜索公开网页，返回带网址的搜索摘要，不保证获取网页全文。

    用于私有新闻之外的公开核验与补充，不替代本地知识库或私有新闻接口。
    网络故障明确报错，由调用方保留其他来源的成功结果。
    """
    try:
        from ..research.runtime import io_capacity
        with io_capacity().slot():
            results = DDGS(timeout=20).text(query, max_results=max_results)
    except Exception as exc:
        logger.warning("Web search failed for query: %s", query, exc_info=True)
        raise ToolException(f"公开搜索失败（{type(exc).__name__}），不是检索零结果") from exc
    documents = [
        Document(
            page_content=item.get("body", ""),
            metadata={"source": item.get("href", ""), "title": item.get("title", ""),
                      "date_hint": date_hint(item),
                      "content_kind": "search_snippet", "source_type": "web_search"},
        )
        for item in results
        if item.get("body")
    ]
    return EvidenceBundle(documents=documents).as_response()
