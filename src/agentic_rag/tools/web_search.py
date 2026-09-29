"""由 DuckDuckGo 提供支持的网络搜索数据源（无需 API 密钥）。"""

import logging

from ddgs import DDGS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def search_web(query: str, max_results: int = 4) -> list[Document]:
    """搜索网络，并将搜索结果摘要作为 Document 返回。

    发生网络错误时返回空列表，使流程图仍能如实回答“不知道”，
    而不是直接崩溃。
    """
    try:
        from ..research.runtime import io_capacity
        with io_capacity().slot():
            results = DDGS(timeout=20).text(query, max_results=max_results)
    except Exception:
        logger.warning("Web search failed for query: %s", query, exc_info=True)
        return []
    return [
        Document(
            page_content=item.get("body", ""),
            metadata={"source": item.get("href", ""), "title": item.get("title", ""),
                      "content_kind": "search_snippet", "source_type": "web_search"},
        )
        for item in results
        if item.get("body")
    ]
