"""新闻混合检索：API 负责召回，向量缓存负责语义排序。"""

import logging
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import settings
from .news_api import NewsAPIError, NewsClient
from .news_plan import clean_queries
from .news_index import (
    SyncStats,
    get_embedding_model,
    open_news_collection,
    search_news,
    upsert_news_items,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NewsRetrievalResult:
    items: list[dict[str, Any]]
    vector_cache_used: bool = False
    cache_stats: SyncStats | None = None
    api_failed: bool = False
    api_error: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)


def _article_id(item: dict[str, Any]) -> str:
    return str(item.get("article_id") or "").strip()


def _diverse_results(items: list[dict[str, Any]], result_k: int) -> list[dict[str, Any]]:
    """优先覆盖各查询因素，避免价格新闻挤掉产能或成本新闻。"""
    queries = dict.fromkeys(q for item in items for q in item.get("retrieval_queries", []))
    selected = []
    for query in queries:
        if any(query in item.get("retrieval_queries", []) for item in selected):
            continue
        match = next((item for item in items if query in item.get("retrieval_queries", [])), None)
        if match is not None and match not in selected:
            selected.append(match)
    selected.extend(item for item in items if item not in selected)
    return selected[:result_k]


def _rank_and_cache(
    items: list[dict[str, Any]],
    semantic_query: str,
    section: str,
    result_k: int,
) -> tuple[list[dict[str, Any]], bool, SyncStats | None]:
    """缓存候选新闻向量，并在有明确语义查询时重新排序。"""
    if not settings.news_vector_cache_enabled or not items:
        return _diverse_results(items, result_k), False, None
    try:
        collection = open_news_collection()
        model = get_embedding_model()
        stats = upsert_news_items(items, collection, model.embed_documents)
        if not semantic_query.strip():
            return items[:result_k], True, stats

        candidate_ids = [_article_id(item) for item in items if _article_id(item)]
        if not candidate_ids:
            return items[:result_k], True, stats
        matches = search_news(
            semantic_query,
            collection,
            model.embed_query,
            k=len(candidate_ids),
            section=section,
            article_ids=candidate_ids,
        )
        by_id = {_article_id(item): item for item in items}
        ranked = [by_id[match["article_id"]] for match in matches if match["article_id"] in by_id]
        ranked_ids = {_article_id(item) for item in ranked}
        ranked.extend(item for item in items if _article_id(item) not in ranked_ids)
        return _diverse_results(ranked, result_k), True, stats
    except Exception:
        logger.warning("新闻向量缓存或重排失败，改用 API 原始顺序", exc_info=True)
        return _diverse_results(items, result_k), False, None


def _cached_fallback(query: str, section: str, result_k: int) -> list[dict[str, Any]]:
    """新闻服务器暂时不可用时，尝试读取已有的本机新闻向量缓存。"""
    if not settings.news_vector_cache_enabled or not query.strip():
        return []
    try:
        collection = open_news_collection()
        if collection.count() == 0:
            return []
        model = get_embedding_model()
        matches = search_news(
            query,
            collection,
            model.embed_query,
            k=result_k,
            section=section,
        )
    except Exception:
        logger.warning("本机新闻向量缓存回退失败", exc_info=True)
        return []
    return [
        {
            **(match.get("metadata") or {}),
            "article_id": match["article_id"],
            "summary": match.get("text") or "",
        }
        for match in matches
    ]


def retrieve_news(
    *,
    semantic_query: str,
    api_query: str = "",
    api_queries: list[str] | None = None,
    start: str = "",
    end: str = "",
    section: str = "",
    client: NewsClient | None = None,
    candidate_k: int | None = None,
    result_k: int | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> NewsRetrievalResult:
    """每页读取最多 20 条 API 候选，再用向量缓存重排并读取入选正文。"""
    candidate_limit = candidate_k or settings.news_candidate_k
    result_limit = result_k or settings.news_retrieval_k
    if candidate_limit < result_limit:
        candidate_limit = result_limit
    news_client = client or NewsClient()
    events: list[dict[str, Any]] = []

    def emit(event: dict[str, Any]) -> None:
        events.append(event)
        if on_event is not None:
            on_event(event)

    queries = clean_queries(api_queries if api_queries is not None else [api_query]) or [""]
    items_by_id = {}
    errors = []
    raw_count = 0
    for query_index, query in enumerate(queries):
        remaining = candidate_limit - raw_count
        if remaining <= 0:
            break
        quota = max(1, remaining // (len(queries) - query_index))

        def query_event(event):
            emit({**event, "query_number": query_index + 1, "query_total": len(queries),
                  "candidate_limit": candidate_limit})

        try:
            for item in news_client.iter_news(
                q=query, start=start, end=end, section=section,
                page_size=20, max_items=quota, include_content=False, on_event=query_event,
            ):
                raw_count += 1
                key = _article_id(item) or str(item.get("canonical_url") or item.get("title") or item)
                stored = items_by_id.setdefault(key, {**item, "retrieval_queries": []})
                if query not in stored["retrieval_queries"]:
                    stored["retrieval_queries"].append(query)
        except (NewsAPIError, ValueError) as exc:
            errors.append(str(exc))
            emit({"kind": "news_error", "message": str(exc), "partial_count": len(items_by_id)})
    items = list(items_by_id.values())
    api_error = "；".join(errors)
    if api_error:
        if not items:
            # 现有向量回退未按发布时间检索，不能把它当成满足日期条件的结果。
            cached = [] if start or end else _cached_fallback(semantic_query or api_query, section, result_limit)
            emit({
                "kind": "news_cache_fallback", "count": len(cached),
                "date_restricted": bool(start or end),
            })
            return NewsRetrievalResult(
                items=cached, vector_cache_used=bool(cached), api_failed=True,
                api_error=api_error, events=events,
            )

    ranked, cache_used, stats = _rank_and_cache(
        items,
        semantic_query or api_query,
        section,
        result_limit,
    )
    emit({
        "kind": "news_ranked", "candidate_count": len(items),
        "raw_count": raw_count,
        "selected_count": len(ranked), "vector_cache_used": cache_used,
        "cache_stats": asdict(stats) if stats is not None else None,
    })

    def read_article(item):
        article_id = _article_id(item)
        if not article_id:
            return item, None
        try:
            full_item = news_client.article(article_id)
            return {**item, **full_item, "_content_scope": "full" if full_item.get("content") else "summary"}, None
        except (NewsAPIError, ValueError, AttributeError) as exc:
            return {**item, "_content_scope": "summary"}, {"kind": "news_article_error", "article_id": article_id, "message": str(exc)}

    selected = []
    with ThreadPoolExecutor(max_workers=max(1, settings.io_concurrency)) as pool:
        for item, event in pool.map(read_article, ranked):
            selected.append(item)
            if event:
                emit(event)
    return NewsRetrievalResult(
        items=selected,
        vector_cache_used=cache_used,
        cache_stats=stats,
        api_failed=bool(api_error),
        api_error=api_error,
        events=events,
    )
