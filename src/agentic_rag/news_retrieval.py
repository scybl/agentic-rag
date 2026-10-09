"""新闻混合检索：API 负责召回，向量缓存负责语义排序。"""

import logging
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.runnables import RunnableLambda

from .config import settings
from .news_api import NewsAPIError, NewsClient
from .news_plan import clean_queries, news_metadata_matches, parse_news_timestamp
from .news_ranking import candidate_key, rank_candidates
from .news_index import (
    SyncStats,
    get_embedding_model,
    open_news_collection,
    prepare_record,
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
    candidates: list[dict[str, Any]] = field(default_factory=list)
    retrieval_complete: bool = True


def _article_id(item: dict[str, Any]) -> str:
    return str(item.get("article_id") or "").strip()


def _diverse_results(items: list[dict[str, Any]], result_k: int) -> list[dict[str, Any]]:
    """优先覆盖各查询因素，避免价格新闻挤掉产能或成本新闻。"""
    queries = dict.fromkeys(q for item in items for q in item.get("retrieval_queries", []))
    selected = []
    selected_keys = set()
    for query in queries:
        if len(selected) >= result_k:
            break
        if any(query in item.get("retrieval_queries", []) for item in selected):
            continue
        match = next((item for item in items if query in item.get("retrieval_queries", [])), None)
        if match is not None and match not in selected:
            selected.append(match)
            selected_keys.add(candidate_key(match))
    for item in items:
        if len(selected) >= result_k:
            break
        if candidate_key(item) not in selected_keys:
            selected.append(item)
            selected_keys.add(candidate_key(item))
    return selected


def _rank_and_cache(
    items: list[dict[str, Any]],
    semantic_query: str,
    section: str,
    sort_by: str = "relevance",
) -> tuple[list[dict[str, Any]], bool, SyncStats | None]:
    """全量候选写入索引后统一评分；分批查询不是候选截断。"""
    scores, stats, cache_used = {}, None, False
    if settings.news_vector_cache_enabled and items:
        try:
            collection = open_news_collection()
            model = get_embedding_model()
            stats = upsert_news_items(items, collection, model.embed_documents)
            candidate_ids = [_article_id(item) for item in items if prepare_record(item) is not None]
            if semantic_query.strip() and candidate_ids:
                vector = model.embed_query(semantic_query)
                for offset in range(0, len(candidate_ids), 256):
                    batch = candidate_ids[offset:offset + 256]
                    for match in search_news(semantic_query, collection, lambda _query: vector,
                                             k=len(batch), section=section, article_ids=batch):
                        distance = float(match["distance"])
                        if not math.isfinite(distance):
                            raise ValueError("新闻索引返回非有限距离")
                        scores[match["article_id"]] = 1 / (1 + max(0, distance))
                if set(scores) != set(candidate_ids):
                    raise ValueError("新闻索引没有返回全部候选的距离")
            cache_used = True
        except Exception:
            # 一批失败不能让只有部分候选享有语义加权优势。
            scores = {}
            logger.warning("新闻向量缓存或重排失败，全部候选改用 TF-IDF 加权排序", exc_info=True)
    return rank_candidates(items, semantic_query, scores, sort_by), cache_used, stats


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
    published_after: str = "",
    published_before: str = "",
    section: str = "",
    source_names: list[str] | None = None,
    sort_by: str = "relevance",
    client: NewsClient | None = None,
    result_k: int | None = None,
    candidate_limit: int = 0,
    ranking_mode: str = "relevance",
    scope_ranker=None,
    summary_screener=None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> NewsRetrievalResult:
    """默认取完摘要页；指定候选数时先限定集合，再统一排序和读取正文。"""
    result_limit = settings.news_retrieval_k if result_k is None else result_k
    if result_limit < 1:
        raise ValueError("正文阅读预算必须为正数")
    events: list[dict[str, Any]] = []

    def emit(event: dict[str, Any]) -> None:
        events.append(event)
        if on_event is not None:
            on_event(event)

    news_client = client or NewsClient(on_retry=emit)

    queries = clean_queries(api_queries if api_queries is not None else [api_query]) or [""]
    if not 0 <= candidate_limit <= 200 or ranking_mode not in {"relevance", "impact"}:
        raise ValueError("候选数量或排序方式无效")
    filter_plan = {"start": start, "end": end, "published_after": published_after,
                   "published_before": published_before, "section": section, "source_names": source_names or []}
    items_by_id = {}
    errors = []
    raw_count = 0
    scope_limited = False
    scope_checked = bool(candidate_limit and sort_by == "newest")
    if scope_checked and scope_ranker is None:
        from .impact_ranking import rank_impact
        from .research.service import store, model_revision
        impact_store, impact_signature = store(), model_revision()
        scope_ranker = lambda values: rank_impact(values, semantic_query, database=impact_store, signature=impact_signature)
    excluded_count = 0
    for query_index, query in enumerate(queries):
        def query_event(event):
            emit({**event, "query_number": query_index + 1, "query_total": len(queries)})

        try:
            accepted_ids = set()
            pending_scope = []
            def accept_batch(batch):
                nonlocal excluded_count
                scored = scope_ranker([{**item, "candidate_id": candidate_key(item)} for item in batch]) if scope_checked else batch
                by_id = {candidate_key(item): item for item in scored}
                for original in batch:
                    item = by_id[candidate_key(original)]
                    if scope_checked and item.get("impact", {}).get("subject_relation") not in {"direct", "driver"}:
                        excluded_count += 1
                        emit({"kind": "news_scope_excluded", "article_id": _article_id(item),
                              "title": item.get("title", ""), "reason": item.get("impact", {}).get("mechanism", "与研究对象无实质关系")})
                        continue
                    if candidate_limit and len(accepted_ids) >= candidate_limit:
                        break
                    stored = items_by_id.setdefault(candidate_key(item), {**item, "retrieval_queries": []})
                    if query not in stored["retrieval_queries"]:
                        stored["retrieval_queries"].append(query)
                    accepted_ids.add(candidate_key(item))
                if scope_checked:
                    emit({"kind": "news_scope_progress", "scanned": raw_count, "accepted": len(accepted_ids),
                          "required": candidate_limit, "excluded": excluded_count})
            for item in news_client.iter_news(
                q=query, start=start, end=end, section=section,
                # 小任务不为凑满20条触发稀疏关键词的全库扫描；仍继续翻页补足有效候选。
                page_size=min(20, candidate_limit) if candidate_limit else 20,
                max_items=None, order="asc" if sort_by == "oldest" else "desc",
                include_content=False, on_event=query_event,
            ):
                raw_count += 1
                # 即使服务端意外附带正文，候选池也只保存轻量索引字段。
                item = {key: item[key] for key in ("article_id", "title", "summary", "published_at",
                        "canonical_url", "url", "section", "source_name", "content_hash") if key in item}
                if "article_id" in item:
                    item["article_id"] = _article_id(item)
                if candidate_limit and not news_metadata_matches(item, filter_plan):
                    if scope_checked and raw_count >= 2000:
                        raise ValueError("已扫描2000条仍未凑足相关候选；触发显式安全上限，不宣称完成")
                    continue
                if candidate_limit and not parse_news_timestamp(str(item.get("published_at") or "")):
                    raise ValueError("限定最新候选时发现无有效发布时间的新闻，无法证明最新顺序")
                pending_scope.append(item)
                scope_batch = min(5, max(1, candidate_limit - len(accepted_ids))) if candidate_limit else 5
                if not scope_checked or len(pending_scope) >= scope_batch:
                    accept_batch(pending_scope)
                    pending_scope = []
                if candidate_limit and len(accepted_ids) >= candidate_limit:
                    scope_limited = True
                    break
                if scope_checked and raw_count >= 2000:
                    raise ValueError("已扫描2000条仍未凑足相关候选；触发显式安全上限，不宣称完成")
            if pending_scope:
                accept_batch(pending_scope)
        except (NewsAPIError, ValueError) as exc:
            errors.append(str(exc))
            if isinstance(exc, NewsAPIError) and pending_scope:
                # 下一页失败不能吞掉前一页已取回、尚未凑满评分批次的有效材料。
                # 若评分本身失败，不再静默重试或把这些材料标作已通过。
                try:
                    accept_batch(pending_scope)
                except ValueError as scope_error:
                    errors.append(f"已取回候选的主题校验失败：{scope_error}")
            emit({"kind": "news_error", "message": str(exc), "partial_count": len(items_by_id),
                  "status_code": getattr(exc, "status_code", None),
                  "error_code": getattr(exc, "error_code", ""), "request_id": getattr(exc, "request_id", ""),
                  "query_number": query_index + 1, "query": query})
    items = list(items_by_id.values())
    source_names = list(source_names or [])
    before_filter = len(items)
    filter_plan = {
        "start": start, "end": end, "published_after": published_after,
        "published_before": published_before, "section": section,
        "source_names": source_names,
    }
    items = [item for item in items if news_metadata_matches(item, filter_plan)]
    if candidate_limit:
        items.sort(key=lambda item: (parse_news_timestamp(item["published_at"]).timestamp(), candidate_key(item)),
                   reverse=sort_by != "oldest")
        items = items[:candidate_limit]
    emit({
        "kind": "news_filtered", "input_count": before_filter,
        "matched_count": len(items), "published_after": published_after,
        "published_before": published_before, "section": section,
        "source_names": source_names,
    })
    api_error = "；".join(errors)
    if candidate_limit:
        emit({"kind": "news_pool", "candidate_count": len(items), "requested_candidates": candidate_limit,
              "scope": "latest_n" if sort_by == "newest" else "bounded_n", "candidates": items})
    if errors:
        if not items:
            # 现有向量回退未按发布时间检索，不能把它当成满足日期条件的结果。
            cached = [] if candidate_limit or ranking_mode == "impact" or start or end or published_after or published_before or source_names else _cached_fallback(
                semantic_query or api_query, section, result_limit)
            emit({
                "kind": "news_cache_fallback", "count": len(cached),
                "date_restricted": bool(start or end or published_after or published_before),
                "scope_restricted": bool(start or end or published_after or published_before or source_names),
            })
            return NewsRetrievalResult(
                items=cached, vector_cache_used=bool(cached), api_failed=True,
                api_error=api_error, events=events, candidates=cached, retrieval_complete=False,
            )

    ranked, cache_used, stats = _rank_and_cache(
        items,
        semantic_query or api_query,
        section,
        sort_by,
    )
    if scope_checked:
        # 不把多个评分批次各自的第1名冒充候选池排名；最新顺序仍由用户排序决定。
        impact_order = sorted(ranked, key=lambda item: -item["impact"]["score"])
        for i, item in enumerate(impact_order, 1):
            item["impact"]["rank"] = i
        if ranking_mode == "impact":
            ranked = impact_order
    elif ranking_mode == "impact":
        from .impact_ranking import rank_impact
        ranked = rank_impact(ranked, semantic_query, emit=emit)
    def read_article(item):
        article_id = _article_id(item)
        if not article_id:
            return item, None
        try:
            full_item = news_client.article(article_id)
            # API 数据不能覆盖本地排序/筛选决定或换掉候选身份。
            if _article_id(full_item) and _article_id(full_item) != article_id:
                raise ValueError("正文返回的文章编号与请求不一致")
            fields = {key: full_item[key] for key in ("title", "content", "summary", "published_at",
                      "canonical_url", "url", "section", "source_name", "content_hash") if key in full_item}
            return {**item, **fields, "_content_scope": "full" if str(fields.get("content") or "").strip() else "summary"}, None
        except (NewsAPIError, ValueError, AttributeError) as exc:
            return {**item, "_content_scope": "summary"}, {
                "kind": "news_article_error", "article_id": article_id, "message": str(exc),
                "status_code": getattr(exc, "status_code", None),
                "error_code": getattr(exc, "error_code", ""), "request_id": getattr(exc, "request_id", ""),
            }

    def read_batch(values):
        return RunnableLambda(read_article).batch(values, config={"max_concurrency": max(1, settings.io_concurrency)})

    if summary_screener is not None and not candidate_limit and ranking_mode == "relevance":
        from .news_screening import select_core_articles
        for offset in range(0, len(ranked), 50):
            emit({"kind": "news_abstract_pool", "offset": offset, "candidates": ranked[offset:offset + 50]})
        selected, ranked = select_core_articles(ranked, result_limit, screen=summary_screener,
                                                read_batch=read_batch, emit=emit, sort_by=sort_by)
        chosen = selected
    else:
        # 用户明确“最新 N 篇中按影响选 M 篇”时保留该母集和排序，不悄悄替换成别的文章。
        chosen = (_diverse_results(ranked, result_limit) if sort_by == "relevance" and ranking_mode != "impact" else ranked[:result_limit])
        selected = []
        for item, event in read_batch(chosen):
            selected.append(item)
            if event:
                emit(event)
    chosen_keys = {candidate_key(item) for item in chosen}
    for item in ranked:
        item["selected_for_reading"] = candidate_key(item) in chosen_keys
        if item["selected_for_reading"]:
            item["selection_reason"] = ("按可核对的潜在影响评分入选" if ranking_mode == "impact" else
                                        "按阅读优先级并兼顾查询覆盖" if sort_by == "relevance" else f"按用户排序 {sort_by}")
        else:
            item["selection_reason"] = (item.get("body_rejection") or
                ("摘要判定无关：" + item["screening"]["reason"] if item.get("screening", {}).get("relation") == "unrelated" else
                 "正文预算外，保留索引待查，不代表无关"))
    # 分块事件随研究编号落盘，工具 artifact 同时返回完整候选，不塞满模型上下文。
    for offset in range(0, len(ranked), 50):
        emit({"kind": "news_candidates", "offset": offset, "candidates": ranked[offset:offset + 50]})
    emit({
        "kind": "news_ranked", "candidate_count": len(items),
        "raw_count": raw_count,
        "selected_count": len(chosen), "deferred_count": len(ranked) - len(chosen),
        "retrieval_complete": not bool(errors), "vector_cache_used": cache_used,
        "ranking_method": "impact-v1" if ranking_mode == "impact" and ranked else ranked[0].get("priority", {}).get("method", "unknown") if ranked else "none",
        "selection_scope": "latest_n" if candidate_limit and sort_by == "newest" else "bounded_n" if candidate_limit else "all_matches",
        "requested_candidates": candidate_limit, "all_matches_exhausted": not scope_limited and not errors,
        "subject_scope_checked": scope_checked, "unrelated_excluded": excluded_count,
        "cache_stats": asdict(stats) if stats is not None else None,
    })
    for item in chosen:
        emit({"kind": "news_selected", "article_id": _article_id(item), "title": item.get("title", ""),
              "priority": item.get("priority", {}), "impact": item.get("impact", {}),
              "selection_reason": next(i["selection_reason"] for i in ranked if candidate_key(i) == candidate_key(item))})
    return NewsRetrievalResult(
        items=selected,
        vector_cache_used=cache_used,
        cache_stats=stats,
        api_failed=bool(errors),
        api_error=api_error,
        events=events,
        candidates=ranked,
        retrieval_complete=not bool(errors),
    )
