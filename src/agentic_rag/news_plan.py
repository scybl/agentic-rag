"""把模型的新闻检索意图编译为可审计、可直接执行的查询计划。"""

from datetime import date, datetime
import re
from typing import Any
from zoneinfo import ZoneInfo


SHANGHAI = ZoneInfo("Asia/Shanghai")


def validate_api_query(query: str, section: str = "") -> None:
    """服务端真实契约：每组最多 8 个 AND 词、200 字符，栏目最多 80 字符。"""
    if not isinstance(query, str) or len(query.strip()) > 200:
        raise ValueError("新闻查询词必须为不超过200字符的字符串")
    if len(query.split()) > 8:
        raise ValueError("每组新闻查询最多8个空格分隔的关键词，请拆成更短的查询")
    if not isinstance(section, str) or len(section.strip()) > 80:
        raise ValueError("新闻栏目必须为不超过80字符的字符串")


def clean_queries(values: list[str], limit: int = 4) -> list[str]:
    """拆开模型误塞进单字段的查询列表，再限制数量并去重。"""
    queries = []
    for value in values:
        for part in re.split(r"[;；\n]+", value):
            query = " ".join(part.split())[:200]
            if query and query not in queries:
                queries.append(query)
    # 有明确主题时不混入“全部最新新闻”；只有全空时才保留不筛关键词。
    return queries[:limit] if queries else ([""] if values else [])


def clean_terms(values, *, limit: int, max_length: int = 100) -> list[str]:
    """清理模型生成的实体/主题/来源约束；顺序保留，便于终端核对。"""
    terms = []
    for value in values or []:
        term = " ".join(str(value).split())[:max_length]
        if term and term not in terms:
            terms.append(term)
    return terms[:limit]


def query_covers_term(query: str, term: str) -> bool:
    """API 的空格词是 AND；顺序不同可以覆盖，但不能跨查询拼成一次匹配。"""
    tokens = term.casefold().split()
    return bool(tokens) and all(token in query.casefold() for token in tokens)


def compile_topic_queries(queries, topics, entities, *, limit=4):
    """主题是检索线索，不是 JSON 合法性条件；在剩余预算内补搜并公开未覆盖项。"""
    actual = list(queries)
    added, deferred = [], []
    anchor = next(iter(entities), "")
    for topic in topics:
        if any(query_covers_term(query, topic) for query in actual):
            continue
        query = f"{anchor} {topic}" if anchor and not query_covers_term(topic, anchor) else topic
        # 有主题时不再执行无关键词的全库请求；不截断长词造成条件丢失。
        candidates = [item for item in actual if item]
        if len(candidates) < limit and len(query) <= 200 and len(query.split()) <= 8:
            actual = [*candidates, query]
            added.append(query)
        else:
            deferred.append(topic)
    return actual, added, deferred


def parse_news_timestamp(value: str) -> datetime | None:
    """解析必须带时区的 ISO 8601 时间；内部统一为上海时区。"""
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("精确新闻时间必须包含时区")
    return parsed.astimezone(SHANGHAI)


def source_matches(value: str, filters: list[str]) -> bool:
    if not filters:
        return True
    aliases = {
        "同花顺": "tonghuashun", "ths": "tonghuashun",
        "路透": "reuters", "路透社": "reuters",
        "彭博": "bloomberg", "彭博社": "bloomberg",
        "新华社": "xinhua", "财联社": "cls",
    }
    haystack = re.sub(r"[^\w\u4e00-\u9fff]+", "", str(value).casefold())
    canonical_haystack = haystack + "".join(
        canonical for alias, canonical in aliases.items()
        if re.sub(r"[^\w\u4e00-\u9fff]+", "", alias.casefold()) in haystack
    )
    for item in filters:
        needle = aliases.get(item.casefold(), item.casefold())
        needle = re.sub(r"[^\w\u4e00-\u9fff]+", "", needle)
        if needle and needle in canonical_haystack:
            return True
    return False


def news_metadata_matches(metadata: dict[str, Any], plan: dict[str, Any]) -> bool:
    """同一套精确时间、栏目和来源规则同时约束实时结果与历史记忆。"""
    published = str(metadata.get("published_at") or "")
    try:
        timestamp = parse_news_timestamp(published)
    except (TypeError, ValueError):
        timestamp = None
    after = parse_news_timestamp(str(plan.get("published_after") or ""))
    before = parse_news_timestamp(str(plan.get("published_before") or ""))
    if (after or before) and timestamp is None:
        return False
    if after and timestamp < after:
        return False
    if before and timestamp > before:
        return False
    if timestamp:
        day = timestamp.date().isoformat()
        if plan.get("start") and day < plan["start"]:
            return False
        if plan.get("end") and day > plan["end"]:
            return False
    elif plan.get("start") or plan.get("end"):
        return False
    section = str(plan.get("section") or "").strip().casefold()
    if section and section not in str(metadata.get("section") or "").casefold():
        return False
    source_value = " ".join(str(metadata.get(key) or "") for key in (
        "source_name", "source", "url",
    ))
    return source_matches(source_value, list(plan.get("source_names") or []))


def prepare_news_plan(
    query: str,
    section: str,
    suggestion: dict[str, str],
    *,
    additional_queries=(),
    people=(),
    organizations=(),
    topics=(),
    source_names=(),
    sort_by: str = "relevance",
    coverage: str = "focused",
    result_limit: int = 5,
) -> dict[str, Any]:
    """保留模型原始意图，同时生成可以直接提交给工具的确定性条件。"""
    people = clean_terms(people, limit=4)
    organizations = clean_terms(organizations, limit=4)
    topics = clean_terms(topics, limit=6)
    source_names = clean_terms(source_names, limit=4)
    raw_queries = [query, *additional_queries]
    queries = clean_queries(raw_queries)
    queries, added_queries, deferred_topics = compile_topic_queries(
        queries, topics, [*people, *organizations],
    )
    plan: dict[str, Any] = {
        "query": query.strip(),
        "raw_queries": raw_queries,
        "queries": queries or [""],
        "added_topic_queries": added_queries,
        "deferred_topics": deferred_topics,
        "people": people,
        "organizations": organizations,
        "topics": topics,
        "source_names": source_names,
        "section": section.strip(),
        "sort_by": sort_by,
        "coverage": coverage,
        "result_limit": result_limit,
        "suggested_time": dict(suggestion),
        "start": "",
        "end": "",
        "published_after": "",
        "published_before": "",
        "time_note": "未设置时间条件",
        "error": "",
    }
    if sort_by not in {"relevance", "newest", "oldest"}:
        plan["error"] = "新闻排序方式无效"
        return plan
    if coverage not in {"focused", "broad", "exhaustive"}:
        plan["error"] = "新闻覆盖模式无效"
        return plan
    if isinstance(result_limit, bool) or not isinstance(result_limit, int) or not 1 <= result_limit <= 100:
        plan["error"] = "新闻入选数量必须为 1–100"
        return plan
    mode = suggestion.get("mode", "unrestricted")
    if mode == "unrestricted":
        if any(suggestion.get(key) for key in ("start", "end", "published_after", "published_before")):
            plan["time_note"] = "模型选择了不限时间，因此忽略附带的日期"
        return plan

    start = suggestion.get("start", "").strip()
    end = suggestion.get("end", "").strip()
    published_after = suggestion.get("published_after", "").strip()
    published_before = suggestion.get("published_before", "").strip()
    try:
        if mode not in {"suggested", "explicit"}:
            raise ValueError("时间类型无法识别")
        if not any((start, end, published_after, published_before)):
            raise ValueError("没有给出起止日期")
        if mode == "suggested" and not suggestion.get("reason", "").strip():
            raise ValueError("没有给出时间建议的理由")
        for value in (start, end):
            if value and date.fromisoformat(value).isoformat() != value:
                raise ValueError("日期必须使用 YYYY-MM-DD")
        if start and end and start > end:
            raise ValueError("开始日期晚于结束日期")
        after = parse_news_timestamp(published_after)
        before = parse_news_timestamp(published_before)
        if after and before and after > before:
            raise ValueError("精确开始时间晚于结束时间")
        # API 先按上海自然日粗筛，随后工具再按精确时间戳硬过滤。
        if after and not start:
            start = after.date().isoformat()
        if before and not end:
            end = before.date().isoformat()
    except ValueError as exc:
        if mode == "explicit":
            plan["error"] = f"用户指定的新闻时间解析无效，停止本次新闻查询：{exc}"
            plan["time_note"] = plan["error"]
        else:
            plan["time_note"] = f"时间建议无效，按不限时间查询：{exc}"
        return plan

    plan.update(start=start, end=end, published_after=published_after, published_before=published_before)
    exact = bool(published_after or published_before)
    prefix = "采用模型建议的" if mode == "suggested" else "采用用户指定的"
    plan["time_note"] = prefix + ("精确时间范围" if exact else "日期范围")
    return plan
