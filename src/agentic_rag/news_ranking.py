"""对全部轻量候选评分；分数是阅读优先级，不是新闻热度或可信度。"""

import hashlib
import json
import math

from .news_plan import parse_news_timestamp


def candidate_key(item: dict) -> str:
    return str(item.get("candidate_id") or item.get("article_id") or item.get("canonical_url") or item.get("url") or
               hashlib.sha256(json.dumps(item, sort_keys=True, ensure_ascii=False).encode()).hexdigest())


def rank_candidates(items: list[dict], query: str, semantic_scores: dict[str, float],
                    sort_by: str = "relevance") -> list[dict]:
    """TF-IDF 使用稀疏字符 n-gram，兼容中文；标题双计权，不读取正文。"""
    if not items:
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [f"{item.get('title', '')} {item.get('title', '')} {item.get('summary', '')}" for item in items]
    lexical = [0.0] * len(items)
    if query.strip():
        try:
            matrix = TfidfVectorizer(analyzer="char", ngram_range=(1, 3),
                                     max_features=50000).fit_transform([query, *texts])
            lexical = (matrix[1:] @ matrix[0].T).toarray().ravel().tolist()
        except ValueError as exc:
            if "empty vocabulary" not in str(exc):
                raise
    queries = {q for item in items for q in item.get("retrieval_queries", []) if q}
    weights = {"semantic": 0.7, "lexical": 0.2, "query_coverage": 0.1} if semantic_scores else {
        "semantic": 0.0, "lexical": 0.9, "query_coverage": 0.1}
    ranked = []
    for item, lexical_score in zip(items, lexical):
        components = {
            "semantic": semantic_scores.get(str(item.get("article_id") or ""), 0.0),
            "lexical": lexical_score,
            "query_coverage": len(set(item.get("retrieval_queries", [])) & queries) / len(queries) if queries else 0.0,
        }
        score = sum(weights[name] * value for name, value in components.items())
        ranked.append({**item, "candidate_id": candidate_key(item), "priority": {
            "score": round(score, 6), "components": {key: round(value, 6) for key, value in components.items()},
            "weights": weights, "method": "semantic+tfidf" if semantic_scores else "tfidf",
            "sort_by": sort_by,
        }})

    def order(item):
        try:
            parsed = parse_news_timestamp(str(item.get("published_at") or ""))
            timestamp = parsed.timestamp() if parsed else None
        except (TypeError, ValueError, OverflowError):
            timestamp = None
        time_key = math.inf if timestamp is None else (timestamp if sort_by == "oldest" else -timestamp)
        score_key = -item["priority"]["score"]
        return ((time_key, score_key) if sort_by in {"newest", "oldest"} else (score_key, time_key)) + (item["candidate_id"],)

    ranked.sort(key=order)
    for rank, item in enumerate(ranked, 1):
        item["priority"]["rank"] = rank
    return ranked
