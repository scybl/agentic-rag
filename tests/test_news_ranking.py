"""全候选加权、稳定排序和分批索引回归，无需在线模型。"""

from dataclasses import replace
from unittest.mock import Mock, patch

from agentic_rag.config import settings
from agentic_rag.news_ranking import rank_candidates
from agentic_rag.news_retrieval import _rank_and_cache


def test_weight_components_are_explicit_and_order_independent():
    items = [{"article_id": "a", "title": "黄金价格", "retrieval_queries": ["黄金"]},
             {"article_id": "b", "title": "体育新闻", "retrieval_queries": ["黄金", "财经"]}]
    scores = {"a": 0.8, "b": 0.1}
    first = rank_candidates(items, "黄金价格", scores)
    reverse = rank_candidates(list(reversed(items)), "黄金价格", scores)
    assert first == reverse
    priority = first[0]["priority"]
    assert priority["weights"] == {"semantic": 0.7, "lexical": 0.2, "query_coverage": 0.1}
    assert abs(priority["score"] - sum(priority["weights"][k] * v for k, v in priority["components"].items())) < 1e-6
    assert first[0]["article_id"] == "a"


def test_explicit_chronology_and_missing_dates():
    items = [{"article_id": "missing", "title": "黄金价格"},
             {"article_id": "old", "title": "黄金", "published_at": "2026-09-29T09:00:00+08:00"},
             {"article_id": "new", "title": "市场", "published_at": "2026-09-30T09:00:00+08:00"}]
    assert [i["article_id"] for i in rank_candidates(items, "黄金", {}, "oldest")] == ["old", "new", "missing"]
    assert [i["article_id"] for i in rank_candidates(items, "黄金", {}, "newest")] == ["new", "old", "missing"]


def test_semantic_ranking_batches_every_candidate_and_embeds_query_once():
    items = [{"article_id": str(i), "title": "新闻"} for i in range(601)]
    model = Mock()
    model.embed_query.return_value = [0.5]
    def search(query, collection, embed_query, **kwargs):
        assert embed_query(query) == [0.5]
        assert kwargs["k"] == len(kwargs["article_ids"]) <= 256
        return [{"article_id": aid, "distance": 0.01 if aid == "600" else 2} for aid in kwargs["article_ids"]]
    with (patch("agentic_rag.news_retrieval.settings", replace(settings, news_vector_cache_enabled=True)),
          patch("agentic_rag.news_retrieval.open_news_collection"),
          patch("agentic_rag.news_retrieval.get_embedding_model", return_value=model),
          patch("agentic_rag.news_retrieval.upsert_news_items") as upsert,
          patch("agentic_rag.news_retrieval.search_news", side_effect=search) as lookup):
        ranked, used, _stats = _rank_and_cache(items, "新闻", "")
    assert used and len(ranked) == 601 and ranked[0]["article_id"] == "600"
    assert lookup.call_count == 3 and model.embed_query.call_count == 1
    assert len(upsert.call_args.args[0]) == 601


def test_partial_semantic_failure_does_not_bias_towards_first_batch():
    items = [{"article_id": str(i), "title": "体育娱乐"} for i in range(256)]
    items.append({"article_id": "tail", "title": "黄金价格"})
    with (patch("agentic_rag.news_retrieval.settings", replace(settings, news_vector_cache_enabled=True)),
          patch("agentic_rag.news_retrieval.open_news_collection"),
          patch("agentic_rag.news_retrieval.get_embedding_model"),
          patch("agentic_rag.news_retrieval.upsert_news_items"),
          patch("agentic_rag.news_retrieval.search_news", side_effect=[
              [{"article_id": str(i), "distance": 0.0} for i in range(256)], RuntimeError("offline")])):
        ranked, used, _stats = _rank_and_cache(items, "黄金价格", "")
    assert not used and ranked[0]["article_id"] == "tail"
    assert all(item["priority"]["weights"]["semantic"] == 0 for item in ranked)
