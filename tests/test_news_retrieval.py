"""新闻候选召回、向量重排和正文按需读取的离线测试。"""

import unittest
from dataclasses import replace
from unittest.mock import patch

from agentic_rag.config import settings
from agentic_rag.news_api import NewsAPIError, NewsClient
from agentic_rag.news_retrieval import _diverse_results, retrieve_news
from agentic_rag.news_plan import clean_queries


class FakeNewsClient:
    iter_news = NewsClient.iter_news

    def __init__(self, items=None, fail=False):
        self.items = items or []
        self.fail = fail
        self.search_calls = []
        self.article_calls = []

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        if self.fail:
            raise NewsAPIError("temporary failure")
        start = int(kwargs.get("cursor") or 0)
        end = start + kwargs["limit"]
        return {
            "items": self.items[start:end],
            "has_more": end < len(self.items),
            "next_cursor": str(end),
        }

    def article(self, article_id):
        self.article_calls.append(article_id)
        return {"article_id": article_id, "content": f"{article_id} 正文"}


class NewsRetrievalTests(unittest.TestCase):
    def test_query_list_is_split_without_changing_and_terms(self):
        self.assertEqual(clean_queries(["生猪; 能繁母猪；玉米 豆粕", "生猪", ""]),
                         ["生猪", "能繁母猪", "玉米 豆粕"])
        self.assertEqual(clean_queries([""]), [""])
        self.assertEqual(clean_queries([]), [])

    def test_query_diversity_survives_ranking(self):
        items = [
            {"article_id": "price1", "retrieval_queries": ["猪价"]},
            {"article_id": "price2", "retrieval_queries": ["猪价"]},
            {"article_id": "supply", "retrieval_queries": ["能繁母猪"]},
            {"article_id": "cost", "retrieval_queries": ["饲料"]},
        ]
        self.assertEqual([item["article_id"] for item in _diverse_results(items, 3)],
                         ["price1", "supply", "cost"])

    def test_multiple_queries_exhaust_pages_then_deduplicate(self):
        items = [{"article_id": str(i), "title": str(i)} for i in range(40)]
        client = FakeNewsClient(items)
        with patch("agentic_rag.news_retrieval._rank_and_cache",
                   side_effect=lambda candidates, *_args: (candidates, False, None)) as rank:
            result = retrieve_news(semantic_query="预测", api_queries=["生猪", "能繁母猪", "饲料"],
                                   client=client)
        self.assertEqual([call["q"] for call in client.search_calls], ["生猪", "生猪", "能繁母猪", "能繁母猪", "饲料", "饲料"])
        self.assertEqual([call["limit"] for call in client.search_calls], [20] * 6)
        candidates = rank.call_args.args[0]
        self.assertEqual(len(candidates), 40)
        self.assertEqual(candidates[0]["retrieval_queries"], ["生猪", "能繁母猪", "饲料"])
        ranked = next(event for event in result.events if event["kind"] == "news_ranked")
        self.assertEqual(ranked["raw_count"], 120)
        self.assertEqual(ranked["candidate_count"], 40)
        self.assertEqual(len(result.candidates), 40)
        self.assertTrue(result.retrieval_complete)
        self.assertEqual(ranked["deferred_count"], 40 - len(result.items))

    def test_one_query_failure_does_not_drop_other_query_results(self):
        client = FakeNewsClient([{"article_id": "one", "title": "事实"}])
        original_search = client.search

        def search(**kwargs):
            if kwargs["q"] == "饲料":
                raise NewsAPIError("HTTP 503", 503)
            return original_search(**kwargs)

        client.search = search
        with patch("agentic_rag.news_retrieval._rank_and_cache",
                   side_effect=lambda candidates, *_args: (candidates, False, None)):
            result = retrieve_news(semantic_query="预测", api_queries=["饲料", "生猪"], client=client)
        self.assertTrue(result.api_failed)
        self.assertFalse(result.retrieval_complete)
        self.assertEqual(len(result.items), 1)
        self.assertEqual(client.article_calls, ["one"])

    def test_fetches_candidates_then_only_selected_full_articles(self):
        items = [
            {"article_id": "one", "title": "第一条"},
            {"article_id": "two", "title": "第二条"},
            {"article_id": "three", "title": "第三条"},
        ]
        client = FakeNewsClient(items)
        with patch(
            "agentic_rag.news_retrieval._rank_and_cache",
            return_value=([items[1], items[0], items[2]], True, None),
        ):
            result = retrieve_news(
                semantic_query="行业影响",
                api_query="生猪",
                client=client,
                result_k=2,
            )

        self.assertEqual(client.search_calls[0]["limit"], 20)
        self.assertFalse(client.search_calls[0]["include_content"])
        self.assertCountEqual(client.article_calls, ["two", "one"])
        self.assertEqual([item["content"] for item in result.items], ["two 正文", "one 正文"])
        self.assertTrue(result.vector_cache_used)
        self.assertEqual(len(result.candidates), 3)
        self.assertFalse(result.candidates[-1]["selected_for_reading"])

    def test_body_budget_does_not_truncate_last_page(self):
        items = [{"article_id": str(index), "title": str(index),
                  "published_at": "2026-09-15T12:00:00+08:00"} for index in range(35)]
        client = FakeNewsClient(items)
        with patch(
            "agentic_rag.news_retrieval._rank_and_cache",
            side_effect=lambda candidates, *_args: (candidates, False, None),
        ) as rank:
            result = retrieve_news(
                semantic_query="生猪",
                api_query="生猪",
                start="2026-09-01",
                client=client,
                result_k=2,
            )

        self.assertEqual([call["limit"] for call in client.search_calls], [20, 20])
        self.assertEqual([call["cursor"] for call in client.search_calls], ["", "20"])
        self.assertTrue(all(call["q"] == "生猪" and call["start"] == "2026-09-01"
                            for call in client.search_calls))
        self.assertEqual(len(rank.call_args.args[0]), 35)
        self.assertCountEqual(client.article_calls, ["0", "1"])
        self.assertEqual(len(result.items), 2)
        self.assertEqual([event["count"] for event in result.events if event["kind"] == "news_page"], [20, 15])
        self.assertEqual(len(result.candidates), 35)

    def test_second_page_failure_keeps_first_page_and_reports_error(self):
        items = [{"article_id": str(index), "title": str(index)} for index in range(35)]
        client = FakeNewsClient(items)
        original_search = client.search

        def search(**kwargs):
            if kwargs.get("cursor"):
                raise NewsAPIError("News API returned HTTP 503", 503)
            return original_search(**kwargs)

        client.search = search
        with patch(
            "agentic_rag.news_retrieval._rank_and_cache",
            side_effect=lambda candidates, *_args: (candidates, False, None),
        ) as rank:
            result = retrieve_news(semantic_query="生猪", client=client, result_k=2)
        self.assertEqual(len(rank.call_args.args[0]), 20)
        self.assertEqual(len(result.items), 2)
        self.assertTrue(result.api_failed)
        self.assertIn("503", result.api_error)
        self.assertFalse(result.retrieval_complete)
        self.assertEqual(len(result.candidates), 20)
        self.assertTrue(any(event.get("partial_count") == 20 for event in result.events))

    def test_api_failure_does_not_use_cache_outside_requested_time(self):
        with patch("agentic_rag.news_retrieval._cached_fallback") as fallback:
            result = retrieve_news(
                semantic_query="生猪", start="2026-09-29", end="2026-09-29",
                client=FakeNewsClient(fail=True),
            )
        fallback.assert_not_called()
        self.assertTrue(result.api_failed)
        self.assertEqual(result.items, [])
        self.assertTrue(result.events[-1]["date_restricted"])

    def test_uses_vector_cache_when_api_is_unavailable(self):
        client = FakeNewsClient(fail=True)
        cached = [{"article_id": "cached", "title": "缓存新闻"}]
        with patch("agentic_rag.news_retrieval._cached_fallback", return_value=cached):
            result = retrieve_news(
                semantic_query="人工智能",
                client=client,
                result_k=3,
            )

        self.assertTrue(result.api_failed)
        self.assertTrue(result.vector_cache_used)
        self.assertEqual(result.items, cached)
        self.assertEqual(client.article_calls, [])

    def test_exact_time_source_and_newest_sort_are_executed_before_full_read(self):
        items = [
            {"article_id": "old", "title": "旧", "published_at": "2026-09-30T08:30:00+08:00", "source_name": "Reuters"},
            {"article_id": "wrong-source", "title": "其他源", "published_at": "2026-09-30T09:50:00+08:00", "source_name": "Other"},
            {"article_id": "new", "title": "新", "published_at": "2026-09-30T09:30:00+08:00", "source_name": "Reuters"},
            {"article_id": "outside", "title": "超时", "published_at": "2026-09-30T10:30:00+08:00", "source_name": "Reuters"},
        ]
        client = FakeNewsClient(items)
        with patch("agentic_rag.news_retrieval.settings", replace(settings, news_vector_cache_enabled=False)):
            result = retrieve_news(
                semantic_query="AI", api_queries=[""], client=client,
                published_after="2026-09-30T08:00:00+08:00",
                published_before="2026-09-30T10:00:00+08:00",
                source_names=["Reuters"], sort_by="newest", result_k=2,
            )
        self.assertEqual([item["article_id"] for item in result.items], ["new", "old"])
        self.assertCountEqual(client.article_calls, ["new", "old"])
        filtered = next(event for event in result.events if event["kind"] == "news_filtered")
        self.assertEqual((filtered["input_count"], filtered["matched_count"]), (4, 2))

    def test_relevant_tail_wins_and_all_candidates_are_returned_and_audited(self):
        items = [{"article_id": f"noise-{i}", "title": "体育娱乐", "summary": "比赛结果"} for i in range(125)]
        items.append({"article_id": "tail", "title": "黄金价格走势", "summary": "黄金价格走势分析",
                      "content": "服务端意外附带的长正文不应进入候选索引"})
        client = FakeNewsClient(items)
        with patch("agentic_rag.news_retrieval.settings", replace(settings, news_vector_cache_enabled=False)):
            result = retrieve_news(semantic_query="黄金价格走势", client=client, result_k=1)
        self.assertEqual(client.article_calls, ["tail"])
        self.assertEqual(len(client.search_calls), 7)
        self.assertEqual(len(result.candidates), 126)
        self.assertTrue(all("content" not in item for item in result.candidates))
        self.assertGreater(result.candidates[0]["priority"]["score"], result.candidates[1]["priority"]["score"])
        archived = [item for event in result.events if event["kind"] == "news_candidates" for item in event["candidates"]]
        self.assertEqual(archived, result.candidates)
        self.assertEqual(sum(item["selected_for_reading"] for item in archived), 1)
        self.assertTrue(all(len(e["candidates"]) <= 50 for e in result.events if e["kind"] == "news_candidates"))

    def test_vector_failure_uses_weighted_ranking_not_api_order(self):
        client = FakeNewsClient([{"article_id": "first", "title": "娱乐比赛"},
                                 {"article_id": "last", "title": "黄金价格走势"}])
        with (patch("agentic_rag.news_retrieval.settings", replace(settings, news_vector_cache_enabled=True)),
              patch("agentic_rag.news_retrieval.open_news_collection", side_effect=RuntimeError("offline"))):
            result = retrieve_news(semantic_query="黄金价格走势", client=client, result_k=1)
        self.assertEqual(client.article_calls, ["last"])
        self.assertFalse(result.vector_cache_used)
        self.assertEqual(result.candidates[0]["priority"]["method"], "tfidf")

    def test_missing_ids_do_not_merge_distinct_articles_with_same_title(self):
        client = FakeNewsClient([{"title": "价格", "summary": "昨日行情"}, {"title": "价格", "summary": "今日行情"}])
        with patch("agentic_rag.news_retrieval.settings", replace(settings, news_vector_cache_enabled=False)):
            result = retrieve_news(semantic_query="今日行情", client=client, result_k=1)
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(sum(c["selected_for_reading"] for c in result.candidates), 1)
        self.assertEqual(client.article_calls, [])

    def test_article_failure_does_not_claim_pagination_failed_or_remove_candidate(self):
        client = FakeNewsClient([{"article_id": "one", "title": "黄金", "summary": "行情"}])
        client.article = lambda _id: (_ for _ in ()).throw(NewsAPIError("503"))
        with patch("agentic_rag.news_retrieval.settings", replace(settings, news_vector_cache_enabled=False)):
            result = retrieve_news(semantic_query="黄金", client=client)
        self.assertTrue(result.retrieval_complete)
        self.assertEqual(result.items[0]["_content_scope"], "summary")
        self.assertEqual(len(result.candidates), 1)

    def test_empty_result_is_complete_and_does_not_read_articles(self):
        client = FakeNewsClient()
        result = retrieve_news(semantic_query="黄金", client=client)
        self.assertTrue(result.retrieval_complete)
        self.assertEqual(result.candidates, [])
        self.assertEqual(client.article_calls, [])


if __name__ == "__main__":
    unittest.main()
