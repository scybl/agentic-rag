"""多数据源规划与新闻节点的离线测试。"""

import unittest
from unittest.mock import ANY, patch

from langchain_core.documents import Document

from agentic_rag.graph import nodes
from agentic_rag.graph.chains import NewsTimeSuggestion, SourcePlan
from agentic_rag.news_plan import prepare_news_plan
from agentic_rag.news_retrieval import NewsRetrievalResult
from agentic_rag.tools import news as news_tools


class NewsGraphTests(unittest.TestCase):
    def setUp(self):
        self.search_plan = prepare_news_plan(
            "生猪", "财经",
            {"mode": "explicit", "start": "2026-09-29", "end": "2026-09-29", "reason": "用户要求今天"},
        )

    def test_planner_can_select_multiple_sources(self):
        plan = SourcePlan(
            task_type="analysis",
            evidence_needs=["新闻事实", "财务传导机制"],
            additional_news_queries=[],
            use_knowledge=True,
            use_news=True,
            use_web=False,
            knowledge_query="生猪价格变化如何影响企业财务",
            news_query="生猪价格",
            news_section="",
            news_time=NewsTimeSuggestion(mode="unrestricted", start="", end="", reason="不限时间"),
            web_query="",
            plan_summary="需要新闻事实和财务分析方法",
        )
        with patch.object(nodes, "get_router") as router:
            router.return_value.invoke.return_value = plan
            result = nodes.route({"question": "今天的生猪新闻会怎样影响企业？"})

        self.assertEqual(result["selected_sources"], ["vectorstore", "news_api"])
        self.assertEqual(result["datasource"], "vectorstore+news_api")
        self.assertEqual(result["source_queries"]["news_api"], "生猪价格")
        self.assertEqual(result["plan_summary"], "需要新闻事实和财务分析方法")
        self.assertEqual(result["news_search_plan"]["start"], "")
        self.assertEqual(result["news_search_plan"]["end"], "")
        self.assertIn("current_date", router.return_value.invoke.call_args.args[0])

    def test_collect_sources_merges_results_and_keeps_partial_success(self):
        knowledge = Document(page_content="财务影响分析方法", metadata={"source": "method.md"})
        news = Document(page_content="今日生猪价格上涨", metadata={"article_id": "news-1"})
        state = {
            "question": "生猪新闻影响",
            "selected_sources": ["vectorstore", "news_api", "web_search"],
            "source_queries": {
                "vectorstore": "财务影响",
                "news_api": "生猪",
                "web_search": "公开核验",
            },
        }
        with (
            patch.object(nodes, "retrieve", return_value={"documents": [knowledge]}),
            patch.object(nodes, "news_api", return_value={"documents": [news]}),
            patch.object(nodes, "web_search", side_effect=RuntimeError("temporary failure")),
        ):
            result = nodes.collect_sources(state)

        self.assertEqual(len(result["documents"]), 2)
        self.assertEqual(knowledge.metadata["source_type"], "vectorstore")
        self.assertEqual(news.metadata["source_type"], "news_api")
        self.assertIn("web_search", result["source_errors"])

    def test_news_node_converts_hybrid_results_to_documents(self):
        retrieval = NewsRetrievalResult(
            items=[
                {
                    "article_id": "news-1",
                    "title": "生猪市场价格上涨",
                    "content": "今日生猪市场价格出现上涨。",
                    "published_at": "2026-09-29T09:00:00+08:00",
                    "section": "财经",
                    "canonical_url": "https://example.com/news-1",
                    "source_name": "同花顺",
                }
            ],
            vector_cache_used=True,
        )
        with (
            patch.object(news_tools, "retrieve_news", return_value=retrieval) as retrieve,
        ):
            result = nodes.news_api({"question": "今天的生猪新闻", "news_search_plan": self.search_plan})

        document = result["documents"][0]
        self.assertIn("生猪市场价格上涨", document.page_content)
        self.assertIn("今日生猪市场价格出现上涨", document.page_content)
        self.assertEqual(document.metadata["article_id"], "news-1")
        self.assertEqual(document.metadata["source_type"], "news_api")
        retrieve.assert_called_once_with(
            semantic_query="今天的生猪新闻",
            api_query="生猪",
            api_queries=["生猪"],
            start="2026-09-29",
            end="2026-09-29",
            section="财经",
            on_event=ANY,
        )

    def test_collect_sources_preserves_news_failure_and_query_plan(self):
        plan = prepare_news_plan("生猪", "", {
            "mode": "suggested", "start": "2026-09-01", "end": "", "reason": "观察本月供需变化",
        })
        with patch.object(news_tools, "retrieve_news", return_value=NewsRetrievalResult(
            items=[], api_failed=True, api_error="News API returned HTTP 503",
        )) as retrieve:
            result = nodes.collect_sources({
                "question": "分析猪肉价格", "selected_sources": ["news_api"],
                "source_queries": {"news_api": "生猪"}, "news_search_plan": plan,
            })
        self.assertEqual(retrieve.call_args.kwargs["start"], "2026-09-01")
        self.assertEqual(retrieve.call_args.kwargs["end"], "")
        self.assertIn("503", result["source_errors"]["news_api"])

    def test_invalid_explicit_date_does_not_run_unrestricted_query(self):
        plan = prepare_news_plan("生猪", "", {
            "mode": "explicit", "start": "2026-02-30", "end": "", "reason": "用户日期解析失败",
        })
        with patch.object(news_tools, "retrieve_news") as retrieve:
            with self.assertRaisesRegex(ValueError, "停止本次新闻查询"):
                nodes.news_api({"question": "生猪", "news_search_plan": plan})
        retrieve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
