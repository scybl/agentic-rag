"""多数据源规划与新闻节点的离线测试。"""

import unittest
from unittest.mock import ANY, patch

from langchain_core.documents import Document
from pydantic import ValidationError

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
        self.assertIn("current_datetime", router.return_value.invoke.call_args.args[0])

    def test_planner_compiles_people_topic_source_time_sort_and_coverage(self):
        plan = SourcePlan(
            task_type="analysis", evidence_needs=["人物表态", "行业影响"],
            use_knowledge=False, use_news=True, use_web=False, knowledge_query="",
            news_query="马斯克 AI", additional_news_queries=["特斯拉 人工智能"],
            news_people=["马斯克"], news_organizations=["特斯拉"], news_topics=["人工智能"],
            news_sources=["路透"], news_section="科技", news_sort_by="newest",
            news_coverage="broad", news_result_limit=10,
            news_time=NewsTimeSuggestion(
                mode="explicit", start="", end="",
                published_after="2026-09-29T12:00:00+08:00",
                published_before="2026-09-30T12:00:00+08:00", reason="用户要求过去24小时"),
            web_query="", plan_summary="按人物、主题、来源和时间联合查询",
        )
        with patch.object(nodes, "get_router") as router:
            router.return_value.invoke.return_value = plan
            result = nodes.route({"question": "过去24小时路透关于马斯克和AI的热点"})
        actual = result["news_search_plan"]
        assert result["selected_sources"] == ["news_api", "web_search"]
        assert "路透" in result["source_queries"]["web_search"]
        assert actual["queries"] == ["马斯克 AI", "特斯拉 人工智能"]
        assert actual["people"] == ["马斯克"] and actual["organizations"] == ["特斯拉"]
        assert actual["topics"] == ["人工智能"] and actual["source_names"] == ["路透"]
        assert actual["sort_by"] == "newest" and actual["coverage"] == "broad"
        assert actual["result_limit"] == 10
        assert (actual["start"], actual["end"]) == ("2026-09-29", "2026-09-30")

    def test_planner_rejects_declared_constraints_that_are_not_executable(self):
        with self.assertRaisesRegex(ValidationError, "没有进入实际查询词"):
            SourcePlan(
                task_type="analysis", evidence_needs=["人物动态"],
                use_knowledge=False, use_news=True, use_web=False, knowledge_query="",
                news_query="人工智能", additional_news_queries=[], news_people=["马斯克"],
                news_section="", news_time=NewsTimeSuggestion(
                    mode="unrestricted", start="", end="", reason="不限时间"),
                web_query="", plan_summary="查询人物动态",
            )

    def test_time_schema_normalizes_iso_timestamp_from_date_slots(self):
        value = NewsTimeSuggestion(
            mode="explicit", start="2026-09-29T12:00:00+08:00",
            end="2026-09-30T12:00:00+08:00", reason="过去24小时",
        )
        self.assertEqual((value.start, value.end), ("2026-09-29", "2026-09-30"))
        self.assertEqual(value.published_after, "2026-09-29T12:00:00+08:00")
        self.assertEqual(value.published_before, "2026-09-30T12:00:00+08:00")

    def test_probability_wording_forces_forecast_and_public_probability_search(self):
        plan = SourcePlan(
            task_type="analysis", evidence_needs=["宏观驱动"], additional_news_queries=[],
            use_knowledge=False, use_news=True, use_web=False, knowledge_query="",
            news_query="美联储", news_section="",
            news_time=NewsTimeSuggestion(mode="unrestricted", start="", end="", reason="不限时间"),
            web_query="", plan_summary="先看宏观驱动",
        )
        with patch.object(nodes, "get_router") as router:
            router.return_value.invoke.return_value = plan
            result = nodes.route({"question": "预测2027年第二季度，美国降息概率", "reading_recipe": "test"})
        self.assertEqual(result["estimate_kind"], "probability")
        self.assertEqual(result["task_type"], "forecast")
        self.assertEqual(set(result["selected_sources"]), {"vectorstore", "news_api", "web_search"})
        self.assertIn("计算方法", result["source_queries"]["web_search"])
        self.assertTrue(any("概率" in need for need in result["evidence_needs"]))
        decisions = {item["tool"]: item for item in result["tool_plan"]}
        self.assertTrue(all(decisions[name]["status"] == "required" for name in (
            "search_knowledge", "search_news", "search_web",
            "validate_model_output", "validate_probability_evidence", "validate_probability_answer",
        )))
        self.assertEqual(decisions["read_news"]["status"], "conditional")

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
            published_after="",
            published_before="",
            section="财经",
            source_names=[],
            sort_by="relevance",
            result_k=5,
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
