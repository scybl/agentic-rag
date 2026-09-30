"""不调用外部模型，验证完整预测流程的补搜和反思闭环。"""

from unittest.mock import patch

from langchain_core.documents import Document

from agentic_rag.graph import nodes
from agentic_rag.graph.build import build_graph
from agentic_rag.graph.chains import EvidenceAssessment, AnswerAssessment, DocumentAssessment
from agentic_rag.config import settings
from agentic_rag.research.store import ResearchStore


def evidence_review(ready=True):
    return EvidenceAssessment(
        ready=ready, usable_evidence_ids=["E1"], covered_factors=["价格基准 E1"],
        missing_factors=[] if ready else ["供给"],
        news_queries=[] if ready else ["能繁母猪"], web_queries=[], summary="需要核对供给" if not ready else "可作条件分析",
    )


def answer_review(complete=True):
    return AnswerAssessment(
        decision="accept" if complete else "revise",
        grounded=True, answers_question=complete, needs_more_evidence=False,
        issues=[] if complete else ["有依据却拒绝推断"],
        revision_instructions="" if complete else "保留事实，写条件预测",
        news_queries=[], web_queries=[],
        concern_checks=[{"concern_id": f"C{i}", "addressed": True, "explanation": "测试替身：答案已交代该项限制"} for i in range(1, 6)],
    )


def test_full_graph_rewrites_unnecessary_refusal_without_searching_again():
    document = Document(page_content="近期出栏价格上涨，需求保持稳定", metadata={"source_type": "news_api"})
    route_result = {"task_type": "forecast", "original_question": "预测2027猪肉价格",
                    "selected_sources": ["news_api"], "source_queries": {"news_api": "生猪"},
                    "evidence_needs": ["供需"], "retries": 0}
    with (patch.object(nodes, "route", return_value=route_result),
          patch.object(nodes, "collect_sources", return_value={"documents": [document]}),
          patch.object(nodes, "get_document_grader") as grader,
          patch.object(nodes, "get_evidence_assessor") as assessor,
          patch.object(nodes, "get_generator") as generator,
          patch.object(nodes, "get_answer_reviewer") as reviewer,
          patch.object(nodes, "AnalysisCache") as cache):
        cache.return_value.get.return_value = None
        grader.return_value.batch.return_value = [DocumentAssessment(decision="keep", subject_match="driver", evidence_role="fact", reason="有供需事实")]
        assessor.return_value.invoke.return_value = evidence_review()
        generator.return_value.invoke.side_effect = ["没有未来预测，不能回答 [E1]", "事实 [E1]。若需求持续，则可能上涨；供给增加则下行。"]
        reviewer.return_value.invoke.side_effect = [answer_review(False), answer_review()]
        result = build_graph(research=False).invoke({"question": "预测2027猪肉价格"})
    assert result["generation_complete"]
    assert result["answer_revisions"] == 1
    assert result["retries"] == 0
    assert "可能上涨" in result["generation"]
    assert "写条件预测" in generator.return_value.invoke.call_args.args[0]["revision_feedback"]
    assert grader.return_value.batch.call_count == 1
    assert cache.return_value.put.call_count == int(settings.analysis_cache_enabled)


def test_supplement_preserves_old_evidence_and_does_not_regrade_it():
    old = Document(page_content="价格基准", metadata={"source_type": "news_api", "article_id": "old"})
    new = Document(page_content="能繁母猪减少", metadata={"source_type": "news_api", "article_id": "new"})
    state = {"question": "猪肉预测", "task_type": "forecast", "documents": [old],
             "retries": 0, "query_history": ["news_api: 生猪"]}
    with (patch.object(nodes, "get_document_grader") as grader,
          patch.object(nodes, "get_evidence_assessor") as assessor,
          patch.object(nodes, "news_api", return_value={"documents": [new]}) as news):
        grader.return_value.batch.return_value = [DocumentAssessment(decision="keep", subject_match="driver", evidence_role="fact", reason="有供需事实")]
        state.update(nodes.grade_documents(state))
        assessor.return_value.invoke.return_value = evidence_review(False)
        state.update(nodes.assess_evidence(state))
        assert state["next_action"] == "supplement"
        state.update(nodes.supplement_sources(state))
        state.update(nodes.grade_documents(state))
    assert {d.metadata["article_id"] for d in state["documents"]} == {"old", "new"}
    assert state["retries"] == 1
    assert grader.return_value.batch.call_count == 2
    assert news.call_args.args[0]["news_search_plan"]["start"] == ""
    assert state["query_history"] == ["news_api: 生猪", "news_api: 能繁母猪"]


def test_insufficient_evidence_stops_supplementing_when_budget_exhausted():
    with patch.object(nodes, "get_evidence_assessor") as assessor:
        assessor.return_value.invoke.return_value = evidence_review(False)
        result = nodes.assess_evidence({"question": "预测", "documents": [], "retries": settings.max_retries})
    assert not result["evidence_assessment"]["ready"]
    assert result["evidence_assessment"]["usable_evidence_ids"] == []
    assert result["next_action"] == "generate"


def test_model_receives_candidate_coverage_without_unread_candidate_content():
    document = Document(page_content="原文事实", metadata={"source_type": "news_api"})
    with patch.object(nodes, "get_evidence_assessor") as assessor:
        assessor.return_value.invoke.return_value = evidence_review()
        result = nodes.assess_evidence({"question": "新闻综述", "documents": [document], "news_trace": [
            {"kind": "news_candidates", "candidates": [{"summary": "未阅读的候选，不得作为证据"}]},
            {"kind": "news_ranked", "candidate_count": 120, "selected_count": 8,
             "deferred_count": 112, "retrieval_complete": False, "ranking_method": "tfidf"},
        ]})
    context = assessor.return_value.invoke.call_args.args[0]["documents"]
    assert '120' in context and '112' in context and '不代表正文全部阅读' in context
    assert '未阅读的候选，不得作为证据' not in context
    assert result['evidence_assessment']['news_retrieval'][0]['retrieval_complete'] is False


def test_probability_evidence_contract_forces_supplement_for_search_snippet():
    document = Document(
        page_content="FedWatch显示降息概率为45%",
        metadata={"source_type": "web_search", "content_kind": "search_snippet", "title": "FedWatch"},
    )
    events = []
    with patch.object(nodes, "get_evidence_assessor") as assessor:
        assessor.return_value.invoke.return_value = evidence_review()
        result = nodes.assess_evidence({
            "question": "预测2027年第二季度，美国降息概率",
            "task_type": "forecast", "estimate_kind": "probability",
            "documents": [document], "retries": 0, "_event_writer": events.append,
        })
    assert not result["evidence_assessment"]["probability_contract"]["passed"]
    assert result["next_action"] == "supplement"
    assert result["pending_web_queries"]
    started = next(event for event in events if event.get("phase") == "started")
    assert started["tool"] == "validate_probability_evidence"
    assert started["caller"] == "主流程/概率证据契约"


def test_probability_contract_uses_read_news_before_repeating_network_search(tmp_path):
    database = ResearchStore(tmp_path / "research.sqlite")
    database.start_run("run", "预测2027年第二季度，美国降息概率")
    original = Document(
        page_content="截至2026-09-30，联邦基金期货隐含概率为45%。",
        metadata={"source_type": "news_api", "content_kind": "news_article", "title": "FedWatch数据"},
    )
    article = database.register(original, 2200)
    note = Document(
        page_content="摘要只说明市场关注降息。",
        metadata={"source_type": "news_api", "content_kind": "news_article", "title": "FedWatch数据",
                  "memory_version": article["version"]},
    )
    with (patch.object(nodes, "get_evidence_assessor") as assessor,
          patch("agentic_rag.research.service.store", return_value=database)):
        assessor.return_value.invoke.return_value = evidence_review()
        result = nodes.assess_evidence({
            "question": "预测2027年第二季度，美国降息概率",
            "task_type": "forecast", "estimate_kind": "probability",
            "target_event": "2027年第二季度至少降息一次",
            "documents": [note], "retries": 0, "run_id": "run", "reading_recipe": "test",
        })
    assert result["evidence_assessment"]["probability_contract"]["passed"]
    assert result["next_action"] == "generate"
    assert "定向原文回读" in result["evidence_context"]
    started = [event["tool"] for event in database.events("run") if event.get("phase") == "started"]
    assert started == ["validate_probability_evidence", "read_news", "validate_probability_evidence"]
